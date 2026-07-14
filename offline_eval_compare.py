"""
Offline evaluation comparison: PID / IQL / DT / GAVE / DGAB using AuctionNet's offline test pipeline.

Uses TestDataLoader + OfflineEnv — compares bids against historical leastWinningCost.
No live competition; opponent behavior is frozen from the logged data.

Usage:
    # Single file
    python offline_eval_compare.py --data ./strategy_train_env/data/traffic/period-12.csv

    # Batch: all CSV files in a folder
    python offline_eval_compare.py --data_dir ./strategy_train_env/data/traffic/

    # Custom settings
    python offline_eval_compare.py --data_dir ./data/logs/ --agents GAVE,DGAB --advertisers 0,4,24 --output my_results
"""
import sys, os, glob, time, argparse, logging
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'strategy_train_env'))

# ── Local imports (no autobidding dependency) ──
from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from bidding_train_env.offline_eval.offline_env import OfflineEnv

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# Config (defaults — override via CLI)
# ═══════════════════════════════════════════════

BLOCK_CONFIG = {
    'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
    'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
    'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
}

GAVE_SAVE_DIR = './saved_model/gave_20k_dense'
DGAB_SAVE_DIR = './saved_model/dgab_v3'
DT_SAVE_DIR   = './saved_model/DTdense'
IQL_SAVE_DIR  = './strategy_train_env/saved_model/IQL_4gpu'

FIXED_SEED = 42

# ═══════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════

def getScore_neurips(reward, cpa, cpa_constraint):
    beta = 2
    if cpa > cpa_constraint:
        penalty = (cpa_constraint / (cpa + 1e-10)) ** beta
    else:
        penalty = 1.0
    return penalty * reward


# ═══════════════════════════════════════════════
# Offline evaluation runner
# ═══════════════════════════════════════════════

def run_offline_eval(agent, data_path, advertiser_key):
    """Run offline evaluation for one advertiser with the given agent."""
    data_loader = TestDataLoader(file_path=data_path)
    env = OfflineEnv()

    num_ticks, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(advertiser_key)

    history = {
        'historyBids': [],
        'historyAuctionResult': [],
        'historyImpressionResult': [],
        'historyLeastWinningCost': [],
        'historyPValueInfo': []
    }
    rewards = np.zeros(num_ticks)
    total_cost = 0.0

    for tick in range(num_ticks):
        pv = pValues[tick]
        sigma = pValueSigmas[tick]
        lwc = leastWinningCosts[tick]

        if agent.remaining_budget < env.min_remaining_budget:
            bid = np.zeros(len(pv))
        else:
            bid = agent.bidding(
                tick, pv, sigma,
                history["historyPValueInfo"],
                history["historyBids"],
                history["historyAuctionResult"],
                history["historyImpressionResult"],
                history["historyLeastWinningCost"])

        # Over-cost handling
        over_cost_ratio = max((np.sum(bid > lwc) * np.mean(lwc) - agent.remaining_budget) /
                            (np.sum(bid > lwc) * np.mean(lwc) + 1e-4), 0)
        loop = 0
        while over_cost_ratio > 0 and loop < 5:
            pv_index = np.where(bid >= lwc)[0]
            if len(pv_index) == 0:
                break
            drop_n = max(1, int(np.ceil(len(pv_index) * over_cost_ratio)))
            dropped = np.random.choice(pv_index, min(drop_n, len(pv_index)), replace=False)
            bid[dropped] = 0
            over_cost_ratio = max((np.sum(bid > lwc) * np.mean(lwc) - agent.remaining_budget) /
                                (np.sum(bid > lwc) * np.mean(lwc) + 1e-4), 0)
            loop += 1

        tick_value, tick_cost_vec, tick_status, tick_conversion = env.simulate_ad_bidding(
            pv, sigma, bid, lwc)

        tick_cost = np.sum(tick_cost_vec)
        agent.remaining_budget -= tick_cost
        total_cost += tick_cost
        rewards[tick] = np.sum(tick_conversion)

        history["historyPValueInfo"].append(
            np.array([(pv[i], sigma[i]) for i in range(len(pv))]))
        history["historyBids"].append(bid)
        history["historyLeastWinningCost"].append(lwc)
        history["historyAuctionResult"].append(
            np.array([(tick_status[i], tick_status[i], tick_cost_vec[i]) for i in range(len(tick_status))]))
        history["historyImpressionResult"].append(
            np.array([(tick_conversion[i], tick_conversion[i]) for i in range(len(tick_conversion))]))

    all_reward = np.sum(rewards)
    cpa_real = total_cost / (all_reward + 1e-10)
    score = getScore_neurips(all_reward, cpa_real, agent.cpa)

    return {
        'reward': int(all_reward),
        'cost': total_cost,
        'cpa_real': cpa_real,
        'cpa_target': agent.cpa,
        'score': score,
        'budget_used': total_cost / agent.budget if agent.budget > 0 else 0,
    }


# ═══════════════════════════════════════════════
# Agent builders
# ═══════════════════════════════════════════════

def build_agent(name, budget, cpa, category, device='cpu'):
    if name == 'PID':
        from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
        return PidBiddingStrategy(budget=budget, cpa=cpa, category=category,
                                  name='PID', exp_tempral_ratio=np.ones(48))
    elif name == 'IQL':
        from simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
        return IqlBiddingStrategy(budget=budget, cpa=cpa, category=category, name='IQL',
                                  model_dir=IQL_SAVE_DIR, device=device)
    elif name == 'DT':
        from simul_bidding_env.strategy.autobidding_agents import DTAuctionNetAgent
        return DTAuctionNetAgent(
            budget=budget, cpa=cpa, category=category, name='DT',
            model_param=dict(save_dir=DT_SAVE_DIR, device=device,
                            target_return=4, scale=2000))
    elif name == 'GAVE':
        from simul_bidding_env.strategy.autobidding_agents import GAVEAuctionNetAgent
        return GAVEAuctionNetAgent(
            budget=budget, cpa=cpa, category=category, name='GAVE',
            model_param=dict(save_dir=GAVE_SAVE_DIR, hidden_size=512, time_dim=8,
                            block_config=BLOCK_CONFIG, device=device,
                            expectile=0.99, score_target_mode='prev'))
    elif name == 'DGAB':
        from simul_bidding_env.strategy.autobidding_agents import DGABFOAuctionNetAgent
        return DGABFOAuctionNetAgent(
            budget=budget, cpa=cpa, category=category, name='DGAB',
            model_param=dict(save_dir=DGAB_SAVE_DIR, hidden_size=512, max_ep_len=96,
                            time_dim=8, block_config=BLOCK_CONFIG, device=device,
                            actor_type='stack', critic_type='sequence'))
    else:
        raise ValueError(f'Unknown agent: {name}')


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description='Offline evaluation comparison')
    ap.add_argument('--data', type=str, default=None,
                    help='Single CSV file to evaluate')
    ap.add_argument('--data_dir', type=str, default=None,
                    help='Folder of CSV files to evaluate (overrides --data)')
    ap.add_argument('--pattern', type=str, default='*.csv',
                    help='Glob pattern for --data_dir (default: *.csv)')
    ap.add_argument('--advertisers', type=str, default='all',
                    help='Comma-separated advertiser indices, or "all"')
    ap.add_argument('--agents', type=str, default='PID,IQL,DT,GAVE,DGAB',
                    help='Comma-separated agent names')
    ap.add_argument('--output', type=str, default='offline_eval',
                    help='Output file prefix (default: offline_eval)')
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: {project}/exp_data)')
    ap.add_argument('--device', type=str, default=None,
                    help='Device (default: cuda:0 if available else cpu)')
    ap.add_argument('--seed', type=int, default=FIXED_SEED)
    # Model paths
    ap.add_argument('--gave_dir', type=str, default=None)
    ap.add_argument('--dgab_dir', type=str, default=None)
    ap.add_argument('--dt_dir', type=str, default=None)
    ap.add_argument('--iql_dir', type=str, default=None)
    return ap.parse_args()


def collect_data_files(data_path, data_dir, pattern):
    """Return list of (label, path) tuples."""
    if data_dir:
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not files:
            logger.error(f'No files matching "{pattern}" in {data_dir}')
            sys.exit(1)
        logger.info(f'Found {len(files)} files in {data_dir}')
        return [(os.path.basename(f), f) for f in files]
    elif data_path:
        return [(os.path.basename(data_path), data_path)]
    else:
        logger.error('Must specify --data or --data_dir')
        sys.exit(1)


def main():
    global GAVE_SAVE_DIR, DGAB_SAVE_DIR, DT_SAVE_DIR, IQL_SAVE_DIR

    args = parse_args()

    DEVICE = args.device or ('cuda:0' if __import__('torch').cuda.is_available() else 'cpu')
    if args.gave_dir:   GAVE_SAVE_DIR = args.gave_dir
    if args.dgab_dir:   DGAB_SAVE_DIR = args.dgab_dir
    if args.dt_dir:     DT_SAVE_DIR   = args.dt_dir
    if args.iql_dir:    IQL_SAVE_DIR  = args.iql_dir

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent_names = [s.strip() for s in args.agents.split(',')]
    data_files = collect_data_files(args.data, args.data_dir, args.pattern)

    logger.info(f'Device: {DEVICE}')
    logger.info(f'Agents: {agent_names}')
    logger.info(f'Data files: {[f[0] for f in data_files]}')

    # ── Output dir ──
    out_dir = args.output_dir or os.path.join(_PROJECT_ROOT, 'exp_data')
    os.makedirs(out_dir, exist_ok=True)

    all_results = []  # flat: one row per (data_file, advertiser, agent)
    total_runs = 0
    t0 = time.time()

    for file_label, file_path in data_files:
        logger.info(f'\n{"="*60}\n  DATA: {file_label}\n{"="*60}')

        loader = TestDataLoader(file_path=file_path)
        all_keys = loader.keys

        if args.advertisers == 'all':
            adv_set = sorted(set(k[1] for k in all_keys))
        else:
            adv_set = [int(x) for x in args.advertisers.split(',')]

        # Read budget/CPA from this data file
        raw = pd.read_csv(file_path)
        adv_info = {}
        for adv in adv_set:
            match = raw[raw['advertiserNumber'] == adv]
            if match.empty:
                logger.warning(f'  Advertiser #{adv} not found in {file_label}, skipping')
                continue
            row = match.iloc[0]
            adv_info[adv] = {
                'budget': row['budget'],
                'cpa': row['CPAConstraint'],
                'category': int(row['advertiserCategoryIndex']),
            }

        for adv in adv_set:
            if adv not in adv_info:
                continue
            info = adv_info[adv]
            # Use the first matching key for this advertiser
            try:
                key = next(k for k in all_keys if k[1] == adv)
            except StopIteration:
                logger.warning(f'  No key for advertiser #{adv} in {file_label}, skipping')
                continue

            logger.info(f'  Advertiser #{adv} (budget={info["budget"]}, cpa={info["cpa"]})')

            for agent_name in agent_names:
                try:
                    agent = build_agent(agent_name, info['budget'], info['cpa'],
                                       info['category'], device=DEVICE)
                except Exception as e:
                    logger.error(f'    {agent_name}: build FAILED — {e}')
                    all_results.append({
                        'data_file': file_label, 'advertiser': adv,
                        'agent': agent_name, 'score': 0, 'reward': 0,
                        'cost': 0, 'cpa_real': float('inf'),
                        'budget_used': 0,
                    })
                    total_runs += 1
                    continue

                agent.remaining_budget = agent.budget
                try:
                    result = run_offline_eval(agent, file_path, key)
                except Exception as e:
                    logger.error(f'    {agent_name}: eval FAILED — {e}')
                    result = {'score': 0, 'reward': 0, 'cost': 0,
                              'cpa_real': float('inf'), 'budget_used': 0}

                total_runs += 1
                result['data_file'] = file_label
                result['advertiser'] = adv
                result['agent'] = agent_name
                all_results.append(result)

                logger.info(f'    {agent_name:6s}: score={result["score"]:.2f}  '
                           f'reward={result["reward"]}  cpa={result["cpa_real"]:.2f}  '
                           f'budget={result["budget_used"]:.0%}')

    elapsed = time.time() - t0
    logger.info(f'\nTotal: {total_runs} runs in {elapsed:.0f}s')

    # ── Save results ──
    df = pd.DataFrame(all_results)
    cols = ['data_file', 'advertiser', 'agent', 'score', 'reward', 'cost',
            'cpa_real', 'budget_used']
    df = df[cols]

    # Detailed results (one row per run)
    out_detailed = os.path.join(out_dir, f'{args.output}_detailed.csv')
    df.to_csv(out_detailed, index=False)
    logger.info(f'Detailed results saved to {out_detailed}')

    # Summary: avg score per (data_file, agent)
    metric_cols = ['score', 'reward', 'cost', 'cpa_real', 'budget_used']
    summary = df.groupby(['data_file', 'agent'])[metric_cols].mean().reset_index()
    out_summary = os.path.join(out_dir, f'{args.output}_summary.csv')
    summary.to_csv(out_summary, index=False)
    logger.info(f'Summary saved to {out_summary}')

    # Pivot: data_file × agent → avg score
    pivot = df.groupby(['data_file', 'agent'])['score'].mean().unstack('agent')
    out_pivot = os.path.join(out_dir, f'{args.output}_pivot.csv')
    pivot.to_csv(out_pivot)
    logger.info(f'Pivot saved to {out_pivot}')

    # ── Console summary ──
    print(f'\n{"="*100}')
    print(f'OFFLINE EVALUATION — {len(data_files)} files × {len(agent_names)} agents')
    print(f'{"="*100}')
    print(pivot.to_string())
    print(f'\nDone. Results in {out_dir}/')


if __name__ == '__main__':
    main()
