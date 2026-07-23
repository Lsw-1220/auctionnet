r"""
Multi-strategy online benchmark: PID / IQL / DT / GAVE / DGAB / GUIDE across pvalue_mean_base levels.

Usage:
    # Full run (8 pv x 48 adv x 6 strategies = 2304 episodes)
    python benchmark_multistrat.py

    # Quick test: 1 pv, 1 advertiser
    python benchmark_multistrat.py --pv 0.001 --advertisers 0

    # Custom subset with server paths
    python benchmark_multistrat.py \
        --gave_dir /data/models/gave_20k_dense \
        --dgab_dir /data/models/dgab_v3 \
        --dt_dir   /data/models/DTdense \
        --iql_dir  /data/models/IQL_4gpu \
        --guide_dir /data/models/GUIDE \
        --output_dir /data/results \
        --output my_test

Output:
    {output_dir}/{prefix}_results.csv      — flat: one row per (pv, strategy, advertiser)
    {output_dir}/{prefix}_summary.csv      — aggregated avg per (pv, strategy)
    {output_dir}/{prefix}_comparison.csv   — pivot: rows=pvalue_mean_base, cols=strategy, val=avg score
"""
import sys, os, time, argparse, logging
import numpy as np
import gin
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ──
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'strategy_train_env'))
sys.path.insert(0, _PROJECT_ROOT)

# ── Pre-import modules that gin needs ──
import run.run_test
import simul_bidding_env.Controller.Controller
import simul_bidding_env.Environment.BiddingEnv

# ── Gin config ──
gin.parse_config_files_and_bindings(["./config/test.gin"], None)

from run.run_test import adjust_over_cost, get_winner
from simul_bidding_env.Controller.Controller import Controller
from simul_bidding_env.strategy.autobidding_agents import (
    GAVEAuctionNetAgent, DGABFOAuctionNetAgent, DTAuctionNetAgent,
)

# ═══════════════════════════════════════════════
# Config (defaults — override via CLI)
# ═══════════════════════════════════════════════

BLOCK_CONFIG = {
    'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
    'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
    'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
}

GAVE_SAVE_DIR = './saved_model/gave_20k_dense'
DGAB_SAVE_DIR   = './saved_model/dgab_v3'
DT_SAVE_DIR     = './saved_model/DTdense'
IQL_SAVE_DIR    = './strategy_train_env/saved_model/IQL_4gpu'
GUIDE_SAVE_DIR  = './strategy_train_env/saved_model/GUIDE'
OUTPUT_DIR      = None  # None → use {_PROJECT_ROOT}/exp_data

NUM_ADVERTISERS = 48
NUM_TICK = 48
FIXED_SEED = 42
RTG_V_CAP = 10

DEFAULT_PV_SWEEP = [0.0003, 0.0005, 0.0007, 0.0009, 0.001, 0.003, 0.005, 0.007]

# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description='Multi-strategy online benchmark')
    ap.add_argument('--pv', nargs='*', type=float, default=None,
                    help='PV values to sweep (default: 8-level sweep)')
    ap.add_argument('--advertisers', nargs='*', type=int, default=None,
                    help='Advertiser indices to test (default: all 48)')
    ap.add_argument('--output', type=str, default='benchmark_multistrat',
                    help='Output file prefix (default: benchmark_multistrat)')
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: {project}/exp_data)')
    ap.add_argument('--device', type=str, default=None,
                    help='Device (default: cuda:0 if available else cpu)')
    ap.add_argument('--seed', type=int, default=FIXED_SEED,
                    help='Random seed (default: 42)')
    # Model paths
    ap.add_argument('--gave_dir', type=str, default=None,
                    help='GAVE model directory')
    ap.add_argument('--dgab_dir', type=str, default=None,
                    help='DGAB model directory')
    ap.add_argument('--dt_dir', type=str, default=None,
                    help='DT model directory')
    ap.add_argument('--iql_dir', type=str, default=None,
                    help='IQL model directory')
    ap.add_argument('--guide_dir', type=str, default=None,
                    help='GUIDE model directory')
    return ap.parse_args()


# ═══════════════════════════════════════════════
# Strategy factories
# ═══════════════════════════════════════════════

def make_pid(budget, cpa, category, **kw):
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    return PidBiddingStrategy(
        budget=budget, cpa=cpa, category=category,
        name='PID', exp_tempral_ratio=np.ones(48),
    )


def make_iql(budget, cpa, category, **kw):
    from simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
    return IqlBiddingStrategy(budget=budget, cpa=cpa, category=category, name='IQL',
                              model_dir=IQL_SAVE_DIR, device=DEVICE)


def make_dt(budget, cpa, category, **kw):
    return DTAuctionNetAgent(
        budget=budget, cpa=cpa, category=category,
        name='DT-Player',
        model_param=dict(
            save_dir=DT_SAVE_DIR,
            device=DEVICE,
            target_return=4,
            scale=2000,
        ),
    )


def make_gave(budget, cpa, category, **kw):
    return GAVEAuctionNetAgent(
        budget=budget, cpa=cpa, category=category,
        name='GAVE-Player',
        model_param=dict(
            save_dir=GAVE_SAVE_DIR,
            hidden_size=512, time_dim=8,
            block_config=BLOCK_CONFIG,
            device=DEVICE,
            expectile=0.99,
            score_target_mode='prev',
        ),
    )


def make_dgab(budget, cpa, category, pvalue_mean_base=None, **kw):
    mp = dict(
        save_dir=DGAB_SAVE_DIR,
        hidden_size=512, max_ep_len=96, time_dim=8,
        block_config=BLOCK_CONFIG,
        device=DEVICE,
        actor_type='stack',
        critic_type='sequence',
        critic_alpha=1.0,
        capture_attention=False,
        rtg_v_cap=RTG_V_CAP,
    )
    if pvalue_mean_base is not None:
        mp['pvalue_mean_base'] = pvalue_mean_base
    return DGABFOAuctionNetAgent(
        budget=budget, cpa=cpa, category=category,
        name='DGAB-FO-Player',
        model_param=mp,
    )


def make_guide(budget, cpa, category, **kw):
    from simul_bidding_env.strategy.guide_bidding_strategy import GUIDEStrategy
    return GUIDEStrategy(
        budget=budget, cpa=cpa, category=category,
        name='GUIDE-Player',
        model_dir=GUIDE_SAVE_DIR,
        device=DEVICE,
    )


STRATEGIES = [
    ('PID',   make_pid),
    ('IQL',   make_iql),
    ('DT',    make_dt),
    ('GAVE',  make_gave),
    ('DGAB',  make_dgab),
    ('GUIDE', make_guide),
]


# ═══════════════════════════════════════════════
# Single-episode runner (simplified — no tick logging)
# ═══════════════════════════════════════════════

def run_one_episode(controller, player_index, agent_factory, pvalue_mean_base):
    """Run one 48-tick episode and return summary metrics dict."""
    envs = controller.biddingEnv
    pv_generator = controller.pvGenerator

    # ── Create and inject player agent ──
    player_agent = agent_factory(
        budget=controller.budget_list[player_index],
        cpa=controller.cpa_constraint_list[player_index],
        category=controller.category[player_index],
        pvalue_mean_base=pvalue_mean_base,
    )
    controller.player_agent = player_agent
    agents = controller.load_agents()

    # ── Reset ──
    controller.reset(episode=0)
    if pvalue_mean_base is not None:
        pv_generator.pvalue_mean_base = pvalue_mean_base
        pv_generator.pv_values, pv_generator.PValueSigmas = pv_generator.generate()

    # ── Init budgets ──
    num_agent = len(agents)
    agents_cpa = np.array([agent.cpa for agent in agents])
    budgets = np.array([agent.budget for agent in agents])
    for i in range(num_agent):
        agents[i].remaining_budget = budgets[i]

    rewards = np.zeros(num_agent)
    costs = np.zeros(num_agent)

    # History buffers (needed for agent.bidding())
    history_pvalue_infos = []
    history_bids = []
    history_auction_results = []
    history_impression_results = []
    history_least_winning_costs = []

    # ── Tick loop ──
    for tick_index in range(NUM_TICK):
        pv_values = pv_generator.pv_values[tick_index]
        pvalue_sigmas = pv_generator.PValueSigmas[tick_index]

        bids = [
            agent.bidding(
                tick_index,
                pv_values[:, i],
                pvalue_sigmas[:, i],
                [x[i] for x in history_pvalue_infos],
                [x[i] for x in history_bids],
                [x[i] for x in history_auction_results],
                [x[i] for x in history_impression_results],
                history_least_winning_costs
            ) if agent.remaining_budget >= envs.min_remaining_budget
            else np.zeros(pv_values.shape[0])
            for i, agent in enumerate(agents)
        ]

        bids = np.array(bids).transpose()
        bids[bids < 0] = 0

        remaining_budget_list = np.array([agent.remaining_budget for agent in agents])

        # Over-cost adjustment loop
        ratio_max = None
        while ratio_max is None or ratio_max > 0:
            if ratio_max and ratio_max > 0:
                over_cost_ratio = np.maximum(
                    (cost - remaining_budget_list) / (cost + 1e-4), 0)
                adjust_over_cost(bids, over_cost_ratio, envs.slot_coefficients, winner_pit)

            (xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit,
             least_winning_cost_pit, market_price_pit) = \
                envs.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)

            real_cost = cost_pit * is_exposed_pit
            cost = real_cost.sum(axis=1)
            reward = conversion_action_pit.sum(axis=1)

            winner_pit = get_winner(slot_pit)
            over_cost_ratio = np.maximum(
                (cost - remaining_budget_list) / (cost + 1e-4), 0)
            ratio_max = over_cost_ratio.max()

        for i, agent in enumerate(agents):
            agent.remaining_budget -= cost[i]

        rewards += reward
        costs += cost

        # Append history
        history_bids.append(bids.transpose())
        history_least_winning_costs.append(least_winning_cost_pit)
        history_pvalue_infos.append(np.stack((pv_values.T, pvalue_sigmas.T), axis=-1))
        history_auction_results.append(np.stack((xi_pit, slot_pit, cost_pit), axis=-1))
        history_impression_results.append(
            np.stack((is_exposed_pit, conversion_action_pit), axis=-1))

    # ── Compute metrics ──
    player_reward = rewards[player_index]
    player_cost = costs[player_index]
    player_cpa_real = player_cost / (player_reward + 1e-10)
    player_cpa_target = agents_cpa[player_index]

    # NeurIPS score
    beta = 2
    if player_cpa_real > player_cpa_target:
        penalty = (player_cpa_target / (player_cpa_real + 1e-10)) ** beta
    else:
        penalty = 1.0
    player_score = penalty * player_reward

    return {
        'score': player_score,
        'reward': int(player_reward),
        'cost': player_cost,
        'cpa_real': player_cpa_real,
        'cpa_target': player_cpa_target,
        'budget_used': player_cost / budgets[player_index],
    }


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    global DEVICE, GAVE_SAVE_DIR, DGAB_SAVE_DIR, DT_SAVE_DIR, IQL_SAVE_DIR, GUIDE_SAVE_DIR, OUTPUT_DIR

    args = parse_args()

    DEVICE = args.device or ('cuda:0' if __import__('torch').cuda.is_available() else 'cpu')
    pv_sweep = args.pv or DEFAULT_PV_SWEEP
    advertisers = args.advertisers if args.advertisers is not None else list(range(NUM_ADVERTISERS))
    output_prefix = args.output
    seed = args.seed

    # Apply CLI path overrides
    if args.gave_dir:   GAVE_SAVE_DIR = args.gave_dir
    if args.dgab_dir:   DGAB_SAVE_DIR = args.dgab_dir
    if args.dt_dir:     DT_SAVE_DIR   = args.dt_dir
    if args.iql_dir:    IQL_SAVE_DIR    = args.iql_dir
    if args.guide_dir:  GUIDE_SAVE_DIR  = args.guide_dir
    OUTPUT_DIR = args.output_dir  # None → use default below

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    logger.info(f'Device: {DEVICE}  Seed: {seed}')
    logger.info(f'PV sweep ({len(pv_sweep)}): {pv_sweep}')
    logger.info(f'Advertisers ({len(advertisers)}): {advertisers if len(advertisers) <= 10 else f"{advertisers[:5]}...{advertisers[-2:]}" }')
    logger.info(f'Strategies: {[n for n, _ in STRATEGIES]}')
    logger.info(f'Total episodes: {len(pv_sweep) * len(advertisers) * len(STRATEGIES)}')
    logger.info(f'Model dirs — GAVE: {GAVE_SAVE_DIR}  DGAB: {DGAB_SAVE_DIR}  DT: {DT_SAVE_DIR}  IQL: {IQL_SAVE_DIR}  GUIDE: {GUIDE_SAVE_DIR}')

    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    dummy_agent = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
    dummy_agent.name += "0"

    # ── Main loop ──
    results = []
    total = len(pv_sweep) * len(advertisers) * len(STRATEGIES)
    done = 0

    for pv_val in pv_sweep:
        logger.info(f'\n{"#"*60}\n  PVALUE_MEAN_BASE = {pv_val:.4f}\n{"#"*60}')

        # One Controller per pv_val (shares same opponent config)
        controller = Controller(player_index=0, player_agent=dummy_agent)

        for adv in advertisers:
            for name, factory in STRATEGIES:
                # Switch player_index by recreating controller with new index
                # (Controller hardcodes player_index at construction)
                controller = Controller(player_index=adv, player_agent=dummy_agent)

                t0 = time.time()
                try:
                    res = run_one_episode(controller, adv, factory, pv_val)
                except Exception as e:
                    logger.error(f'  [{name}] adv={adv} pv={pv_val} FAILED: {e}')
                    res = {'score': 0, 'reward': 0, 'cost': 0,
                           'cpa_real': float('inf'), 'cpa_target': 0, 'budget_used': 0}

                elapsed = time.time() - t0
                done += 1
                res['pvalue_mean_base'] = pv_val
                res['strategy'] = name
                res['advertiser'] = adv
                results.append(res)

                if True:  # log every episode
                    logger.info(f'  [{done}/{total}] {name} adv={adv} pv={pv_val:.4f} '
                                f'score={res["score"]:.1f} reward={res["reward"]} '
                                f'cpa={res["cpa_real"]:.2f} ({elapsed:.1f}s)')

    # ── Save results ──
    out_dir = OUTPUT_DIR if OUTPUT_DIR else os.path.join(_PROJECT_ROOT, 'exp_data')
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(results)
    out_results = os.path.join(out_dir, f'{output_prefix}_results.csv')
    df.to_csv(out_results, index=False)
    logger.info(f'Results saved to {out_results}')

    # Summary: avg per (pv, strategy)
    metric_cols = ['score', 'reward', 'cost', 'cpa_real', 'budget_used']
    summary = df.groupby(['pvalue_mean_base', 'strategy'])[metric_cols].mean().reset_index()
    out_summary = os.path.join(out_dir, f'{output_prefix}_summary.csv')
    summary.to_csv(out_summary, index=False)
    logger.info(f'Summary saved to {out_summary}')

    # Comparison pivot: avg score by (pv x strategy)
    pivot = df.groupby(['pvalue_mean_base', 'strategy'])['score'].mean().unstack('strategy')
    out_comp = os.path.join(out_dir, f'{output_prefix}_comparison.csv')
    pivot.to_csv(out_comp)
    logger.info(f'Comparison pivot saved to {out_comp}')

    print('\n=== Avg Score Comparison ===')
    print(pivot.to_string())
    print(f'\nDone. {done} episodes total.')


if __name__ == '__main__':
    main()
