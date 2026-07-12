"""
General benchmark: compare strategies vs original on multiple period CSVs.

Usage:
    python benchmark_general.py data/traffic/period-7.csv data/traffic/period-8.csv
    python benchmark_general.py strategy_train_env/data/traffic/period-*.csv
"""

import sys, os, time, argparse, glob
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'strategy_train_env'))
sys.path.insert(0, _PROJECT_ROOT)

DGAB_DIR = 'D:/research/Experiment/autobidding/saved_model/dgab_400k_sparse'
GAVE_DIR = 'D:/research/Experiment/autobidding/saved_model/gave_400k_sparse'
DT_DIR = './saved_model/DTtest'
DEVICE = 'cuda:0' if __import__('torch').cuda.is_available() else 'cpu'
BLOCK_CONFIG = {
    'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
    'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
    'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
}
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'exp_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Strategy factories ────────────────────────────

def make_dgab(budget, cpa, category):
    from simul_bidding_env.strategy.autobidding_agents import DGABFOAuctionNetAgent
    return DGABFOAuctionNetAgent(
        budget=budget, cpa=cpa, category=category, name='DGAB',
        model_param=dict(save_dir=DGAB_DIR, hidden_size=512, max_ep_len=96,
                         time_dim=8, block_config=BLOCK_CONFIG, device=DEVICE,
                         actor_type='stack', critic_type='sequence'))

def make_iql(budget, cpa, category):
    from simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
    return IqlBiddingStrategy(budget=budget, cpa=cpa, category=category, name='IQL')

def make_dt(budget, cpa, category):
    from simul_bidding_env.strategy.autobidding_agents import DTAuctionNetAgent
    return DTAuctionNetAgent(
        budget=budget, cpa=cpa, category=category, name='DT',
        model_param=dict(save_dir=DT_DIR, device=DEVICE, target_return=4, scale=2000))

def make_gave(budget, cpa, category):
    from simul_bidding_env.strategy.autobidding_agents import GAVEAuctionNetAgent
    return GAVEAuctionNetAgent(
        budget=budget, cpa=cpa, category=category, name='GAVE',
        model_param=dict(save_dir=GAVE_DIR, hidden_size=512, time_dim=8,
                         block_config=BLOCK_CONFIG, device=DEVICE,
                         expectile=0.99, score_target_mode='prev'))

#STRATEGIES = [('DGAB', make_dgab), ('GAVE', make_gave), ('IQL', make_iql), ('DT', make_dt)]
STRATEGIES = [ ('GAVE', make_gave)]


# ── Helpers ────────────────────────────────────────

def extract_period_number(csv_path):
    """Extract period number from filename like period-7.csv → 7."""
    import re
    m = re.search(r'period[_-](\d+)', os.path.basename(csv_path))
    return int(m.group(1)) if m else -1


def load_pv_data(csv_path):
    """Reconstruct per-tick pv_values (n_pv, 48) and pValueSigmas from CSV."""
    df = pd.read_csv(csv_path)
    ticks = sorted(df['timeStepIndex'].unique())
    pv_values_list, pv_sigmas_list = [], []
    for tick in ticks:
        td = df[df['timeStepIndex'] == tick]
        pv_v = td.pivot_table(index='pvIndex', columns='advertiserNumber',
                              values='pValue', aggfunc='first').values.astype(np.float64)
        pv_s = td.pivot_table(index='pvIndex', columns='advertiserNumber',
                              values='pValueSigma', aggfunc='first').values.astype(np.float64)
        pv_values_list.append(pv_v)
        pv_sigmas_list.append(pv_s)
    return pv_values_list, pv_sigmas_list, int(df['advertiserNumber'].nunique()), len(ticks), df


def extract_original_performance(df):
    """Extract per-advertiser original metrics."""
    gw = df[df['isExposed'] == 1]
    total_cost = gw.groupby('advertiserNumber')['cost'].sum()
    total_reward = df.groupby('advertiserNumber')['conversionAction'].sum()
    budget = df.groupby('advertiserNumber')['budget'].first()
    cpa_target = df.groupby('advertiserNumber')['CPAConstraint'].first()
    cpa_real = total_cost / (total_reward + 1e-10)
    penalty = np.where(cpa_real > cpa_target,
                       (cpa_target / (cpa_real + 1e-10)) ** 2, 1.0)
    score = penalty * total_reward
    return score, cpa_real, total_cost / budget * 100, total_reward, total_cost, penalty


# ── Run replay ─────────────────────────────────────

def run_replay(strategy_factory, player_index, pv_values_list, pv_sigmas_list,
               budgets, cpas, categories, num_ticks, num_agents):
    from simul_bidding_env.Environment.BiddingEnv import BiddingEnv
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    from simul_bidding_env.strategy.player_agent_wrapper import PlayerAgentWrapper

    agents = []
    for i in range(num_agents):
        a = PidBiddingStrategy(budget=budgets[i], cpa=cpas[i], category=categories[i],
                               name=f'PID-{i}', exp_tempral_ratio=np.ones(num_agents))
        a.budget = budgets[i]; a.cpa = cpas[i]; a.category = categories[i]
        a.remaining_budget = budgets[i]
        agents.append(a)

    player = strategy_factory(budget=budgets[player_index],
                              cpa=cpas[player_index], category=categories[player_index])
    agents[player_index] = PlayerAgentWrapper(player_agent=player)

    envs = BiddingEnv(); envs.reset(episode=0)
    rewards = np.zeros(num_agents); costs = np.zeros(num_agents)
    hp, hb, ha, hi, hl = [], [], [], [], []
    tick_recs = []

    for tick in range(num_ticks):
        pv_v = pv_values_list[tick]; pv_s = pv_sigmas_list[tick]
        pv_s = np.maximum(np.nan_to_num(pv_s, nan=0.0), 1e-8)

        bids = []
        for i, a in enumerate(agents):
            b = a.bidding(tick, pv_v[:, i], pv_s[:, i],
                          [x[i] for x in hp], [x[i] for x in hb],
                          [x[i] for x in ha], [x[i] for x in hi], hl) \
                if a.remaining_budget >= envs.min_remaining_budget else np.zeros(pv_v.shape[0])
            bids.append(b)
        bids = np.array(bids).transpose(); bids[bids < 0] = 0
        rem = np.array([a.remaining_budget for a in agents])

        rm = None; wp = None
        while rm is None or rm > 0:
            if rm and rm > 0:
                ocr = np.maximum((cost - rem) / (cost + 1e-4), 0)
                _adj(bids, ocr, envs.slot_coefficients, wp)
            xi, sl, cp, ex, cv, lw, _ = envs.simulate_ad_bidding(pv_v, pv_s, bids)
            cost = (cp * ex).sum(axis=1); reward = cv.sum(axis=1)
            wp = _winner(sl)
            ocr = np.maximum((cost - rem) / (cost + 1e-4), 0); rm = ocr.max()

        for i in range(num_agents):
            agents[i].remaining_budget -= cost[i]

        rewards += reward; costs += cost
        pv_mean = float(np.mean(pv_v[:, player_index]))
        bid_mean = float(np.mean(bids[:, player_index]))
        tick_recs.append({
            'tick': tick,
            'alpha': bid_mean / (pv_mean + 1e-8) if pv_mean > 0 else 0.0,
            'tick_reward': int(reward[player_index]),
            'tick_cost': float(cost[player_index]),
            'budget_left': float(agents[player_index].remaining_budget),
        })
        hb.append(bids.transpose()); hl.append(lw)
        hp.append(np.stack((pv_v.T, pv_s.T), axis=-1))
        ha.append(np.stack((xi, sl, cp), axis=-1))
        hi.append(np.stack((ex, cv), axis=-1))

    pr = rewards[player_index]; pc = costs[player_index]
    cpr = pc / (pr + 1e-10)
    pen = (cpas[player_index] / (cpr + 1e-10)) ** 2 if cpr > cpas[player_index] else 1.0
    return pen * pr, pr, pc, cpr, pen, tick_recs


def _winner(s):
    s = s.T
    w = np.full((s.shape[0], 3), -1, dtype=int)
    for p in [1, 2, 3]:
        wi = np.argwhere(s == p)
        if wi.size:
            pv_i, a_i = wi.T
            w[pv_i, p - 1] = a_i
    return w


def _adj(b, o, sl, w):
    for ai in np.where(o > 0)[0]:
        for si in range(len(sl)):
            wi = w[:, si]
            pi = np.where(wi == ai)[0]
            n = int(np.ceil(pi.size * o[ai]))
            if n:
                b[np.random.default_rng(1).choice(pi, n, replace=False), ai] = 0


# ── Main ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_files', nargs='+', help='CSV files to benchmark')
    parser.add_argument('--output', default='benchmark_general',
                        help='Output file prefix in exp_data/')
    parser.add_argument('--advertisers', type=int, nargs='*', default=None,
                        help='Specific advertisers to test (default: all 48)')
    args = parser.parse_args()

    test_advs = args.advertisers if args.advertisers else list(range(48))
    all_summaries, all_ticks = [], []

    for csv_path in args.csv_files:
        period = extract_period_number(csv_path)
        logger.info(f'\n{"="*60}')
        logger.info(f'Period {period}: {csv_path}')
        logger.info(f'{"="*60}')

        pv_vl, pv_sl, n_a, n_t, df = load_pv_data(csv_path)
        budgets = df.groupby('advertiserNumber')['budget'].first().values
        cpas = df.groupby('advertiserNumber')['CPAConstraint'].first().values
        cats = df.groupby('advertiserNumber')['advertiserCategoryIndex'].first().values.astype(int)
        o_s, o_cpa, o_bu, o_rw, o_ct, o_pn = extract_original_performance(df)

        for s_name, s_fact in STRATEGIES:
            for adv in test_advs:
                t0 = time.time()
                score, rw, ct, cpr, pen, ticks = run_replay(
                    s_fact, adv, pv_vl, pv_sl, budgets, cpas, cats, n_t, n_a)
                all_summaries.append({
                    'period': period, 'strategy': s_name, 'advertiser': adv,
                    'budget': budgets[adv], 'cpa_target': cpas[adv],
                    'score': score, 'total_reward': int(rw), 'total_cost': ct,
                    'cpa_real': cpr, 'penalty': pen, 'budget_used': ct / budgets[adv] * 100,
                    'orig_score': o_s[adv], 'orig_cpa_real': o_cpa[adv],
                    'orig_budget_used': o_bu[adv], 'orig_reward': o_rw[adv],
                })
                for t in ticks:
                    t['period'] = period; t['strategy'] = s_name; t['advertiser'] = adv
                    all_ticks.append(t)
                logger.info(f'  [{s_name}] adv={adv} score={score:.1f} (orig={o_s[adv]:.1f}) '
                            f't={time.time()-t0:.1f}s')
        # Also record original tick data
        for adv in test_advs:
            adv_df = df[df['advertiserNumber'] == adv]
            for t in range(n_t):
                td = adv_df[adv_df['timeStepIndex'] == t]
                pvm = td['pValue'].mean()
                bm = td['bid'].mean()
                all_ticks.append({
                    'period': period, 'strategy': 'ORIGINAL', 'advertiser': adv,
                    'tick': t, 'tick_reward': int(td['conversionAction'].sum()),
                    'tick_cost': float(td[td['isExposed']==1]['cost'].sum()),
                    'alpha': bm / (pvm + 1e-8) if pvm > 0 else 0.0,
                    'budget_left': float(td['remainingBudget'].iloc[-1]),
                })

    # ── Save ──
    prefix = os.path.join(OUTPUT_DIR, args.output)
    pd.DataFrame(all_summaries).to_csv(f'{prefix}_summary.csv', index=False)
    pd.DataFrame(all_ticks).to_csv(f'{prefix}_tick.csv', index=False)
    logger.info(f'Saved: {prefix}_summary.csv, {prefix}_tick.csv')

    # ── Comparison ──
    dfs = pd.DataFrame(all_summaries)
    comp = dfs.groupby(['period', 'strategy'])['score'].mean().unstack()
    if 'ORIGINAL' not in comp.columns:
        orig_scores = dfs[['period', 'advertiser', 'orig_score']].drop_duplicates()
        comp['ORIGINAL'] = orig_scores.groupby('period')['orig_score'].mean().values
    print(f'\n{"="*80}')
    print(f'AVERAGE SCORE BY PERIOD × STRATEGY')
    print(f'{"="*80}')
    #cols = [c for c in ['ORIGINAL', 'DGAB', 'IQL', 'DT'] if c in comp.columns]
    strat_names = ['ORIGINAL'] + [s[0] for s in STRATEGIES]
    cols = [c for c in strat_names if c in comp.columns]
    print(comp[cols].to_string(float_format=lambda x: f'{x:.1f}'))
    print()

    comp.to_csv(f'{prefix}_comparison.csv')
    logger.info(f'Saved: {prefix}_comparison.csv')


if __name__ == '__main__':
    main()
