"""
Benchmark: compare IQL/DGAB/DT vs original strategy on period-12.csv

For each advertiser, extract the original performance from the CSV,
then run IQL/DGAB/DT on the same PV data via offline replay.
Compare per-advertiser Score, CPA, budget, per-tick alpha.
"""

import sys, os, time
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'strategy_train_env'))
sys.path.insert(0, _PROJECT_ROOT)

CSV_PATH = os.path.join(_PROJECT_ROOT, 'strategy_train_env', 'data', 'traffic', 'period-12.csv')
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'exp_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DGAB_DIR = 'D:/research/Experiment/autobidding/saved_model/dgab_400k_sparse'
GAVE_DIR = 'D:/research/Experiment/autobidding/saved_model/gave_400k_rc'
DT_DIR = './saved_model/DTtest'
DEVICE = 'cuda:0' if __import__('torch').cuda.is_available() else 'cpu'
BLOCK_CONFIG = {
    'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
    'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
    'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
}


# ── Load PV data from CSV ──────────────────────────

def load_pv_data(csv_path):
    """Reconstruct per-tick pv_values (n_pv, 48) and pValueSigmas from CSV."""
    logger.info(f'Loading {csv_path} ...')
    df = pd.read_csv(csv_path)
    ticks = sorted(df['timeStepIndex'].unique())
    num_agents = int(df['advertiserNumber'].nunique())

    pv_values_list, pv_sigmas_list = [], []
    for tick in ticks:
        td = df[df['timeStepIndex'] == tick]
        pv_v = td.pivot_table(index='pvIndex', columns='advertiserNumber',
                              values='pValue', aggfunc='first').values.astype(np.float64)
        pv_s = td.pivot_table(index='pvIndex', columns='advertiserNumber',
                              values='pValueSigma', aggfunc='first').values.astype(np.float64)
        pv_values_list.append(pv_v)
        pv_sigmas_list.append(pv_s)

    return pv_values_list, pv_sigmas_list, num_agents, len(ticks)


# ── Extract original performance from CSV ──────────

def extract_original_performance(df):
    """Extract per-advertiser original metrics from the auction log."""
    gw = df[df['isExposed'] == 1]
    total_cost = gw.groupby('advertiserNumber')['cost'].sum()
    total_reward = df.groupby('advertiserNumber')['conversionAction'].sum()
    budget = df.groupby('advertiserNumber')['budget'].first()
    cpa_target = df.groupby('advertiserNumber')['CPAConstraint'].first()

    cpa_real = total_cost / (total_reward + 1e-10)
    penalty = np.where(cpa_real > cpa_target,
                       (cpa_target / (cpa_real + 1e-10)) ** 2, 1.0)
    score = penalty * total_reward
    budget_used = total_cost / budget * 100

    # Per-tick alpha: bid_mean / pv_mean per (advertiser, tick)
    alpha_per_tick = df.groupby(['advertiserNumber', 'timeStepIndex']).apply(
        lambda g: np.mean(g['bid']) / (np.mean(g['pValue']) + 1e-10)
    ).reset_index(name='alpha')

    return score, cpa_real, budget_used, total_reward, total_cost, penalty, alpha_per_tick


# ── Strategy factories ─────────────────────────────

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

STRATEGIES = [
    ('DGAB', make_dgab),
    ('IQL',  make_iql),
    ('DT',   make_dt),
]

TEST_ADVERTISERS = [0]  # only test first advertiser


# ── Run replay ─────────────────────────────────────

def run_replay(strategy_factory, player_index, pv_values_list, pv_sigmas_list,
               budgets, cpas, categories, num_ticks, num_agents):
    """Run offline replay with given player strategy, return per-tick data."""

    from simul_bidding_env.Environment.BiddingEnv import BiddingEnv
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    from simul_bidding_env.strategy.player_agent_wrapper import PlayerAgentWrapper

    # Build agent lineup: player + 47 PID defaults
    agents = []
    for i in range(num_agents):
        agent = PidBiddingStrategy(budget=budgets[i], cpa=cpas[i],
                                   category=categories[i],
                                   name=f'PID-{i}',
                                   exp_tempral_ratio=np.ones(num_agents))
        agent.budget = budgets[i]
        agent.cpa = cpas[i]
        agent.category = categories[i]
        agent.remaining_budget = budgets[i]
        agents.append(agent)

    player_agent = strategy_factory(budget=budgets[player_index],
                                    cpa=cpas[player_index],
                                    category=categories[player_index])
    agents[player_index] = PlayerAgentWrapper(player_agent=player_agent)

    envs = BiddingEnv()
    envs.reset(episode=0)

    rewards = np.zeros(num_agents)
    costs = np.zeros(num_agents)

    history_pv = []; history_bids = []; history_auc = []
    history_imp = []; history_lwc = []

    tick_records = []

    for tick in range(num_ticks):
        pv_v = pv_values_list[tick]
        pv_s = pv_sigmas_list[tick]
        pv_s = np.nan_to_num(pv_s, nan=0.0)
        pv_s = np.maximum(pv_s, 1e-8)

        bids = []
        for i, a in enumerate(agents):
            if a.remaining_budget >= envs.min_remaining_budget:
                b = a.bidding(tick, pv_v[:, i], pv_s[:, i],
                              [x[i] for x in history_pv],
                              [x[i] for x in history_bids],
                              [x[i] for x in history_auc],
                              [x[i] for x in history_imp],
                              history_lwc)
            else:
                b = np.zeros(pv_v.shape[0])
            bids.append(b)
        bids = np.array(bids).transpose()
        bids[bids < 0] = 0

        rem_budgets = np.array([a.remaining_budget for a in agents])

        # Cost-adjustment loop
        ratio_max = None
        wp = None
        while ratio_max is None or ratio_max > 0:
            if ratio_max and ratio_max > 0:
                ocr = np.maximum((cost - rem_budgets) / (cost + 1e-4), 0)
                _adjust_over_cost(bids, ocr, envs.slot_coefficients, wp)

            xi, slot, cost_pit, exp, conv, lwc, mp = envs.simulate_ad_bidding(pv_v, pv_s, bids)
            cost = (cost_pit * exp).sum(axis=1)
            reward = conv.sum(axis=1)
            wp = _get_winner(slot)
            ocr = np.maximum((cost - rem_budgets) / (cost + 1e-4), 0)
            ratio_max = ocr.max()

        for i in range(num_agents):
            agents[i].remaining_budget -= cost[i]

        rewards += reward
        costs += cost

        # Record
        pv_mean = float(np.mean(pv_v[:, player_index]))
        bid_mean = float(np.mean(bids[:, player_index]))
        alpha = bid_mean / (pv_mean + 1e-8) if pv_mean > 0 else 0.0
        tick_records.append({
            'tick': tick, 'alpha': alpha,
            'tick_reward': int(reward[player_index]),
            'tick_cost': float(cost[player_index]),
            'budget_left': float(agents[player_index].remaining_budget),
        })

        history_bids.append(bids.transpose())
        history_lwc.append(lwc)
        history_pv.append(np.stack((pv_v.T, pv_s.T), axis=-1))
        history_auc.append(np.stack((xi, slot, cost_pit), axis=-1))
        history_imp.append(np.stack((exp, conv), axis=-1))

    player_reward = rewards[player_index]
    player_cost = costs[player_index]
    cpa_real = player_cost / (player_reward + 1e-10)
    penalty = (cpas[player_index] / (cpa_real + 1e-10)) ** 2 if cpa_real > cpas[player_index] else 1.0
    score = penalty * player_reward

    return score, player_reward, player_cost, cpa_real, penalty, tick_records


def _get_winner(slot_pit):
    slot_pit = slot_pit.T
    n_pv, n_a = slot_pit.shape
    w = np.full((n_pv, 3), -1, dtype=int)
    for pos in range(1, 4):
        wi = np.argwhere(slot_pit == pos)
        if wi.size > 0:
            pv_i, a_i = wi.T; w[pv_i, pos - 1] = a_i
    return w


def _adjust_over_cost(bids, ocr, slots, wp):
    for ai in np.where(ocr > 0)[0]:
        for si in range(len(slots)):
            wi = wp[:, si]; pvi = np.where(wi == ai)[0]
            nd = int(np.ceil(pvi.size * ocr[ai]))
            if nd > 0:
                dropped = np.random.default_rng(seed=1).choice(pvi, nd, replace=False)
                bids[dropped, ai] = 0


# ── Main ───────────────────────────────────────────

def main():
    pv_values_list, pv_sigmas_list, num_agents, num_ticks = load_pv_data(CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    # Extract original metrics
    orig_score, orig_cpa, orig_bu, orig_rew, orig_cost, orig_pen, orig_alpha = \
        extract_original_performance(df)

    # Get budget/CPA from data
    budgets = df.groupby('advertiserNumber')['budget'].first().values
    cpas = df.groupby('advertiserNumber')['CPAConstraint'].first().values
    categories = df.groupby('advertiserNumber')['advertiserCategoryIndex'].first().values.astype(int)

    all_summaries = []
    all_tick_comparisons = []

    for s_name, s_factory in STRATEGIES:
        logger.info(f'\n{"="*50}')
        logger.info(f'Running {s_name} on all {num_agents} advertisers...')
        logger.info(f'{"="*50}')

        s_scores, s_cpas, s_bus, s_rews, s_costs, s_pens = [], [], [], [], [], []

        for adv in TEST_ADVERTISERS:
            t0 = time.time()
            score, reward, cost, cpa_r, pen, tick_recs = run_replay(
                s_factory, adv, pv_values_list, pv_sigmas_list,
                budgets, cpas, categories, num_ticks, num_agents)

            s_scores.append(score); s_cpas.append(cpa_r); s_bus.append(cost / budgets[adv] * 100)
            s_rews.append(reward); s_costs.append(cost); s_pens.append(pen)

            all_summaries.append({
                'strategy': s_name, 'advertiser': adv,
                'budget': budgets[adv], 'cpa_target': cpas[adv],
                'score': score, 'total_reward': int(reward),
                'total_cost': cost, 'cpa_real': cpa_r,
                'penalty': pen, 'budget_used': cost / budgets[adv] * 100,
            })

            # Compare tick-level alpha with original
            orig_adv_alpha = orig_alpha[orig_alpha['advertiserNumber'] == adv]
            for tr in tick_recs:
                all_tick_comparisons.append({
                    'strategy': s_name, 'advertiser': adv,
                    **tr,
                })

            if adv % 8 == 0:
                logger.info(f'  [{s_name}] adv {adv}/{num_agents} ...')

        logger.info(f'  [{s_name}] DONE. Avg score={np.mean(s_scores):.1f}, '
                    f'Avg CPA={np.mean(s_cpas):.1f}, Avg BU={np.mean(s_bus):.1f}%')

    # ── Summary with original as baseline ──
    print(f'\n{"="*90}')
    print(f'BENCHMARK: IQL / DGAB / DT vs Original (period-12.csv, {num_agents} advertisers)')
    print(f'{"="*90}')
    headers = ['Strategy', 'Avg Score', 'Median Score', 'Avg CPA Real', 'Avg Penalty',
               'Avg Budget%', 'Avg Reward', 'Score>Orig']
    print(f'  {headers[0]:<12} {headers[1]:>10} {headers[2]:>12} {headers[3]:>12} '
          f'{headers[4]:>11} {headers[5]:>11} {headers[6]:>11} {headers[7]:>10}')
    print(f'  {"-"*88}')

    for label, score_arr, cpa_arr, pen_arr, bu_arr, rew_arr in [
        ('ORIGINAL', orig_score, orig_cpa, orig_pen, orig_bu, orig_rew),
    ]:
        beats = '-'
        print(f'  {label:<12} {score_arr.mean():>10.1f} {score_arr.median():>12.1f} '
              f'{cpa_arr.mean():>12.1f} {pen_arr.mean():>11.4f} {bu_arr.mean():>11.1f} '
              f'{rew_arr.mean():>11.1f} {beats:>10}')

    df_sum = pd.DataFrame(all_summaries)
    for s_name, _ in STRATEGIES:
        sub = df_sum[df_sum['strategy'] == s_name]
        beats = (sub['score'].values > orig_score.values).sum()
        print(f'  {s_name:<12} {sub["score"].mean():>10.1f} {sub["score"].median():>12.1f} '
              f'{sub["cpa_real"].mean():>12.1f} {sub["penalty"].mean():>11.4f} '
              f'{sub["budget_used"].mean():>11.1f} {sub["total_reward"].mean():>11.1f} '
              f'{beats}/{num_agents}:>10')

    # Save
    df_sum.to_csv(os.path.join(OUTPUT_DIR, 'benchmark_period12_summary.csv'), index=False)
    pd.DataFrame(all_tick_comparisons).to_csv(
        os.path.join(OUTPUT_DIR, 'benchmark_period12_tick.csv'), index=False)
    logger.info(f'\nSaved to {OUTPUT_DIR}/benchmark_period12_*.csv')


if __name__ == '__main__':
    main()
