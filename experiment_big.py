"""
Large-scale experiment: pvalue_mean_base × strategy × advertiser.

  10 pvalue values: [0.0001 : 0.0001 : 0.001]
  4 strategies:     GAVE, DT, DGAB, IQL
  48 advertisers:   0..47

Records per-tick: state (16-dim), alpha, RTG (if available).
Compares average scores across strategies.
"""

import sys, os, time
import numpy as np
import pandas as pd
import gin
import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'strategy_train_env'))
sys.path.insert(0, _PROJECT_ROOT)

import run.run_test
import simul_bidding_env.Controller.Controller
import simul_bidding_env.Environment.BiddingEnv

gin_file = ["./config/test.gin"]
gin.parse_config_files_and_bindings(gin_file, None)

from simul_bidding_env.Controller.Controller import Controller
from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy

# ── Config ──────────────────────────────────────────

DGAB_SAVE_DIR = 'D:/research/Experiment/autobidding/saved_model/dgab_400k_rc'
GAVE_SAVE_DIR = 'D:/research/Experiment/autobidding/saved_model/gave_400k_rc'
DT_SAVE_DIR = './saved_model/DTtest'
DEVICE = 'cuda:0' if __import__('torch').cuda.is_available() else 'cpu'
NUM_TICK = 48
FIXED_SEED = 42
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'exp_data')

BLOCK_CONFIG = {
    'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
    'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
    'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
}

PVALUES = np.arange(0.0001, 0.0011, 0.0001)  # [0.0001, 0.0002, ..., 0.0010]
NUM_ADVERTISERS = 48

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Strategy factories ──────────────────────────────

def make_dgab(budget, cpa, category):
    from simul_bidding_env.strategy.autobidding_agents import DGABFOAuctionNetAgent
    return DGABFOAuctionNetAgent(
        budget=budget, cpa=cpa, category=category, name='DGAB-FO',
        model_param=dict(
            save_dir=DGAB_SAVE_DIR, hidden_size=512, max_ep_len=96, time_dim=8,
            block_config=BLOCK_CONFIG, device=DEVICE,
            actor_type='stack', critic_type='sequence',
        ),
    )


def make_gave(budget, cpa, category):
    from simul_bidding_env.strategy.autobidding_agents import GAVEAuctionNetAgent
    return GAVEAuctionNetAgent(
        budget=budget, cpa=cpa, category=category, name='GAVE',
        model_param=dict(
            save_dir=GAVE_SAVE_DIR, hidden_size=512, time_dim=8,
            block_config=BLOCK_CONFIG, device=DEVICE,
            expectile=0.99, score_target_mode='prev',
        ),
    )


def make_dt(budget, cpa, category):
    from simul_bidding_env.strategy.autobidding_agents import DTAuctionNetAgent
    return DTAuctionNetAgent(
        budget=budget, cpa=cpa, category=category, name='DT',
        model_param=dict(
            save_dir=DT_SAVE_DIR, device=DEVICE,
            target_return=4, scale=2000,
        ),
    )


def make_iql(budget, cpa, category):
    from simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
    return IqlBiddingStrategy(budget=budget, cpa=cpa, category=category, name='IQL')


STRATEGIES = [
    ('DGAB', make_dgab),
    ('GAVE', make_gave),
    ('DT',   make_dt),
    ('IQL',  make_iql),
]


# ── Helpers ─────────────────────────────────────────

def compute_score(reward, cost, cpa_target):
    eps = 1e-10
    cpa_real = cost / (reward + eps)
    beta = 2
    if cpa_real > cpa_target:
        penalty = (cpa_target / (cpa_real + eps)) ** beta
    else:
        penalty = 1.0
    return penalty * reward, cpa_real


# ── Main ────────────────────────────────────────────

def main():
    np.random.seed(FIXED_SEED)
    import torch
    torch.manual_seed(FIXED_SEED)

    all_summaries = []
    all_tick_records = []

    total_runs = len(PVALUES) * NUM_ADVERTISERS
    run_idx = 0

    for pv_val in PVALUES:
        pv_label = f'{pv_val:.4f}'
        logger.info(f'\n{"="*70}')
        logger.info(f'PVALUE_MEAN_BASE = {pv_label}')
        logger.info(f'{"="*70}')

        for adv in range(NUM_ADVERTISERS):
            run_idx += 1
            # ── Create controller with fixed seed for this (pv, adv) ──
            seed = FIXED_SEED * 10000 + int(pv_val * 100000) + adv
            np.random.seed(seed)
            torch.manual_seed(seed)

            dummy_agent = PidBiddingStrategy(exp_tempral_ratio=np.ones(NUM_ADVERTISERS))
            controller = Controller(player_index=adv, player_agent=dummy_agent)
            envs = controller.biddingEnv
            pv_gen = controller.pvGenerator

            controller.reset(episode=0)
            # Patch pvalue_mean_base AFTER reset (__init__ hardcodes 0.001)
            pv_gen.pvalue_mean_base = pv_val
            pv_gen.pv_values, pv_gen.PValueSigmas = pv_gen.generate()

            budget = controller.budget_list[adv]
            cpa_target = controller.cpa_constraint_list[adv]
            category = controller.category[adv]

            for s_name, s_factory in STRATEGIES:
                # ── Build fresh strategy agent ──
                player_agent = s_factory(budget=budget, cpa=cpa_target, category=category)
                controller.player_agent = player_agent
                agents = controller.load_agents()
                num_agent = len(agents)
                # Reset all agents' remaining budgets
                for i in range(num_agent):
                    agents[i].remaining_budget = controller.budget_list[i]

                costs_arr = np.zeros(num_agent)
                rewards_arr = np.zeros(num_agent)

                history_pvalue_infos = []
                history_bids = []
                history_auction_results = []
                history_impression_results = []
                history_least_winning_costs = []

                for tick in range(NUM_TICK):
                    pv_values = pv_gen.pv_values[tick]
                    pvalue_sigmas = pv_gen.PValueSigmas[tick]

                    bids = [
                        agent.bidding(
                            tick, pv_values[:, i], pvalue_sigmas[:, i],
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

                    remaining_budget_list = np.array([a.remaining_budget for a in agents])

                    ratio_max = None
                    winner_pit = None
                    while ratio_max is None or ratio_max > 0:
                        if ratio_max and ratio_max > 0:
                            over_cost_ratio = np.maximum(
                                (cost - remaining_budget_list) / (cost + 1e-4), 0)
                            from run.run_test import adjust_over_cost
                            adjust_over_cost(bids, over_cost_ratio, envs.slot_coefficients, winner_pit)

                        (xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit,
                         least_winning_cost_pit, market_price_pit) = \
                            envs.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)

                        real_cost = cost_pit * is_exposed_pit
                        cost = real_cost.sum(axis=1)
                        reward = conversion_action_pit.sum(axis=1)

                        from run.run_test import get_winner
                        winner_pit = get_winner(slot_pit)
                        over_cost_ratio = np.maximum(
                            (cost - remaining_budget_list) / (cost + 1e-4), 0)
                        ratio_max = over_cost_ratio.max()

                    for i in range(num_agent):
                        agents[i].remaining_budget -= cost[i]

                    tick_reward = int(reward[adv])
                    tick_cost = float(cost[adv])
                    costs_arr += cost
                    rewards_arr += reward

                    # ── Capture state ──
                    state_vec = None
                    try:
                        if s_name in ('DGAB',):
                            state_vec = player_agent._build_state(tick, pv_values[:, adv])
                        elif s_name in ('GAVE', 'DT'):
                            from simul_bidding_env.strategy.autobidding_agents import _build_state_16
                            state_vec = _build_state_16(
                                tick, agents[adv].remaining_budget, budget,
                                pv_values[:, adv], history_bids, history_auction_results,
                                history_impression_results, history_least_winning_costs,
                                history_pvalue_infos)
                        else:  # IQL
                            # IQL builds state differently; skip per-tick capture
                            pass
                    except Exception:
                        pass

                    # ── Capture RTG ──
                    rtg_list = None
                    try:
                        if s_name == 'DGAB':
                            r = player_agent._rollout.rtg.cpu().numpy()
                            rtg_list = [float(r[0]), float(r[1])]
                        elif s_name == 'DT':
                            r = player_agent._dt_model.eval_target_return
                            rtg_list = [float(r[0, -1])] if r is not None else []
                        elif s_name == 'GAVE':
                            pass  # GAVE doesn't expose RTG this way
                    except Exception:
                        pass

                    pv_mean = float(np.mean(pv_values[:, adv]))
                    bid_mean = float(np.mean(bids[:, adv]))
                    alpha_est = bid_mean / (pv_mean + 1e-8) if pv_mean > 0 else 0.0

                    rec = {
                        'pvalue_mean_base': pv_val,
                        'advertiser': adv,
                        'strategy': s_name,
                        'tick': tick,
                        'alpha': alpha_est,
                        'tick_reward': tick_reward,
                        'tick_cost': tick_cost,
                        'cum_reward': int(rewards_arr[adv]),
                        'cum_cost': float(costs_arr[adv]),
                        'budget_left': float(agents[adv].remaining_budget),
                    }
                    if state_vec is not None:
                        for j in range(16):
                            rec[f'state_{j}'] = float(state_vec[j])
                    if rtg_list is not None:
                        if len(rtg_list) >= 1:
                            rec['rtg_v'] = rtg_list[0]
                        if len(rtg_list) >= 2:
                            rec['rtg_c'] = rtg_list[1]

                    all_tick_records.append(rec)

                    history_bids.append(bids.transpose())
                    history_least_winning_costs.append(least_winning_cost_pit)
                    history_pvalue_infos.append(
                        np.stack((pv_values.T, pvalue_sigmas.T), axis=-1))
                    history_auction_results.append(
                        np.stack((xi_pit, slot_pit, cost_pit), axis=-1))
                    history_impression_results.append(
                        np.stack((is_exposed_pit, conversion_action_pit), axis=-1))

                # ── Final score ──
                final_reward = rewards_arr[adv]
                final_cost = costs_arr[adv]
                score, cpa_real = compute_score(final_reward, final_cost, cpa_target)

                all_summaries.append({
                    'pvalue_mean_base': pv_val,
                    'advertiser': adv,
                    'strategy': s_name,
                    'budget': float(budget),
                    'cpa_target': float(cpa_target),
                    'total_reward': int(final_reward),
                    'total_cost': float(final_cost),
                    'cpa_real': float(cpa_real),
                    'score': float(score),
                    'budget_used': float(final_cost / budget),
                })

            pct = 100.0 * run_idx / total_runs
            logger.info(f'  [{run_idx}/{total_runs} ({pct:.0f}%)] '
                        f'pv={pv_label} adv={adv} done')

    # ── Save ──
    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, 'exp_big_summary.csv'), index=False)

    df_tick = pd.DataFrame(all_tick_records)
    df_tick.to_csv(os.path.join(OUTPUT_DIR, 'exp_big_tick_detail.csv'), index=False)

    # ── Comparison: avg score per strategy per pvalue ──
    comparison = df_summary.groupby(['pvalue_mean_base', 'strategy'])['score'].mean().unstack()
    comparison.columns.name = None
    comparison = comparison.reset_index()

    print(f'\n{"="*90}')
    print('AVERAGE SCORE PER STRATEGY × PVALUE (48 advertisers)')
    print(f'{"="*90}')
    print(comparison.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print()

    comparison.to_csv(os.path.join(OUTPUT_DIR, 'exp_big_comparison.csv'), index=False)

    # Overall avg per strategy
    print('OVERALL AVERAGE SCORE PER STRATEGY:')
    overall = df_summary.groupby('strategy')['score'].mean()
    for s in overall.index:
        print(f'  {s}: {overall[s]:.2f}')
    print()

    logger.info(f'Done. Files saved to {OUTPUT_DIR}/')
    logger.info(f'  exp_big_summary.csv       — {len(df_summary)} rows')
    logger.info(f'  exp_big_tick_detail.csv   — {len(df_tick)} rows')
    logger.info(f'  exp_big_comparison.csv    — avg score per pvalue × strategy')


if __name__ == '__main__':
    main()
