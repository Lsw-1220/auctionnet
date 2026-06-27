"""
pvalue_mean_base comparison experiment.

Compares two pvalue density settings:
  Exp A: pvalue_mean_base = 0.0005
  Exp B: pvalue_mean_base = 0.001

Records per-tick data: RTG, alpha, reward, cost, state (16-dim).
Saves results to ./exp_data/ folder.
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
from simul_bidding_env.strategy.autobidding_agents import DGABFOAuctionNetAgent

# ── Config ──────────────────────────────────────────

DGAB_SAVE_DIR = 'D:/research/Experiment/autobidding/saved_model/dgab_400k_sparse'
DEVICE = 'cuda:0' if __import__('torch').cuda.is_available() else 'cpu'
NUM_TICK = 48
FIXED_SEED = 42
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'exp_data')
BLOCK_CONFIG = {
    'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
    'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
    'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


# ── Experiment runner ───────────────────────────────

def run_experiment(exp_name, pvalue_mean_base, player_index=0):
    """Run one experiment and return per-tick records + final summary."""

    np.random.seed(FIXED_SEED)
    import torch
    torch.manual_seed(FIXED_SEED)

    dummy_agent = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
    controller = Controller(player_index=player_index, player_agent=dummy_agent)
    envs = controller.biddingEnv
    pv_generator = controller.pvGenerator

    # Build DGAB agent
    dgab_agent = DGABFOAuctionNetAgent(
        budget=controller.budget_list[player_index],
        cpa=controller.cpa_constraint_list[player_index],
        category=controller.category[player_index],
        name='DGAB-FO',
        model_param=dict(
            save_dir=DGAB_SAVE_DIR, hidden_size=512, max_ep_len=96, time_dim=8,
            block_config=BLOCK_CONFIG, device=DEVICE,
            actor_type='stack', critic_type='sequence',
        ),
    )
    controller.player_agent = dgab_agent
    agents = controller.load_agents()
    num_agent = len(agents)

    # controller.reset() calls pvGenerator.reset() which re-inits via __init__
    # → hardcodes pvalue_mean_base=0.001. Patch instance AFTER reset, then regenerate.
    controller.reset(episode=0)
    pv_generator.pvalue_mean_base = pvalue_mean_base
    pv_generator.pv_values, pv_generator.pValueSigmas = pv_generator.generate()
    logger.info(f'[{exp_name}] pvalue_mean_base = {pv_generator.pvalue_mean_base}')

    # State tracking
    rewards = np.zeros(num_agent)
    costs = np.zeros(num_agent)
    budgets = np.array([a.budget for a in agents])

    history_pvalue_infos = []
    history_bids = []
    history_auction_results = []
    history_impression_results = []
    history_least_winning_costs = []

    records = []

    for tick_index in range(NUM_TICK):
        pv_values = pv_generator.pv_values[tick_index]
        pvalue_sigmas = pv_generator.pValueSigmas[tick_index]

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

        remaining_budget_list = np.array([a.remaining_budget for a in agents])

        ratio_max = None
        winner_pit = None
        while ratio_max is None or ratio_max > 0:
            if ratio_max and ratio_max > 0:
                over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 1e-4), 0)
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
            over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 1e-4), 0)
            ratio_max = over_cost_ratio.max()

        for i, agent in enumerate(agents):
            agent.remaining_budget -= cost[i]

        tick_reward = int(reward[player_index])
        tick_cost = float(cost[player_index])
        rewards += reward
        costs += cost

        # ── Capture DGAB internal state ──
        rtg_norm = None
        state_raw = None
        try:
            rtg_norm = dgab_agent._rollout.rtg.cpu().numpy().copy()
        except Exception:
            pass
        try:
            state_raw = dgab_agent._build_state(tick_index, pv_values[:, player_index])
        except Exception:
            pass

        pv_mean = float(np.mean(pv_values[:, player_index]))
        bid_mean = float(np.mean(bids[:, player_index]))
        alpha_est = bid_mean / (pv_mean + 1e-8) if pv_mean > 0 else 0.0

        rec = {
            'tick': tick_index,
            'rtg_v_norm': float(rtg_norm[0]) if rtg_norm is not None else np.nan,
            'rtg_c_norm': float(rtg_norm[1]) if rtg_norm is not None else np.nan,
            'alpha': alpha_est,
            'tick_reward': tick_reward,
            'tick_cost': tick_cost,
            'cum_reward': int(rewards[player_index]),
            'cum_cost': float(costs[player_index]),
            'budget_left': float(agents[player_index].remaining_budget),
            'pv_mean': pv_mean,
        }
        if state_raw is not None:
            for j in range(16):
                rec[f'state_{j}'] = float(state_raw[j])

        records.append(rec)
        logger.info(f'  [{exp_name}] tick={tick_index:02d} '
                    f'alpha={alpha_est:.1f} reward={tick_reward} cost={tick_cost:.1f} '
                    f'cum_reward={int(rewards[player_index])}')

        history_bids.append(bids.transpose())
        history_least_winning_costs.append(least_winning_cost_pit)
        history_pvalue_infos.append(np.stack((pv_values.T, pvalue_sigmas.T), axis=-1))
        history_auction_results.append(np.stack((xi_pit, slot_pit, cost_pit), axis=-1))
        history_impression_results.append(np.stack((is_exposed_pit, conversion_action_pit), axis=-1))

    # Final score
    player_reward = rewards[player_index]
    player_cost = costs[player_index]
    score, cpa_real = compute_score(player_reward, player_cost,
                                     controller.cpa_constraint_list[player_index])

    summary = {
        'exp_name': exp_name,
        'pvalue_mean_base': pvalue_mean_base,
        'budget': float(budgets[player_index]),
        'cpa_target': float(controller.cpa_constraint_list[player_index]),
        'total_reward': int(player_reward),
        'total_cost': float(player_cost),
        'cpa_real': float(cpa_real),
        'score': float(score),
        'budget_used': float(player_cost / budgets[player_index]),
    }

    return records, summary


# ── Main ────────────────────────────────────────────

def main():
    experiments = [
        ('pvalue_0.0005', 0.0005),
        ('pvalue_0.001',  0.001),
    ]

    all_summaries = []

    for exp_name, pv_val in experiments:
        logger.info(f'\n{"="*60}')
        logger.info(f'Experiment: {exp_name}')
        logger.info(f'{"="*60}')

        records, summary = run_experiment(exp_name, pv_val)
        all_summaries.append(summary)

        df = pd.DataFrame(records)
        csv_path = os.path.join(OUTPUT_DIR, f'{exp_name}_tick_detail.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f'Saved tick detail to {csv_path}')

    df_summary = pd.DataFrame(all_summaries)
    summary_path = os.path.join(OUTPUT_DIR, 'pvalue_experiment_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    logger.info(f'Saved summary to {summary_path}')

    print(f'\n{"="*80}')
    print('EXPERIMENT RESULTS')
    print(f'{"="*80}')
    print(df_summary.to_string(index=False))
    print()


if __name__ == '__main__':
    main()
