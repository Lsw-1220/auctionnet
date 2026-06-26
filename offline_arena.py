"""
Offline Arena: Replay historical pv data with custom strategy lineup.
All strategies compete for the same traffic, online auction style.

Usage:
    python offline_arena.py
"""
import sys, os
# Resolve project root robustly
_script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(_script_dir, 'run')):
    # Fallback: use cwd
    _script_dir = os.getcwd()
sys.path.insert(0, os.path.join(_script_dir, 'strategy_train_env'))
sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd
import logging
import time
import gin
from collections import defaultdict

# Pre-import so gin can resolve config imports
import run.run_test
import simul_bidding_env.Controller.Controller
import simul_bidding_env.Environment.BiddingEnv

# Parse gin so Controller gets correct num_agent=48 etc.
gin.parse_config_files_and_bindings(['./config/test.gin'], None)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────
CSV_PATH = 'strategy_train_env/data/traffic/period-12.csv'
PLAYER_INDEX = 0  # which agent slot gets DGAB
# ────────────────────────────────────────────────────────────────


def load_pv_data(csv_path):
    """Load CSV and reconstruct per-tick pv_values and pValueSigmas (n_pv, 48)."""
    logger.info(f'Loading {csv_path} ...')
    df = pd.read_csv(csv_path)
    ticks = sorted(df['timeStepIndex'].unique())
    num_ticks = len(ticks)
    num_agents = int(df['advertiserNumber'].nunique())
    logger.info(f'  {len(df)} rows, {num_ticks} ticks, {num_agents} agents')

    pv_values_list = []
    pv_sigmas_list = []
    lwc_list = []

    for tick in ticks:
        tick_df = df[df['timeStepIndex'] == tick]
        # Pivot: (n_pv, 48) — group by pvIndex, sort by advertiserNumber
        pv_pivot = tick_df.pivot_table(
            index='pvIndex', columns='advertiserNumber',
            values='pValue', aggfunc='first')
        sigma_pivot = tick_df.pivot_table(
            index='pvIndex', columns='advertiserNumber',
            values='pValueSigma', aggfunc='first')
        lwc_pivot = tick_df.pivot_table(
            index='pvIndex', columns='advertiserNumber',
            values='leastWinningCost', aggfunc='first')

        pv_values_list.append(pv_pivot.values.astype(np.float64))
        pv_sigmas_list.append(sigma_pivot.values.astype(np.float64))
        lwc_list.append(lwc_pivot.values.astype(np.float64))

    return pv_values_list, pv_sigmas_list, lwc_list, num_ticks, num_agents


def build_agent_lineup(player_agent_factory, player_index, num_agents, pv_values_list):
    """Build strategy lineup: player at slot, others use project defaults."""
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    from simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
    from simul_bidding_env.strategy.bc_bidding_strategy import BcBiddingStrategy
    from simul_bidding_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy
    from simul_bidding_env.strategy.player_agent_wrapper import PlayerAgentWrapper
    from simul_bidding_env.Controller.Controller import Controller

    # Get budget/CPA from Controller
    ctrl = Controller(player_index=player_index, player_agent=PidBiddingStrategy(exp_tempral_ratio=np.ones(48)))
    budgets = ctrl.budget_list
    cpas = ctrl.cpa_constraint_list
    categories = np.arange(num_agents) // 8

    # Use project default strategies for all non-player slots
    default_agents = ctrl.agent_list  # this has 48 agents from initialize_agents()

    # Replace player slot
    player_agent = player_agent_factory(
        budget=budgets[player_index],
        cpa=cpas[player_index],
        category=categories[player_index])
    player_agent.budget = budgets[player_index]
    player_agent.cpa = cpas[player_index]
    player_agent.category = categories[player_index]
    default_agents[player_index] = PlayerAgentWrapper(player_agent=player_agent)

    # Assign budget/CPA to all
    for i in range(num_agents):
        default_agents[i].budget = budgets[i]
        default_agents[i].cpa = cpas[i]
        default_agents[i].category = categories[i]
        default_agents[i].remaining_budget = budgets[i]

    return default_agents, budgets, cpas, categories


def run_arena(player_agent_factory, player_index, pv_values_list, pv_sigmas_list, num_ticks):
    """Run online auction with custom agent lineup against historical pv data."""
    from simul_bidding_env.Environment.BiddingEnv import BiddingEnv
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    from simul_bidding_env.strategy.player_agent_wrapper import PlayerAgentWrapper
    from simul_bidding_env.Controller.Controller import Controller

    num_agents = pv_values_list[0].shape[1]

    # Build agent lineup
    agents, budgets, cpas, categories = build_agent_lineup(
        player_agent_factory, player_index, num_agents, pv_values_list)

    envs = BiddingEnv()
    envs.reset(episode=0)

    rewards = np.zeros(num_agents)
    costs = np.zeros(num_agents)

    history_pvalue_infos = []
    history_bids = []
    history_auction_results = []
    history_impression_results = []
    history_least_winning_costs = []

    t0 = time.time()
    for tick_index in range(num_ticks):
        pv_values = pv_values_list[tick_index]
        pv_sigmas = pv_sigmas_list[tick_index]
        # Ensure no NaN/zero sigma (truncnorm requires positive scale)
        pv_sigmas = np.nan_to_num(pv_sigmas, nan=0.0)
        pv_sigmas = np.maximum(pv_sigmas, 1e-8)

        bids = [
            agent.bidding(
                tick_index, pv_values[:, i], pv_sigmas[:, i],
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
                _adjust_over_cost(bids, over_cost_ratio,
                                  envs.slot_coefficients, winner_pit)

            xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, \
                least_winning_cost_pit, market_price_pit = \
                envs.simulate_ad_bidding(pv_values, pv_sigmas, bids)

            cost = (cost_pit * is_exposed_pit).sum(axis=1)
            reward = conversion_action_pit.sum(axis=1)
            winner_pit = _get_winner(slot_pit)
            over_cost_ratio = np.maximum(
                (cost - remaining_budget_list) / (cost + 1e-4), 0)
            ratio_max = over_cost_ratio.max()

        for i, agent in enumerate(agents):
            agent.remaining_budget -= cost[i]

        rewards += reward
        costs += cost

        history_bids.append(bids.transpose())
        history_least_winning_costs.append(least_winning_cost_pit)
        history_pvalue_infos.append(
            np.stack((pv_values.T, pv_sigmas.T), axis=-1))
        history_auction_results.append(
            np.stack((xi_pit, slot_pit, cost_pit), axis=-1))
        history_impression_results.append(
            np.stack((is_exposed_pit, conversion_action_pit), axis=-1))

    elapsed = time.time() - t0
    logger.info(f'  Done in {elapsed:.1f}s')

    return rewards, costs, budgets, cpas, categories


def _get_winner(slot_pit):
    slot_pit = slot_pit.T
    num_pv, num_agent = slot_pit.shape
    winner = np.full((num_pv, 3), -1, dtype=int)
    for pos in range(1, 4):
        winning = np.argwhere(slot_pit == pos)
        if winning.size > 0:
            pv_idx, agent_idx = winning.T
            winner[pv_idx, pos - 1] = agent_idx
    return winner


def _adjust_over_cost(bids, over_cost_ratio, slot_coefs, winner_pit):
    import math
    overcost_indices = np.where(over_cost_ratio > 0)[0]
    for agent_index in overcost_indices:
        for i in range(len(slot_coefs)):
            winner_indices = winner_pit[:, i]
            pv_indices = np.where(winner_indices == agent_index)[0]
            rng = np.random.default_rng(seed=1)
            num_drop = math.ceil(pv_indices.size * over_cost_ratio[agent_index])
            if num_drop > 0:
                dropped = rng.choice(pv_indices, num_drop, replace=False)
                bids[dropped, agent_index] = 0


def compute_score(reward, cost, cpa_target):
    """Same score as AuctionNet."""
    eps = 1e-10
    cpa_real = cost / (reward + eps)
    beta = 2
    if cpa_real > cpa_target:
        penalty = (cpa_target / (cpa_real + eps)) ** beta
    else:
        penalty = 1.0
    return penalty * reward, cpa_real


def compare(player_index, agents_config):
    """
    Run arena for each agent config and print comparison.
    agents_config: list of (name, factory_function)
    """
    pv_values_list, pv_sigmas_list, lwc_list, num_ticks, num_agents = load_pv_data(CSV_PATH)

    all_results = {}
    for name, factory in agents_config:
        logger.info(f'\n{"="*50}')
        logger.info(f'Running {name} at slot {player_index} ...')
        logger.info(f'{"="*50}')
        rewards, costs, budgets, cpas, categories = run_arena(
            factory, player_index, pv_values_list, pv_sigmas_list, num_ticks)

        score, cpa_real = compute_score(
            rewards[player_index], costs[player_index], cpas[player_index])
        all_results[name] = {
            'reward': rewards[player_index],
            'cost': costs[player_index],
            'score': score,
            'cpa_real': cpa_real,
            'cpa_target': cpas[player_index],
            'budget': budgets[player_index],
            'budget_used': costs[player_index] / budgets[player_index],
        }
        logger.info(f'  {name}: reward={int(rewards[player_index])}, '
                    f'cost={costs[player_index]:.1f}, '
                    f'cpa={cpa_real:.2f} (target={cpas[player_index]}), '
                    f'score={score:.2f}')

    # ── Summary ──
    print(f'\n{"="*90}')
    print(f'ARENA RESULTS — Advertiser #{player_index} (budget={all_results[agents_config[0][0]]["budget"]}, '
          f'CPA={all_results[agents_config[0][0]]["cpa_target"]})')
    print(f'{"="*90}')
    header = f'{"Strategy":<20} {"Score":>10} {"Reward":>8} {"Cost":>10} {"CPA":>8} {"Budget%":>8}'
    print(header)
    print('-' * len(header))
    for name, _ in agents_config:
        r = all_results[name]
        print(f'{name:<20} {r["score"]:>10.2f} {int(r["reward"]):>8} '
              f'{r["cost"]:>10.1f} {r["cpa_real"]:>8.2f} {r["budget_used"]:>7.1%}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--player', type=int, default=0, help='advertiser slot for test strategies')
    args = p.parse_args()
    PLAYER_INDEX = args.player

    # ── Strategy lineup ─────────────────────────────────────────
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy

    def _pid(budget, cpa, category):
        return PidBiddingStrategy(budget=budget, cpa=cpa, category=category,
                                  name='PID', exp_tempral_ratio=np.ones(48))

    from simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
    def _iql(budget, cpa, category):
        return IqlBiddingStrategy(budget=budget, cpa=cpa, category=category, name='IQL')

    from simul_bidding_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy
    def _onlinelp(budget, cpa, category):
        return OnlineLpBiddingStrategy(budget=budget, cpa=cpa, category=category,
                                       name='OnlineLP', episode=0)

    # Try importing GAVE and DGAB
    try:
        from simul_bidding_env.strategy.autobidding_agents import (
            GAVEAuctionNetAgent, DGABFOAuctionNetAgent)
        BLOCK_CONFIG = {
            'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
            'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
            'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
        }
        GAVE_DIR = 'D:/research/Experiment/autobidding/saved_model/gave_400k_sparse'
        DGAB_DIR = 'D:/research/Experiment/autobidding/saved_model/dgab_400k_sparse'
        DEVICE = 'cpu'

        def _gave(budget, cpa, category):
            return GAVEAuctionNetAgent(
                budget=budget, cpa=cpa, category=category, name='GAVE',
                model_param=dict(
                    save_dir=GAVE_DIR, hidden_size=512, time_dim=8,
                    block_config=BLOCK_CONFIG, device=DEVICE,
                    expectile=0.99, score_target_mode='prev'))

        def _dgab(budget, cpa, category):
            return DGABFOAuctionNetAgent(
                budget=budget, cpa=cpa, category=category, name='DGAB-FO',
                model_param=dict(
                    save_dir=DGAB_DIR, hidden_size=512, max_ep_len=96,
                    time_dim=8, block_config=BLOCK_CONFIG, device=DEVICE,
                    actor_type='stack', critic_type='sequence'))
        HAS_CUSTOM = True
    except Exception as e:
        logger.warning(f'Custom agents not available: {e}')
        HAS_CUSTOM = False

    AGENTS = [('PID', _pid), ('IQL', _iql), ('OnlineLP', _onlinelp)]
    if HAS_CUSTOM:
        AGENTS.extend([('GAVE', _gave), ('DGAB-FO', _dgab)])

    compare(PLAYER_INDEX, AGENTS)
