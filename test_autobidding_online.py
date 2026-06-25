"""
Online comparison test: GAVE vs DGAB-FO in the same AuctionNet environment.

Usage:
    cd D:\research\Experiment\AuctionNet-main
    python test_autobidding_online.py

What it does:
    1. Fixes random seeds so traffic (PVs) is identical across runs
    2. Replaces the player agent with GAVE, runs one episode → records results
    3. Same episode config, replaces with DGAB-FO, runs again → records results
    4. Prints side-by-side comparison

Requirements:
    - GAVE checkpoint at saved_model/GAVE/complete_train.pt (autobidding project)
    - DGAB-FO checkpoint at saved_model/GAVEv2_csv_*/complete_train.pt
    - Adjust save_dir paths below to match your actual checkpoint locations
"""
import sys, os, time, math
import numpy as np
import gin
import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ──
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# NOTE: strategy_train_env must go LAST so root's 'run' is found before strategy_train_env's 'run'
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'strategy_train_env'))
sys.path.insert(0, _PROJECT_ROOT)

# ── Pre-import modules that gin needs (bypasses gin's internal import) ──
import run.run_test
import simul_bidding_env.Controller.Controller
import simul_bidding_env.Environment.BiddingEnv

# ── Gin config ──
gin_file = ["./config/test.gin"]
gin.parse_config_files_and_bindings(gin_file, None)

from run.run_test import (
    initialize_player_analysis, adjust_over_cost, get_winner,
    log_memory_usage,
)
from simul_bidding_env.Tracker.BiddingTracker import BiddingTracker
from simul_bidding_env.Controller.Controller import Controller

# ── Our agents ──
from simul_bidding_env.strategy.autobidding_agents import (
    GAVEAuctionNetAgent, DGABFOAuctionNetAgent,
)

# ═══════════════════════════════════════════════
# Config — adjust these paths
# ═══════════════════════════════════════════════

BLOCK_CONFIG = {
    'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
    'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
    'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
}

GAVE_SAVE_DIR = 'D:/research/Experiment/autobidding/saved_model/gave_cpu'
DGAB_SAVE_DIR = 'D:/research/Experiment/autobidding/saved_model/dgab_v2_scale'  # <-- UPDATE THIS

DEVICE = 'cuda:0' if __import__('torch').cuda.is_available() else 'cpu'
NUM_EPISODE = 1          # number of episodes to run per agent
FIXED_SEED = 42          # fixed seed for reproducibility


# ═══════════════════════════════════════════════
# Build agent factories
# ═══════════════════════════════════════════════

def make_gave_agent(budget, cpa, category):
    return GAVEAuctionNetAgent(
        budget=budget, cpa=cpa, category=category,
        name='GAVE-Player',
        model_param=dict(
            save_dir=GAVE_SAVE_DIR,
            hidden_size=512, time_dim=8,
            block_config=BLOCK_CONFIG,
            device=DEVICE,
            expectile=0.99,
            score_target_mode='prev',   # GAoVE (prev) mode
        ),
    )


def make_dgab_agent(budget, cpa, category):
    return DGABFOAuctionNetAgent(
        budget=budget, cpa=cpa, category=category,
        name='DGAB-FO-Player',
        model_param=dict(
            save_dir=DGAB_SAVE_DIR,
            hidden_size=512, max_ep_len=96, time_dim=8,
            block_config=BLOCK_CONFIG,
            device=DEVICE,
            actor_type='stack',          # or 'cross_attn'
            critic_type='sequence',
        ),
    )


def make_pid_agent(budget, cpa, category):
    """Classical PID controller."""
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    return PidBiddingStrategy(
        budget=budget, cpa=cpa, category=category,
        name='PID', exp_tempral_ratio=np.ones(48),
    )


def make_abid_agent(budget, cpa, category):
    """Fixed bid-rate strategy."""
    from simul_bidding_env.strategy.abid_bidding_strategy import AbidBiddingStrategy
    return AbidBiddingStrategy(
        budget=budget, cpa=cpa, category=category,
        name='Abid', exp_tempral_ratio=np.ones(48),
    )


def make_iql_agent(budget, cpa, category):
    """Implicit Q-Learning (offline RL)."""
    from simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
    return IqlBiddingStrategy(budget=budget, cpa=cpa, category=category, name='IQL')


def make_bc_agent(budget, cpa, category):
    """Behavioral Cloning."""
    from simul_bidding_env.strategy.bc_bidding_strategy import BcBiddingStrategy
    return BcBiddingStrategy(budget=budget, cpa=cpa, category=category, name='BC')


def make_bcq_agent(budget, cpa, category):
    """Batch-Constrained Q-Learning."""
    from simul_bidding_env.strategy.bcq_bidding_strategy import BcqBiddingStrategy
    return BcqBiddingStrategy(budget=budget, cpa=cpa, category=category, name='BCQ')


def make_td3bc_agent(budget, cpa, category):
    """TD3 + Behavioral Cloning."""
    from simul_bidding_env.strategy.td3_bc_bidding_strategy import TD3_BCBiddingStrategy
    return TD3_BCBiddingStrategy(budget=budget, cpa=cpa, category=category, name='TD3_BC')


def make_cql_agent(budget, cpa, category):
    """Conservative Q-Learning."""
    from simul_bidding_env.strategy.cql_bidding_strategy import CqlBiddingStrategy
    return CqlBiddingStrategy(budget=budget, cpa=cpa, category=category, name='CQL')


def make_mbrl_mopo_agent(budget, cpa, category):
    """Model-Based RL (MOPO)."""
    from simul_bidding_env.strategy.mbrl_mopo_bidding_strategy import MbrlMopoBiddingStrategy
    return MbrlMopoBiddingStrategy(budget=budget, cpa=cpa, category=category, name='MBRL_MOPO')


def make_mbrl_combo_agent(budget, cpa, category):
    """Model-Based RL (ComboMicro)."""
    from simul_bidding_env.strategy.mbrl_combomicro_bidding_strategy import MbrlComboMicroBiddingStrategy
    return MbrlComboMicroBiddingStrategy(budget=budget, cpa=cpa, category=category, name='MBRL_Combo')


def make_onlinelp_agent(budget, cpa, category):
    """Online Linear Programming (episode 0)."""
    from simul_bidding_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy
    return OnlineLpBiddingStrategy(budget=budget, cpa=cpa, category=category, name='OnlineLP', episode=0)


# ── All strategies for comparison ──
ALL_AGENTS = [
    # User's custom strategies
    ('GAVE',        make_gave_agent),
    ('DGAB-FO',     make_dgab_agent),
    # Project baselines — RL-based
    ('IQL',         make_iql_agent),
    ('BC',          make_bc_agent),
    ('BCQ',         make_bcq_agent),
    ('TD3_BC',      make_td3bc_agent),
    ('CQL',         make_cql_agent),
    ('MBRL_MOPO',   make_mbrl_mopo_agent),
    ('MBRL_Combo',  make_mbrl_combo_agent),
    # Project baselines — classical
    ('OnlineLP',    make_onlinelp_agent),
    ('PID',         make_pid_agent),
    ('Abid',        make_abid_agent),
]


# ═══════════════════════════════════════════════
# Single-episode runner
# ═══════════════════════════════════════════════

def run_one_episode(controller, envs, pv_generator, tracker,
                    episode, player_index, num_tick=48, generate_log=True):
    """Run a single episode. Returns (total_reward, total_cost, cpa_real, win_pv_ratio)."""
    agents = controller.agents
    num_agent = len(agents)
    agents_cpa = np.array([agent.cpa for agent in agents])
    agents_category = np.array([agent.category for agent in agents])

    if generate_log:
        tracker.reset()

    rewards = np.zeros(num_agent)
    costs = np.zeros(num_agent)
    budgets = np.array([agent.budget for agent in agents])

    history_pvalue_infos = []
    history_bids = []
    history_auction_results = []
    history_impression_results = []
    history_least_winning_costs = []

    controller.reset(episode=episode)
    total_pv_num = 0

    for tick_index in range(num_tick):
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

        remaining_budget_list = np.array([agent.remaining_budget for agent in agents])
        done_list = np.ones(len(agents), dtype=int) if tick_index == (num_tick - 1) else (
            remaining_budget_list < envs.min_remaining_budget
        ).astype(int)

        ratio_max = None
        while ratio_max is None or ratio_max > 0:
            if ratio_max and ratio_max > 0:
                over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 1e-4), 0)
                adjust_over_cost(bids, over_cost_ratio, envs.slot_coefficients, winner_pit)

            (xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit,
             least_winning_cost_pit, market_price_pit) = \
                envs.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)

            real_cost = cost_pit * is_exposed_pit
            cost = real_cost.sum(axis=1)
            reward = conversion_action_pit.sum(axis=1)

            winner_pit = get_winner(slot_pit)
            over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 1e-4), 0)
            ratio_max = over_cost_ratio.max()

        for i, agent in enumerate(agents):
            agent.remaining_budget -= cost[i]

        rewards += reward
        costs += cost

        history_bids.append(bids.transpose())
        history_least_winning_costs.append(least_winning_cost_pit)
        pvalue_info = np.stack((pv_values.T, pvalue_sigmas.T), axis=-1)
        history_pvalue_infos.append(pvalue_info)
        auction_info = np.stack((xi_pit, slot_pit, cost_pit), axis=-1)
        history_auction_results.append(auction_info)
        impression_info = np.stack((is_exposed_pit, conversion_action_pit), axis=-1)
        history_impression_results.append(impression_info)

        if generate_log:
            tracker.train_logging(
                episode, tick_index, pv_values, budgets, agents_cpa, agents_category,
                remaining_budget_list, total_pv_num, pvalue_sigmas, bids,
                xi_pit, slot_pit, cost_pit, is_exposed_pit,
                conversion_action_pit, least_winning_cost_pit, done_list
            )

    # Compute metrics
    player_reward = rewards[player_index]
    player_cost = costs[player_index]
    player_cpa_real = player_cost / (player_reward + 1e-10)
    player_cpa_target = agents_cpa[player_index]

    # Score (same as getScore_nips)
    beta = 2
    if player_cpa_real > player_cpa_target:
        penalty = (player_cpa_target / (player_cpa_real + 1e-10)) ** beta
    else:
        penalty = 1.0
    player_score = penalty * player_reward

    # Win PV ratio
    total_pv = getattr(pv_generator, 'pv_num', getattr(pv_generator, 'PV_NUM', 0))
    total_won = 0
    for h in history_auction_results:
        total_won += int((h[player_index][:, 0] > 0.5).sum())  # xi col 0

    win_pv_ratio = total_won / (total_pv * num_tick * num_agent)  # across all agents

    result = {
        'score': player_score,
        'reward': int(player_reward),
        'cost': player_cost,
        'cpa_real': player_cpa_real,
        'cpa_target': player_cpa_target,
        'win_pv_ratio': win_pv_ratio,
        'budget_used': player_cost / budgets[player_index],
    }
    return result


# ═══════════════════════════════════════════════
# Main comparison
# ═══════════════════════════════════════════════

def main():
    # Fix seeds for reproducibility
    np.random.seed(FIXED_SEED)
    import torch
    torch.manual_seed(FIXED_SEED)

    logger.info(f'Device: {DEVICE}')
    logger.info(f'Seed: {FIXED_SEED}')
    logger.info(f'Episodes per agent: {NUM_EPISODE}')
    logger.info(f'GAVE save_dir: {GAVE_SAVE_DIR}')
    logger.info(f'DGAB save_dir: {DGAB_SAVE_DIR}')

    # We need a dummy default agent to initialize the Controller
    # (the Controller will replace it with our agent)
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    dummy_agent = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
    dummy_agent.name += "0"

    player_index = 0

    all_results = {name: [] for name, _ in ALL_AGENTS}
    tracker = BiddingTracker("comparison_tracker")

    for ep in range(NUM_EPISODE):
        logger.info(f'\n{"="*60}')
        logger.info(f'Episode {ep}')
        logger.info(f'{"="*60}')

        for agent_name, agent_factory in ALL_AGENTS:
            # Fresh controller each run (same episode index → same PVs)
            controller = Controller(
                player_index=player_index,
                player_agent=dummy_agent,
            )
            envs = controller.biddingEnv
            pv_generator = controller.pvGenerator

            # Replace player with our agent
            player_agent = agent_factory(
                budget=controller.budget_list[player_index],
                cpa=controller.cpa_constraint_list[player_index],
                category=controller.category[player_index],
            )
            controller.player_agent = player_agent
            # Re-load agents so the player is wrapped correctly
            controller.agents = controller.load_agents()

            logger.info(f'  Running {agent_name} ... (budget={player_agent.budget}, '
                        f'cpa={player_agent.cpa})')
            t0 = time.time()
            result = run_one_episode(
                controller, envs, pv_generator, tracker,
                episode=ep, player_index=player_index,
            )
            elapsed = time.time() - t0
            logger.info(f'  {agent_name} done in {elapsed:.1f}s')
            logger.info(f'    score={result["score"]:.2f}  reward={result["reward"]}  '
                        f'cpa_real={result["cpa_real"]:.4f}  '
                        f'budget_used={result["budget_used"]:.1%}')
            all_results[agent_name].append(result)

    # ── Summary ──
    agents_order = [name for name, _ in ALL_AGENTS]
    col_w = 15
    sep_w = 85 + (len(agents_order) - 3) * col_w
    print(f'\n{"="*sep_w}')
    print('COMPARISON SUMMARY')
    print(f'{"="*sep_w}')
    header = f'{"Metric":<25}' + ''.join(f'{name:>{col_w}}' for name in agents_order)
    print(header)
    print('-' * len(header))

    for metric, key, fmt in [
        ('Score', 'score', '.2f'),
        ('Reward (conversions)', 'reward', 'd'),
        ('Cost', 'cost', '.2f'),
        ('CPA real', 'cpa_real', '.4f'),
        ('CPA target', 'cpa_target', '.2f'),
        ('Budget used %', 'budget_used', '.1%'),
    ]:
        row = f'{metric:<25}'
        for name in agents_order:
            v = np.mean([r[key] for r in all_results[name]])
            if fmt == 'd':
                v = int(v)
            row += f'{v:>{col_w}{fmt}}'
        print(row)

    print()
    logger.info('Done.')


if __name__ == '__main__':
    main()
