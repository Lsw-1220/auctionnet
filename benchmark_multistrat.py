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
import sys, os, time, argparse, logging, hashlib, json, datetime
import numpy as np
try:
    import gin
except ImportError:
    # The SemBid environment has the required CUDA/NumPy stack; reuse only the
    # lightweight gin package from the existing benchmark environment.
    sys.path.append(r'C:\Users\22397\.conda\envs\nips-Gebidding-env\Lib\site-packages')
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
    GASAuctionNetAgent, GASPaperAuctionNetAgent,
)
from simul_bidding_env.strategy.vgab_bidding_strategy import VGABStrategy
from simul_bidding_env.strategy.dgabshare_bidding_strategy import DGABShareStrategy

# ═══════════════════════════════════════════════
# Config (defaults — override via CLI)
# ═══════════════════════════════════════════════

BLOCK_CONFIG = {
    'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
    'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
    'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
}

GAVE_SAVE_DIR = '../GAVE/GAVE/code/saved_model/DTtest_20260901154322'
DGAB_SAVE_DIR   = './saved_model/dgab_v3'
VGAB_SAVE_DIR   = './saved_model/vgab'
DT_SAVE_DIR     = './saved_model/DTdense'
IQL_SAVE_DIR    = './saved_model/IQL/checkpoint_00006000'
BC_SAVE_DIR     = './saved_model/BC/checkpoint_00004000'
BCQ_SAVE_DIR    = './saved_model/BCQ/checkpoint_00001000'
TD3_BC_SAVE_DIR = './saved_model/td3_bc/checkpoint_00010000'
GUIDE_SAVE_DIR  = './strategy_train_env/saved_model/9000'
DGABSHARE_SAVE_DIR = './saved_model/dgabshare_full'
DGABSHARE_CKPT = 'step_15000.pt'
EXPLORATION_SCALE = 0.0
EXPLORATION_RHO = 0.8
V_GOAL_MULTIPLIER = 1.0
GAS_SAVE_DIR = 'D:/research/Experiment/GAS_WWW-25/results/gas_dt_reweight_b48_s400000/checkpoints/step_280000'
QGA_SAVE_DIR = './saved_model/QGA_dense/QGA'
QGA_CKPT_STEP = 6000
SEMBID_ROOT = 'D:/research/Experiment/SemBid-CIKM2026'
SEMBID_SAVE_DIR = os.path.join(SEMBID_ROOT, 'reproduction', 'sembid_200k', 'model_batch48_200k', 'checkpoint_50000')
SEMBID_EMBEDDING_LOOKUP = os.path.join(SEMBID_ROOT, 'reproduction', 'sembid_200k', 'embedding_lookup.pkl')
OUTPUT_DIR      = None  # None → use {_PROJECT_ROOT}/exp_data

NUM_ADVERTISERS = 48
NUM_TICK = 48
FIXED_SEED = 42
RTG_V_CAP = 10

#DEFAULT_PV_SWEEP = [0.0003, 0.0005, 0.0007, 0.0009, 0.001, 0.003, 0.005, 0.007]
DEFAULT_PV_SWEEP = [0.0005]
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
    ap.add_argument('--pv_num', type=int, default=500000,
                    help='Total simulated PVs per environment episode')
    ap.add_argument('--budget_rate', type=float, default=1.0,
                    help='Multiplier applied to every advertiser budget (default: 1.0)')
    ap.add_argument('--episodes', nargs='*', type=int, default=[0],
                    help='AuctionNet environment episode ids (default: 0)')
    ap.add_argument('--strategies', nargs='*', default=['DGABShare'],
                    help='Strategies to run; default: DGABShare')
    ap.add_argument('--exploration_scale', type=float, default=0.0,
                    help='DGABShare AR(1) log-action exploration std; 0 disables it')
    ap.add_argument('--exploration_rho', type=float, default=0.8,
                    help='DGABShare AR(1) exploration correlation')
    ap.add_argument('--v_goal_multiplier', type=float, default=V_GOAL_MULTIPLIER,
                    help='DGABShare value-goal multiplier (default: 1.0)')
    ap.add_argument('--trajectory_output', type=str, default='',
                    help='Optional per-tick CSV path for post-training trajectories')
    ap.add_argument('--fail_fast', action='store_true',
                    help='Raise the first evaluation error instead of recording a failed row')
    # Model paths
    ap.add_argument('--gave_dir', type=str, default=None,
                    help='GAVE model directory')
    ap.add_argument('--dgab_dir', type=str, default=None,
                    help='DGAB model directory')
    ap.add_argument('--vgab_dir', type=str, default=None,
                    help='VGAB model directory')
    ap.add_argument('--dt_dir', type=str, default=None,
                    help='DT model directory')
    ap.add_argument('--iql_dir', type=str, default=None,
                    help='IQL model directory')
    ap.add_argument('--bc_dir', type=str, default=None,
                    help='BC model directory')
    ap.add_argument('--bcq_dir', type=str, default=None,
                    help='BCQ model directory')
    ap.add_argument('--td3_bc_dir', type=str, default=None,
                    help='TD3+BC model directory')
    ap.add_argument('--guide_dir', type=str, default=None,
                    help='GUIDE model directory')
    ap.add_argument('--dgabshare_dir', type=str, default=None,
                    help='DGABShare model directory (actor.pt + normalize_dict.pkl)')
    ap.add_argument('--dgabshare_ckpt', type=str, default=DGABSHARE_CKPT,
                    help='DGABShare checkpoint filename (default: step_15000.pt)')
    ap.add_argument('--gas_dir', type=str, default=None,
                    help='GAS model directory (dt.pt + normalize_dict.pkl)')
    ap.add_argument('--qga_dir', type=str, default=None,
                    help='QGA model directory (normalize_dict.pkl + checkpoints/)')
    ap.add_argument('--qga_ckpt_step', type=int, default=QGA_CKPT_STEP,
                    help='QGA actor/critic checkpoint step (default: 13000)')
    ap.add_argument('--sembid_dir', type=str, default=None)
    ap.add_argument('--sembid_embedding_lookup', type=str, default=None)
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
    from bidding_train_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
    return IqlBiddingStrategy(budget=budget, cpa=cpa, category=category, name='IQL',
                              model_dir=IQL_SAVE_DIR)


def make_bc(budget, cpa, category, **kw):
    from bidding_train_env.strategy.bc_bidding_strategy import BcBiddingStrategy
    return BcBiddingStrategy(budget=budget, cpa=cpa, category=category,
                             name='BC', model_dir=BC_SAVE_DIR)


def make_bcq(budget, cpa, category, **kw):
    from bidding_train_env.strategy.bcq_bidding_strategy import BcqBiddingStrategy
    return BcqBiddingStrategy(budget=budget, cpa=cpa, category=category,
                              name='BCQ', model_dir=BCQ_SAVE_DIR)


def make_td3_bc(budget, cpa, category, **kw):
    from bidding_train_env.strategy.td3_bc_bidding_strategy import TD3_BCBiddingStrategy
    return TD3_BCBiddingStrategy(budget=budget, cpa=cpa, category=category,
                                 name='TD3_BC', model_dir=TD3_BC_SAVE_DIR)


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
            ckpt_name='2000.pt',
            hidden_size=512, time_dim=8,
            block_config=BLOCK_CONFIG,
            device=DEVICE,
            expectile=0.99,
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


def make_vgab(budget, cpa, category, pvalue_mean_base=None, **kw):
    mp = dict(
        save_dir=VGAB_SAVE_DIR,
        device=DEVICE
    )
    if pvalue_mean_base is not None:
        mp['pvalue_mean_base'] = pvalue_mean_base
    return VGABStrategy(
        budget=budget, cpa=cpa, category=category,
        name='VGAB-Player',
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


def make_dgabshare(budget, cpa, category, **kw):
    return DGABShareStrategy(
        budget=budget, cpa=cpa, category=category,
        name='DGABShare',
        model_param=dict(
            save_dir=DGABSHARE_SAVE_DIR, ckpt_name=DGABSHARE_CKPT, device=DEVICE,
            K=20, v_goal_multiplier=V_GOAL_MULTIPLIER,
            exploration_scale=EXPLORATION_SCALE,
            exploration_rho=EXPLORATION_RHO,
            exploration_seed=int(kw.get('exploration_seed', 0)),
            exploration_min_ratio=0.8, exploration_max_ratio=1.2,
        ),
    )


def make_gas(budget, cpa, category, **kw):
    return GASPaperAuctionNetAgent(
        budget=budget, cpa=cpa, category=category,
        name='GAS',
        model_param=dict(
            bundle_dir=GAS_SAVE_DIR,
            device=DEVICE,
            action_num=5,
            seed=int(kw.get('exploration_seed', 42)),
        ),
    )


def make_qga(budget, cpa, category, **kw):
    """Build the QGA agent and adapt its (bids, alpha) return to AuctionNet."""
    from bidding_train_env.strategy.qga_bidding_strategy import QGAStrategy

    actor_path = os.path.join(
        QGA_SAVE_DIR, 'checkpoints', f'actor_step_{QGA_CKPT_STEP}.pt')
    critic_path = os.path.join(
        QGA_SAVE_DIR, 'checkpoints', f'critic_step_{QGA_CKPT_STEP}.pt')

    class QGAAuctionNetAdapter(QGAStrategy):
        def bidding(self, *args, **kwargs):
            bids, _alpha = super().bidding(*args, **kwargs)
            return bids

    return QGAAuctionNetAdapter(
        budget=budget, cpa=cpa, category=category, name='QGA',
        load_dir=QGA_SAVE_DIR, actor_path=actor_path, critic_path=critic_path,
        device=DEVICE,
    )


def make_sembid(budget, cpa, category, **kw):
    """Load ckpt_50000 and adapt SemBid's evaluator to AuctionNet's online API."""
    import importlib.util
    import pickle
    testing_dir = os.path.join(SEMBID_ROOT, 'code', 'Testing')
    for path in (testing_dir, os.path.join(SEMBID_ROOT, 'code', 'Algorithms'),
                 os.path.join(SEMBID_ROOT, 'code')):
        if path not in sys.path:
            sys.path.insert(0, path)
    module_name = '_sembid_auctionnet_test'
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(testing_dir, 'test.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]
    with open(SEMBID_EMBEDDING_LOOKUP, 'rb') as handle:
        embedding_lookup = pickle.load(handle)
    generator_cls = module.load_language_generator('sembid_templates_high')

    class SemBidAuctionNetAdapter(module.BiddingStrategy):
        def __init__(self):
            super().__init__(model_dir=SEMBID_SAVE_DIR, budget=budget, cpa=cpa,
                             category=category, device=DEVICE, language_emb_dim=2048,
                             embedding_lookup=embedding_lookup,
                             language_generator_cls=generator_cls)
            self.name = 'SemBid'

        def reset(self):
            self.remaining_budget = self.budget
            self.bid_mean_hist, self.conv_mean_hist = [], []
            self.precomputed_stats = None
            self.reset_buffer()

        def bidding(self, timeStepIndex, pValues, pValueSigmas,
                    historyPValueInfo, historyBid, historyAuctionResult,
                    historyImpressionResult, historyLeastWinningCost):
            if timeStepIndex > len(self.bid_mean_hist) and historyBid:
                self.bid_mean_hist.append(float(np.mean(historyBid[-1])))
                conversions = np.asarray(historyImpressionResult[-1])[..., 1]
                self.conv_mean_hist.append(float(np.mean(conversions)))
                self.last_reward = float(np.sum(conversions))
            horizon = NUM_TICK
            pvalue_mean = np.zeros(horizon, np.float32)
            lwc_mean = np.zeros(horizon, np.float32)
            xi_mean = np.zeros(horizon, np.float32)
            volume = np.zeros(horizon, np.float32)
            for t, info in enumerate(historyPValueInfo[-horizon:]):
                arr = np.asarray(info); pvalue_mean[t] = np.mean(arr[..., 0]); volume[t] = arr.shape[0]
            for t, costs in enumerate(historyLeastWinningCost[-horizon:]):
                lwc_mean[t] = np.mean(costs)
            for t, result in enumerate(historyAuctionResult[-horizon:]):
                xi_mean[t] = np.mean(np.asarray(result)[..., 0])
            pvalue_mean[timeStepIndex] = np.mean(pValues)
            volume[timeStepIndex] = len(pValues)
            historical_volume = np.array([np.sum(volume[:t]) for t in range(horizon)], np.float32)
            last3_volume = np.array([np.sum(volume[max(0, t-3):t]) for t in range(horizon)], np.float32)
            self.precomputed_stats = dict(pvalue_mean=pvalue_mean, lwc_mean=lwc_mean,
                xi_mean=xi_mean, volume=volume, historical_volume=historical_volume,
                last3_volume=last3_volume)
            return super().bidding(timeStepIndex, pValues, pValueSigmas, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost)

    return SemBidAuctionNetAdapter()


ALL_STRATEGIES = [
    ('PID', make_pid),
    ('GAVE', make_gave),
    ('BC', make_bc),
    ('BCQ', make_bcq),
    ('IQL', make_iql),
    ('TD3_BC', make_td3_bc),
    ('DGABShare', make_dgabshare),
    ('QGA', make_qga),
    ('SemBid', make_sembid),
]


# ═══════════════════════════════════════════════
# Single-episode runner (simplified — no tick logging)
# ═══════════════════════════════════════════════

def run_one_episode(controller, player_index, agent_factory, pvalue_mean_base,
                    episode=0, exploration_seed=0):
    """Run one 48-tick episode and return summary metrics dict."""
    envs = controller.biddingEnv
    pv_generator = controller.pvGenerator

    # ── Create and inject player agent ──
    player_agent = agent_factory(
        budget=controller.budget_list[player_index],
        cpa=controller.cpa_constraint_list[player_index],
        category=controller.category[player_index],
        pvalue_mean_base=pvalue_mean_base,
        exploration_seed=exploration_seed,
    )
    controller.player_agent = player_agent
    agents = controller.load_agents()

    # ── Reset ──
    controller.reset(episode=episode)
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
    tick_rows = []

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

        diagnostics = dict(getattr(agents[player_index], 'last_diagnostics', {}))
        tick_rows.append(dict(
            episode=episode, advertiser=player_index, timestep=tick_index,
            pvalue_mean_base=pvalue_mean_base,
            budget=float(budgets[player_index]),
            cpa_target=float(agents_cpa[player_index]),
            tick_conversion=float(reward[player_index]),
            tick_cost=float(cost[player_index]),
            cumulative_conversion=float(rewards[player_index]),
            cumulative_cost=float(costs[player_index]),
            remaining_budget=float(agents[player_index].remaining_budget),
            slot1_win=int(np.sum(slot_pit[player_index] == 1)),
            slot2_win=int(np.sum(slot_pit[player_index] == 2)),
            slot3_win=int(np.sum(slot_pit[player_index] == 3)),
            exposure_count=int(np.sum(is_exposed_pit[player_index])),
            least_winning_cost_mean=float(np.mean(least_winning_cost_pit)),
            done=bool(tick_index == NUM_TICK - 1 or
                      agents[player_index].remaining_budget < envs.min_remaining_budget),
            **diagnostics,
        ))

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

    result = {
        'score': player_score,
        'reward': int(player_reward),
        'cost': player_cost,
        'cpa_real': player_cpa_real,
        'cpa_target': player_cpa_target,
        'budget_used': player_cost / budgets[player_index],
    }
    return result, tick_rows


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    global DEVICE, GAVE_SAVE_DIR, DGAB_SAVE_DIR, VGAB_SAVE_DIR, DT_SAVE_DIR, IQL_SAVE_DIR, BC_SAVE_DIR, BCQ_SAVE_DIR, TD3_BC_SAVE_DIR, GUIDE_SAVE_DIR, DGABSHARE_SAVE_DIR, DGABSHARE_CKPT, GAS_SAVE_DIR, QGA_SAVE_DIR, QGA_CKPT_STEP, SEMBID_SAVE_DIR, SEMBID_EMBEDDING_LOOKUP, OUTPUT_DIR, EXPLORATION_SCALE, EXPLORATION_RHO, V_GOAL_MULTIPLIER

    args = parse_args()

    DEVICE = args.device or ('cuda:0' if __import__('torch').cuda.is_available() else 'cpu')
    pv_sweep = args.pv or DEFAULT_PV_SWEEP
    advertisers = args.advertisers if args.advertisers is not None else list(range(NUM_ADVERTISERS))
    output_prefix = args.output
    seed = args.seed
    episodes = args.episodes
    if not np.isfinite(args.budget_rate) or args.budget_rate <= 0:
        ap_error = '--budget_rate must be a finite number greater than 0'
        raise ValueError(ap_error)

    # Apply CLI path overrides
    if args.gave_dir:   GAVE_SAVE_DIR = args.gave_dir
    if args.dgab_dir:   DGAB_SAVE_DIR = args.dgab_dir
    if args.vgab_dir:   VGAB_SAVE_DIR = args.vgab_dir
    if args.dt_dir:     DT_SAVE_DIR   = args.dt_dir
    if args.iql_dir:    IQL_SAVE_DIR    = args.iql_dir
    if args.bc_dir:     BC_SAVE_DIR     = args.bc_dir
    if args.bcq_dir:    BCQ_SAVE_DIR    = args.bcq_dir
    if args.td3_bc_dir: TD3_BC_SAVE_DIR = args.td3_bc_dir
    if args.guide_dir:  GUIDE_SAVE_DIR  = args.guide_dir
    if args.dgabshare_dir: DGABSHARE_SAVE_DIR = args.dgabshare_dir
    DGABSHARE_CKPT = args.dgabshare_ckpt
    EXPLORATION_SCALE = args.exploration_scale
    EXPLORATION_RHO = args.exploration_rho
    V_GOAL_MULTIPLIER = args.v_goal_multiplier
    if args.gas_dir: GAS_SAVE_DIR = args.gas_dir
    if args.qga_dir: QGA_SAVE_DIR = args.qga_dir
    QGA_CKPT_STEP = args.qga_ckpt_step
    if args.sembid_dir: SEMBID_SAVE_DIR = args.sembid_dir
    if args.sembid_embedding_lookup: SEMBID_EMBEDDING_LOOKUP = args.sembid_embedding_lookup
    OUTPUT_DIR = args.output_dir  # None → use default below

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    logger.info(f'Device: {DEVICE}  Seed: {seed}')
    logger.info(f'PV sweep ({len(pv_sweep)}): {pv_sweep}')
    logger.info(f'Advertisers ({len(advertisers)}): {advertisers if len(advertisers) <= 10 else f"{advertisers[:5]}...{advertisers[-2:]}" }')
    strategy_map = dict(ALL_STRATEGIES)
    unknown = sorted(set(args.strategies) - set(strategy_map))
    if unknown:
        raise ValueError(f'Unknown strategies: {unknown}; choices={sorted(strategy_map)}')
    strategies = [(name, strategy_map[name]) for name in args.strategies]
    logger.info(f'Strategies: {[n for n, _ in strategies]}')
    logger.info(f'Environment episodes: {episodes}')
    logger.info(f'Budget rate: {args.budget_rate}')
    logger.info(f'Total episodes: {len(pv_sweep) * len(advertisers) * len(strategies) * len(episodes)}')
    logger.info(f'Model dirs — GAVE: {GAVE_SAVE_DIR}  DGAB: {DGAB_SAVE_DIR}  DT: {DT_SAVE_DIR}  IQL: {IQL_SAVE_DIR}  GUIDE: {GUIDE_SAVE_DIR}')

    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    dummy_agent = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
    dummy_agent.name += "0"

    # ── Main loop ──
    results = []
    trajectory_rows = []
    total = len(pv_sweep) * len(advertisers) * len(strategies) * len(episodes)
    done = 0

    for pv_val in pv_sweep:
        logger.info(f'\n{"#"*60}\n  PVALUE_MEAN_BASE = {pv_val:.4f}\n{"#"*60}')

        for episode in episodes:
          for adv in advertisers:
            for name, factory in strategies:
                # Switch player_index by recreating controller with new index
                # (Controller hardcodes player_index at construction)
                controller = Controller(player_index=adv, player_agent=dummy_agent,
                                        pv_num=args.pv_num)
                controller.budget_list = [
                    budget * args.budget_rate for budget in controller.budget_list
                ]

                t0 = time.time()
                try:
                    res, ticks = run_one_episode(
                        controller, adv, factory, pv_val, episode=episode,
                        exploration_seed=seed * 1000003 + episode * 1009 + adv)
                    status, error = 'ok', ''
                except Exception as e:
                    logger.error(f'  [{name}] adv={adv} pv={pv_val} FAILED: {e}')
                    if args.fail_fast:
                        raise
                    res = {'score': 0, 'reward': 0, 'cost': 0,
                           'cpa_real': float('inf'), 'cpa_target': 0, 'budget_used': 0}
                    ticks, status, error = [], 'failed', repr(e)

                elapsed = time.time() - t0
                done += 1
                res['pvalue_mean_base'] = pv_val
                res['budget_rate'] = args.budget_rate
                res['strategy'] = name
                res['advertiser'] = adv
                res['episode'] = episode
                res['status'] = status
                res['error'] = error
                results.append(res)
                for row in ticks:
                    row['strategy'] = name
                    trajectory_rows.append(row)

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
    ok_df = df[df['status'] == 'ok']
    summary = ok_df.groupby(['pvalue_mean_base', 'strategy'])[metric_cols].mean().reset_index()
    out_summary = os.path.join(out_dir, f'{output_prefix}_summary.csv')
    summary.to_csv(out_summary, index=False)
    logger.info(f'Summary saved to {out_summary}')

    # Comparison pivot: avg score by (pv x strategy)
    pivot = ok_df.groupby(['pvalue_mean_base', 'strategy'])['score'].mean().unstack('strategy')
    out_comp = os.path.join(out_dir, f'{output_prefix}_comparison.csv')
    pivot.to_csv(out_comp)
    logger.info(f'Comparison pivot saved to {out_comp}')
    if args.trajectory_output:
        trajectory_path = os.path.abspath(args.trajectory_output)
        os.makedirs(os.path.dirname(trajectory_path), exist_ok=True)
        pd.DataFrame(trajectory_rows).to_csv(trajectory_path, index=False)
        logger.info(f'Per-tick trajectories saved to {trajectory_path}')

    ckpt_path = os.path.join(DGABSHARE_SAVE_DIR, DGABSHARE_CKPT)
    manifest = dict(
        created_at=datetime.datetime.now().isoformat(), command=' '.join(sys.argv),
        model_dir=os.path.abspath(DGABSHARE_SAVE_DIR), checkpoint=DGABSHARE_CKPT,
        checkpoint_sha256=(hashlib.sha256(open(ckpt_path, 'rb').read()).hexdigest()
                           if os.path.isfile(ckpt_path) else None),
        episodes=episodes, advertisers=advertisers, pv_sweep=pv_sweep,
        seed=seed, exploration_scale=EXPLORATION_SCALE,
        exploration_rho=EXPLORATION_RHO, pv_num=args.pv_num,
        budget_rate=args.budget_rate,
    )
    with open(os.path.join(out_dir, f'{output_prefix}_manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print('\n=== Avg Score Comparison ===')
    print(pivot.to_string())
    print(f'\nDone. {done} episodes total.')


if __name__ == '__main__':
    main()
