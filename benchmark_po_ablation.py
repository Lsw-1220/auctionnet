"""
DGAB-PO online ablation benchmark (R0-R5) in the AuctionNet live auction.

Runs each PO ablation config as the player agent against the standard
47-opponent AuctionNet lineup. Traffic, exposure and conversion draws are
deterministic per episode (NeurIPSPvGen / BiddingEnv use fixed per-episode
seeds), and np/torch are re-seeded before every run, so all configs face
IDENTICAL market conditions — differences in score are attributable to the
ablation factor alone.

Ablation matrix (config is self-described by each checkpoint's normalize_dict.pkl):
    r0 : rl-only             + stack DT       (base_state=2)    纯自有量能
    r1 : rl + stat_t         + stack DT       (base_state=18)   +窗口统计特征
    r2 : rl + cls(obs)       + stack DT       (base_state=2+H)  +原始曝光集(cls池化)
    r3 : rl + bidformer(obs) + stack DT       (base_state=2+H)  +分布结构编码
    r4 : rl + stat_t         + cross-attn DT  (base_state=18)   只动结构(RTG做query)
    r5 : rl + stat_t + bidformer + cross-attn (base_state=18+H) 完整方案
Optional reference anchors: pid (classical), fo (DGAB-FO, 完全可观测上参照).
r2r5 (ensemble): R2 (dense, cls) + R5 (sparse, bidformer) 稀疏度自适应.

Usage:
    python benchmark_po_ablation.py
    python benchmark_po_ablation.py --configs r0,r1,r4 --episodes 2
    python benchmark_po_ablation.py --configs r2,r5,r2r5 --pv 0.0003 0.001 0.005 0.007
    python benchmark_po_ablation.py --configs r0,r1,r2,r3,r4,r5,r2r5,pid,fo \
        --pv 0.0005 0.001 --device cuda:0
    python benchmark_po_ablation.py --player all         # 循环全部 48 个广告主
    python benchmark_po_ablation.py --player 0,4,24      # 指定广告主子集
    python benchmark_po_ablation.py --model_root D:/path/to/saved_model
    python benchmark_po_ablation.py --configs r2r5 --sparsity_threshold -0.5
"""
import sys, os, time, argparse, logging
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ──
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# NOTE: strategy_train_env must go LAST so root's 'run' is found before strategy_train_env's 'run'
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'strategy_train_env'))
sys.path.insert(0, _PROJECT_ROOT)

# ── Pre-import modules that gin needs (bypasses gin's internal import) ──
import gin
import run.run_test
import simul_bidding_env.Controller.Controller
import simul_bidding_env.Environment.BiddingEnv

gin.parse_config_files_and_bindings([os.path.join(_PROJECT_ROOT, 'config', 'test.gin')], None)

from run.run_test import adjust_over_cost, get_winner
from simul_bidding_env.Controller.Controller import Controller

# ═══════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════

BLOCK_CONFIG = {
    'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
    'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
    'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
}

DEFAULT_MODEL_ROOT = 'D:/research/Experiment/autobidding/saved_model'
DEFAULT_FO_DIR     = 'D:/research/Experiment/autobidding/saved_model/dgab_v3_20260701062347'

PO_CONFIGS = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7']
ENSEMBLE_CONFIGS = ['r2r5']
FUSION_CONFIGS = ['fusion']


# ═══════════════════════════════════════════════
# Agent factories
# ═══════════════════════════════════════════════

def make_po_agent(cfg, budget, cpa, category, args):
    from simul_bidding_env.strategy.autobidding_agents import DGABPOAuctionNetAgent
    # Allow per-config path override (e.g. --r5_dir)
    override_attr = f'{cfg}_dir'
    if hasattr(args, override_attr) and getattr(args, override_attr):
        save_dir = getattr(args, override_attr)
    else:
        save_dir = os.path.join(args.model_root, f'dgab_po_{cfg}')
    mp = dict(
        save_dir=save_dir,
        hidden_size=512, max_ep_len=96, time_dim=8,
        block_config=BLOCK_CONFIG,
        device=args.device,
        K=20,
        max_imp=args.max_imp,
    )
    if args.rtg_v_cap is not None:
        mp['rtg_v_cap'] = args.rtg_v_cap
    return DGABPOAuctionNetAgent(
        budget=budget, cpa=cpa, category=category,
        name=f'DGAB-PO-{cfg.upper()}', model_param=mp)


def make_fo_agent(budget, cpa, category, args):
    """DGAB-FO — 完全可观测(16维手工状态)参照,信息上界的近似锚点。"""
    from simul_bidding_env.strategy.autobidding_agents import DGABFOAuctionNetAgent
    mp = dict(
        save_dir=args.fo_dir,
        hidden_size=512, max_ep_len=96, time_dim=8,
        block_config=BLOCK_CONFIG,
        device=args.device,
        actor_type='stack', critic_type='sequence',
        critic_alpha=1.0,
    )
    if args.rtg_v_cap is not None:
        mp['rtg_v_cap'] = args.rtg_v_cap
    return DGABFOAuctionNetAgent(
        budget=budget, cpa=cpa, category=category,
        name='DGAB-FO', model_param=mp)


def make_pid_agent(budget, cpa, category, args):
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    return PidBiddingStrategy(budget=budget, cpa=cpa, category=category,
                              name='PID', exp_tempral_ratio=np.ones(48))


def make_ensemble_agent(budget, cpa, category, args):
    """R2 (dense, cls-token) + R5 (sparse, BidFormer) sparsity-gated ensemble."""
    from simul_bidding_env.strategy.autobidding_agents import DGABEnsembleAuctionNetAgent
    mp = dict(
        hidden_size=512, max_ep_len=96, time_dim=8,
        block_config=BLOCK_CONFIG,
        device=args.device,
        K=20,
        max_imp=args.max_imp,
        dense_model=dict(
            save_dir=os.path.join(args.model_root, 'dgab_po_r2'),
        ),
        sparse_model=dict(
            save_dir=os.path.join(args.model_root, 'dgab_po_r5'),
        ),
        sparsity_threshold=getattr(args, 'sparsity_threshold', 0.0),
    )
    if args.rtg_v_cap is not None:
        mp['rtg_v_cap'] = args.rtg_v_cap
    if hasattr(args, 'r2_dir') and args.r2_dir:
        mp['dense_model']['save_dir'] = args.r2_dir
    if hasattr(args, 'r5_dir') and args.r5_dir:
        mp['sparse_model']['save_dir'] = args.r5_dir
    return DGABEnsembleAuctionNetAgent(
        budget=budget, cpa=cpa, category=category,
        name='DGAB-Ensemble-R2R5', model_param=mp)


def make_fusion_agent(budget, cpa, category, args):
    """DGAB-Fusion (Plan B): dual-encoder + heuristic gate."""
    from simul_bidding_env.strategy.autobidding_agents import DGABFusionAuctionNetAgent
    mp = dict(
        hidden_size=512, max_ep_len=96, time_dim=8,
        block_config=BLOCK_CONFIG,
        device=args.device,
        K=20,
        max_imp=args.max_imp,
        save_dir=os.path.join(args.model_root, 'DGAB_fusion'),
        gate_temperature=getattr(args, 'gate_temperature', 1.0),
    )
    if args.rtg_v_cap is not None:
        mp['rtg_v_cap'] = args.rtg_v_cap
    if hasattr(args, 'fusion_dir') and args.fusion_dir:
        mp['save_dir'] = args.fusion_dir
    return DGABFusionAuctionNetAgent(
        budget=budget, cpa=cpa, category=category,
        name='DGAB-Fusion', model_param=mp)


def build_agent(cfg, budget, cpa, category, args):
    if cfg in PO_CONFIGS:
        return make_po_agent(cfg, budget, cpa, category, args)
    elif cfg in ENSEMBLE_CONFIGS:
        return make_ensemble_agent(budget, cpa, category, args)
    elif cfg in FUSION_CONFIGS:
        return make_fusion_agent(budget, cpa, category, args)
    elif cfg == 'fo':
        return make_fo_agent(budget, cpa, category, args)
    elif cfg == 'pid':
        return make_pid_agent(budget, cpa, category, args)
    raise ValueError(f'Unknown config: {cfg}')


# ═══════════════════════════════════════════════
# Single-episode runner
# ═══════════════════════════════════════════════

def run_one_episode(controller, episode, player_index,
                    num_tick=48, pvalue_mean_base=None, pv_num=None, log_every=8):
    """Run one live-auction episode; return player metrics dict."""
    envs = controller.biddingEnv
    pv_generator = controller.pvGenerator
    agents = controller.agents
    num_agent = len(agents)
    agents_cpa = np.array([agent.cpa for agent in agents])

    rewards = np.zeros(num_agent)
    costs = np.zeros(num_agent)
    budgets = np.array([agent.budget for agent in agents])

    history_pvalue_infos = []
    history_bids = []
    history_auction_results = []
    history_impression_results = []
    history_least_winning_costs = []

    controller.reset(episode=episode)
    # Override PV density / volume AFTER reset (NeurIPSPvGen.reset re-inits
    # itself with the defaults: pvalue_mean_base=0.0005, pv_num=500000).
    # Regenerate values AND sigmas (fixes the attribute-case bug in
    # test_autobidding_online.py where only pv_values was refreshed).
    regen = False
    if pv_num is not None and pv_generator.PV_NUM != pv_num:
        pv_generator.PV_NUM = pv_num
        regen = True
    if pvalue_mean_base is not None and \
            abs(pvalue_mean_base - pv_generator.pvalue_mean_base) > 1e-12:
        pv_generator.pvalue_mean_base = pvalue_mean_base
        regen = True
    if regen:
        pv_generator.pv_values, pv_generator.pValueSigmas = pv_generator.generate()

    total_player_pv = 0
    player_won = 0
    alpha_ests = []

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
        # Sanitize: a NaN bid would poison market prices for ALL agents
        nonfinite = ~np.isfinite(bids)
        if nonfinite.any():
            logger.warning(f'  [tick={tick_index:02d}] {int(nonfinite.sum())} '
                           f'non-finite bids zeroed')
            bids[nonfinite] = 0
        bids[bids < 0] = 0

        remaining_budget_list = np.array([agent.remaining_budget for agent in agents])

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

        history_bids.append(bids.transpose())
        history_least_winning_costs.append(least_winning_cost_pit)
        history_pvalue_infos.append(
            np.stack((pv_values.T, pvalue_sigmas.T), axis=-1))
        history_auction_results.append(
            np.stack((xi_pit, slot_pit, cost_pit), axis=-1))
        history_impression_results.append(
            np.stack((is_exposed_pit, conversion_action_pit), axis=-1))

        # ── Player per-tick stats ──
        total_player_pv += pv_values.shape[0]
        player_won += int((xi_pit[player_index] > 0.5).sum())
        pv_mean = float(np.mean(pv_values[:, player_index]))
        bid_mean = float(np.mean(bids[:, player_index]))
        alpha_est = bid_mean / (pv_mean + 1e-8) if pv_mean > 0 else 0.0
        alpha_ests.append(alpha_est)

        if tick_index % log_every == 0 or tick_index == num_tick - 1:
            logger.info(f'  [tick={tick_index:02d}] '
                        f'budget_left={agents[player_index].remaining_budget:.0f} '
                        f'tick_cost={cost[player_index]:.1f} '
                        f'tick_conv={int(reward[player_index])} '
                        f'cum_reward={int(rewards[player_index])} '
                        f'cum_cost={costs[player_index]:.1f} '
                        f'alpha≈{alpha_est:.1f}')

    # ── Player metrics ──
    player_reward = rewards[player_index]
    player_cost = costs[player_index]
    player_cpa_real = player_cost / (player_reward + 1e-10)
    player_cpa_target = agents_cpa[player_index]

    beta = 2
    if player_cpa_real > player_cpa_target:
        penalty = (player_cpa_target / (player_cpa_real + 1e-10)) ** beta
    else:
        penalty = 1.0

    return {
        'score': penalty * player_reward,
        'reward': int(player_reward),
        'cost': player_cost,
        'cpa_real': player_cpa_real,
        'cpa_target': player_cpa_target,
        'budget_used': player_cost / budgets[player_index],
        'win_rate': player_won / (total_player_pv + 1e-10),
        'alpha_mean': float(np.mean(alpha_ests)),
    }


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description='DGAB-PO online ablation benchmark (R0-R5)')
    ap.add_argument('--configs', type=str, default=','.join(PO_CONFIGS),
                    help=f'Comma-separated configs: {PO_CONFIGS} + pid/fo '
                         f'(default: all R0-R5)')
    ap.add_argument('--model_root', type=str, default=DEFAULT_MODEL_ROOT,
                    help='Dir containing dgab_po_r0 ... dgab_po_r5 checkpoints')
    ap.add_argument('--r5_dir', type=str, default=None,
                    help='Override path for r5 checkpoint (bypasses model_root/dgab_po_r5)')
    ap.add_argument('--r6_dir', type=str, default=None,
                    help='Override path for r6 checkpoint (bypasses model_root/dgab_po_r6)')
    ap.add_argument('--r7_dir', type=str, default=None,
                    help='Override path for r7 checkpoint (bypasses model_root/dgab_po_r7)')
    ap.add_argument('--r2_dir', type=str, default=None,
                    help='Override path for r2 checkpoint (for r2r5 ensemble)')
    ap.add_argument('--sparsity_threshold', type=float, default=0.0,
                    help='Sparsity threshold for r2r5 ensemble (default 0.0)')
    ap.add_argument('--gate_temperature', type=float, default=1.0,
                    help='Gate temperature for fusion config (default 1.0)')
    ap.add_argument('--fusion_dir', type=str, default=None,
                    help='Override path for DGAB-Fusion checkpoint')
    ap.add_argument('--fo_dir', type=str, default=DEFAULT_FO_DIR,
                    help='DGAB-FO checkpoint dir (for config "fo")')
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--pv', nargs='*', type=float, default=[0.0005],
                    help='pvalue_mean_base sweep (default 0.0005 = training density)')
    ap.add_argument('--player', type=str, default='0',
                    help='Player advertiser index/indices: single ("0"), '
                         'comma-separated ("0,4,24"), or "all" for all 48')
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_imp', type=int, default=512,
                    help='Obs padding cap for R2/R3/R5 encoders')
    ap.add_argument('--rtg_v_cap', type=float, default=None,
                    help='Optional V_goal cap applied uniformly to all configs')
    ap.add_argument('--pv_num', type=int, default=None,
                    help='Override total PV volume (default: gin PVNUM=500000; '
                         'use e.g. 20000 for a quick smoke run)')
    ap.add_argument('--output', type=str, default='po_ablation',
                    help='Output file prefix under the output dir')
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: {project}/exp_data)')
    return ap.parse_args()


def main():
    args = parse_args()
    import torch
    import pandas as pd

    args.device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    ALL_CONFIGS = PO_CONFIGS + ENSEMBLE_CONFIGS + FUSION_CONFIGS + ['pid', 'fo']
    configs = [c.strip().lower() for c in args.configs.split(',') if c.strip()]
    for c in configs:
        if c not in ALL_CONFIGS:
            raise SystemExit(f'Unknown config "{c}" (choose from {ALL_CONFIGS})')

    # ── Resolve player advertiser list ──
    num_agent_total = 48   # gin: NUM_CATERORY(6) × NUM_AGENT_CATERORY(8)
    if args.player.strip().lower() == 'all':
        players = list(range(num_agent_total))
    else:
        players = [int(x) for x in args.player.split(',') if x.strip()]
        bad = [p for p in players if not 0 <= p < num_agent_total]
        if bad:
            raise SystemExit(f'Invalid player index/indices {bad} (valid: 0-{num_agent_total - 1})')

    logger.info(f'Device: {args.device}  Seed: {args.seed}')
    logger.info(f'Configs: {configs}')
    logger.info(f'Model root: {args.model_root}')
    logger.info(f'PV sweep: {args.pv}  Episodes: {args.episodes}  '
                f'Players: {players if len(players) <= 10 else f"all {len(players)}"}')
    n_runs = len(args.pv) * args.episodes * len(players) * len(configs)
    logger.info(f'Total runs: {n_runs} (~60s each at full volume on CPU)')

    # Dummy agent for Controller construction (replaced by the real player)
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy

    rows = []
    t_start = time.time()

    for pv_val in args.pv:
        pv_label = f'{pv_val:.4f}'
        logger.info(f'\n{"#" * 60}\n# PVALUE_MEAN_BASE = {pv_label}\n{"#" * 60}')

        for ep in range(args.episodes):
            for player_idx in players:
                for cfg in configs:
                    # Re-seed before EVERY run: identical opponents/traffic per
                    # (pv, episode) across all configs and players.
                    np.random.seed(args.seed)
                    torch.manual_seed(args.seed)

                    dummy = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
                    dummy.name += '0'
                    controller = Controller(player_index=player_idx, player_agent=dummy)

                    budget = controller.budget_list[player_idx]
                    cpa = controller.cpa_constraint_list[player_idx]
                    category = controller.category[player_idx]

                    try:
                        player_agent = build_agent(cfg, budget, cpa, category, args)
                    except Exception as e:
                        logger.error(f'[{cfg}] adv#{player_idx}: agent build FAILED — {e}')
                        rows.append(dict(pvalue_mean_base=pv_val, episode=ep,
                                         advertiser=player_idx, config=cfg,
                                         score=0.0, reward=0, cost=0.0,
                                         cpa_real=float('inf'), cpa_target=cpa,
                                         budget_used=0.0, win_rate=0.0,
                                         alpha_mean=0.0, elapsed_s=0.0))
                        continue

                    controller.player_agent = player_agent
                    agents = controller.load_agents()
                    for i in range(len(agents)):
                        agents[i].remaining_budget = controller.budget_list[i]

                    logger.info(f'\n[{cfg.upper()}] adv#{player_idx} ep={ep} pv={pv_label} '
                                f'budget={budget:.0f} cpa={cpa:.0f}')
                    t0 = time.time()
                    result = run_one_episode(
                        controller, episode=ep, player_index=player_idx,
                        pvalue_mean_base=pv_val, pv_num=args.pv_num)
                    elapsed = time.time() - t0

                    logger.info(f'[{cfg.upper()}] adv#{player_idx} done {elapsed:.1f}s '
                                f'score={result["score"]:.2f} reward={result["reward"]} '
                                f'cpa={result["cpa_real"]:.2f}/{result["cpa_target"]:.0f} '
                                f'budget={result["budget_used"]:.0%} '
                                f'win_rate={result["win_rate"]:.2%}')

                    rows.append(dict(pvalue_mean_base=pv_val, episode=ep,
                                     advertiser=player_idx, config=cfg,
                                     elapsed_s=round(elapsed, 1), **result))

                    # Free GPU memory between configs
                    del player_agent, controller
                    if args.device.startswith('cuda'):
                        torch.cuda.empty_cache()

    # ── Save results ──
    out_dir = args.output_dir or os.path.join(_PROJECT_ROOT, 'exp_data')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)

    detailed_path = os.path.join(out_dir, f'{args.output}_detailed.csv')
    df.to_csv(detailed_path, index=False)
    logger.info(f'Detailed results saved to {detailed_path}')

    pivot = df.groupby(['pvalue_mean_base', 'config'])['score'].mean().unstack('config')
    # keep declared config order in columns
    pivot = pivot[[c for c in configs if c in pivot.columns]]
    pivot_path = os.path.join(out_dir, f'{args.output}_pivot.csv')
    pivot.to_csv(pivot_path)
    logger.info(f'Pivot saved to {pivot_path}')

    summary = df.groupby(['pvalue_mean_base', 'config'])[
        ['score', 'reward', 'cost', 'cpa_real', 'budget_used', 'win_rate', 'alpha_mean']
    ].mean().reset_index()
    summary_path = os.path.join(out_dir, f'{args.output}_summary.csv')
    summary.to_csv(summary_path, index=False)
    logger.info(f'Summary saved to {summary_path}')

    # Per-advertiser pivot (advertiser × config) when testing multiple players
    adv_pivot = None
    if len(players) > 1:
        adv_pivot = df.groupby(['advertiser', 'config'])['score'].mean().unstack('config')
        adv_pivot = adv_pivot[[c for c in configs if c in adv_pivot.columns]]
        adv_pivot_path = os.path.join(out_dir, f'{args.output}_adv_pivot.csv')
        adv_pivot.to_csv(adv_pivot_path)
        logger.info(f'Per-advertiser pivot saved to {adv_pivot_path}')

    print(f'\n{"=" * 80}')
    print(f'DGAB-PO ONLINE ABLATION — score '
          f'(mean over {args.episodes} episode(s) × {len(players)} advertiser(s))')
    print(f'{"=" * 80}')
    print(pivot.to_string())
    if adv_pivot is not None:
        print(f'\n── Per-advertiser score (mean over pv levels & episodes) ──')
        print(adv_pivot.to_string())
    print(f'\n{summary.to_string(index=False)}')
    print(f'\nTotal wall time: {time.time() - t_start:.0f}s. Results in {out_dir}/')


if __name__ == '__main__':
    main()
