"""
Batch generation script for multi-period bidding logs.
Supports parallel execution and multi-dimensional variation across periods.

Usage:
    # Single process: generate periods 0-99
    python batch_generate.py --start 0 --end 100

    # Parallel: launch multiple processes
    python batch_generate.py --start 0 --end 5000 &
    python batch_generate.py --start 5000 --end 10000 &

    # Dry run: print config without generating
    python batch_generate.py --start 0 --end 10 --dry_run
"""

import argparse
import sys
import time
import os
import logging
import numpy as np
import gin

sys.path.append("./strategy_train_env")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

from strategy_config import get_world_type, build_agents_for_period


def parse_args():
    parser = argparse.ArgumentParser(description="Batch generation of bidding logs")
    parser.add_argument("--start", type=int, required=True,
                        help="Start period index (inclusive)")
    parser.add_argument("--end", type=int, required=True,
                        help="End period index (exclusive)")
    parser.add_argument("--pv_num", type=int, default=500000,
                        help="Number of pvs per period")
    parser.add_argument("--generator", type=str, default="mix",
                        choices=["modelPvGen", "neuripsPvGen", "mix"],
                        help="PV generator: 'mix' alternates model/neurips each period")
    parser.add_argument("--output_dir", type=str, default="data/log_batch",
                        help="Output directory for CSV files")
    parser.add_argument("--budget_perturb", type=float, default=0.08,
                        help="Per-period budget perturbation ratio (0=disabled)")
    parser.add_argument("--config", type=str, default="./config/test.gin",
                        help="Path to gin config file")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print configuration without running simulation")
    return parser.parse_args()


def run_single_period(period_index, pv_num, generator_type, output_dir, budget_perturb):
    """Run a single period of bidding simulation with all 48 agents randomly assigned strategies."""
    from simul_bidding_env.Tracker.BiddingTracker import BiddingTracker
    from simul_bidding_env.Environment.BiddingEnv import BiddingEnv
    from simul_bidding_env.PvGenerator.NeurIPSPvGen import NeurIPSPvGen
    from simul_bidding_env.PvGenerator.ModelPvGen import ModelPvGenerator

    # ── Determine world type for this period ────────────────────
    world_type = get_world_type(period_index)
    rng = np.random.default_rng(period_index * 99991)

    # ── Build ALL 48 agents from strategy config (no special "player") ──
    agents, budget_mult, cpa_shuffle = build_agents_for_period(
        period_index, world_type)

    # ── PV generator ────────────────────────────────────────────
    gen_type = generator_type
    if gen_type == "mix":
        gen_type = "modelPvGen" if period_index % 2 == 0 else "neuripsPvGen"
    num_tick = 48
    num_agent = 48
    num_agent_category = 8
    num_category = 6
    if gen_type == "neuripsPvGen":
        pv_generator = NeurIPSPvGen(
            episode=period_index, num_tick=num_tick, num_agent=num_agent,
            num_agent_category=num_agent_category, num_category=num_category,
            pv_num=pv_num)
    else:
        start_cat = (period_index * 6) % 44
        all_cats = np.arange(1, 45)
        if start_cat + 6 <= 44:
            select_category = list(all_cats[start_cat:start_cat + 6])
        else:
            select_category = list(all_cats[start_cat:]) + list(all_cats[:start_cat + 6 - 44])
        pv_generator = ModelPvGenerator(
            num_tick=num_tick, num_agent_category=num_agent_category,
            select_category=select_category, pv_num=pv_num,
            episodic_std=0.1, episode=period_index)

    # ── Budget / CPA / Category assignment ──────────────────────
    BASE_BUDGETS = [
        2900, 4350, 3000, 2400, 4800, 2000, 2050, 3500,
        4600, 2000, 2800, 2350, 2050, 2900, 4750, 3450,
        2000, 3500, 2200, 2700, 3100, 2100, 4850, 4100,
        2000, 4800, 3050, 4250, 2850, 2250, 2000, 3900,
        2000, 3250, 4450, 3550, 2700, 2100, 4650, 2000,
        3400, 2650, 2300, 4100, 4800, 4450, 2000, 2050,
    ]
    BASE_CPAS = [
        100, 70, 90, 110, 60, 130, 120, 80,
        70, 130, 100, 110, 120, 90, 60, 80,
        130, 80, 110, 100, 90, 120, 60, 70,
        120, 60, 90, 70, 100, 110, 130, 80,
        120, 90, 70, 80, 100, 110, 60, 130,
        90, 100, 110, 80, 60, 70, 130, 120,
    ]

    budgets = np.array(BASE_BUDGETS, dtype=float)
    cpas = np.array(BASE_CPAS, dtype=float)

    # Budget-stressed world
    if world_type == "budget_stressed":
        budgets *= budget_mult

    # Apply per-period perturbation
    for i in range(num_agent):
        budgets[i] *= rng.uniform(1 - budget_perturb, 1 + budget_perturb)

    # CPA variation: shuffle for cpa_varied world, slight perturb otherwise
    if cpa_shuffle:
        rng.shuffle(cpas)
    for i in range(num_agent):
        cpas[i] *= rng.uniform(0.9, 1.1)

    categories = np.arange(num_agent) // num_agent_category

    for i in range(num_agent):
        agents[i].budget = budgets[i]
        agents[i].cpa = cpas[i]
        agents[i].category = categories[i]
        agents[i].remaining_budget = budgets[i]

    # ── Bidding environment ─────────────────────────────────────
    envs = BiddingEnv()

    # ── Trackers ────────────────────────────────────────────────
    tracker = BiddingTracker("batch_tracker")

    logger.info(f"Period {period_index:05d}: world={world_type}, gen={gen_type}, "
                f"pv_num={pv_generator.pv_num}, "
                f"min_budget={budgets.min():.0f}, max_budget={budgets.max():.0f}")

    # ── Run simulation ──────────────────────────────────────────
    total_pv_num = 0
    rewards = np.zeros(num_agent)
    costs = np.zeros(num_agent)
    agents_category = np.array([a.category for a in agents])
    agents_cpa = np.array([a.cpa for a in agents])
    initial_budgets = np.array([a.budget for a in agents])

    history_pvalue_infos = []
    history_bids = []
    history_auction_results = []
    history_impression_results = []
    history_least_winning_costs = []

    # Reset environment
    envs.reset(episode=period_index)
    for a in agents:
        a.reset()
        a.remaining_budget = a.budget

    t0 = time.time()
    for tick_index in range(num_tick):
        pv_values = pv_generator.pv_values[tick_index]
        pvalue_sigmas = pv_generator.pValueSigmas[tick_index]

        bids = [
            agent.bidding(
                tick_index, pv_values[:, i], pvalue_sigmas[:, i],
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
        done_list = np.ones(len(agents), dtype=int) if tick_index == (num_tick - 1) else (
            remaining_budget_list < envs.min_remaining_budget
        ).astype(int)

        # Over-cost adjustment loop
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
                envs.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)

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
            np.stack((pv_values.T, pvalue_sigmas.T), axis=-1))
        history_auction_results.append(
            np.stack((xi_pit, slot_pit, cost_pit), axis=-1))
        history_impression_results.append(
            np.stack((is_exposed_pit, conversion_action_pit), axis=-1))

        tracker.train_logging(
            period_index, tick_index, pv_values, initial_budgets,
            agents_cpa, agents_category, remaining_budget_list,
            total_pv_num, pvalue_sigmas, bids, xi_pit, slot_pit,
            cost_pit, is_exposed_pit, conversion_action_pit,
            least_winning_cost_pit, done_list)

    # ── Save CSV ────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"period-{period_index:05d}.csv")
    tracker.generate_train_data(csv_path)

    elapsed = time.time() - t0
    logger.info(f"Period {period_index:05d} done: {elapsed:.1f}s, "
                f"total_reward={rewards.sum():.0f}, "
                f"total_cost={costs.sum():.0f}, "
                f"active_agents={(costs > 0).sum()}/48")

    return {
        "period": period_index,
        "world": world_type,
        "generator": gen_type,
        "total_reward": float(rewards.sum()),
        "total_cost": float(costs.sum()),
        "active_agents": int((costs > 0).sum()),
        "elapsed": elapsed,
    }


def _get_winner(slot_pit):
    slot_pit = slot_pit.T
    num_pv, num_agent = slot_pit.shape
    num_slot = 3
    winner = np.full((num_pv, num_slot), -1, dtype=int)
    for pos in range(1, num_slot + 1):
        winning = np.argwhere(slot_pit == pos)
        if winning.size > 0:
            pv_idx, agent_idx = winning.T
            winner[pv_idx, pos - 1] = agent_idx
    return winner


def _adjust_over_cost(bids, over_cost_ratio, slot_coefs, winner_pit):
    import math
    overcost_agent_indices = np.where(over_cost_ratio > 0)[0]
    for agent_index in overcost_agent_indices:
        for i, _ in enumerate(slot_coefs):
            winner_indices = winner_pit[:, i]
            pv_indices = np.where(winner_indices == agent_index)[0]
            rng = np.random.default_rng(seed=1)
            num_drop = math.ceil(pv_indices.size * over_cost_ratio[agent_index])
            if num_drop > 0:
                dropped = rng.choice(pv_indices, num_drop, replace=False)
                bids[dropped, agent_index] = 0


def main():
    args = parse_args()
    gin_file = [args.config]
    gin.parse_config_files_and_bindings(gin_file, None)

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        for period in range(args.start, min(args.end, args.start + 20)):
            wt = get_world_type(period)
            gt = args.generator
            if gt == "mix":
                gt = "modelPvGen" if period % 2 == 0 else "neuripsPvGen"
            logger.info(f"  Period {period:05d}: world={wt}, generator={gt}")
        logger.info(f"  ... (total {args.end - args.start} periods)")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    t_total = time.time()
    for period in range(args.start, args.end):
        try:
            result = run_single_period(
                period_index=period,
                pv_num=args.pv_num,
                generator_type=args.generator,
                output_dir=args.output_dir,
                budget_perturb=args.budget_perturb,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Period {period} FAILED: {e}", exc_info=True)

    total_elapsed = time.time() - t_total
    successful = len(results)
    if successful > 0:
        logger.info(
            f"Batch done: {successful}/{args.end - args.start} periods "
            f"in {total_elapsed:.0f}s "
            f"({total_elapsed / successful:.1f}s/period)")

        # Summary by world type
        from collections import Counter
        world_counts = Counter(r["world"] for r in results)
        logger.info(f"World distribution: {dict(world_counts)}")


if __name__ == "__main__":
    main()
