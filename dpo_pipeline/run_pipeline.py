"""
Main Pipeline Orchestrator.

Executes the full DPO data generation pipeline:
  Step 1: Configure scenario → modify PV generation parameters.
  Step 2: For each lambda probe, run online simulation.
  Step 3: Score results and extract preference pairs.
  Step 4: Format DPO JSONL records with CoT reasoning.

Usage:
    python -m dpo_pipeline.run_pipeline

    Or programmatically:
    from dpo_pipeline.run_pipeline import run_pipeline
    run_pipeline()
"""

import copy
import json
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure repository root is on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch

from simul_bidding_env.Controller.Controller import Controller
from simul_bidding_env.PvGenerator.NeurIPSPvGen import NeurIPSPvGen

from .scenario_config import SCENARIOS, ScenarioConfig
from .fixed_lambda_strategy import ProbeConfig, FixedLambdaBiddingStrategy
from .scoring import ScoringConfig, compute_score
from .simulation_runner import run_single_simulation, SimulationResult
from .preference_extractor import (
    LambdaResult, PreferencePair, extract_preferences
)
from .dpo_formatter import format_dpo_record


# ===========================================================================
# Default configuration
# ===========================================================================

DEFAULT_PV_NUM = 250000     # Total PVs per episode (production)
# DEFAULT_PV_NUM = 50000    # Reduced for development/testing
DEFAULT_NUM_TICK = 48
DEFAULT_NUM_AGENT = 48
DEFAULT_NUM_AGENT_CATEGORY = 8
DEFAULT_NUM_CATEGORY = 6

CANDIDATE_LAMBDAS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

DEFAULT_OUTPUT_PATH = "dpo_bidding_data.jsonl"


# ===========================================================================
# Lightweight PV array holder (replaces full PV generator)
# ===========================================================================

class _PVArrayHolder:
    """Holds pre-computed pv_values and pValueSigmas arrays.

    This avoids creating a full NeurIPSPvGen for every lambda probe.
    The simulation runner only accesses .pv_values[tick] and
    .pValueSigmas[tick], so this minimal wrapper suffices.
    """
    def __init__(self, pv_values: List[np.ndarray],
                 pvalue_sigmas: List[np.ndarray]):
        self.pv_values = pv_values
        self.pValueSigmas = pvalue_sigmas


# ===========================================================================
# Scenario PV generation
# ===========================================================================

def _create_scenario_pv_data(
    scenario: ScenarioConfig,
    pv_num: int = DEFAULT_PV_NUM,
    num_tick: int = DEFAULT_NUM_TICK,
    num_agent: int = DEFAULT_NUM_AGENT,
    num_agent_category: int = DEFAULT_NUM_AGENT_CATEGORY,
    num_category: int = DEFAULT_NUM_CATEGORY,
    episode: int = 0,
) -> _PVArrayHolder:
    """Create PV arrays for a given scenario.

    Uses NeurIPSPvGen to generate base PV data, then applies scenario-
    specific post-processing multipliers to pv_values and pValueSigmas.
    Traffic volume is controlled by scaling pv_num before generation.

    Args:
        scenario: ScenarioConfig with multiplier settings.
        pv_num: Base total PV count.
        num_tick: Number of ticks per episode.
        num_agent: Number of competing advertisers.
        num_agent_category: Agents per category.
        num_category: Number of categories.
        episode: Seed-determining episode number.

    Returns:
        _PVArrayHolder with scenario-modified arrays.
    """
    # Scale traffic volume
    scaled_pv_num = int(pv_num * scenario.traffic_volume_multiplier)

    # Generate base PV data
    base_gen = NeurIPSPvGen(
        episode=episode,
        num_tick=num_tick,
        num_agent=num_agent,
        num_agent_category=num_agent_category,
        num_category=num_category,
        pv_num=scaled_pv_num,
    )

    # Post-process: apply scenario multipliers
    pv_values_modified = []
    pvalue_sigmas_modified = []

    for tick in range(num_tick):
        pv = base_gen.pv_values[tick].copy()
        sigma = base_gen.pValueSigmas[tick].copy()

        # Scale pValue means and clamp to [0, 1]
        pv = np.clip(pv * scenario.pvalue_mean_multiplier, 0.0, 1.0)

        # Scale pValue sigmas and clamp to [0, 0.3]
        sigma = np.clip(sigma * scenario.pvalue_sigma_multiplier, 0.0, 0.3)

        pv_values_modified.append(pv)
        pvalue_sigmas_modified.append(sigma)

    return _PVArrayHolder(pv_values_modified, pvalue_sigmas_modified)


# ===========================================================================
# Multi-probe simulation for one scenario
# ===========================================================================

def _run_lambda_probes(
    scenario: ScenarioConfig,
    pv_data: _PVArrayHolder,
    probe_config: ProbeConfig,
    pv_num: int = DEFAULT_PV_NUM,
    num_tick: int = DEFAULT_NUM_TICK,
) -> List[LambdaResult]:
    """Run simulations for all candidate lambda values.

    Each lambda probe sees the same PV data (deep-copied from pv_data)
    to ensure fair comparison.

    Args:
        scenario: ScenarioConfig with reserve_price, budget, cpa_constraint.
        pv_data: Pre-generated PV arrays.
        probe_config: ProbeConfig specifying strategy type and candidates.
        pv_num: Base PV count (passed to Controller).
        num_tick: Number of ticks per episode.

    Returns:
        List of LambdaResult, one per candidate value.
    """
    results = []

    for candidate_value in probe_config.candidate_values:
        lam = candidate_value
        print(f"    lambda={lam:<5} ...", end=" ", flush=True)

        # Create strategy for this candidate value
        strategy = probe_config.create_strategy(
            candidate_value,
            budget=scenario.budget,
            cpa=scenario.cpa_constraint,
            category=0,
        )

        # Create Controller with our strategy injected
        controller = Controller(
            player_index=0,
            player_agent=strategy,
            num_tick=num_tick,
            num_agent_category=DEFAULT_NUM_AGENT_CATEGORY,
            num_category=DEFAULT_NUM_CATEGORY,
            pv_num=pv_num,
            pv_generator_type="neuripsPvGen",
        )

        # Override reserve price for scenario-specific competition level
        controller.biddingEnv.reserve_pv_price = scenario.reserve_price

        # Inject scenario PV data (deep-copy so each run is independent)
        pv_holder = _PVArrayHolder(
            pv_values=copy.deepcopy(pv_data.pv_values),
            pvalue_sigmas=copy.deepcopy(pv_data.pValueSigmas),
        )
        controller.pvGenerator = pv_holder

        # Override player agent's budget and CPA (already set in strategy,
        # but Controller.load_agents() may overwrite — re-set here)
        player_agent = controller.agents[controller.player_index]
        player_agent.budget = scenario.budget
        player_agent.cpa = scenario.cpa_constraint
        player_agent.remaining_budget = scenario.budget

        # Run simulation
        sim_result = run_single_simulation(controller, num_tick=num_tick)

        lambda_result = LambdaResult(
            lambda_value=lam,
            total_cost=sim_result.total_cost,
            total_conversion=sim_result.total_conversion,
            cpa_constraint=sim_result.cpa_constraint,
            budget=sim_result.budget,
        )

        results.append(lambda_result)

        print(f"cost={sim_result.total_cost:,.1f}, "
              f"conv={sim_result.total_conversion:.0f}, "
              f"cpa={sim_result.real_cpa:.2f}")

    return results


# ===========================================================================
# Main pipeline entry point
# ===========================================================================

def run_pipeline(
    scenarios: Optional[Dict[str, ScenarioConfig]] = None,
    probe_config: Optional[ProbeConfig] = None,
    scoring_config: Optional[ScoringConfig] = None,
    output_path: str = DEFAULT_OUTPUT_PATH,
    pv_num: int = DEFAULT_PV_NUM,
    num_tick: int = DEFAULT_NUM_TICK,
    episode: int = 0,
) -> List[dict]:
    """Execute the full DPO data generation pipeline.

    Args:
        scenarios: Dict of scenario_key → ScenarioConfig.
            Defaults to all SCENARIOS.
        probe_config: ProbeConfig specifying strategy type and candidate
            lambda values. Defaults to fixed_lambda with [0.5..2.0].
        scoring_config: ScoringConfig for the preference scoring formula.
            Defaults to CPA soft-constraint with beta=2.
        output_path: Path for the output JSONL file.
        pv_num: Base total PV count per episode.
        num_tick: Number of ticks per episode.
        episode: Seed-determining episode number (for reproducibility).

    Returns:
        List of formatted DPO records (also written to output_path).
    """
    scenarios = scenarios or SCENARIOS
    if probe_config is None:
        probe_config = ProbeConfig()
    if scoring_config is None:
        scoring_config = ScoringConfig(use_cpa_constraint=True,
                                       cpa_penalty_beta=2.0)

    all_records = []
    skipped_count = 0
    start_time = time.time()

    # Title line
    print("=" * 70)
    print("  AuctionNet DPO Data Pipeline")
    print(f"  Scenarios: {len(scenarios)}  |  "
          f"Lambdas: {probe_config.candidate_values}  |  "
          f"PV={pv_num}  |  Ticks={num_tick}")
    print(f"  Scoring: use_cpa={scoring_config.use_cpa_constraint}, "
          f"beta={scoring_config.cpa_penalty_beta}")
    print("=" * 70)

    for scenario_key, scenario_config in scenarios.items():
        print(f"\n{'─' * 70}")
        print(f"Scenario: {scenario_config.name} ({scenario_key})")
        print(f"  {scenario_config.description[:80]}...")
        print(f"  Budget={scenario_config.budget:.0f}, "
              f"CPA={scenario_config.cpa_constraint:.0f}, "
              f"Reserve={scenario_config.reserve_price:.6f}")

        # Step 1: Generate scenario PV data
        print("  [Step 1] Generating scenario PV data...")
        pv_data = _create_scenario_pv_data(
            scenario_config, pv_num=pv_num, num_tick=num_tick, episode=episode
        )

        # Step 2: Run lambda probes
        print(f"  [Step 2] Running {len(probe_config.candidate_values)} "
              f"lambda probes...")
        lambda_results = _run_lambda_probes(
            scenario_config, pv_data, probe_config,
            pv_num=pv_num, num_tick=num_tick,
        )

        # Step 3: Extract preferences
        print("  [Step 3] Extracting preference pair...")
        pair = extract_preferences(lambda_results, scoring_config)

        if pair is None:
            print(f"  ⚠ WARNING: Could not form preference pair. Skipping.")
            skipped_count += 1
            continue

        # Print score summary
        for lr in sorted(lambda_results, key=lambda r: r.score, reverse=True):
            marker = ""
            if lr == pair.chosen:
                marker = " ← CHOSEN"
            elif lr == pair.rejected:
                marker = " ← REJECTED"
            if lr.total_conversion > 0:
                cpa = lr.total_cost / lr.total_conversion
            else:
                cpa = float('inf')
            print(f"      λ={lr.lambda_value:.1f}  "
                  f"score={lr.score:.1f}  "
                  f"cost={lr.total_cost:.1f}  "
                  f"conv={lr.total_conversion:.0f}  "
                  f"cpa={cpa:.2f}{marker}")
        print(f"    Rejection reason: {pair.rejection_reason}")

        # Step 4: Format DPO record
        print("  [Step 4] Formatting DPO record...")
        record = format_dpo_record(scenario_config, pair, scoring_config)
        all_records.append(record)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_full_path = Path(_repo_root) / output_path
    output_full_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_full_path, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Pipeline complete in {elapsed:.0f}s")
    print(f"  Scenarios processed: {len(all_records)}")
    print(f"  Scenarios skipped:   {skipped_count}")
    print(f"  Total records:       {len(all_records)}")
    print(f"  Output:              {output_full_path.resolve()}")
    print(f"{'=' * 70}")

    return all_records


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    # Fix seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)

    # Use development PV count for quick testing
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate DPO training data from AuctionNet simulations")
    parser.add_argument("--pv-num", type=int, default=DEFAULT_PV_NUM,
                        help=f"Total PV count (default: {DEFAULT_PV_NUM})")
    parser.add_argument("--dev", action="store_true",
                        help="Use reduced PV count (50000) for quick testing")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH,
                        help="Output JSONL path")
    parser.add_argument("--no-cpa", action="store_true",
                        help="Disable CPA constraint (pure conversion max)")
    args = parser.parse_args()

    if args.dev:
        pv_num = 50000
        print(f"[DEV MODE] PV_NUM = {pv_num}")
    else:
        pv_num = args.pv_num

    scoring = ScoringConfig(
        use_cpa_constraint=not args.no_cpa,
        cpa_penalty_beta=2.0,
    )

    run_pipeline(
        pv_num=pv_num,
        scoring_config=scoring,
        output_path=args.output,
    )
