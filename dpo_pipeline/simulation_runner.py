"""
Step 3: Custom Online Simulation Runner.

Runs a single-episode simulation on the AuctionNet environment using the
provided Controller. Unlike the original run_test(), this runner:
  - Does NOT use PlayerAnalysis (aggregates cost/conversion in-place).
  - Does NOT generate CSV logs (no BiddingTracker overhead).
  - Does NOT call initialize_player_agent() (strategy is already injected).
  - Only tracks the player agent's metrics (for efficiency).

The core auction logic (GSP, over-cost adjustment, history tracking) is
preserved from run_test() to ensure identical competitor behavior.

Usage:
    controller = Controller(player_agent=strategy, ...)
    controller.pvGenerator = scenario_pv_gen_copy   # inject scenario PVs
    result = run_single_simulation(controller)
    # result: SimulationResult(total_cost, total_conversion, cpa_constraint, budget)
"""

import math
import sys
import os

# Ensure repository root is on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from dataclasses import dataclass
from typing import List

import numpy as np


# ===========================================================================
# Result data class
# ===========================================================================

@dataclass
class SimulationResult:
    """Aggregated result of a single simulation run (one episode).

    Attributes:
        total_cost: Sum of costs incurred by the player agent across all ticks.
        total_conversion: Sum of conversion actions for the player agent.
        cpa_constraint: The player's CPA target constraint.
        budget: The player's initial budget.
        remaining_budget: Budget left after the simulation.
    """
    total_cost: float
    total_conversion: float
    cpa_constraint: float
    budget: float
    remaining_budget: float = 0.0

    @property
    def real_cpa(self) -> float:
        """Actual CPA = total_cost / total_conversion."""
        return self.total_cost / (self.total_conversion + 1e-10)

    @property
    def cpa_exceedance_rate(self) -> float:
        """Relative CPA exceedance: (real_cpa - constraint) / constraint."""
        return (self.real_cpa - self.cpa_constraint) / (self.cpa_constraint + 1e-10)

    @property
    def budget_consumption_ratio(self) -> float:
        """Fraction of budget spent."""
        return self.total_cost / (self.budget + 1e-10)

    def __repr__(self) -> str:
        return (
            f"SimulationResult(cost={self.total_cost:.1f}, "
            f"conv={self.total_conversion:.1f}, "
            f"cpa={self.real_cpa:.2f}/{self.cpa_constraint:.1f}, "
            f"budget_use={self.budget_consumption_ratio:.1%})"
        )


# ===========================================================================
# Helper functions (extracted from run_test.py)
# ===========================================================================

def _get_winner(slot_pit: np.ndarray) -> np.ndarray:
    """Determines which agent won each slot for each PV.

    Args:
        slot_pit: shape (num_agent, num_pv), slot assignment per agent per PV.
            Slot values: 0=lost, 1=slot1, 2=slot2, 3=slot3.

    Returns:
        winner: shape (num_pv, 3), agent index for each slot per PV.
            -1 means no winner for that slot.
    """
    slot_pit = slot_pit.T  # (num_pv, num_agent)
    num_pv, num_agent = slot_pit.shape
    num_slot = 3
    winner = np.full((num_pv, num_slot), -1, dtype=int)

    for pos in range(1, num_slot + 1):
        winning_indices = np.argwhere(slot_pit == pos)
        if winning_indices.size > 0:
            pv_indices, agent_indices = winning_indices.T
            winner[pv_indices, pos - 1] = agent_indices
    return winner


def _adjust_over_cost(bids: np.ndarray,
                      over_cost_ratio: np.ndarray,
                      envs_slots: List[float],
                      winner_pit: np.ndarray) -> None:
    """Randomly drops winning bids for agents that exceeded their budget.

    Modifies `bids` in-place. For each agent with over_cost_ratio > 0,
    a proportion of their winning bids are set to zero so the agent
    no longer exceeds budget when the auction is re-run.

    Args:
        bids: shape (num_pv, num_agent), current bid matrix (modified in-place).
        over_cost_ratio: shape (num_agent,), proportion of over-spend.
        envs_slots: Exposure coefficients for each slot, e.g. [1.0, 0.8, 0.6].
        winner_pit: shape (num_pv, 3), agent index per slot per PV.
    """
    overcost_agent_indices = np.where(over_cost_ratio > 0)[0]
    for agent_index in overcost_agent_indices:
        for i, _coefficient in enumerate(envs_slots):
            winner_indices = winner_pit[:, i]
            pv_indices = np.where(winner_indices == agent_index)[0]
            rng = np.random.default_rng(seed=1)
            num_to_drop = math.ceil(pv_indices.size * over_cost_ratio[agent_index])
            if num_to_drop > 0:
                dropped_pv_indices = rng.choice(
                    pv_indices, num_to_drop, replace=False)
                bids[dropped_pv_indices, agent_index] = 0


# ===========================================================================
# Main simulation function
# ===========================================================================

def run_single_simulation(controller,
                          num_tick: int = 48) -> SimulationResult:
    """Run one episode of the ad auction simulation.

    Reuses the Controller's agents, BiddingEnv, and PV generator.
    Only the player agent's cost and conversions are tracked.

    The simulation loop mirrors run_test() from run/run_test.py:
      1. For each tick: collect bids from all agents.
      2. Over-cost adjustment loop: GSP auction + drop overspent bids.
      3. Deduct costs, accumulate player metrics.
      4. Update history buffers (needed by competitor strategies).

    Args:
        controller: A fully initialized Controller instance with:
            - controller.agents (48 agents, player at index 0)
            - controller.biddingEnv (BiddingEnv instance)
            - controller.pvGenerator (PV generator with pv_values set)
            - controller.player_index (int, default 0)
        num_tick: Number of ticks per episode (default 48).

    Returns:
        SimulationResult with player's total_cost, total_conversion,
        cpa_constraint, and budget.
    """
    agents = controller.agents
    num_agent = len(agents)
    bidding_env = controller.biddingEnv
    pv_generator = controller.pvGenerator
    player_index = controller.player_index

    # ------------------------------------------------------------------
    # Manually reset all agents (do NOT call controller.reset() because
    # that would also reset the PV generator and regenerate traffic).
    # ------------------------------------------------------------------
    for agent in agents:
        agent.reset()

    # ------------------------------------------------------------------
    # Per-episode accumulators
    # ------------------------------------------------------------------
    player_total_cost = 0.0
    player_total_conversion = 0.0

    # History buffers — required by competitor strategies (PID, IQL, BC, etc.)
    # Each element is a list-of-arrays from one tick, shaped per-agent.
    history_pvalue_infos = []       # list of (num_agent, num_pv, 2)
    history_bids = []                # list of (num_agent, num_pv)
    history_auction_results = []     # list of (num_agent, num_pv, 3)
    history_impression_results = []  # list of (num_agent, num_pv, 2)
    history_least_winning_costs = [] # list of (num_pv,)

    # Environment constants
    min_remaining_budget = bidding_env.min_remaining_budget
    slot_coefficients = bidding_env.slot_coefficients  # [1.0, 0.8, 0.6]

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------
    for tick_index in range(num_tick):
        pv_values = pv_generator.pv_values[tick_index]        # (num_pv, num_agent)
        pvalue_sigmas = pv_generator.pValueSigmas[tick_index]  # (num_pv, num_agent)
        num_pv = pv_values.shape[0]

        # --- Collect bids from all agents ---
        bids_list = []
        for i, agent in enumerate(agents):
            if agent.remaining_budget >= min_remaining_budget:
                # Agent still active — call its bidding strategy
                bid = agent.bidding(
                    tick_index,
                    pv_values[:, i],
                    pvalue_sigmas[:, i],
                    # Per-agent history slices
                    [x[i] for x in history_pvalue_infos],
                    [x[i] for x in history_bids],
                    [x[i] for x in history_auction_results],
                    [x[i] for x in history_impression_results],
                    history_least_winning_costs,
                )
            else:
                # Budget exhausted — bid zero
                bid = np.zeros(num_pv)
            bids_list.append(bid)

        bids = np.array(bids_list).transpose()  # (num_pv, num_agent)
        bids[bids < 0] = 0  # clip negative bids

        # Current remaining budgets (before this tick's auction)
        remaining_budget_list = np.array(
            [agent.remaining_budget for agent in agents])

        # --- Over-cost adjustment loop ---
        # Re-run auction if any agent's cost exceeds their remaining budget.
        # This loop reduces bids for overspent agents and retries until
        # no agent is over budget.
        ratio_max = None
        winner_pit = None
        cost = np.zeros(num_agent)
        reward = np.zeros(num_agent)

        while ratio_max is None or ratio_max > 0:
            if ratio_max is not None and ratio_max > 0:
                # Some agents are over budget — drop some of their winning bids
                over_cost_ratio = np.maximum(
                    (cost - remaining_budget_list) / (cost + 1e-4), 0)
                _adjust_over_cost(bids, over_cost_ratio,
                                  slot_coefficients, winner_pit)

            # Run the GSP auction
            (xi_pit, slot_pit, cost_pit, is_exposed_pit,
             conversion_action_pit, least_winning_cost_pit,
             market_price_pit) = bidding_env.simulate_ad_bidding(
                 pv_values, pvalue_sigmas, bids)

            # Compute per-agent cost and reward
            real_cost = cost_pit * is_exposed_pit  # only pay if exposed
            cost = real_cost.sum(axis=1)            # (num_agent,)
            reward = conversion_action_pit.sum(axis=1)  # (num_agent,)

            # Determine winners per slot (needed for over-cost adjustment)
            winner_pit = _get_winner(slot_pit)

            # Check if any agent still overspent
            over_cost_ratio = np.maximum(
                (cost - remaining_budget_list) / (cost + 1e-4), 0)
            ratio_max = over_cost_ratio.max()

        # --- Deduct costs from agent budgets ---
        for i, agent in enumerate(agents):
            agent.remaining_budget -= cost[i]

        # --- Accumulate player metrics ---
        player_total_cost += cost[player_index]
        player_total_conversion += reward[player_index]

        # --- Update history buffers (competitor strategies need these) ---
        # Transpose bids back to (num_agent, num_pv) for history storage
        history_bids.append(bids.transpose())
        history_least_winning_costs.append(least_winning_cost_pit)

        # pvalue_info: (num_agent, num_pv, 2) — [pValue, sigma]
        pvalue_info = np.stack((pv_values.T, pvalue_sigmas.T), axis=-1)
        history_pvalue_infos.append(pvalue_info)

        # auction_info: (num_agent, num_pv, 3) — [xi, slot, cost]
        auction_info = np.stack((xi_pit, slot_pit, cost_pit), axis=-1)
        history_auction_results.append(auction_info)

        # impression_info: (num_agent, num_pv, 2) — [is_exposed, conversion]
        impression_info = np.stack(
            (is_exposed_pit, conversion_action_pit), axis=-1)
        history_impression_results.append(impression_info)

    # ------------------------------------------------------------------
    # Build and return result
    # ------------------------------------------------------------------
    player_agent = agents[player_index]
    return SimulationResult(
        total_cost=player_total_cost,
        total_conversion=player_total_conversion,
        cpa_constraint=player_agent.cpa,
        budget=player_agent.budget,
        remaining_budget=player_agent.remaining_budget,
    )
