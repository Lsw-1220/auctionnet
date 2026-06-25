"""
Step 2: Probe Strategies.

Provides FixedLambdaBiddingStrategy — the default probe strategy that bids
a constant lambda * pValue for every impression. Also defines ProbeConfig
for future extensibility to other strategy types (PID, time-varying, etc.).

All strategies inherit from AuctionNet's BaseBiddingStrategy and are
directly injectable into Controller(player_agent=...).
"""

import sys
import os

# Ensure the simul_bidding_env package is importable (repository root)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from dataclasses import dataclass, field
from typing import Dict, Any, List

from simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy


# ===========================================================================
# Fixed-Lambda Strategy (default probe)
# ===========================================================================

class FixedLambdaBiddingStrategy(BaseBiddingStrategy):
    """Memoryless strategy: bid = lambda_value * pValues every tick.

    This is the simplest possible probe strategy — the only control knob
    is the lambda coefficient, which makes preference extraction clean
    and interpretable.

    Inherits from BaseBiddingStrategy and overrides reset() and bidding().
    """

    def __init__(self, lambda_value=1.0, budget=100,
                 name="FixedLambda", cpa=2, category=1):
        """Initialize the fixed-lambda strategy.

        Args:
            lambda_value: Coefficient multiplying pValues to produce bids.
                lambda < 1.0 = conservative (underbid relative to pValue)
                lambda = 1.0 = neutral (bid = pValue)
                lambda > 1.0 = aggressive (overbid relative to pValue)
            budget: Initial budget for the delivery period.
            name: Strategy display name.
            cpa: CPA constraint (cost-per-action target).
            category: Advertiser industry category index.
        """
        super().__init__(budget, name, cpa, category)
        self.lambda_value = lambda_value
        # Append lambda to name for traceability in logs
        self.name = f"{name}_lambda_{lambda_value}"

    def reset(self):
        """Reset remaining budget to initial state (called per episode)."""
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas,
                historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult,
                historyLeastWinningCost):
        """Return bids for all PVs in this tick.

        All history parameters are ignored by this memoryless strategy.
        The bid is simply lambda * pValue for each impression opportunity.

        Args:
            timeStepIndex: Current tick index (0-47).
            pValues: shape (num_pv,), conversion probability per PV.
            pValueSigmas: shape (num_pv,), prediction uncertainty per PV.
            history*: Historical data (unused).

        Returns:
            bids: shape (num_pv,), bid price for each PV.
        """
        return self.lambda_value * pValues


# ===========================================================================
# Probe Configuration (extensibility point)
# ===========================================================================

@dataclass
class ProbeConfig:
    """Configuration for the probe strategy used in multi-probe simulation.

    This decouples the pipeline from a specific strategy type. To add a
    new probe type (e.g. PID controller), implement the corresponding
    BaseBiddingStrategy subclass and register it here.

    Attributes:
        strategy_type: Identifier for the probe strategy class.
            Currently supported: "fixed_lambda"
            Planned: "pid", "time_varying_lambda"
        strategy_params: Keyword arguments passed to the strategy constructor
            (shared across all candidate values).
        candidate_values: List of parameter values to sweep over.
            For "fixed_lambda": list of lambda values.
            For "pid": list of initial alpha values.
    """
    strategy_type: str = "fixed_lambda"
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    candidate_values: List[float] = field(default_factory=lambda: [
        0.5, 0.8, 1.0, 1.2, 1.5, 2.0
    ])

    def create_strategy(self, candidate_value: float, **extra_params):
        """Factory method: create a strategy instance for a candidate value.

        Args:
            candidate_value: The parameter value for this probe run.
            **extra_params: Additional kwargs merged with strategy_params.

        Returns:
            A BaseBiddingStrategy instance ready for simulation.

        Raises:
            ValueError: If strategy_type is unknown.
        """
        params = {**self.strategy_params, **extra_params}

        if self.strategy_type == "fixed_lambda":
            return FixedLambdaBiddingStrategy(
                lambda_value=candidate_value,
                **params,
            )
        else:
            raise ValueError(
                f"Unknown strategy_type '{self.strategy_type}'. "
                f"Supported: 'fixed_lambda'."
            )
