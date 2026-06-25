"""
Step 4: Preference Extraction.

Given a set of (lambda, SimulationResult) pairs for one scenario,
compute scores via the configurable ScoringConfig, then select the
optimal (chosen) and worst (rejected) lambda as a DPO preference pair.

Supports multiple scoring objectives:
  - CPA soft-constraint: NeurIPS score = penalty^beta * conversions
  - Unconstrained: score = conversions (pure volume optimization)
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .scoring import ScoringConfig, compute_score


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class LambdaResult:
    """One lambda probe result."""
    lambda_value: float
    total_cost: float
    total_conversion: float
    cpa_constraint: float
    budget: float
    score: float = 0.0  # Populated after scoring


@dataclass
class PreferencePair:
    """A DPO preference pair: chosen (optimal) vs rejected (worst).

    Attributes:
        chosen: The LambdaResult selected as optimal.
        rejected: The LambdaResult selected as worst.
        rejection_reason: Why the rejected lambda was chosen.
            - "cpa_violation": Rejected lambda's real CPA exceeds constraint.
            - "too_conservative": Rejected lambda spent very little.
            - "suboptimal": Rejected lambda scored lower without specific failure.
        all_results: All LambdaResult objects with scores, for debugging.
    """
    chosen: LambdaResult
    rejected: LambdaResult
    rejection_reason: str
    all_results: List[LambdaResult] = field(default_factory=list)


# ===========================================================================
# Preference extraction
# ===========================================================================

def extract_preferences(
    lambda_results: List[LambdaResult],
    scoring_config: ScoringConfig,
) -> Optional[PreferencePair]:
    """Extract a preference pair from a set of lambda probe results.

    Algorithm:
      1. Compute score for each result via compute_score().
      2. chosen  = argmax(score)  — highest scoring lambda.
      3. rejected = argmin(score) — lowest scoring lambda.
      4. Classify rejection_reason based on rejected's CPA and spend.

    Args:
        lambda_results: List of LambdaResult from different lambda probes.
            Must have at least 2 entries.
        scoring_config: ScoringConfig controlling the scoring formula.

    Returns:
        PreferencePair if a valid pair can be formed, None otherwise.
    """
    if len(lambda_results) < 2:
        return None

    # Compute scores
    for lr in lambda_results:
        # Build a lightweight result-like object for compute_score
        class _ResultProxy:
            pass
        proxy = _ResultProxy()
        proxy.total_cost = lr.total_cost
        proxy.total_conversion = lr.total_conversion
        proxy.cpa_constraint = lr.cpa_constraint
        lr.score = compute_score(proxy, scoring_config)

    # Sort by score descending
    sorted_results = sorted(lambda_results,
                            key=lambda r: r.score, reverse=True)

    chosen = sorted_results[0]
    rejected = sorted_results[-1]

    # Degenerate check
    if chosen.lambda_value == rejected.lambda_value:
        return None

    # Determine rejection reason
    if rejected.total_conversion > 0:
        rejected_cpa = rejected.total_cost / rejected.total_conversion
    else:
        rejected_cpa = float('inf')

    if rejected_cpa > rejected.cpa_constraint:
        reason = "cpa_violation"
    elif rejected.total_cost < rejected.budget * 0.1:
        reason = "too_conservative"
    else:
        reason = "suboptimal"

    return PreferencePair(
        chosen=chosen,
        rejected=rejected,
        rejection_reason=reason,
        all_results=lambda_results,
    )
