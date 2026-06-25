"""
Multi-objective scoring for bid decision evaluation.

Different bidding tasks have different objectives:
  - Unconstrained: purely maximize conversions (brand campaigns).
  - CPA soft-constraint: NeurIPS-style score = penalty^beta * conversions,
    where penalty = min(1, cpa_constraint / real_cpa).

The ScoringConfig dataclass allows switching between these modes.
The compute_score() function is used by preference_extractor to rank
lambda candidates.
"""

from dataclasses import dataclass


@dataclass
class ScoringConfig:
    """Configuration for the scoring / evaluation function.

    Attributes:
        use_cpa_constraint: If False, score = total_conversion (unconstrained).
            If True, applies CPA soft-penalty per the NeurIPS formula.
        cpa_penalty_beta: Exponent for the CPA penalty term.
            beta=2 is the NeurIPS default (quadratic penalty).
            beta=1 gives linear penalty.
            Higher beta = more severe penalty for CPA violations.
    """
    use_cpa_constraint: bool = True
    cpa_penalty_beta: float = 2.0


def compute_score(result, config: ScoringConfig) -> float:
    """Compute the composite score for a single simulation result.

    The score represents the "goodness" of a bidding strategy:
      - Higher score = better strategy.
      - Score is always non-negative.

    Unconstrained mode:
        score = total_conversion
        (Pure volume optimization — spend budget, maximize conversions.)

    CPA soft-constraint mode (NeurIPS formula):
        if real_cpa <= cpa_constraint:
            score = total_conversion
        else:
            penalty = cpa_constraint / real_cpa    # in (0, 1)
            score = penalty^beta * total_conversion

        This means: slightly exceeding CPA is penalized but not rejected;
        severely exceeding CPA receives heavy penalty.

    Args:
        result: A SimulationResult (or any object with total_cost,
            total_conversion, cpa_constraint attributes).
        config: ScoringConfig specifying the scoring mode.

    Returns:
        float: Computed score.
    """
    conversions = result.total_conversion

    if not config.use_cpa_constraint or conversions <= 0:
        return max(0.0, conversions)

    real_cpa = result.total_cost / (conversions + 1e-10)

    if real_cpa <= result.cpa_constraint:
        # CPA compliant — full reward
        return conversions
    else:
        # CPA violated — apply soft penalty
        penalty_ratio = result.cpa_constraint / (real_cpa + 1e-10)
        penalty = penalty_ratio ** config.cpa_penalty_beta
        return penalty * conversions
