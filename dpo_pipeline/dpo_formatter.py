"""
Step 5: DPO Data Formatter.

Formats a PreferencePair into a JSONL-ready record following the DPO
schema with auto-generated Chain-of-Thought (CoT) reasoning text.

Reasoning is template-based (not LLM-generated) to avoid circular
dependency and ensure determinism. Templates are parameterized with
real simulation numbers (CPA, conversions, budget usage, score).
"""

from .scenario_config import ScenarioConfig
from .scoring import ScoringConfig
from .preference_extractor import PreferencePair


# ===========================================================================
# Instruction template (shared across all records)
# ===========================================================================

INSTRUCTION = (
    "作为高级出价智能体，请根据当前宏观市场环境和微观账户约束，"
    "输出最优的出价系数 lambda，使得在满足CPA约束的前提下最大化总转化量。"
    "请给出完整的推理过程。"
)


# ===========================================================================
# Context builder
# ===========================================================================

def _build_context(scenario: ScenarioConfig, pair: PreferencePair) -> str:
    """Build the 'context' field combining macro environment and micro constraints."""
    ch = pair.chosen
    lines = [
        f"【宏观环境】{scenario.description}",
        f"【微观约束】当前总预算 {ch.budget:.1f}，CPA目标约束为 {ch.cpa_constraint:.1f}。",
    ]
    return "\n".join(lines)


# ===========================================================================
# CoT reasoning generators
# ===========================================================================

def _brief_trait(scenario: ScenarioConfig) -> str:
    """One-line scenario trait for rejected reasoning snippets."""
    trait_map = {
        "深夜冲动消费期": "深夜低流量、低竞争、高转化意愿",
        "双十一抢量期": "超高流量、超高竞争、转化稀释",
        "正常工作日": "常规流量、常规竞争、稳定转化",
        "节假日低竞争期": "流量略降、竞争减弱、转化意图明确",
        "高峰期高竞争期": "流量高峰、竞争激烈、底价上浮",
    }
    return trait_map.get(scenario.name, "常规状态")


def _build_chosen_reasoning(scenario: ScenarioConfig,
                            pair: PreferencePair) -> str:
    """Generate CoT reasoning for the chosen (optimal) lambda."""
    lam = pair.chosen.lambda_value
    cpa = pair.chosen.total_cost / (pair.chosen.total_conversion + 1e-10)
    constraint = pair.chosen.cpa_constraint
    conversions = pair.chosen.total_conversion
    cost = pair.chosen.total_cost
    budget_usage = cost / (pair.chosen.budget + 1e-10) * 100
    cpa_margin = (constraint - cpa) / (constraint + 1e-10) * 100
    score = pair.chosen.score

    # Scenario-specific reasoning templates
    templates = {
        "深夜冲动消费期": (
            f"深夜流量总量下降但用户转化意愿强烈，且竞争减弱。"
            f"选择lambda={lam}，真实CPA={cpa:.2f}（约束{constraint:.1f}，"
            f"安全边际{cpa_margin:.1f}%），预算利用率{budget_usage:.1f}%，"
            f"实现了{conversions:.0f}次转化。在当前低竞争窗口期以适度出价"
            f"高效捕获冲动流量，综合评分{score:.1f}，为最优策略。"
        ),
        "双十一抢量期": (
            f"双十一大促流量暴涨但竞争极度激烈，转化率被稀释。"
            f"lambda={lam}在激烈竞价中保持竞争力，真实CPA={cpa:.2f}"
            f"（约束{constraint:.1f}），转化量{conversions:.0f}次，"
            f"预算利用率{budget_usage:.1f}%。在抢量与控成本之间取得最佳平衡，"
            f"综合评分{score:.1f}。"
        ),
        "正常工作日": (
            f"工作日常规流量下，lambda={lam}维持真实CPA={cpa:.2f}"
            f"（约束{constraint:.1f}，安全边际{cpa_margin:.1f}%），"
            f"转化{conversions:.0f}次，预算利用率{budget_usage:.1f}%。"
            f"在稳定环境中实现了转化量与成本控制的最优平衡，"
            f"综合评分{score:.1f}。"
        ),
        "节假日低竞争期": (
            f"节假日竞争减弱，lambda={lam}利用低底价窗口，"
            f"真实CPA={cpa:.2f}（约束{constraint:.1f}），"
            f"转化{conversions:.0f}次，预算利用率{budget_usage:.1f}%。"
            f"以合理出价高效拿量，综合评分{score:.1f}。"
        ),
        "高峰期高竞争期": (
            f"高峰时段竞争激烈、底价上浮，lambda={lam}在CPA约束内"
            f"保持出价竞争力，真实CPA={cpa:.2f}（约束{constraint:.1f}），"
            f"转化{conversions:.0f}次，预算利用率{budget_usage:.1f}%。"
            f"在高竞争环境中实现稳健拿量，综合评分{score:.1f}。"
        ),
    }

    reasoning = templates.get(
        scenario.name,
        f"lambda={lam}实现转化{conversions:.0f}次，CPA={cpa:.2f}，"
        f"综合评分{score:.1f}，为所有候选参数中的最优选择。"
    )
    return reasoning


def _build_rejected_reasoning(scenario: ScenarioConfig,
                              pair: PreferencePair) -> str:
    """Generate CoT reasoning for the rejected (worst) lambda."""
    lam = pair.rejected.lambda_value
    trait = _brief_trait(scenario)
    constraint = pair.rejected.cpa_constraint
    score = pair.rejected.score

    if pair.rejected.total_conversion > 0:
        cpa = pair.rejected.total_cost / pair.rejected.total_conversion
    else:
        cpa = float('inf')

    conversions = pair.rejected.total_conversion
    cost = pair.rejected.total_cost
    budget_usage = cost / (pair.rejected.budget + 1e-10) * 100

    if pair.rejection_reason == "cpa_violation":
        cpa_exceed_pct = (cpa - constraint) / (constraint + 1e-10) * 100
        return (
            f"当前环境为{scenario.name}，流量特征为{trait}。"
            f"选择lambda={lam}导致真实CPA={cpa:.2f}，"
            f"超出约束{constraint:.1f}达{cpa_exceed_pct:.1f}%。"
            f"这种激进策略造成成本失控，违反了广告主的核心约束，"
            f"存在严重的预算超支风险，综合评分仅{score:.1f}。"
        )
    elif pair.rejection_reason == "too_conservative":
        return (
            f"当前环境为{scenario.name}，流量特征为{trait}。"
            f"选择lambda={lam}虽然可能满足CPA约束，"
            f"但仅获得{conversions:.0f}次转化，预算仅消耗{budget_usage:.1f}%。"
            f"出价过于保守导致大量预算和转化机会被浪费，"
            f"综合评分仅{score:.1f}。"
        )
    else:  # suboptimal
        return (
            f"当前环境为{scenario.name}，流量特征为{trait}。"
            f"选择lambda={lam}，真实CPA={cpa:.2f}，转化{conversions:.0f}次，"
            f"预算利用率{budget_usage:.1f}%。"
            f"在当前环境下该出价策略综合表现不佳，"
            f"综合评分{score:.1f}，低于最优策略。"
        )


# ===========================================================================
# Main formatter
# ===========================================================================

def format_dpo_record(scenario: ScenarioConfig,
                      pair: PreferencePair,
                      scoring_config: ScoringConfig = None) -> dict:
    """Format a single DPO preference pair into a JSONL-ready record.

    Args:
        scenario: The ScenarioConfig used for this simulation.
        pair: The PreferencePair (chosen + rejected lambdas).
        scoring_config: Optional ScoringConfig for metadata (not embedded).

    Returns:
        dict with keys: instruction, context, chosen, rejected.
            Ready for json.dumps(..., ensure_ascii=False).
    """
    chosen_text = (
        f"【推理过程】{_build_chosen_reasoning(scenario, pair)}\n"
        f"【决策出价】建议设定系数 lambda = {pair.chosen.lambda_value}。"
    )

    rejected_text = (
        f"【推理过程】{_build_rejected_reasoning(scenario, pair)}\n"
        f"【决策出价】建议设定系数 lambda = {pair.rejected.lambda_value}。"
    )

    return {
        "instruction": INSTRUCTION,
        "context": _build_context(scenario, pair),
        "chosen": chosen_text,
        "rejected": rejected_text,
    }
