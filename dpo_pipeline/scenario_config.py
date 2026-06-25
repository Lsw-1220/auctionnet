"""
Step 1: Scenario Configurator.

Maps macro environment descriptions (e.g. "深夜冲动消费期", "双十一抢量期")
to concrete simulation parameter overrides for the AuctionNet environment.

Each scenario modifies five dimensions:
  - Traffic volume    → pv_num multiplier
  - Conversion quality → pv_values post-process multiplier
  - Uncertainty       → pValueSigmas post-process multiplier
  - Competition       → reserve_price (GSP floor price)
  - Player constraints → budget + cpa_constraint
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ScenarioConfig:
    """Configuration for one macro bidding scenario.

    Attributes:
        name: Short human-readable scenario name (Chinese).
        description: Detailed macro environment description used in DPO context.
        traffic_volume_multiplier: Scales total PV count per episode.
            < 1.0 = less traffic, > 1.0 = more traffic.
        pvalue_mean_multiplier: Scales the mean of pValue (conversion prob).
            < 1.0 = lower conversion quality, > 1.0 = higher.
        pvalue_sigma_multiplier: Scales the uncertainty (sigma) of pValue.
            < 1.0 = more predictable, > 1.0 = more uncertain.
        reserve_price: GSP auction reserve/floor price.
            Lower = easier to win, Higher = more expensive to compete.
        budget: Player advertiser's total budget for the delivery period.
        cpa_constraint: Player advertiser's CPA (cost-per-action) target.
        extra_params: Extension point for future scenario-specific parameters.
    """
    name: str
    description: str
    traffic_volume_multiplier: float = 1.0
    pvalue_mean_multiplier: float = 1.0
    pvalue_sigma_multiplier: float = 1.0
    reserve_price: float = 0.0001
    budget: float = 2900.0
    cpa_constraint: float = 100.0
    extra_params: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Preset Scenarios (5 typical macro environments)
# ===========================================================================

SCENARIOS: Dict[str, ScenarioConfig] = {
    # ------------------------------------------------------------------
    "night_impulse": ScenarioConfig(
        name="深夜冲动消费期",
        description=(
            "星期日晚上十一点，深夜冲动消费期。整体流量下降约40%，"
            "但在线用户转化意愿强烈、行为方差大。竞争对手出价温和，"
            "大盘底价降低约30%。"
        ),
        traffic_volume_multiplier=0.6,
        pvalue_mean_multiplier=1.1,
        pvalue_sigma_multiplier=1.3,
        reserve_price=0.00007,
        budget=2900.0,
        cpa_constraint=100.0,
    ),

    # ------------------------------------------------------------------
    "double_11": ScenarioConfig(
        name="双十一抢量期",
        description=(
            "双十一大促最后两小时，全平台流量暴涨约3倍。"
            "大量价格敏感型浏览用户涌入，整体转化率被稀释。"
            "竞争极度激烈，大盘底价指数级攀升。"
        ),
        traffic_volume_multiplier=3.0,
        pvalue_mean_multiplier=0.8,
        pvalue_sigma_multiplier=0.9,
        reserve_price=0.0003,
        budget=5000.0,
        cpa_constraint=130.0,
    ),

    # ------------------------------------------------------------------
    "normal_workday": ScenarioConfig(
        name="正常工作日",
        description=(
            "普通工作日上午，流量平稳，用户行为模式规律。"
            "竞争处于常规水平，大盘底价标准。"
        ),
        traffic_volume_multiplier=1.0,
        pvalue_mean_multiplier=1.0,
        pvalue_sigma_multiplier=1.0,
        reserve_price=0.0001,
        budget=2900.0,
        cpa_constraint=100.0,
    ),

    # ------------------------------------------------------------------
    "holiday_low": ScenarioConfig(
        name="节假日低竞争期",
        description=(
            "法定节假日，用户多在线下活动，线上流量下降约20%。"
            "但浏览用户购买意图更明确，转化率略高。"
            "多数广告主降低预算，竞争减弱。"
        ),
        traffic_volume_multiplier=0.8,
        pvalue_mean_multiplier=1.05,
        pvalue_sigma_multiplier=0.8,
        reserve_price=0.00008,
        budget=2900.0,
        cpa_constraint=90.0,
    ),

    # ------------------------------------------------------------------
    "peak_hour": ScenarioConfig(
        name="高峰期高竞争期",
        description=(
            "工作日晚间8-10点流量高峰，在线用户数增加约30%。"
            "大量广告主集中投放，竞争激烈，大盘底价上浮约40%。"
        ),
        traffic_volume_multiplier=1.3,
        pvalue_mean_multiplier=0.9,
        pvalue_sigma_multiplier=1.1,
        reserve_price=0.00014,
        budget=3500.0,
        cpa_constraint=120.0,
    ),
}
