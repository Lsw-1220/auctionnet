"""
Strategy configuration: defines "world types" for agent composition variation.
Each world type specifies a different set of competing strategies.
"""

import numpy as np

# ── Strategy factory functions ──────────────────────────────────────

def _make_pid(period_seed, **overrides):
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    rng = np.random.default_rng(period_seed)
    # Vary base_action (default=15) and temporal ratio
    base_action = overrides.get("base_action", rng.uniform(8, 25))
    # Smooth temporal ratio: some agents aggressive early, some late
    ratio_type = rng.choice(["flat", "early", "late", "midday"])
    if ratio_type == "flat":
        ratio = np.ones(48)
    elif ratio_type == "early":
        ratio = np.exp(-np.arange(48) / 10)
    elif ratio_type == "late":
        ratio = np.exp(-(47 - np.arange(48)) / 10)
    else:  # midday
        ratio = np.exp(-np.abs(np.arange(48) - 24) / 8)
    ratio = ratio / ratio.sum() * 48
    agent = PidBiddingStrategy(exp_tempral_ratio=ratio)
    agent.base_action = base_action
    return agent

def _make_abid(period_seed, **overrides):
    from simul_bidding_env.strategy.abid_bidding_strategy import AbidBiddingStrategy
    rng = np.random.default_rng(period_seed)
    ratio_type = rng.choice(["flat", "early", "late", "midday"])
    if ratio_type == "flat":
        ratio = np.ones(48)
    elif ratio_type == "early":
        ratio = np.exp(-np.arange(48) / 10)
    elif ratio_type == "late":
        ratio = np.exp(-(47 - np.arange(48)) / 10)
    else:
        ratio = np.exp(-np.abs(np.arange(48) - 24) / 8)
    ratio = ratio / ratio.sum() * 48
    return AbidBiddingStrategy(exp_tempral_ratio=ratio)

def _make_iql(period_seed, **overrides):
    from simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
    return IqlBiddingStrategy()

def _make_td3bc(period_seed, **overrides):
    from simul_bidding_env.strategy.td3_bc_bidding_strategy import TD3_BCBiddingStrategy
    return TD3_BCBiddingStrategy()

def _make_bc(period_seed, **overrides):
    from simul_bidding_env.strategy.bc_bidding_strategy import BcBiddingStrategy
    return BcBiddingStrategy()

def _make_bcq(period_seed, **overrides):
    from simul_bidding_env.strategy.bcq_bidding_strategy import BcqBiddingStrategy
    return BcqBiddingStrategy()

def _make_cql(period_seed, **overrides):
    from simul_bidding_env.strategy.cql_bidding_strategy import CqlBiddingStrategy
    return CqlBiddingStrategy()

def _make_mbrl_mopo(period_seed, **overrides):
    from simul_bidding_env.strategy.mbrl_mopo_bidding_strategy import MbrlMopoBiddingStrategy
    return MbrlMopoBiddingStrategy()

def _make_mbrl_combo(period_seed, **overrides):
    from simul_bidding_env.strategy.mbrl_combomicro_bidding_strategy import MbrlComboMicroBiddingStrategy
    return MbrlComboMicroBiddingStrategy()

def _make_onlinelp(period_seed, **overrides):
    from simul_bidding_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy
    rng = np.random.default_rng(period_seed)
    ep = overrides.get("onlinelp_episode", rng.integers(0, 7))
    return OnlineLpBiddingStrategy(episode=ep)


# ── Strategy registry ───────────────────────────────────────────────

STRATEGY_REGISTRY = {
    "pid":     _make_pid,
    "abid":    _make_abid,
    "iql":     _make_iql,
    "td3bc":   _make_td3bc,
    "bc":      _make_bc,
    "bcq":     _make_bcq,
    "cql":     _make_cql,
    "mbrl_m":  _make_mbrl_mopo,
    "mbrl_c":  _make_mbrl_combo,
    "onlinelp": _make_onlinelp,
}


# ── World type definitions ──────────────────────────────────────────
# Each world = 6 categories × 8 agents = 48 strategies.
# Period index deterministic mapping: world_type = hash(period) % weights

WORLD_TYPES = {}

# World 0: Standard (current baseline) — 35%
WORLD_TYPES["standard"] = {
    "weight": 35,
    "description": "Baseline: PID + RL + OnlineLP mix",
    "agent_specs": [
        # category 0 (even)
        ["pid", "iql", "td3bc", "onlinelp", "onlinelp", "cql", "bc", "mbrl_m"],
        # category 1 (odd)
        ["pid", "bcq", "mbrl_m", "onlinelp", "onlinelp", "td3bc", "iql", "mbrl_c"],
        # category 2 (even)
        ["pid", "iql", "td3bc", "onlinelp", "onlinelp", "cql", "bc", "mbrl_m"],
        # category 3 (odd)
        ["pid", "bcq", "mbrl_m", "onlinelp", "onlinelp", "td3bc", "iql", "mbrl_c"],
        # category 4 (even)
        ["pid", "iql", "td3bc", "onlinelp", "onlinelp", "cql", "bc", "mbrl_m"],
        # category 5 (odd)
        ["pid", "bcq", "mbrl_m", "onlinelp", "onlinelp", "td3bc", "iql", "mbrl_c"],
    ],
}

# World 1: Simple-only (no RL) — 15%
WORLD_TYPES["simple"] = {
    "weight": 15,
    "description": "Simple strategies only: PID + Abid + OnlineLP",
    "agent_specs": [
        ["pid", "pid", "pid", "abid", "abid", "onlinelp", "onlinelp", "onlinelp"],
        ["pid", "abid", "pid", "abid", "onlinelp", "onlinelp", "onlinelp", "pid"],
        ["abid", "pid", "abid", "pid", "onlinelp", "onlinelp", "pid", "onlinelp"],
        ["pid", "pid", "abid", "onlinelp", "abid", "onlinelp", "onlinelp", "pid"],
        ["abid", "abid", "pid", "pid", "onlinelp", "onlinelp", "pid", "onlinelp"],
        ["pid", "abid", "pid", "onlinelp", "abid", "pid", "onlinelp", "onlinelp"],
    ],
}

# World 2: RL-heavy — 15%
WORLD_TYPES["rl_heavy"] = {
    "weight": 15,
    "description": "RL-dominant: mostly RL strategies, few PID/OnlineLP",
    "agent_specs": [
        ["iql", "td3bc", "cql", "bcq", "bc", "mbrl_m", "mbrl_c", "iql"],
        ["td3bc", "bcq", "iql", "cql", "mbrl_m", "mbrl_c", "bc", "td3bc"],
        ["cql", "iql", "bcq", "td3bc", "mbrl_c", "bc", "mbrl_m", "cql"],
        ["bcq", "mbrl_m", "td3bc", "iql", "bc", "cql", "iql", "mbrl_c"],
        ["mbrl_m", "cql", "iql", "bc", "bcq", "td3bc", "cql", "mbrl_m"],
        ["iql", "bcq", "mbrl_c", "cql", "td3bc", "mbrl_m", "bc", "iql"],
    ],
}

# World 3: Budget-stressed — 15% (handled by budget multiplier in batch script)
WORLD_TYPES["budget_stressed"] = {
    "weight": 15,
    "description": "Standard strategies but with tight budgets (×0.3-0.6)",
    "agent_specs": WORLD_TYPES["standard"]["agent_specs"],  # same as standard
}

# World 4: Aggressive bidding — 15%
WORLD_TYPES["aggressive"] = {
    "weight": 15,
    "description": "Higher base_action PID + aggressive RL mix",
    "agent_specs": [
        ["pid", "pid", "td3bc", "onlinelp", "iql", "bcq", "mbrl_m", "pid"],
        ["pid", "iql", "pid", "onlinelp", "td3bc", "bcq", "pid", "mbrl_c"],
        ["td3bc", "pid", "iql", "onlinelp", "pid", "bcq", "cql", "pid"],
        ["pid", "bcq", "td3bc", "onlinelp", "pid", "iql", "mbrl_m", "pid"],
        ["iql", "pid", "bcq", "onlinelp", "pid", "td3bc", "pid", "mbrl_c"],
        ["pid", "pid", "mbrl_m", "onlinelp", "bcq", "pid", "td3bc", "iql"],
    ],
}

# World 5: Mixed CPA constraints — 5% (extra variation on CPA assignment)
WORLD_TYPES["cpa_varied"] = {
    "weight": 5,
    "description": "Standard strategies but with shuffled CPA constraints",
    "agent_specs": WORLD_TYPES["standard"]["agent_specs"],
}


def get_world_type(period_index):
    """Deterministically select a world type based on period index."""
    rng = np.random.default_rng(period_index * 77773)
    names = list(WORLD_TYPES.keys())
    weights = [WORLD_TYPES[n]["weight"] for n in names]
    # Normalize
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    idx = rng.choice(len(names), p=weights)
    return names[idx]


def build_agents_for_period(period_index, world_type):
    """Build the 48 agent list for a given period and world type.

    Returns:
        agents: list of 48 agent instances
        budget_multiplier: float
        cpa_shuffle: bool
    """
    specs = WORLD_TYPES[world_type]["agent_specs"]
    rng = np.random.default_rng(period_index * 55579)

    agents = []
    for cat_idx, cat_specs in enumerate(specs):
        for slot_idx, strat_name in enumerate(cat_specs):
            seed = period_index * 10000 + cat_idx * 100 + slot_idx
            factory = STRATEGY_REGISTRY[strat_name]
            agent = factory(seed)
            # Inject PID base_action variation for aggressive world
            if world_type == "aggressive" and strat_name == "pid":
                agent.base_action = rng.uniform(18, 35)
            agent.name += f"_p{period_index}_c{cat_idx}_s{slot_idx}"
            agents.append(agent)

    budget_multiplier = 1.0
    if world_type == "budget_stressed":
        budget_multiplier = rng.uniform(0.3, 0.6)

    cpa_shuffle = (world_type == "cpa_varied")

    return agents, budget_multiplier, cpa_shuffle
