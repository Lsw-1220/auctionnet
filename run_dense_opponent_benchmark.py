"""Run the multi-strategy benchmark with dense trained opponent models."""

import argparse
import builtins
import os
import sys

import numpy as np

import benchmark_multistrat as benchmark

from bidding_train_env.strategy.bc_bidding_strategy import BcBiddingStrategy
from bidding_train_env.strategy.bcq_bidding_strategy import BcqBiddingStrategy
from bidding_train_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
from bidding_train_env.strategy.td3_bc_bidding_strategy import TD3_BCBiddingStrategy
from simul_bidding_env.strategy.dgabshare_bidding_strategy import DGABShareStrategy
from simul_bidding_env.strategy.dgab_ablation import DGABAblationStrategy


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--advertisers", nargs="+", type=int)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["BC", "BCQ", "DT", "GUIDE", "TD3_BC", "IQL", "DGABShare_26000"],
    )
    parser.add_argument("--pv", type=float, default=0.005)
    parser.add_argument("--pv_num", type=int, default=500000)
    parser.add_argument("--budget_rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--gave_dir", default=benchmark.GAVE_SAVE_DIR)
    parser.add_argument("--gave_ckpt", default="step_5000.pt")
    parser.add_argument(
        "--gas_dir",
        default=project_path("saved_model", "GAS_dense", "gas_paper_400k_2729782"),
    )
    parser.add_argument(
        "--dgabshare_dir",
        default=project_path("saved_model", "dgab_shared_20260912135525"),
    )
    parser.add_argument("--dgabshare_ckpt", default="step_26000.pt")
    parser.add_argument("--v_goal_multiplier", type=float, default=15.0)
    parser.add_argument("--action_multiplier", type=float, default=1.0)
    parser.add_argument("--ablation_model_root", default=project_path("saved_model", "dgab_ablation"))
    parser.add_argument(
        "--qga_dir",
        default=project_path("saved_model", "QGA_dense", "QGA"),
    )
    parser.add_argument("--qga_ckpt_step", type=int, default=6000)
    return parser.parse_args()


args = parse_args()

BC_DIR = project_path("saved_model", "BC_dense", "checkpoint_00011000")
BCQ_DIR = project_path("saved_model", "BCQ_dense", "checkpoint_00008000")
DT_DIR = project_path("saved_model", "DT_dense", "checkpoint_00001000")
GUIDE_DIR = project_path("saved_model", "guide_dense", "12000")
GUIDE_NORMALIZE = project_path("saved_model", "guide_dense", "normalize_dict.pkl")
TD3_BC_DIR = project_path("saved_model", "td3_bc_dense", "checkpoint_00015000")
IQL_DIR = project_path("saved_model", "IQL_dense", "checkpoint_00010000")


def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)


required_paths = [
    # The dense opponent population always uses these four trained policies.
    os.path.join(BC_DIR, "bc_model.pth"),
    os.path.join(BC_DIR, "normalize_dict.pkl"),
    os.path.join(BCQ_DIR, "bcq_model.pth"),
    os.path.join(BCQ_DIR, "normalize_dict.pkl"),
    os.path.join(TD3_BC_DIR, "td3_bc_model.pth"),
    os.path.join(TD3_BC_DIR, "normalize_dict.pkl"),
    os.path.join(IQL_DIR, "iql_model.pth"),
    os.path.join(IQL_DIR, "normalize_dict.pkl"),
]

selected_strategies = set(args.strategies)
ABLATION_NAMES = {f"v{i}_dense" for i in range(1, 6)}
for name in sorted(selected_strategies.intersection(ABLATION_NAMES)):
    directory = os.path.join(args.ablation_model_root, name)
    required_paths.extend([os.path.join(directory, f) for f in ("best.pt", "normalize_dict.pkl", "model_spec.json")])
if "GAVE" in selected_strategies:
    required_paths.extend([
        os.path.join(args.gave_dir, args.gave_ckpt),
        os.path.join(args.gave_dir, "normalize_dict.pkl"),
    ])
if "GAS" in selected_strategies:
    required_paths.append(os.path.join(args.gas_dir, "best_model.json"))
    for gas_kind in ("policy", "critic_101", "critic_202", "critic_303"):
        gas_step_dir = os.path.join(
            args.gas_dir, gas_kind, "checkpoints", "step_250000")
        required_paths.extend([
            os.path.join(gas_step_dir, "dt.pt"),
            os.path.join(gas_step_dir, "model.json"),
            os.path.join(gas_step_dir, "normalization.npz"),
        ])
if "DT" in selected_strategies:
    required_paths.extend([
        os.path.join(DT_DIR, "dt.pt"),
        os.path.join(DT_DIR, "normalize_dict.pkl"),
    ])
if "GUIDE" in selected_strategies:
    required_paths.extend([
        os.path.join(GUIDE_DIR, "GUIDE.pt"),
        os.path.join(GUIDE_DIR, "GUIDE_critic_inverse.pt"),
        os.path.join(GUIDE_DIR, "GUIDE_idm.pt"),
        GUIDE_NORMALIZE,
    ])
if selected_strategies.intersection({"DGABShare", "DGABShare_26000"}):
    required_paths.extend([
        os.path.join(args.dgabshare_dir, args.dgabshare_ckpt),
        os.path.join(args.dgabshare_dir, "normalize_dict.pkl"),
    ])
if "QGA" in selected_strategies:
    required_paths.extend([
        os.path.join(args.qga_dir, "normalize_dict.pkl"),
        os.path.join(args.qga_dir, "checkpoints", f"actor_step_{args.qga_ckpt_step}.pt"),
        os.path.join(args.qga_dir, "checkpoints", f"critic_step_{args.qga_ckpt_step}.pt"),
    ])

for required_path in required_paths:
    require_file(required_path)

original_initialize_agents = benchmark.Controller.initialize_agents


def initialize_dense_opponents(self):
    agents = original_initialize_agents(self)
    for index in (6, 22, 38):
        agents[index] = BcBiddingStrategy(name="BC-dense-opponent", model_dir=BC_DIR)
    for index in (9, 25, 41):
        agents[index] = BcqBiddingStrategy(name="BCQ-dense-opponent", model_dir=BCQ_DIR)
    for index in (2, 13, 18, 29, 34, 45):
        agents[index] = TD3_BCBiddingStrategy(
            name="TD3_BC-dense-opponent", model_dir=TD3_BC_DIR
        )
    for index in (1, 14, 17, 30, 33, 46):
        agents[index] = IqlBiddingStrategy(name="IQL-dense-opponent", model_dir=IQL_DIR)
    return agents


benchmark.Controller.initialize_agents = initialize_dense_opponents


def make_gave_checkpoint(budget, cpa, category, **kwargs):
    return benchmark.GAVEAuctionNetAgent(
        budget=budget,
        cpa=cpa,
        category=category,
        name="GAVE-Player",
        model_param={
            "save_dir": args.gave_dir,
            "ckpt_name": args.gave_ckpt,
            "hidden_size": 512,
            "time_dim": 8,
            "block_config": benchmark.BLOCK_CONFIG,
            "device": benchmark.DEVICE,
            "expectile": 0.99,
        },
    )

def make_guide_dense(budget, cpa, category, **kwargs):
    original_open = builtins.open
    checkpoint_normalize = os.path.abspath(os.path.join(GUIDE_DIR, "normalize_dict.pkl"))

    def redirected_open(path, *open_args, **open_kwargs):
        if os.path.abspath(os.fspath(path)) == checkpoint_normalize:
            path = GUIDE_NORMALIZE
        return original_open(path, *open_args, **open_kwargs)

    builtins.open = redirected_open
    try:
        from simul_bidding_env.strategy.guide_bidding_strategy import GUIDEStrategy

        return GUIDEStrategy(
            budget=budget,
            cpa=cpa,
            category=category,
            name="GUIDE-Player",
            model_dir=GUIDE_DIR,
            device=benchmark.DEVICE,
        )
    finally:
        builtins.open = original_open


class CalibratedDGABShareStrategy(DGABShareStrategy):
    def __init__(self, *strategy_args, action_multiplier=1.0, **strategy_kwargs):
        self.action_multiplier = float(action_multiplier)
        if not np.isfinite(self.action_multiplier) or self.action_multiplier <= 0:
            raise ValueError("action_multiplier must be finite and > 0")
        super().__init__(*strategy_args, **strategy_kwargs)

    def bidding(self, *bid_args, **bid_kwargs):
        pvalues = np.asarray(bid_args[1], dtype=np.float32)
        super().bidding(*bid_args, **bid_kwargs)
        original_alpha = float(self.last_diagnostics["alpha"])
        calibrated_alpha = float(np.clip(
            original_alpha * self.action_multiplier, 0.0, self.action_upper))
        self.last_diagnostics.update(
            uncalibrated_alpha=original_alpha,
            action_multiplier=self.action_multiplier,
            alpha=calibrated_alpha,
            executed_alpha=calibrated_alpha,
        )
        return calibrated_alpha * pvalues


def make_dgabshare_v15(budget, cpa, category, exploration_seed=0, **kwargs):
    return CalibratedDGABShareStrategy(
        budget=budget,
        cpa=cpa,
        category=category,
        name="DGABShare",
        action_multiplier=args.action_multiplier,
        model_param={
            "save_dir": args.dgabshare_dir,
            "ckpt_name": args.dgabshare_ckpt,
            "device": benchmark.DEVICE,
            "K": 20,
            "v_goal_multiplier": args.v_goal_multiplier,
            "exploration_scale": 0.0,
            "exploration_rho": 0.8,
            "exploration_seed": int(exploration_seed),
            "exploration_min_ratio": 0.8,
            "exploration_max_ratio": 1.2,
        },
    )


strategy_map = dict(benchmark.ALL_STRATEGIES)
strategy_map["GAVE"] = make_gave_checkpoint
strategy_map["GAS"] = benchmark.make_gas
strategy_map["PID"] = benchmark.make_pid
strategy_map["QGA"] = benchmark.make_qga
strategy_map["DT"] = benchmark.make_dt
strategy_map["GUIDE"] = make_guide_dense
strategy_map["DGABShare"] = make_dgabshare_v15
strategy_map["DGABShare_26000"] = make_dgabshare_v15
def make_ablation_factory(name):
    def factory(budget, cpa, category, **kwargs):
        return DGABAblationStrategy(budget=budget, cpa=cpa, category=category, name=name, device=benchmark.DEVICE, model_dir=os.path.join(args.ablation_model_root, name))
    return factory
for ablation_name in sorted(ABLATION_NAMES):
    strategy_map[ablation_name] = make_ablation_factory(ablation_name)
benchmark.ALL_STRATEGIES = list(strategy_map.items())

benchmark_args = [
    "benchmark_multistrat.py",
    "--pv", str(args.pv),
    "--pv_num", str(args.pv_num),
    "--budget_rate", str(args.budget_rate),
    "--episodes", *[str(episode) for episode in args.episodes],
    "--strategies", *args.strategies,
    "--gave_dir", args.gave_dir,
    "--gas_dir", args.gas_dir,
    "--bc_dir", BC_DIR,
    "--bcq_dir", BCQ_DIR,
    "--dt_dir", DT_DIR,
    "--guide_dir", GUIDE_DIR,
    "--td3_bc_dir", TD3_BC_DIR,
    "--iql_dir", IQL_DIR,
    "--dgabshare_dir", args.dgabshare_dir,
    "--dgabshare_ckpt", args.dgabshare_ckpt,
    "--qga_dir", args.qga_dir,
    "--qga_ckpt_step", str(args.qga_ckpt_step),
    "--device", args.device,
    "--seed", str(args.seed),
    "--output", args.output,
    "--fail_fast",
]

if args.advertisers is not None:
    benchmark_args.extend(["--advertisers", *[str(index) for index in args.advertisers]])
if args.output_dir:
    benchmark_args.extend(["--output_dir", args.output_dir])

sys.argv = benchmark_args

benchmark.main()
