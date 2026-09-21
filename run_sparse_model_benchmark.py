"""Run sparse-model generalization benchmarks with default AuctionNet opponents."""

import argparse
import json
import os
import sys

import benchmark_multistrat as benchmark

from simul_bidding_env.strategy.dgabshare_bidding_strategy import DGABShareStrategy


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--advertisers", nargs="+", type=int)
    parser.add_argument("--strategies", nargs="+", required=True)
    parser.add_argument("--pv", type=float, required=True)
    parser.add_argument("--pv_num", type=int, default=500000)
    parser.add_argument("--budget_rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model_root", default=project_path("saved_model", "sparse_benchmark"))
    parser.add_argument("--v_goal_multiplier", type=float, required=True)
    return parser.parse_args()


args = parse_args()
root = os.path.abspath(args.model_root)

BC_DIR = os.path.join(root, "BC", "checkpoint_00004000")
BCQ_DIR = os.path.join(root, "BCQ", "checkpoint_00001000")
IQL_DIR = os.path.join(root, "IQL", "checkpoint_00006000")
TD3_BC_DIR = os.path.join(root, "td3_bc", "checkpoint_00010000")
DT_DIR = os.path.join(root, "DT", "checkpoint_00008000")
GUIDE_DIR = os.path.join(root, "GUIDE", "9000")
DGAB_DIR = os.path.join(root, "DGAB_Pretrain")
GAVE_DIR = os.path.join(root, "GAVE", "gave_400k_2571370")
GAS_DIR = os.path.join(root, "GAS", "gas_paper_400k_2571447")
QGA_DIR = os.path.join(root, "QGA")


def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)


selected = set(args.strategies)
requirements = {
    "BC": [os.path.join(BC_DIR, "bc_model.pth"),
           os.path.join(BC_DIR, "normalize_dict.pkl")],
    "BCQ": [os.path.join(BCQ_DIR, "bcq_model.pth"),
            os.path.join(BCQ_DIR, "normalize_dict.pkl")],
    "IQL": [os.path.join(IQL_DIR, "iql_model.pth"),
            os.path.join(IQL_DIR, "normalize_dict.pkl")],
    "TD3_BC": [os.path.join(TD3_BC_DIR, "td3_bc_model.pth"),
               os.path.join(TD3_BC_DIR, "normalize_dict.pkl")],
    "DT": [os.path.join(DT_DIR, "dt.pt"),
           os.path.join(DT_DIR, "normalize_dict.pkl")],
    "GUIDE": [os.path.join(GUIDE_DIR, "GUIDE.pt"),
              os.path.join(GUIDE_DIR, "GUIDE_critic_inverse.pt"),
              os.path.join(GUIDE_DIR, "GUIDE_idm.pt"),
              os.path.join(GUIDE_DIR, "normalize_dict.pkl")],
    "DGABShare": [os.path.join(DGAB_DIR, "step_15000.pt"),
                  os.path.join(DGAB_DIR, "normalize_dict.pkl")],
    "GAVE": [os.path.join(GAVE_DIR, "step_95000.pt"),
             os.path.join(GAVE_DIR, "normalize_dict.pkl")],
    "QGA": [os.path.join(QGA_DIR, "normalize_dict.pkl"),
            os.path.join(QGA_DIR, "checkpoints", "actor_step_11000.pt"),
            os.path.join(QGA_DIR, "checkpoints", "critic_step_11000.pt")],
}
for strategy in selected:
    for path in requirements.get(strategy, []):
        require_file(path)

if "GAS" in selected:
    best_path = os.path.join(GAS_DIR, "best_model.json")
    require_file(best_path)
    with open(best_path, "r", encoding="utf-8") as f:
        gas_step = int(json.load(f)["best_step"])
    for kind in ("policy", "critic_101", "critic_202", "critic_303"):
        step_dir = os.path.join(GAS_DIR, kind, "checkpoints", f"step_{gas_step}")
        for filename in ("dt.pt", "model.json", "normalization.npz"):
            require_file(os.path.join(step_dir, filename))


def make_gave_sparse(budget, cpa, category, **kwargs):
    return benchmark.GAVEAuctionNetAgent(
        budget=budget, cpa=cpa, category=category, name="GAVE-Player",
        model_param={
            "save_dir": GAVE_DIR,
            "ckpt_name": "step_95000.pt",
            "hidden_size": 512,
            "time_dim": 8,
            "block_config": benchmark.BLOCK_CONFIG,
            "device": benchmark.DEVICE,
            "expectile": 0.99,
        },
    )


def make_guide_sparse(budget, cpa, category, **kwargs):
    from simul_bidding_env.strategy.guide_bidding_strategy import GUIDEStrategy
    return GUIDEStrategy(
        budget=budget, cpa=cpa, category=category, name="GUIDE-Player",
        model_dir=GUIDE_DIR, device=benchmark.DEVICE)


def make_dgab_pretrain(budget, cpa, category, exploration_seed=0, **kwargs):
    return DGABShareStrategy(
        budget=budget, cpa=cpa, category=category, name="DGABShare",
        model_param={
            "save_dir": DGAB_DIR,
            "ckpt_name": "step_15000.pt",
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
strategy_map.update({
    "PID": benchmark.make_pid,
    "DT": benchmark.make_dt,
    "GUIDE": make_guide_sparse,
    "DGABShare": make_dgab_pretrain,
    "GAVE": make_gave_sparse,
    "GAS": benchmark.make_gas,
    "QGA": benchmark.make_qga,
})
benchmark.ALL_STRATEGIES = list(strategy_map.items())

benchmark_args = [
    "benchmark_multistrat.py",
    "--pv", str(args.pv),
    "--pv_num", str(args.pv_num),
    "--budget_rate", str(args.budget_rate),
    "--episodes", *[str(value) for value in args.episodes],
    "--strategies", *args.strategies,
    "--bc_dir", BC_DIR,
    "--bcq_dir", BCQ_DIR,
    "--dt_dir", DT_DIR,
    "--guide_dir", GUIDE_DIR,
    "--td3_bc_dir", TD3_BC_DIR,
    "--iql_dir", IQL_DIR,
    "--gave_dir", GAVE_DIR,
    "--gas_dir", GAS_DIR,
    "--qga_dir", QGA_DIR,
    "--qga_ckpt_step", "11000",
    "--device", args.device,
    "--seed", str(args.seed),
    "--output_dir", args.output_dir,
    "--output", args.output,
    "--fail_fast",
]
if args.advertisers is not None:
    benchmark_args.extend(
        ["--advertisers", *[str(value) for value in args.advertisers]])

sys.argv = benchmark_args
benchmark.main()
