"""Merge and validate the 70 Slurm shards for one budget-rate run."""

import argparse
import glob
import json
import os

import pandas as pd


STRATEGIES = ["BC", "BCQ", "DT", "GUIDE", "TD3_BC", "IQL", "DGABShare_26000"]
METRICS = ["score", "reward", "cost", "cpa_real", "budget_used"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget_rate", type=float, required=True)
    parser.add_argument("--budget_tag", required=True)
    parser.add_argument("--root", default="exp_data/budget_sweep")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = os.path.join(args.root, args.budget_tag)
    shard_dir = os.path.join(run_dir, "shards")
    paths = sorted(glob.glob(os.path.join(shard_dir, "*_results.csv")))
    if len(paths) != 70:
        raise RuntimeError(f"Expected 70 result shards, found {len(paths)} in {shard_dir}")

    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    expected_rows = 10 * len(STRATEGIES) * 48
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(frame)}")
    if set(frame["strategy"]) != set(STRATEGIES):
        raise RuntimeError(f"Unexpected strategies: {sorted(set(frame['strategy']))}")
    if set(frame["episode"]) != set(range(10)):
        raise RuntimeError(f"Unexpected episodes: {sorted(set(frame['episode']))}")
    if set(frame["advertiser"]) != set(range(48)):
        raise RuntimeError("Advertiser coverage is not 0..47")
    if (frame["status"] != "ok").any():
        raise RuntimeError("At least one shard contains a failed evaluation")
    duplicate = frame.duplicated(["budget_rate", "strategy", "episode", "advertiser"])
    if duplicate.any():
        raise RuntimeError(f"Found {int(duplicate.sum())} duplicate result keys")
    if not (frame["budget_rate"] == args.budget_rate).all():
        raise RuntimeError("Result budget_rate does not match --budget_rate")

    frame = frame.sort_values(["episode", "strategy", "advertiser"])
    prefix = f"dense_budget_{args.budget_tag}_ep0_9"
    frame.to_csv(os.path.join(run_dir, f"{prefix}_results.csv"), index=False)

    summary = frame.groupby("strategy")[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.sort_values("score_mean", ascending=False).reset_index()
    summary.insert(0, "rank", range(1, len(summary) + 1))
    summary.to_csv(os.path.join(run_dir, f"{prefix}_summary.csv"), index=False)

    by_episode = frame.groupby(["strategy", "episode"])[METRICS].mean().reset_index()
    by_episode.to_csv(os.path.join(run_dir, f"{prefix}_by_episode.csv"), index=False)
    comparison = by_episode.pivot(index="episode", columns="strategy", values="score")
    comparison.to_csv(os.path.join(run_dir, f"{prefix}_comparison.csv"))

    manifest = {
        "budget_rate": args.budget_rate,
        "budget_tag": args.budget_tag,
        "rows": len(frame),
        "shards": len(paths),
        "episodes": list(range(10)),
        "advertisers": list(range(48)),
        "strategies": STRATEGIES,
        "failures": 0,
        "duplicate_keys": 0,
    }
    with open(os.path.join(run_dir, f"{prefix}_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(summary[["rank", "strategy", "score_mean", "cpa_real_mean", "budget_used_mean"]])


if __name__ == "__main__":
    main()
