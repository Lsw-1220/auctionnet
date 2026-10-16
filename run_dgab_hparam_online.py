"""Run one DGAB hyperparameter configuration/episode in AuctionNet.

Task IDs 0..199 cover 20 offline-selected configurations x episodes 0..9.
The checkpoint and normalization file remain in the DGAB project; no copy is
needed when the two repositories share the same server filesystem.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


EXPECTILES = ((0.51, 0.49), (0.7, 0.3), (0.8, 0.2),
             (0.9, 0.1), (0.99, 0.01))
OTHER_ALPHAS = (0.0, 0.25, 0.5, 2.0, 5.0)
PROJECT_ROOT = Path(__file__).resolve().parent


def configuration(slot):
    if slot not in range(20):
        raise ValueError("slot must be in 0..19")
    environment = "dense" if slot < 10 else "sparse"
    local = slot % 10
    if local < 5:
        tau_v, tau_c = EXPECTILES[local]
        return environment, "expectile", f"tau_v_{tau_v:g}_tau_c_{tau_c:g}"
    return environment, "alpha", f"alpha_{OTHER_ALPHAS[local - 5]:g}"


def selected_checkpoint(selection_root, slot):
    environment, sweep, name = configuration(slot)
    selection_file = selection_root / environment / sweep / name / "best_checkpoint.json"
    if not selection_file.is_file():
        raise FileNotFoundError(selection_file)
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if (selection.get("slot"), selection.get("environment"),
            selection.get("sweep"), selection.get("name")) != (slot, environment, sweep, name):
        raise ValueError(f"selection identity mismatch: {selection_file}")
    evaluation = selection["evaluation"]
    if (float(evaluation["budget_rate"]) != 1.0 or
            float(evaluation["v_goal_multiplier"]) != 1.0 or
            int(evaluation["seed"]) != 44):
        raise ValueError(f"offline selection protocol mismatch: {selection_file}")
    checkpoint = Path(selection["best"]["checkpoint_path"]).resolve()
    normalization = checkpoint.parent / "normalize_dict.pkl"
    if checkpoint.name != selection["best"]["checkpoint"]:
        raise ValueError(f"checkpoint filename mismatch: {selection_file}")
    for path in (checkpoint, normalization):
        if not path.is_file():
            raise FileNotFoundError(path)
    return environment, sweep, name, selection_file, checkpoint, normalization


def validate_results(path, episode):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if (len(rows) != 48 or
            {int(row["advertiser"]) for row in rows} != set(range(48)) or
            {int(row["episode"]) for row in rows} != {episode} or
            {row["status"] for row in rows} != {"ok"}):
        raise RuntimeError(f"incomplete or failed online evaluation: {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "exp_data" / "dgab_hparam_online")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task-id", type=int)
    mode.add_argument("--check-all", action="store_true")
    args = parser.parse_args()
    selection_root = args.selection_root.resolve()

    if args.check_all:
        for slot in range(20):
            env, sweep, name, _, checkpoint, _ = selected_checkpoint(selection_root, slot)
            print(f"slot={slot:02d} {env}/{sweep}/{name}: {checkpoint}")
        return

    if args.task_id not in range(200):
        parser.error("--task-id must be in 0..199")
    slot, episode = divmod(args.task_id, 10)
    env, sweep, name, selection_file, checkpoint, normalization = selected_checkpoint(
        selection_root, slot)
    output_dir = args.output_root.resolve() / env / sweep / name / f"episode_{episode}"
    prefix = f"DGABShare_{env}_{sweep}_{name}_ep{episode}"
    result_file = output_dir / f"{prefix}_results.csv"
    success_file = output_dir / "success.json"
    if success_file.is_file():
        previous = json.loads(success_file.read_text(encoding="utf-8"))
        if (previous.get("checkpoint") == str(checkpoint) and
                previous.get("selection_file") == str(selection_file)):
            validate_results(result_file, episode)
            print(f"SKIP completed: {result_file}", flush=True)
            return
        raise RuntimeError(f"existing success marker has another checkpoint: {success_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable,
               str(PROJECT_ROOT / ("run_dense_opponent_benchmark.py" if env == "dense"
                                   else "run_sparse_model_benchmark.py")),
               "--budget_rate", "1.0", "--episodes", str(episode),
               "--strategies", "DGABShare",
               "--pv", "0.005" if env == "dense" else "0.0005",
               "--pv_num", "500000", "--seed", "42", "--device", "cpu",
               "--dgabshare_dir", str(checkpoint.parent),
               "--dgabshare_ckpt", checkpoint.name,
               "--v_goal_multiplier", "25" if env == "dense" else "1.0",
               "--output_dir", str(output_dir), "--output", prefix]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    rows = validate_results(result_file, episode)
    marker = dict(status="complete", slot=slot, episode=episode,
                  environment=env, sweep=sweep, config=name,
                  checkpoint=str(checkpoint), normalization=str(normalization),
                  selection_file=str(selection_file), command=command,
                  result_file=str(result_file), advertisers=len(rows))
    temporary = success_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    os.replace(temporary, success_file)
    print(f"COMPLETE {result_file}", flush=True)


if __name__ == "__main__":
    main()
