"""Offline evaluation for packaged DGAB ablation checkpoints."""
import argparse, csv, glob, json, math, os, random
import numpy as np
import pandas as pd
import torch

from simul_bidding_env.strategy.dgab_ablation import DGABAblationStrategy
from strategy_train_env.bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from strategy_train_env.bidding_train_env.offline_eval.offline_env import OfflineEnv


def score(reward, cpa, target):
    return reward if cpa <= target else reward * (target / (cpa + 1e-10)) ** 2


def evaluate(path, model_dir, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    loader, env = TestDataLoader(file_path=path), OfflineEnv()
    meta_df = pd.read_csv(path, usecols=["deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex", "budget", "CPAConstraint"]).drop_duplicates(["deliveryPeriodIndex", "advertiserNumber"])
    metadata = {(row.deliveryPeriodIndex, row.advertiserNumber): (float(row.budget), float(row.CPAConstraint), int(row.advertiserCategoryIndex)) for row in meta_df.itertuples(index=False)}
    rows = []
    for key in loader.keys:
        steps, pvs, sigmas, lwcs = loader.mock_data(key)
        budget, cpa, category = metadata[key]
        agent = DGABAblationStrategy(budget=budget, cpa=cpa, category=category,
                                     model_dir=model_dir, device="cpu")
        reward = cost = 0.0
        hpv, hb, ha, hi, hl = [], [], [], [], []
        for tick in range(steps):
            bid = (np.zeros_like(pvs[tick]) if agent.remaining_budget < env.min_remaining_budget
                   else agent.bidding(tick, pvs[tick], sigmas[tick], hpv, hb, ha, hi, hl))
            value, tick_cost, status, conversion = env.simulate_ad_bidding(
                pvs[tick], sigmas[tick], bid, lwcs[tick])
            over = max((tick_cost.sum() - agent.remaining_budget) /
                       (tick_cost.sum() + 1e-4), 0)
            while over > 0:
                won = np.where(status == 1)[0]
                drop = min(int(math.ceil(len(won) * over)), len(won))
                bid[np.random.choice(won, drop, replace=False)] = 0
                value, tick_cost, status, conversion = env.simulate_ad_bidding(
                    pvs[tick], sigmas[tick], bid, lwcs[tick])
                over = max((tick_cost.sum() - agent.remaining_budget) /
                           (tick_cost.sum() + 1e-4), 0)
            agent.remaining_budget -= tick_cost.sum()
            reward += conversion.sum(); cost += tick_cost.sum()
            hpv.append(np.asarray([(pvs[tick][i], sigmas[tick][i]) for i in range(len(pvs[tick]))]))
            hb.append(bid); hl.append(lwcs[tick])
            ha.append(np.asarray([(pvs[tick][i], sigmas[tick][i], status[i], 0.,
                                   tick_cost[i], status[i], conversion[i])
                                  for i in range(len(status))]))
            hi.append(np.asarray([(conversion[i], conversion[i]) for i in range(len(conversion))]))
        real_cpa = cost / (reward + 1e-10)
        rows.append(dict(score=score(reward, real_cpa, cpa), conversion=reward,
                         exceeded=float(real_cpa > cpa), spend_rate=cost / budget))
    return dict(score=float(np.mean([r["score"] for r in rows])),
                conversion=float(np.mean([r["conversion"] for r in rows])),
                exceed_rate=float(np.mean([r["exceeded"] for r in rows])),
                mean_spend_rate=float(np.mean([r["spend_rate"] for r in rows])),
                advertisers=len(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--environment", choices=("dense", "sparse"), required=True)
    p.add_argument("--variant", choices=tuple(f"v{i}" for i in range(6)), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data", default=""); p.add_argument("--data_glob", default="")
    p.add_argument("--model_root", default="saved_model/dgab_ablation")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    files = [a.data] if a.data else sorted(glob.glob(a.data_glob))
    if not files: raise FileNotFoundError(a.data or a.data_glob)
    model_name = f"{a.variant}_{a.environment}"
    model_dir = os.path.join(a.model_root, model_name)
    results = []
    for path in files:
        result = evaluate(path, model_dir, a.seed)
        result.update(strategy=model_name, environment=a.environment,
                      seed=a.seed, data=os.path.abspath(path), model_dir=os.path.abspath(model_dir))
        results.append(result); print(json.dumps(result), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    with open(os.path.splitext(a.output)[0] + "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dict(command=" ".join(__import__("sys").argv), results=results), f, indent=2)


if __name__ == "__main__": main()
