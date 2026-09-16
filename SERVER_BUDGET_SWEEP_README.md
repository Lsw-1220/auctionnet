# Dense budget-rate benchmark on Slurm

Each submission evaluates one budget rate. The array has 70 shards: 10 episodes times 7 strategies. Each shard uses one CPU and evaluates all 48 advertisers.

## Submit

```bash
sbatch --export=ALL,BUDGET_RATE=0.5,BUDGET_TAG=050 run_dense_budget.sbatch
sbatch --export=ALL,BUDGET_RATE=0.75,BUDGET_TAG=075 run_dense_budget.sbatch
sbatch --export=ALL,BUDGET_RATE=1.25,BUDGET_TAG=125 run_dense_budget.sbatch
sbatch --export=ALL,BUDGET_RATE=1.5,BUDGET_TAG=150 run_dense_budget.sbatch
```

## Merge one completed budget rate

```bash
python merge_dense_budget_results.py --budget_rate 0.5 --budget_tag 050
```

Change the rate and tag for the other three runs. The merger requires exactly 70 successful shards and rejects missing coverage or duplicate keys.

## Resource tuning

The initial configuration is `--array=0-69%16`, one CPU and 24 GB per shard. Run a smoke shard first and inspect `MaxRSS` with `sacct`; reduce concurrency if aggregate node memory is insufficient.
