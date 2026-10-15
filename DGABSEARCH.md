# dgabsearch benchmark strategy

Run from the AuctionNet project root in nips-Gebidding-env:

```powershell
python benchmark_multistrat.py --strategies dgabsearch --dgabsearch_dir ../dgab/saved_model/dgab_shared_20260927172551 --dgabsearch_ckpt step_15000.pt --pv 0.0005 --advertisers 0 --episodes 0 --device cuda:0 --fail_fast --output dgabsearch_test --trajectory_output exp_data/dgabsearch_test_ticks.csv
```

Omit `--advertisers` to test all 48. `--episodes 0 1` selects multiple environment
episodes, and `--pv` accepts multiple means. To compare the original and feedback
inference, use `--strategies DGABShare dgabsearch` and pass the same model directory
through both `--dgabshare_dir` and `--dgabsearch_dir` (and matching checkpoint flags).
Without `--dgabsearch_dir`, the model directory falls back to DGABShare's directory.
The default search checkpoint is step_15000.pt. Existing strategy defaults remain.

The new strategy generates one deterministic action, evaluates the clipped action
with Q, then compares Q next-return versus target-minus-actual at the next decision.
Score uses accumulated actual conversion and exposed cost, including the preceding
tick. Both return channels are selected together; ties retain the residual. Negative
returns are clamped to zero; nonfinite Q falls back to the residual. Initial targets
use `--v_goal_multiplier` (default 1.0) and budget. AR action noise is disabled for
dgabsearch. The environment may subsequently adjust individual bids for budget.

`--trajectory_output` retains feedback candidates, scores, selected source, and
accumulated outcomes. Feedback fields describe the choice before the current action;
no feedback is needed after the final action. The manifest includes a separate
dgabsearch model path, checkpoint and SHA256. Shared linear-head AWBC checkpoints
are required, as supported by the existing AuctionNet DGABShare loader.

Checks: `python -m unittest discover -s tests -p test_dgabsearch_feedback.py`.

## Server full-scenario run

`slurm_dgabsearch_full_scenarios.sbatch` evaluates only the new inference mode;
it does not retrain or change opponents. Dense uses the trained dense opponent
population and `saved_model/dgab_shared_20260927172519/step_4000.pt` with
`v_goal_multiplier=25`. Sparse uses the existing sparse opponent population and
`saved_model/dgab_sparse_20260927172551/DGAB_Pretrain/step_15000.pt` with
`v_goal_multiplier=1.0`. Both use CPU, seed 42, all 48 advertisers, episodes
0--9, five budget ratios at reference traffic, and four non-reference traffic
scales at budget ratio 1.0. The overlapping reference condition is run once.

From the server AuctionNet project root, create `log` before submission
(Slurm opens output paths before the script starts), verify both model
directories contain the checkpoint and `normalize_dict.pkl`, then run:

```bash
mkdir -p log
python -m unittest discover -s tests -p test_dgabsearch_feedback.py
sbatch --array=20,110 --export=ALL,SMOKE_ADVERTISER=0,OUTPUT_ROOT=exp_data/dgabsearch_smoke slurm_dgabsearch_full_scenarios.sbatch
```

Array IDs 20 and 110 are dense and sparse reference-traffic, 100%-budget,
episode-0 smoke tests, restricted to advertiser 0 and a separate output root.
After both finish successfully, submit the full array:

```bash
sbatch slurm_dgabsearch_full_scenarios.sbatch
```

The full job has 180 array elements with at most 24 simultaneous CPU tasks.
Results go to `exp_data/dgabsearch_full_scenarios/{dense,sparse}/`.
