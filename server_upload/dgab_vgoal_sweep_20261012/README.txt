Required code changes:
  benchmark_multistrat.py
  run_sparse_model_benchmark.py

Required server model paths (already deployed):
  saved_model/dgab_shared_20260927172519/step_4000.pt
  saved_model/dgab_shared_20260927172519/normalize_dict.pkl
  saved_model/dgab_sparse_20260927172551/DGAB_Pretrain/step_15000.pt
  saved_model/dgab_sparse_20260927172551/DGAB_Pretrain/normalize_dict.pkl

Submit from the AuctionNet project root:
  sbatch server_upload/dgab_vgoal_sweep_20261012/slurm_dgab_vgoal_dense.sbatch
  sbatch server_upload/dgab_vgoal_sweep_20261012/slurm_dgab_vgoal_sparse.sbatch

Each array has 630 tasks:
  7 v_goal_multiplier values x 90 tasks per multiplier.
  Per multiplier: 50 budget tasks + 40 non-baseline PV tasks.
  Each parameter setting uses episodes 0-9 and evaluates 48 advertisers.

Budget experiments:
  dense PV fixed at 0.005
  sparse PV fixed at 0.0005

PV experiments:
  budget_rate fixed at 1.0
  baseline PV results are reused from the budget_rate=1.0 tasks.

New per-advertiser result columns:
  budget
  v_goal_multiplier
  conversion_goal
  conversion_completion_rate

New summary columns:
  mean_conversion_completion_rate
  total_conversion
  total_conversion_goal
  aggregate_conversion_completion_rate
