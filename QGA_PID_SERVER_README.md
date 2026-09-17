# QGA step6000 and PID server benchmark

The QGA checkpoint was selected from `D:/research/Experiment/QGA/dense_model/best_checkpoint.json`. The selected pair is `actor_step_6000.pt` and `critic_step_6000.pt`.

Upload the contents of `server_qga_pid_models_20260917` into the server-side AuctionNet project root. This creates:

```text
saved_model/QGA_dense/best_checkpoint.json
saved_model/QGA_dense/QGA/normalize_dict.pkl
saved_model/QGA_dense/QGA/checkpoints/actor_step_6000.pt
saved_model/QGA_dense/QGA/checkpoints/critic_step_6000.pt
```

Run the CPU smoke test before submitting the full array:

```bash
python run_dense_opponent_benchmark.py --budget_rate 1.0 --episodes 0 --advertisers 0 --strategies QGA PID --pv 0.005 --pv_num 500000 --seed 42 --device cpu --qga_dir saved_model/QGA_dense/QGA --qga_ckpt_step 6000 --output_dir exp_data/qga_pid_smoke --output qga_pid_smoke
```

Submit all 100 strategy/episode/budget shards with:

```bash
sbatch run_qga_pid_budget.sbatch
```
