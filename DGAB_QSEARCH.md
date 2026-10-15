# dgabQsearch

Independent strategy, separate from DGABShare and dgabsearch. Only benchmark
registration, CLI options and manifest tracking are added to existing code.
Search starts at t=0: four conversion targets [1, 1.2, 1.4, 1.6] times current
target, fixed current cost target; original policy action plus two Gaussian
perturbations per R; twelve clipped candidates ranked only by Q conversion.
Selected input R and normalized executed action are committed to history.
Next step subtracts actual conversion/cost before searching again. The Q return
does not replace the selected input R. No score feedback or extra AR noise.

Run from the AuctionNet root, using nips-Gebidding-env:

```powershell
python benchmark_multistrat.py --strategies dgabQsearch --dgabqsearch_dir ../dgab/saved_model/dgab_shared_20260927172551 --dgabqsearch_ckpt step_15000.pt --pv 0.00055 --episodes 0 --device cuda:0 --fail_fast --output dgabQsearch_test --trajectory_output exp_data/dgabQsearch_test_ticks.csv
```

Default: all 48 advertisers, 500000 PV, seed=42, goal multiplier=1.
Add `--advertisers 0 --pv_num 20000` for a quick integration check.
Use `--dgabqsearch_noise_std` to change std (default .1 in original bid coefficient
units, before clipping). Model directory falls back to `--dgabshare_dir`;
checkpoint defaults to step_15000.pt. Strategy name is case sensitive.
Existing linear-head shared AWBC checkpoints are supported by AuctionNet's loader.

Private noise seed: seed * 1000003 + episode * 1009 + advertiser. Candidate order
is R then action; ties select the first. Invalid action/Q_v candidates are excluded;
all invalid raises an error. Original Q_v values are ranked without clamping.
Cost Q never affects ranking. Both RTG channels retain their training scaling.
Q evaluates each candidate under the R used to generate it. Environment budget
enforcement may subsequently drop individual bids.

Tick CSV diagnostics include base/selected R, selected multiplier/action variant,
Q conversion and all 12 candidates as JSON in `qsearch_candidate_details`.
Candidate R/Q diagnostics are in original units; standard rtg_v/rtg_c columns
remain in scaled model units. The manifest records the search config and model.

Tests: `python -m unittest discover -s tests -p test_dgab_qsearch.py`.

## Sparse full-scenario server run

The sparse benchmark wrapper preserves the existing AuctionNet opponent
population. `slurm_dgabqsearch_sparse_full.sbatch` uses the same sparse DGAB
checkpoint (`saved_model/dgab_sparse_20260927172551/DGAB_Pretrain/step_15000.pt`)
and normalization file as the earlier sparse evaluation. The base
conversion-target multiplier is fixed at 1.0, while QSearch internally tests
its four target candidates. Search noise std is 0.1 in raw action units.

The job uses CPU and seed 42. It covers budget ratios 0.5, 0.75, 1.0, 1.25,
1.5 at PV 0.0005, plus PV 0.0004, 0.00045, 0.00055, 0.0006 at budget ratio 1.0,
each for episodes 0--9 and all 48 advertisers. The reference configuration
is run once: 90 array elements, at most 24 simultaneous single-core tasks.

From the server AuctionNet root, after syncing the code and verifying the
checkpoint and `normalize_dict.pkl` under the directory above:

```bash
mkdir -p log
python -m unittest discover -s tests -p test_dgab_qsearch.py
sbatch --array=20 --export=ALL,SMOKE_ADVERTISER=0,OUTPUT_ROOT=exp_data/dgabqsearch_sparse_smoke slurm_dgabqsearch_sparse_full.sbatch
```

The smoke task tests advertiser 0, episode 0, budget ratio 1.0 and PV 0.0005.
After it passes, run `sbatch slurm_dgabqsearch_sparse_full.sbatch`. Full results
go to `exp_data/dgabqsearch_sparse_full/`.
