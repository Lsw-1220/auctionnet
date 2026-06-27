"""
Batch generate auction data + immediate POMDP conversion (no raw CSV retained).

Usage:
    python batch_generate_pomdp.py --start 0 --end 1000 --pv_num 500000 --output_dir data/POMDP
"""

import argparse, sys, os, time, logging, glob
import numpy as np
import pandas as pd
import gin

sys.path.append("./strategy_train_env")

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from batch_generate import run_single_period, parse_args as batch_parse_args
from bidding_train_env.train_data_generator.POMDP_data_generator import POMDPDataGenerator


def main():
    parser = argparse.ArgumentParser(description="Batch generate + POMDP conversion")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--pv_num", type=int, default=500000)
    parser.add_argument("--generator", type=str, default="neuripsPvGen")
    parser.add_argument("--output_dir", type=str, default="data/POMDP")
    parser.add_argument("--config", type=str, default="./config/test.gin")
    parser.add_argument("--keep_raw", action="store_true",
                        help="Keep raw CSV files (default: delete after convert)")
    parser.add_argument("--merge_every", type=int, default=200,
                        help="Merge POMDP data every N periods to limit memory")
    args = parser.parse_args()

    gin.parse_config_files_and_bindings([args.config], None)

    raw_dir = os.path.join(args.output_dir, '_raw_temp')
    pkl_dir = os.path.join(args.output_dir, 'pkl')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(pkl_dir, exist_ok=True)

    converter = POMDPDataGenerator()
    all_merged = []

    t_total = time.time()
    success, fail = 0, 0

    for period in range(args.start, args.end):
        try:
            result = run_single_period(
                period_index=period,
                pv_num=args.pv_num,
                generator_type=args.generator,
                output_dir=raw_dir,
                budget_perturb=0.08,
            )
            csv_path = os.path.join(raw_dir, f"period-{period:05d}.csv")

            # Convert to POMDP
            df_pomdp, pkl_path = converter.convert_csv_to_pickle(csv_path, pkl_dir)
            all_merged.append(df_pomdp)

            # Delete raw CSV
            if not args.keep_raw:
                os.remove(csv_path)

            success += 1

        except Exception as e:
            logger.error(f"Period {period:05d} FAILED: {e}")
            fail += 1

        # Periodic merge
        if (period - args.start + 1) % args.merge_every == 0:
            merged = pd.concat(all_merged, ignore_index=True)
            merged.to_pickle(os.path.join(pkl_dir, f'POMDP_{args.start:05d}_{period:05d}.pkl'))
            logger.info(f"  Merged {len(all_merged)} batches -> {len(merged)} rows")
            all_merged = []  # free memory

    # Final merge
    if all_merged:
        merged = pd.concat(all_merged, ignore_index=True)
        merged.to_pickle(os.path.join(pkl_dir, f'POMDP_{args.start:05d}_{args.end-1:05d}.pkl'))
        logger.info(f"  Final merge: {len(merged)} rows")

    # Clean up temp dir
    if not args.keep_raw:
        for f in glob.glob(os.path.join(raw_dir, '*.csv')):
            os.remove(f)
        os.rmdir(raw_dir)

    elapsed = time.time() - t_total
    logger.info(f"Done: {success}/{args.end - args.start} periods "
                f"in {elapsed:.0f}s ({elapsed/success:.1f}s/period)")


if __name__ == '__main__':
    main()
