"""
Merge all .pkl files in a directory into a single DataFrame pickle.

Usage:
    python merge_pkl.py --input_dir data/POMDP/pkl --output data/POMDP/training_data.pkl

    # Keep only certain columns to save memory
    python merge_pkl.py --input_dir data/POMDP/pkl --output data/POMDP/training.pkl --cols deliveryPeriodIndex,advertiserNumber,timeStepIndex,resourceLeft,observations,action,reward,budget,CPAConstraint,done
"""
import argparse, os, glob, sys, time
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge .pkl files into one training corpus")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing .pkl files to merge")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path (e.g. data/training_data.pkl)")
    parser.add_argument("--pattern", type=str, default="*.pkl",
                        help="Glob pattern for input files (default: *.pkl)")
    parser.add_argument("--cols", type=str, default=None,
                        help="Comma-separated columns to keep (default: keep all)")
    parser.add_argument("--sort_by", type=str, default=None,
                        help="Columns to sort by after merge, comma-separated "
                             "(e.g. deliveryPeriodIndex,advertiserNumber,timeStepIndex)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"ERROR: --input_dir not found: {args.input_dir}")

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        sys.exit(f"ERROR: no files matching '{args.pattern}' in {args.input_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    print(f"Input:  {args.input_dir}")
    print(f"Output: {os.path.abspath(args.output)}")
    print(f"Files:  {len(files)}")
    if args.cols:
        keep = [c.strip() for c in args.cols.split(",") if c.strip()]
        print(f"Cols:   {keep}")
    else:
        keep = None

    t0 = time.time()
    chunks = []
    total_rows = 0
    for i, f in enumerate(files):
        df = pd.read_pickle(f)
        if keep is not None:
            df = df[[c for c in keep if c in df.columns]]
        total_rows += len(df)
        chunks.append(df)
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f"  [{i + 1}/{len(files)}] read {total_rows:,} rows ...")

    merged = pd.concat(chunks, ignore_index=True)

    if args.sort_by:
        sort_cols = [c.strip() for c in args.sort_by.split(",") if c.strip()]
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    merged.to_pickle(args.output)
    elapsed = time.time() - t0
    print(f"Done: {len(merged):,} rows saved in {elapsed:.0f}s "
          f"({elapsed / len(files):.1f}s/file)")
    print(f"Output size: {os.path.getsize(args.output) / 1024**2:.0f} MB")


if __name__ == "__main__":
    main()
