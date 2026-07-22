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
    parser.add_argument("--chunk_size", type=int, default=0,
                        help="Number of INPUT files per output chunk (0 = single merged file). "
                             "When set, produces training_000.pkl, training_001.pkl, ... "
                             "each with at most chunk_size input files merged. "
                             "Use this when the full corpus exceeds available RAM "
                             "(e.g. --chunk_size 200 gives ~10 GB chunks).")
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
    if args.chunk_size > 0:
        n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size
        print(f"Chunks: {n_chunks} (up to {args.chunk_size} files each)")
    if args.cols:
        keep = [c.strip() for c in args.cols.split(",") if c.strip()]
        print(f"Cols:   {keep}")
    else:
        keep = None

    t0 = time.time()

    if args.chunk_size > 0:
        # ── Chunked mode: write one output file per chunk, low memory ──
        base, ext = os.path.splitext(args.output)
        n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size
        total_rows = 0
        for ci in range(n_chunks):
            batch = []
            batch_rows = 0
            lo = ci * args.chunk_size
            hi = min(lo + args.chunk_size, len(files))
            for i in range(lo, hi):
                df = pd.read_pickle(files[i])
                if keep is not None:
                    df = df[[c for c in keep if c in df.columns]]
                batch_rows += len(df)
                batch.append(df)
            chunk_df = pd.concat(batch, ignore_index=True)
            if args.sort_by:
                sort_cols = [c.strip() for c in args.sort_by.split(",") if c.strip()]
                chunk_df = chunk_df.sort_values(sort_cols).reset_index(drop=True)
            out_path = f"{base}_{ci:03d}{ext}"
            chunk_df.to_pickle(out_path)
            total_rows += len(chunk_df)
            size_mb = os.path.getsize(out_path) / 1024**2
            print(f"  [{ci + 1}/{n_chunks}] {out_path}: "
                  f"{len(chunk_df):,} rows ({size_mb:.0f} MB)")
            del batch, chunk_df

        elapsed = time.time() - t0
        print(f"Done: {total_rows:,} rows across {n_chunks} chunks in {elapsed:.0f}s")

    else:
        # ── Single-file mode (original behavior) ──
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
