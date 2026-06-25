"""
EDA on real bidding logs (period 7-12).

Extracts per-period aggregate statistics to identify natural scenario
clusters, then produces scenario descriptions + parameter mappings.
"""
import pandas as pd
import numpy as np
import os
import json

DATA_DIR = "D:/research/Experiment/autobidding/data/MDP/traffic"
PERIODS = list(range(7, 13))
OUTPUT = "dpo_pipeline/scenario_eda_results.json"

CHUNK_SIZE = 200000  # rows per chunk


def extract_period_stats(csv_path, period_id):
    """Extract aggregate stats from one period CSV using chunked reading."""
    print(f"  Processing period-{period_id}...")

    # Accumulators
    tick_stats = {}  # tick -> {pv_count, pValue_sum, ...}
    total_rows = 0

    reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE)

    for chunk in reader:
        total_rows += len(chunk)

        # Per-tick aggregation within this chunk
        for tick, grp in chunk.groupby('timeStepIndex'):
            tick = int(tick)
            if tick not in tick_stats:
                tick_stats[tick] = {
                    'pv_set': set(),
                    'pValue_sum': 0.0, 'pValue_cnt': 0,
                    'pValueSigma_sum': 0.0,
                    'bid_sum': 0.0, 'bid_cnt': 0,
                    'leastWinCost_sum': 0.0, 'leastWinCost_cnt': 0,
                    'xi_sum': 0, 'xi_cnt': 0,
                    'isExposed_sum': 0,
                    'conversionAction_sum': 0,
                    'cost_sum': 0.0, 'cost_win_cnt': 0,
                }
            ts = tick_stats[tick]
            ts['pv_set'].update(grp['pvIndex'].unique().tolist())
            ts['pValue_sum'] += grp['pValue'].sum()
            ts['pValue_cnt'] += len(grp)
            ts['pValueSigma_sum'] += grp['pValueSigma'].sum()
            ts['bid_sum'] += (grp['bid'] * (grp['bid'] > 0)).sum()
            ts['bid_cnt'] += (grp['bid'] > 0).sum()
            ts['leastWinCost_sum'] += grp['leastWinningCost'].sum()
            ts['leastWinCost_cnt'] += len(grp)
            ts['xi_sum'] += grp['xi'].sum()
            ts['xi_cnt'] += len(grp)
            ts['isExposed_sum'] += grp['isExposed'].sum()
            ts['conversionAction_sum'] += grp['conversionAction'].sum()
            ts['cost_sum'] += (grp['cost'] * (grp['isExposed'] > 0)).sum()
            ts['cost_win_cnt'] += (grp['cost'] > 0).sum()

        if total_rows % 5000000 == 0:
            print(f"    {total_rows / 1e6:.0f}M rows...", flush=True)

    # --- Compute final per-tick aggregates ---
    tick_rows = []
    for tick in sorted(tick_stats.keys()):
        ts = tick_stats[tick]
        pv_count = len(ts['pv_set'])
        pValue_mean = ts['pValue_sum'] / ts['pValue_cnt'] if ts['pValue_cnt'] else 0
        pValueSigma_mean = ts['pValueSigma_sum'] / ts['pValue_cnt'] if ts['pValue_cnt'] else 0
        bid_mean = ts['bid_sum'] / ts['bid_cnt'] if ts['bid_cnt'] else 0
        leastWinCost_mean = ts['leastWinCost_sum'] / ts['leastWinCost_cnt'] if ts['leastWinCost_cnt'] else 0
        win_rate = ts['xi_sum'] / ts['xi_cnt'] if ts['xi_cnt'] else 0
        exposed_rate = ts['isExposed_sum'] / ts['xi_cnt'] if ts['xi_cnt'] else 0
        conv_rate = ts['conversionAction_sum'] / ts['xi_cnt'] if ts['xi_cnt'] else 0
        cost_per_win = ts['cost_sum'] / ts['cost_win_cnt'] if ts['cost_win_cnt'] else 0

        tick_rows.append({
            'period': period_id, 'tick': tick,
            'pv_count': pv_count,
            'pValue_mean': pValue_mean,
            'pValueSigma_mean': pValueSigma_mean,
            'bid_mean': bid_mean,
            'leastWinCost_mean': leastWinCost_mean,
            'win_rate': win_rate,
            'exposed_rate': exposed_rate,
            'conv_rate': conv_rate,
            'cost_per_win': cost_per_win,
        })

    # --- Period-level summary ---
    df_tick = pd.DataFrame(tick_rows)
    summary = {
        'period': period_id,
        'total_rows': total_rows,
        'num_ticks': len(tick_rows),
        'pv_total': df_tick['pv_count'].sum(),
        'pv_per_tick_avg': df_tick['pv_count'].mean(),
        'pv_per_tick_std': df_tick['pv_count'].std(),
        'pValue_mean': df_tick['pValue_mean'].mean(),
        'pValue_mean_std': df_tick['pValue_mean'].std(),
        'pValueSigma_mean': df_tick['pValueSigma_mean'].mean(),
        'bid_mean': df_tick['bid_mean'].mean(),
        'leastWinCost_mean': df_tick['leastWinCost_mean'].mean(),
        'win_rate': df_tick['win_rate'].mean(),
        'exposed_rate': df_tick['exposed_rate'].mean(),
        'conv_rate': df_tick['conv_rate'].mean(),
        'cost_per_win': df_tick['cost_per_win'].mean(),
        'tick_details': tick_rows,
    }
    return summary


def main():
    all_summaries = []
    for p in PERIODS:
        path = os.path.join(DATA_DIR, f"period-{p}.csv")
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found")
            continue
        summary = extract_period_stats(path, p)
        all_summaries.append(summary)

    # --- Cross-period comparison ---
    df_period = pd.DataFrame([{k: v for k, v in s.items()
                               if k != 'tick_details'}
                              for s in all_summaries])
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 240)
    pd.set_option('display.float_format', '{:.6f}'.format)
    print("\n" + "=" * 80)
    print("PERIOD-LEVEL SUMMARY")
    print("=" * 80)
    print(df_period.to_string(index=False))

    # --- Range analysis ---
    print("\n" + "=" * 80)
    print("CROSS-PERIOD RANGES (min / max / mean / ratio)")
    print("=" * 80)
    key_cols = ['pv_per_tick_avg', 'pValue_mean', 'leastWinCost_mean',
                'bid_mean', 'win_rate', 'conv_rate', 'cost_per_win']
    for col in key_cols:
        vals = df_period[col]
        print(f"  {col:25s}: {vals.min():.6f} / {vals.max():.6f} / "
              f"{vals.mean():.6f}  (max/min={vals.max()/max(vals.min(),1e-10):.2f}x)")

    # --- Save for later use ---
    import json
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved detailed results to {OUTPUT}")


if __name__ == "__main__":
    main()
