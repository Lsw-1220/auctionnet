"""
Dataset quality comparison: data/log/0.csv vs strategy_train_env/data/traffic/period-12.csv

"Expert" dataset indicators (higher = more expert):
  1. Score (AuctionNet official metric)
  2. CPA control (lower CPA exceedance penalty → better)
  3. Budget utilization (closer to 100% without overspend)
  4. Win rate & total reward
  5. Alpha distribution (less extreme = more reasonable)
  6. Budget pacing (smooth spending over time)
"""

import pandas as pd
import numpy as np
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CSV1 = os.path.join(_PROJECT_ROOT, 'data', 'log', '0.csv')
CSV2 = os.path.join(_PROJECT_ROOT, 'strategy_train_env', 'data', 'traffic', 'period-12.csv')


def analyze_dataset(df, label):
    """Compute per-advertiser metrics."""
    df_win = df[df['isExposed'] == 1]

    total_cost = df_win.groupby('advertiserNumber')['cost'].sum()
    total_reward = df.groupby('advertiserNumber')['conversionAction'].sum()
    budget = df.groupby('advertiserNumber')['budget'].first()
    cpa_target = df.groupby('advertiserNumber')['CPAConstraint'].first()

    cpa_real = total_cost / (total_reward + 1e-10)
    penalty = np.where(cpa_real > cpa_target,
                       (cpa_target / (cpa_real + 1e-10)) ** 2, 1.0)
    score = penalty * total_reward
    budget_used = total_cost / budget * 100

    compete_per_adv = df.groupby('advertiserNumber').size()
    win_per_adv = df_win.groupby('advertiserNumber').size()
    win_rate = win_per_adv / (compete_per_adv + 1e-10) * 100

    # Alpha: bid / pValue
    alpha = df['bid'] / (df['pValue'] + 1e-10)
    alpha_median = alpha.groupby(df['advertiserNumber']).median()

    n_agents = len(total_reward)

    print(f'\n{"="*70}')
    print(f'DATASET: {label}  ({n_agents} agents, {len(df):,} rows)')
    print(f'{"="*70}')

    print(f'\n  {"Metric":<28} {"Mean":>10} {"Std":>10} {"Median":>10} {"Min":>10} {"Max":>10}')
    print(f'  {"-"*70}')
    for name, arr in [
        ('Score', score),
        ('Total Reward', total_reward),
        ('Total Cost', total_cost),
        ('CPA Real', cpa_real),
        ('CPA Target', cpa_target),
        ('Budget Used %', budget_used),
        ('Win Rate %', win_rate),
        ('CPA Penalty', penalty),
    ]:
        print(f'  {name:<28} {arr.mean():>10.2f} {arr.std():>10.2f} '
              f'{np.median(arr):>10.2f} {arr.min():>10.2f} {arr.max():>10.2f}')

    # Budget utilization bands
    print(f'\n  Budget Utilization Bands:')
    for lo, hi in [(0, 50), (50, 80), (80, 95), (95, 99), (99, 100), (100, 105)]:
        cnt = ((budget_used >= lo) & (budget_used < hi)).sum()
        bar = '█' * (cnt * 40 // n_agents)
        print(f'    [{lo:>3}%-{hi:>3}%): {cnt:>3}/{n_agents}  {bar}')

    # CPA penalty bands
    print(f'\n  CPA Penalty Distribution:')
    for lo, hi in [(1.0, 1.0), (0.8, 1.0), (0.5, 0.8), (0.2, 0.5), (0.0, 0.2)]:
        cnt = ((penalty >= lo) & (penalty <= hi)).sum()
        bar = '█' * (cnt * 40 // n_agents)
        print(f'    [{lo:.1f}-{hi:.1f}]: {cnt:>3}/{n_agents}  {bar}')

    # Budget pacing curve
    budget_left_by_tick = df.groupby(['advertiserNumber', 'timeStepIndex'])['remainingBudget'].last()
    budget_left_avg = budget_left_by_tick.groupby('timeStepIndex').mean()
    max_budget = budget.mean()
    print(f'\n  Budget Pacing (avg remaining / max budget):')
    print(f'    {"Tick":<6} {"Remaining":>10} {"%":>7}')
    for t in sorted(budget_left_avg.index):
        pct = budget_left_avg[t] / budget.max() * 100  # per-tick vs own budget
        print(f'    {int(t):<6} {budget_left_avg[t]:>10.0f} {pct:>7.1f}%')

    # Alpha distribution
    print(f'\n  Alpha (bid/pValue) Distribution:')
    print(f'    Mean={alpha.mean():.2f}  Median={alpha.median():.2f}  Std={alpha.std():.2f}')
    for q in [1, 5, 25, 75, 95, 99]:
        print(f'    P{q:02d}: {np.percentile(alpha, q):.2f}')

    # pValue distribution
    print(f'\n  pValue Distribution:')
    print(f'    Mean={df["pValue"].mean():.6f}  Std={df["pValue"].std():.6f}  '
          f'Median={df["pValue"].median():.6f}')

    # Zero bid
    zero_bid = (df['bid'] < 1e-10).mean() * 100
    print(f'\n  Zero bid: {zero_bid:.2f}%')

    return {
        'score': score, 'penalty': penalty, 'budget_used': budget_used,
        'cpa_real': cpa_real, 'total_reward': total_reward, 'total_cost': total_cost,
        'win_rate': win_rate, 'alpha_median': alpha_median,
        'alpha': alpha, 'n_agents': n_agents,
    }


# ── Run ──
res1 = analyze_dataset(pd.read_csv(CSV1), 'data/log/0.csv')
res2 = analyze_dataset(pd.read_csv(CSV2), 'traffic/period-12.csv')

# ── Head-to-head ──
print(f'\n{"="*70}')
print(f'HEAD-TO-HEAD')
print(f'{"="*70}')
print(f'\n  {"Metric":<28} {"0.csv":>14} {"period-12":>14} {"Winner":>10}')
print(f'  {"-"*68}')
for name, v1, v2, higher_better in [
    ('Avg Score', res1['score'].mean(), res2['score'].mean(), True),
    ('Median Score', res1['score'].median(), res2['score'].median(), True),
    ('Avg CPA Penalty', res1['penalty'].mean(), res2['penalty'].mean(), True),
    ('Avg Budget Used %', res1['budget_used'].mean(), res2['budget_used'].mean(), None),
    ('Avg CPA Real', res1['cpa_real'].mean(), res2['cpa_real'].mean(), False),
    ('Avg Total Reward', res1['total_reward'].mean(), res2['total_reward'].mean(), True),
    ('Avg Win Rate %', res1['win_rate'].mean(), res2['win_rate'].mean(), True),
    ('Median Alpha', res1['alpha_median'].median(), res2['alpha_median'].median(), None),
]:
    if higher_better is True:
        w = '0.csv' if v1 > v2 else 'period-12'
    elif higher_better is False:
        w = '0.csv' if v1 < v2 else 'period-12'
    else:
        w = '-'
    print(f'  {name:<28} {v1:>14.2f} {v2:>14.2f} {w:>10}')

# Percents of agents where 0.csv wins
print(f'\n  Agent-level win rates (0.csv > period-12):')
for name, a1, a2, higher_better in [
    ('Score', res1['score'], res2['score'], True),
    ('CPA Penalty', res1['penalty'], res2['penalty'], True),
    ('Budget Used %', res1['budget_used'], res2['budget_used'], None),
]:
    if higher_better is True:
        wins = (a1 > a2).sum()
    elif higher_better is False:
        wins = (a1 < a2).sum()
    else:
        wins = None
    if wins is not None:
        print(f'    {name}: {wins}/{res1["n_agents"]} ({wins/res1["n_agents"]*100:.1f}%)')

print()
