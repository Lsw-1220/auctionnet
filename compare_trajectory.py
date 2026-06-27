"""
Compare two RL trajectory datasets:
  trajectory_data.csv vs trajectory_data_1.csv

Episode-level metrics: Score, CPA, budget utilization, action quality.
"""
import pandas as pd
import numpy as np
import os, ast

F1 = 'D:/research/Experiment/autobidding/data/MDP/trajectory/trajectory_data.csv'
F2 = 'D:/research/Experiment/autobidding/data/MDP/trajectory/trajectory_data_1.csv'


def analyze(fpath, label):
    print(f'\n{"="*70}')
    print(f'{label}')
    print(f'  Loading...')
    df = pd.read_csv(fpath)

    # Per-episode summary: last row per (period, advertiser) has final realAllCost/Conversion
    ep = df.groupby(['deliveryPeriodIndex', 'advertiserNumber']).agg(
        budget=('budget', 'first'),
        cpa_target=('CPAConstraint', 'first'),
        total_cost=('realAllCost', 'first'),
        total_reward=('realAllConversion', 'first'),
        n_steps=('timeStepIndex', 'count'),
    ).reset_index()

    # Compute metrics
    ep['cpa_real'] = ep['total_cost'] / (ep['total_reward'] + 1e-10)
    ep['penalty'] = np.where(
        ep['cpa_real'] > ep['cpa_target'],
        (ep['cpa_target'] / (ep['cpa_real'] + 1e-10)) ** 2, 1.0)
    ep['score'] = ep['penalty'] * ep['total_reward']
    ep['budget_used'] = ep['total_cost'] / ep['budget'] * 100

    n_ep = len(ep)
    n_periods = ep['deliveryPeriodIndex'].nunique()
    n_advs = ep['advertiserNumber'].nunique()

    print(f'  Episodes: {n_ep}  Periods: {n_periods}  Advertisers: {n_advs}')
    print(f'  Rows: {len(df):,}')

    # Overall metrics
    print(f'\n  {"Metric":<28} {"Mean":>10} {"Std":>10} {"Median":>10} {"Min":>10} {"Max":>10}')
    print(f'  {"-"*70}')
    for name, arr in [
        ('Score', ep['score']),
        ('Total Reward', ep['total_reward']),
        ('Total Cost', ep['total_cost']),
        ('CPA Real', ep['cpa_real']),
        ('CPA Target', ep['cpa_target']),
        ('Budget Used %', ep['budget_used']),
        ('CPA Penalty', ep['penalty']),
    ]:
        print(f'  {name:<28} {arr.mean():>10.2f} {arr.std():>10.2f} '
              f'{arr.median():>10.2f} {arr.min():>10.2f} {arr.max():>10.2f}')

    # Budget utilization bands
    print(f'\n  Budget Utilization:')
    for lo, hi in [(0, 50), (50, 80), (80, 95), (95, 99), (99, 100), (100, 105)]:
        cnt = ((ep['budget_used'] >= lo) & (ep['budget_used'] < hi)).sum()
        bar = '█' * (cnt * 40 // max(n_ep, 1))
        print(f'    [{lo:>3}%-{hi:>3}%): {cnt:>5}/{n_ep}  {bar}')

    # CPA penalty
    print(f'\n  CPA Penalty:')
    for lo, hi in [(1.0, 1.0), (0.8, 1.0), (0.5, 0.8), (0.2, 0.5), (0.0, 0.2)]:
        cnt = ((ep['penalty'] >= lo) & (ep['penalty'] <= hi)).sum()
        bar = '█' * (cnt * 40 // max(n_ep, 1))
        print(f'    [{lo:.1f}-{hi:.1f}]: {cnt:>5}/{n_ep}  {bar}')

    # Action distribution
    print(f'\n  Action Distribution:')
    acts = df['action']
    print(f'    Mean={acts.mean():.2f}  Median={acts.median():.2f}  Std={acts.std():.2f}')
    for q in [1, 5, 25, 50, 75, 95, 99]:
        print(f'    P{q:02d}: {np.percentile(acts, q):.2f}')

    # Avg per-advertiser score
    adv_score = ep.groupby('advertiserNumber')['score'].mean()
    print(f'\n  Avg Score per Advertiser: mean={adv_score.mean():.2f} std={adv_score.std():.2f} '
          f'min={adv_score.min():.2f} max={adv_score.max():.2f}')

    # Per-period score trend
    period_score = ep.groupby('deliveryPeriodIndex')['score'].mean()
    print(f'\n  Score Trend (first 10 / last 5 periods):')
    for i, (p, s) in enumerate(period_score.head(10).items()):
        print(f'    period {int(p)}: avg_score={s:.2f}')
    print(f'    ...')
    for p, s in period_score.tail(5).items():
        print(f'    period {int(p)}: avg_score={s:.2f}')

    return ep, df


ep1, df1 = analyze(F1, 'trajectory_data.csv')
ep2, df2 = analyze(F2, 'trajectory_data_1.csv')

# Head-to-head
print(f'\n{"="*70}')
print(f'HEAD-TO-HEAD')
print(f'{"="*70}')
print(f'\n  {"Metric":<28} {"traj_data":>14} {"traj_data_1":>14} {"Winner":>10}')
print(f'  {"-"*68}')
for name, v1, v2, higher_better in [
    ('Avg Score', ep1['score'].mean(), ep2['score'].mean(), True),
    ('Median Score', ep1['score'].median(), ep2['score'].median(), True),
    ('Avg CPA Penalty', ep1['penalty'].mean(), ep2['penalty'].mean(), True),
    ('Avg Budget Used %', ep1['budget_used'].mean(), ep2['budget_used'].mean(), None),
    ('Avg CPA Real', ep1['cpa_real'].mean(), ep2['cpa_real'].mean(), False),
    ('Avg Total Reward', ep1['total_reward'].mean(), ep2['total_reward'].mean(), True),
    ('Median Action', df1['action'].median(), df2['action'].median(), None),
]:
    w = ''
    if higher_better is True:
        w = 'traj_data' if v1 > v2 else 'traj_data_1'
    elif higher_better is False:
        w = 'traj_data' if v1 < v2 else 'traj_data_1'
    print(f'  {name:<28} {v1:>14.2f} {v2:>14.2f} {w:>12}')

print()
