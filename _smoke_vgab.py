"""Smoke test: run VGAB through the real benchmark episode path (small sim)."""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'strategy_train_env'))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from simul_bidding_env.Controller.Controller import Controller
from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
from benchmark_multistrat import run_one_episode, make_vgab

# Shrink sim by passing pv_num explicitly (overrides gin's 500000).
dummy = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
dummy.name += '0'
controller = Controller(player_index=0, player_agent=dummy, pv_num=2000)

res = run_one_episode(controller, 0, make_vgab, 0.001)
print('VGAB episode result:', res)
assert res['score'] >= 0, 'VGAB produced invalid score'
print('SMOKE OK')
