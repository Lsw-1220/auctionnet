"""Cost target multiplier must leave the original rollout unchanged at 1.0."""

import numpy as np
import unittest
import torch

from simul_bidding_env.strategy.dgabshare_bidding_strategy import (
    DGABShareStrategy,
)
from simul_bidding_env.strategy.dgabshare.model import DGABRollout
from simul_bidding_env.strategy.dgabshare.shared_model import DGABSharedRollout


class DummyActor:
    act_dim = 1
    state_dim = 16


class CostMultiplierTests(unittest.TestCase):
    def test_unit_multiplier_matches_original_cost_target(self):
        for rollout_class in (DGABRollout, DGABSharedRollout):
            with self.subTest(rollout_class=rollout_class.__name__):
                budget, conversion_goal, scale = 120.0, 7.0, 1000.0
                original = rollout_class(DummyActor(), conversion_goal, budget,
                                         rtg_scale=scale)
                multiplied = rollout_class(DummyActor(), conversion_goal,
                                           1.0 * budget, rtg_scale=scale)
                np.testing.assert_array_equal(original.rtg.numpy(),
                                              multiplied.rtg.numpy())
                self.assertAlmostEqual(multiplied.rtg[1].item(), budget / scale)
                original.update_rtg(2.0, 15.0)
                multiplied.update_rtg(2.0, 15.0)
                np.testing.assert_array_equal(original.rtg.numpy(),
                                              multiplied.rtg.numpy())


    def test_reset_applies_cost_multiplier_without_changing_budget(self):
        class CaptureRollout:
            actor = object()
            K = 20
            rtg_scale = 1000.0
            device = 'cpu'

            def __init__(self, actor=None, **kwargs):
                self.kwargs = kwargs

        strategy = object.__new__(DGABShareStrategy)
        strategy.budget = 100.0
        strategy.cpa = 2.0
        strategy.v_goal_multiplier = 1.0
        strategy.ctg_multiplier = 1.3
        strategy.exploration_seed = 42
        strategy.rollout = CaptureRollout()
        strategy.reset()
        self.assertEqual(strategy.remaining_budget, 100.0)
        self.assertAlmostEqual(strategy.rollout.kwargs['C_goal'], 130.0)
        self.assertAlmostEqual(strategy.rollout.kwargs['V_goal'], 50.0)

    def test_subunit_target_replenishes_from_live_remaining_budget(self):
        strategy = object.__new__(DGABShareStrategy)
        strategy.ctg_multiplier = 0.8
        strategy.remaining_budget = 40.0
        strategy.ctg_replenishment_count = 0
        strategy.rollout = type('Rollout', (), {
            'rtg': torch.tensor([0.01, 0.0]), 'rtg_scale': 1000.0})()
        strategy._replenish_ctg_if_needed()
        self.assertAlmostEqual(strategy.rollout.rtg[1].item(), 0.032)
        self.assertEqual(strategy.ctg_replenishment_count, 1)
        strategy._replenish_ctg_if_needed()
        self.assertEqual(strategy.ctg_replenishment_count, 1)

    def test_unit_target_does_not_replenish(self):
        strategy = object.__new__(DGABShareStrategy)
        strategy.ctg_multiplier = 1.0
        strategy.remaining_budget = 40.0
        strategy.ctg_replenishment_count = 0
        strategy.rollout = type('Rollout', (), {
            'rtg': torch.tensor([0.01, 0.0]), 'rtg_scale': 1000.0})()
        strategy._replenish_ctg_if_needed()
        self.assertEqual(strategy.rollout.rtg[1].item(), 0.0)
        self.assertEqual(strategy.ctg_replenishment_count, 0)

    def test_no_replenishment_after_budget_is_exhausted(self):
        strategy = object.__new__(DGABShareStrategy)
        strategy.ctg_multiplier = 0.8
        strategy.remaining_budget = 0.0
        strategy.ctg_replenishment_count = 0
        strategy.rollout = type('Rollout', (), {
            'rtg': torch.tensor([0.01, 0.0]), 'rtg_scale': 1000.0})()
        strategy._replenish_ctg_if_needed()
        self.assertEqual(strategy.rollout.rtg[1].item(), 0.0)
        self.assertEqual(strategy.ctg_replenishment_count, 0)


if __name__ == '__main__':
    unittest.main()
