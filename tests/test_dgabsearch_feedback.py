"""Timing, score selection, and executed-action regression checks."""
import unittest
import torch

from simul_bidding_env.strategy.dgabsearch_bidding_strategy import DualReturnFeedbackRollout


class FakeModel:
    state_dim = act_dim = 1

    def __call__(self, rtg, states, actions, times, mask):
        # Q changes with the action token, exposing placeholder mistakes.
        return {'action': torch.ones_like(actions),
                'q': torch.cat([actions * 100, actions * 50], dim=-1)}


class FeedbackTests(unittest.TestCase):
    def rollout(self):
        return DualReturnFeedbackRollout(FakeModel(), V_goal=100, C_goal=100,
                                         rtg_scale=10, K=2, cpa_target=1)

    def test_q_wins_without_subtracting_actual_twice(self):
        rollout = self.rollout()
        rollout.pending_q = torch.tensor([11., 8.])
        rollout.update_rtg(10, 20)
        torch.testing.assert_close(rollout.rtg, torch.tensor([11., 8.]))
        torch.testing.assert_close(rollout.past, torch.tensor([1., 2.]))
        self.assertEqual(rollout.feedback_diagnostics['feedback_source'], 'q')

    def test_cpa_penalty_keeps_residual(self):
        rollout = self.rollout()
        rollout.pending_q = torch.tensor([12., 30.])
        rollout.update_rtg(10, 20)
        torch.testing.assert_close(rollout.rtg, torch.tensor([9., 8.]))
        self.assertEqual(rollout.feedback_diagnostics['feedback_source'], 'residual')

    def test_tie_and_nonfinite_keep_residual(self):
        for candidate in ([9., 8.], [float('nan'), 8.]):
            rollout = self.rollout()
            rollout.pending_q = torch.tensor(candidate)
            rollout.update_rtg(10, 20)
            torch.testing.assert_close(rollout.rtg, torch.tensor([9., 8.]))

    def test_q_uses_executed_action_and_keeps_history(self):
        rollout = self.rollout()
        rollout.act([0.])
        rollout.predict_next_return([0.2])
        torch.testing.assert_close(rollout.pending_q, torch.tensor([20., 10.]))
        torch.testing.assert_close(rollout.actions[-1], torch.tensor([0.2]))
        rollout.update_rtg(10, 20)
        rollout.act([1.])
        self.assertEqual(rollout.t, 1)
        torch.testing.assert_close(rollout.rtgs[-1], torch.tensor([20., 10.]))

    def test_episode_reset_clears_feedback(self):
        rollout = self.rollout()
        rollout.pending_q = torch.tensor([11., 8.])
        rollout.update_rtg(10, 20)
        rollout.__init__(rollout.actor, V_goal=100, C_goal=100,
                         rtg_scale=10, K=2)
        self.assertEqual(rollout.cpa_target, 1.)
        self.assertIsNone(rollout.pending_q)
        self.assertEqual(rollout.t, 0)
        torch.testing.assert_close(rollout.past, torch.zeros(2))


if __name__ == '__main__':
    unittest.main()
