import json
import unittest

import torch

from simul_bidding_env.strategy.dgabqsearch_bidding_strategy import DGABQSearchRollout


class FakeModel:
    state_dim = act_dim = 1

    def __init__(self):
        self.calls = []

    def __call__(self, rtg, states, actions, times, mask):
        self.calls.append((rtg.clone(), actions.clone(), times.clone(), mask.clone()))
        # Highest conversion condition wins despite much higher predicted cost.
        q_v = rtg[..., 0:1] + actions
        return dict(action=rtg[..., 0:1] * 0 + 0.25,
                    q=torch.cat([q_v, q_v * 10000], dim=-1))


class QSearchTests(unittest.TestCase):
    def rollout(self, **kwargs):
        return DGABQSearchRollout(FakeModel(), V_goal=100, C_goal=200,
                                 rtg_scale=10, K=2, **kwargs)

    def test_searches_initial_step_and_commits_input_r(self):
        rollout = self.rollout(noise_std=0)
        rollout.act([0.])
        self.assertEqual([len(c[0]) for c in rollout.model.calls], [4, 12])
        actor_rtgs = rollout.model.calls[0][0][:, -1]
        torch.testing.assert_close(actor_rtgs[:, 0], torch.tensor([10., 12., 14., 16.]))
        torch.testing.assert_close(actor_rtgs[:, 1], torch.full((4,), 20.))
        torch.testing.assert_close(rollout.rtg, torch.tensor([16., 20.]))
        torch.testing.assert_close(rollout.rtgs[-1], rollout.rtg)
        self.assertEqual(rollout.search_diagnostics['qsearch_selected_index'], 9)
        self.assertNotEqual(rollout.search_diagnostics['qsearch_selected_q_v'], 160.)
        rollout.update_rtg(10, 30)
        torch.testing.assert_close(rollout.rtg, torch.tensor([15., 17.]))
        rollout.act([1.])
        torch.testing.assert_close(rollout.rtg, torch.tensor([24., 17.]))
        self.assertEqual(len(rollout.rtgs), 2)
        torch.testing.assert_close(rollout.rtgs[0], torch.tensor([16., 20.]))

    def test_noise_in_raw_action_units_and_original_retained(self):
        rollout = self.rollout(action_mean=5, action_std=2, noise_std=0.1, search_seed=42)
        rollout.act([0.])
        details = json.loads(rollout.search_diagnostics['qsearch_candidate_details'])
        for offset in (0, 3, 6, 9):
            self.assertAlmostEqual(details[offset]['alpha'], 5.5, places=5)
        # First standard-normal draw for default_rng(42) is 0.304717...
        self.assertAlmostEqual(details[1]['alpha'], 5.5304717, places=5)
        evaluated = rollout.model.calls[1][1][:, -1, 0]
        torch.testing.assert_close(evaluated, torch.tensor([(d['alpha'] - 5) / 2 for d in details]))

    def test_clips_before_q_evaluation(self):
        rollout = self.rollout(action_mean=5, action_upper=2, noise_std=0.1)
        rollout.act([0.])
        evaluated = rollout.model.calls[1][1][:, -1, 0]
        torch.testing.assert_close(evaluated, torch.full((12,), -3.))
        self.assertAlmostEqual(float(rollout.actions[-1]), -3.)

    def test_reset_reproduces_noise_and_clears_history(self):
        rollout = self.rollout(action_mean=5, action_std=2, noise_std=0.1, search_seed=77)
        rollout.act([0.])
        details = rollout.search_diagnostics['qsearch_candidate_details']
        rollout.update_rtg(10, 20)
        rollout.__init__(rollout.actor, V_goal=100, C_goal=200, rtg_scale=10, K=2)
        rollout.act([0.])
        self.assertEqual(details, rollout.search_diagnostics['qsearch_candidate_details'])
        self.assertEqual(len(rollout.states), 1)
        self.assertEqual(rollout.t, 0)

    def test_only_conversion_q_affects_ranking(self):
        rollout = self.rollout(noise_std=0)
        rollout.act([0.])
        self.assertEqual(rollout.search_diagnostics['qsearch_r_multiplier'], 1.6)

    def test_negative_q_values_are_ranked_without_zero_clamping(self):
        class NegativeQ(FakeModel):
            def __call__(self, *args):
                output = super().__call__(*args)
                output['q'][..., 0] -= 1000
                return output
        rollout = self.rollout(noise_std=0)
        rollout.model = rollout.actor = NegativeQ()
        rollout.act([0.])
        self.assertEqual(rollout.search_diagnostics['qsearch_r_multiplier'], 1.6)


if __name__ == '__main__':
    unittest.main()
