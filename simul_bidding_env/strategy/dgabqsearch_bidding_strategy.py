"""Q-conversion-only search over four conversion targets and twelve actions."""
import json

import numpy as np
import torch

from .dgabshare.shared_model import DGABSharedRollout


class DGABQSearchRollout(DGABSharedRollout):
    R_MULTIPLIERS = (1.0, 1.2, 1.4, 1.6)

    def __init__(self, *args, action_mean=None, action_std=None, action_upper=None,
                 noise_std=None, search_seed=None, **kwargs):
        # The existing strategy resets rollouts by calling __init__ again.
        self.action_mean = float(action_mean if action_mean is not None else getattr(self, 'action_mean', 0.0))
        self.action_std = float(action_std if action_std is not None else getattr(self, 'action_std', 1.0))
        self.action_upper = float(action_upper if action_upper is not None else getattr(self, 'action_upper', np.inf))
        self.noise_std = float(noise_std if noise_std is not None else getattr(self, 'noise_std', 0.1))
        self.search_seed = int(search_seed if search_seed is not None else getattr(self, 'search_seed', 42))
        if not np.isfinite(self.action_mean) or not np.isfinite(self.action_std) or self.action_std <= 0:
            raise ValueError('Invalid action normalization')
        if np.isnan(self.action_upper) or self.action_upper <= 0:
            raise ValueError('action_upper must be positive')
        if not np.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError('noise_std must be finite and nonnegative')
        super().__init__(*args, **kwargs)
        if self.act_dim != 1:
            raise ValueError('dgabQsearch requires a scalar bidding coefficient')
        self.rng = np.random.default_rng(self.search_seed)
        self.search_diagnostics = {}

    @torch.no_grad()
    def act(self, state):
        self.states.append(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        self.rtgs.append(self.rtg.clone())
        self.actions.append(torch.zeros(self.act_dim, device=self.device))
        self.timesteps.append(torch.tensor(self.t, dtype=torch.long, device=self.device))
        base = self.rtg.clone()
        rtg = self._pad(self.rtgs, (2,))[None].repeat(4, 1, 1)
        states = self._pad(self.states, (self.state_dim,))[None].repeat(4, 1, 1)
        actions = self._pad(self.actions, (self.act_dim,))[None].repeat(4, 1, 1)
        times = self._pad(self.timesteps, ()).long()[None].repeat(4, 1)
        mask = self._mask(len(self.rtgs))[None].repeat(4, 1)
        rtg[:, -1, 0] = base[0] * base.new_tensor(self.R_MULTIPLIERS)

        # Four actor conditions; retain their original (unclipped) actions.
        predicted = self.model(rtg, states, actions, times, mask)['action'][:, -1, 0]
        raw = predicted * self.action_std + self.action_mean
        noise = np.zeros((4, 3), dtype=np.float32)
        noise[:, 1:] = self.rng.normal(0.0, self.noise_std, size=(4, 2))
        candidates = raw[:, None] + torch.as_tensor(noise, device=self.device)
        # Additive Gaussian noise uses ORIGINAL bidding coefficient units.
        candidates = candidates.clamp(min=0, max=self.action_upper).reshape(12)
        normalized = (candidates - self.action_mean) / self.action_std

        q_rtgs = rtg.repeat_interleave(3, dim=0)
        q_actions = actions.repeat_interleave(3, dim=0)
        q_actions[:, -1, 0] = normalized
        q = self.model(q_rtgs, states.repeat_interleave(3, dim=0), q_actions,
                       times.repeat_interleave(3, dim=0), mask.repeat_interleave(3, dim=0))['q'][:, -1]
        # Cost output does not affect ranking. Preserve raw Q_v, including negatives.
        valid = torch.isfinite(q[:, 0]) & torch.isfinite(normalized)
        if not bool(valid.any()):
            raise RuntimeError('dgabQsearch has no finite action/Q_v candidates')
        best = int(q[:, 0].masked_fill(~valid, -torch.inf).argmax())
        self.rtg = q_rtgs[best, -1].clone()
        self.rtgs[-1] = self.rtg.clone()
        self.actions[-1] = normalized[best].reshape(1).clone()
        details = []
        for i in range(12):
            details.append(dict(
                r_multiplier=self.R_MULTIPLIERS[i // 3], action_variant=i % 3,
                rtg_v=float(q_rtgs[i, -1, 0]) * self.rtg_scale,
                alpha=float(candidates[i]) if bool(torch.isfinite(candidates[i])) else None,
                q_v=float(q[i, 0]) * self.rtg_scale if bool(valid[i]) else None))
        self.search_diagnostics = dict(
            inference_mode='conversion_q_search', qsearch_candidates=12,
            qsearch_valid_candidates=int(valid.sum()), qsearch_selected_index=best,
            qsearch_r_multiplier=self.R_MULTIPLIERS[best // 3],
            qsearch_action_variant=best % 3,
            qsearch_base_r_v=float(base[0]) * self.rtg_scale,
            qsearch_base_r_c=float(base[1]) * self.rtg_scale,
            qsearch_selected_r_v=float(self.rtg[0]) * self.rtg_scale,
            qsearch_selected_r_c=float(self.rtg[1]) * self.rtg_scale,
            qsearch_selected_q_v=float(q[best, 0]) * self.rtg_scale,
            qsearch_noise_std=self.noise_std,
            qsearch_candidate_details=json.dumps(details, allow_nan=False),
        )
        return self.actions[-1].cpu().numpy()


from .dgabshare_bidding_strategy import DGABShareStrategy


class DGABQSearchStrategy(DGABShareStrategy):
    def __init__(self, budget=100, cpa=2, category=1, name='dgabQsearch', model_param=None):
        params = dict(model_param or {})
        params.setdefault('v_goal_multiplier', 1.0)
        params['exploration_scale'] = 0.0
        self.qsearch_noise_std = float(params.get('qsearch_noise_std', 0.1))
        self.qsearch_seed = int(params.get('qsearch_seed', 42))
        super().__init__(budget=budget, cpa=cpa, category=category, name=name, model_param=params)
        if not isinstance(self.rollout, DGABSharedRollout):
            raise ValueError('dgabQsearch requires a shared Q/V AWBC checkpoint')

    def reset(self):
        super().reset()
        previous = self.rollout
        self.rollout = DGABQSearchRollout(
            previous.actor, V_goal=self.v_goal_multiplier * self.budget / (self.cpa + 1e-10),
            C_goal=self.budget, K=previous.K, rtg_scale=previous.rtg_scale,
            device=previous.device, action_mean=self.action_mean, action_std=self.action_std,
            action_upper=self.action_upper, noise_std=self.qsearch_noise_std,
            search_seed=self.qsearch_seed)

    def bidding(self, *args, **kwargs):
        bids = super().bidding(*args, **kwargs)
        self.last_diagnostics.update(self.rollout.search_diagnostics)
        return bids


