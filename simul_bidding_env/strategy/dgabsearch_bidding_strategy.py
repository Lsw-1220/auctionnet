"""dgabsearch: score-selected dual-return feedback, without action search.

AuctionNet history parsing comes from DGABShareStrategy. Q predicts next RTG;
compare it against target-minus-actual using cumulative realized V/C and beta=2.
"""
import torch
from .dgabshare.model import global_score
from .dgabshare.shared_model import DGABSharedRollout
from .dgabshare_bidding_strategy import DGABShareStrategy


class DualReturnFeedbackRollout(DGABSharedRollout):
    def __init__(self, *args, cpa_target=None, **kwargs):
        if cpa_target is None:
            cpa_target = getattr(self, 'cpa_target', None)
        if cpa_target is None:
            raise ValueError('cpa_target is required')
        super().__init__(*args, **kwargs)
        self.cpa_target = float(cpa_target)
        self.past = torch.zeros(2, device=self.device)
        self.pending_q = None
        self.feedback_diagnostics = {'feedback_source': 'initial'}

    @torch.no_grad()
    def predict_next_return(self, executed_action):
        """Evaluate the actual clipped action, not the actor's zero placeholder."""
        self.actions[-1] = torch.as_tensor(
            executed_action, dtype=torch.float32, device=self.device).reshape(self.act_dim)
        output = self.model(
            self._pad(self.rtgs, (2,))[None],
            self._pad(self.states, (self.state_dim,))[None],
            self._pad(self.actions, (self.act_dim,))[None],
            self._pad(self.timesteps, ()).long()[None],
            self._mask(len(self.rtgs))[None])
        self.pending_q = output['q'][0, -1].detach().clone()

    @torch.no_grad()
    def update_rtg(self, v_t, c_t):
        actual = self.rtg.new_tensor([v_t, c_t]) / self.rtg_scale
        self.past += actual
        residual = (self.rtg - actual).clamp_min(0)
        if self.pending_q is None:
            raise RuntimeError('Missing Q prediction for the previous executed action')
        q_return = self.pending_q.clamp_min(0)

        def score(candidate):
            return global_score(self.past[0], self.past[1], candidate[0],
                                candidate[1], self.cpa_target)

        score_residual = score(residual)
        q_valid = bool(torch.isfinite(q_return).all())
        score_q = score(q_return) if q_valid else self.rtg.new_tensor(float('nan'))
        use_q = q_valid and bool(score_q > score_residual)
        self.rtg = (q_return if use_q else residual).clone()
        # Scores and returns in diagnostics use original units.
        self.feedback_diagnostics = dict(
            feedback_source='q' if use_q else 'residual',
            feedback_q_valid=q_valid,
            feedback_q_v=float(q_return[0]) * self.rtg_scale,
            feedback_q_c=float(q_return[1]) * self.rtg_scale,
            feedback_residual_v=float(residual[0]) * self.rtg_scale,
            feedback_residual_c=float(residual[1]) * self.rtg_scale,
            feedback_score_q=float(score_q) * self.rtg_scale,
            feedback_score_residual=float(score_residual) * self.rtg_scale,
            feedback_past_v=float(self.past[0]) * self.rtg_scale,
            feedback_past_c=float(self.past[1]) * self.rtg_scale,
        )
        self.pending_q = None
        self.t += 1


class DGABSearchStrategy(DGABShareStrategy):
    def __init__(self, budget=100, cpa=2, category=1,
                 name='dgabsearch', model_param=None):
        params = dict(model_param or {})
        params.setdefault('v_goal_multiplier', 1.0)
        params['exploration_scale'] = 0.0
        super().__init__(budget=budget, cpa=cpa, category=category,
                         name=name, model_param=params)
        if not isinstance(self.rollout, DGABSharedRollout):
            raise ValueError('Dual-return feedback requires a shared Q/V AWBC checkpoint')

    def reset(self):
        super().reset()
        previous = self.rollout
        self.rollout = DualReturnFeedbackRollout(
            previous.actor, V_goal=self.v_goal_multiplier * self.budget / (self.cpa + 1e-10),
            C_goal=self.budget, K=previous.K, rtg_scale=previous.rtg_scale,
            device=previous.device, cpa_target=self.cpa)

    def bidding(self, *args, **kwargs):
        bids = super().bidding(*args, **kwargs)
        action = (self.last_diagnostics['alpha'] - self.action_mean) / self.action_std
        self.rollout.predict_next_return([action])
        self.last_diagnostics.update(self.rollout.feedback_diagnostics)
        self.last_diagnostics['inference_mode'] = 'dual_return_feedback'
        return bids


