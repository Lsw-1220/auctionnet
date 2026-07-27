"""
DGAB-Fusion: sparsity-aware dual-encoder ensemble with heuristic gate.

Architecture:
  - ClsTokenObsEncoder (from R2, frozen)     → snap_dense  (512,)
  - BidFormerEncoder   (from R5, frozen)     → snap_sparse (512,)
  - HeuristicGate      (non-learnable)        → g ∈ (0,1)
  - snap_fused = g * snap_sparse + (1-g) * snap_dense
  - state = [base_state | snap_fused]          → (530,)
  - DGABCrossAttnActor (from R5, trainable)   → α, β
  - SequenceCritic     (from R5, trainable)   → V_pred, C_pred

Training: gate randomly sampled from Beta(0.5, 0.5) to expose actor to
  the full blending spectrum.
Inference: gate = sigmoid(-pv_mean_z / temperature), deterministic.
"""

import numpy as np
import torch
import torch.nn as nn

from bidding_train_env.baseline.dgab.model_po import (
    ClsTokenObsEncoder, BidFormerEncoder,
    DGABCrossAttnActor, SequenceCritic, DGABRollout,
    EPS, CPA_PENALTY_BETA,
)

RL_DIM = 2
STAT_DIM = 16
OBS_DIM = 9


# ═══════════════════════════════════════════════
# Heuristic Gate (non-learnable)
# ═══════════════════════════════════════════════

def compute_gate(base_state, stat_mean_1, stat_std_1, temperature=1.0):
    """Compute sparsity gate from base_state.

    Uses z-scored pv_mean (stat[1]):
      pv_mean_z = (raw_pv_mean - stat_mean[1]) / stat_std[1]
      g = sigmoid(-pv_mean_z / temperature)

    g → 1: pv lower than training → sparse → bias toward BidFormer
    g → 0: pv higher than training → dense → bias toward ClsToken

    Args:
        base_state: (B*T, 18) = [norm(rl) | norm(stat)]
        stat_mean_1: float, training mean of stat[1]
        stat_std_1:  float, training std of stat[1]
        temperature: float, sigmoid sharpness (smaller = harder switch)
    Returns:
        g: (B*T, 1) in (0,1)
    """
    # stat[1] is the z-scored pv_mean.
    # But base_state already has z-scored stat! stat[1] in base_state is
    # (raw_pv_mean - stat_mean[1]) / stat_std[1] — already z-scored.
    # So pv_mean_z = base_state[:, 1:2] (index RL_DIM + 1 = 2 + 1 - 1 = 2 wait)
    # base_state layout: [rl(2) | stat(16)]
    # stat[1] is base_state[:, 2 + 1] = base_state[:, 3]
    # Actually: rl has indices 0,1; stat has indices 2..17
    # stat[0] = base_state[:, 2] = log(n_imp+1)
    # stat[1] = base_state[:, 3] = pv_mean
    pv_mean_z = base_state[:, 3:4]  # already z-scored during data prep
    g = torch.sigmoid(-pv_mean_z / temperature)
    return g  # (B*T, 1)


# ═══════════════════════════════════════════════
# DGABFusion — main model
# ═══════════════════════════════════════════════

class DGABFusion(nn.Module):
    """Dual-encoder fusion model with heuristic sparsity gate.

    Constructor parallels DGAB.__init__ with additional fusion parameters.
    """

    def __init__(self,
                 base_state_dim=18,
                 act_dim=1,
                 hidden_size=512,
                 max_ep_len=96,
                 time_dim=8,
                 block_config=None,
                 tau_v=0.99, tau_c=0.05,
                 alpha=3.0,
                 lambda_critic=1.0, lambda_actor=1.0,
                 rtg_dropout=0.0,
                 learning_rate=1e-4, weight_decay=1e-4,
                 critic_type='sequence',
                 obs_dim=9, macro_dim=2,
                 device='cpu',
                 # Fusion-specific
                 gate_temperature=1.0,
                 gate_beta_alpha=0.5,
                 ):
        super().__init__()
        self.base_state_dim = base_state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len
        self.macro_dim = macro_dim
        self.obs_dim = obs_dim
        self.tau_v = tau_v
        self.tau_c = tau_c
        self.alpha = alpha
        self.lambda_critic = lambda_critic
        self.lambda_actor = lambda_actor
        self.rtg_dropout = rtg_dropout
        self.device = device

        # Gate config
        self.gate_temperature = gate_temperature
        self.gate_beta_alpha = gate_beta_alpha

        # ── Dual encoders (frozen during training) ──
        self.cls_encoder = ClsTokenObsEncoder(
            obs_dim=obs_dim, hidden_size=hidden_size, n_heads=4)
        self.bidformer_encoder = BidFormerEncoder(
            hidden_size=hidden_size, macro_dim=macro_dim, n_heads=4, n_layers=2)

        # ── Actor + Critic ──
        snap_dim = hidden_size  # 512
        state_dim = base_state_dim + snap_dim  # 530

        self.actor = DGABCrossAttnActor(
            state_dim=state_dim, act_dim=act_dim,
            hidden_size=hidden_size, max_ep_len=max_ep_len,
            time_dim=time_dim,
            n_head=block_config['n_head'],
            n_layer=block_config['n_layer'],
            block_config=block_config,
        )

        if critic_type == 'sequence':
            self.critic = SequenceCritic(
                state_dim=state_dim, act_dim=act_dim,
                hidden_size=hidden_size, max_ep_len=max_ep_len,
                time_dim=time_dim, block_config=block_config)
        else:
            from bidding_train_env.baseline.dgab.model_po import DualHeadCritic
            self.critic = DualHeadCritic(state_dim=state_dim, act_dim=act_dim)

        self.critic_type = critic_type
        self.snap_dim = snap_dim
        self.state_dim = state_dim

        # flag set by load_pretrained_encoders
        self._stat_mean_1 = 0.0
        self._stat_std_1 = 1.0

        # sentinel: DGABRollout.act() checks this to know obs encoding is needed
        self.obs_encoder = True

        self.to(device)  # move all modules to target device

        # ── Optimizer (only actor + critic params) ──
        self.optimizer = torch.optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate, weight_decay=weight_decay)

    def load_pretrained_encoders(self, r2_ckpt_path, r5_ckpt_path,
                                  stat_mean_1=0.0, stat_std_1=1.0):
        """Load frozen encoder weights from pretrained R2 and R5 checkpoints.

        Args:
            r2_ckpt_path: path to R2 complete_train.pt
            r5_ckpt_path: path to R5 complete_train.pt
            stat_mean_1: training mean of stat[1] (for heuristic gate)
            stat_std_1:  training std of stat[1]
        """
        # ── Load R2 cls_encoder ──
        r2_sd = torch.load(r2_ckpt_path, map_location=self.device)
        cls_sd = {k[len('obs_encoder.'):]: v
                  for k, v in r2_sd.items() if k.startswith('obs_encoder.')}
        self.cls_encoder.load_state_dict(cls_sd, strict=True)
        for p in self.cls_encoder.parameters():
            p.requires_grad = False
        print(f"[Fusion] Loaded cls_encoder from R2: {len(cls_sd)} params (frozen)")

        # ── Load R5 bidformer_encoder + actor + critic ──
        r5_sd = torch.load(r5_ckpt_path, map_location=self.device)
        bf_sd = {k[len('obs_encoder.'):]: v
                 for k, v in r5_sd.items() if k.startswith('obs_encoder.')}
        self.bidformer_encoder.load_state_dict(bf_sd, strict=True)
        for p in self.bidformer_encoder.parameters():
            p.requires_grad = False
        print(f"[Fusion] Loaded bidformer_encoder from R5: {len(bf_sd)} params (frozen)")

        actor_sd = {k[len('actor.'):]: v
                    for k, v in r5_sd.items() if k.startswith('actor.')}
        self.actor.load_state_dict(actor_sd, strict=True)
        print(f"[Fusion] Loaded actor from R5: {len(actor_sd)} params (trainable)")

        critic_sd = {k[len('critic.'):]: v
                     for k, v in r5_sd.items() if k.startswith('critic.')}
        self.critic.load_state_dict(critic_sd, strict=True)
        print(f"[Fusion] Loaded critic from R5: {len(critic_sd)} params (trainable)")

        # Gate parameters
        self._stat_mean_1 = stat_mean_1
        self._stat_std_1 = stat_std_1
        print(f"[Fusion] Gate config: stat_mean[1]={stat_mean_1:.6f}, "
              f"stat_std[1]={stat_std_1:.6f}, "
              f"temperature={self.gate_temperature}, "
              f"beta_alpha={self.gate_beta_alpha}")

    def _encode_obs(self, obs_padded, obs_mask, base_states, training=False):
        """Dual-path obs encoding with gate-controlled fusion.

        Args:
            obs_padded:  (B, T, M, obs_dim)
            obs_mask:    (B, T, M)
            base_states: (B, T, base_state_dim) = [norm(rl) | norm(stat)]
        Returns:
            state_aug:   (B, T, state_dim) = [base_states | snap_fused]
        """
        B, T = obs_padded.shape[:2]
        M = obs_padded.shape[2]
        BxT = B * T

        obs_flat = obs_padded.reshape(BxT, M, self.obs_dim)
        mask_flat = obs_mask.reshape(BxT, M)
        base_flat = base_states.reshape(BxT, self.base_state_dim)

        # Path 1: Dense (cls-token), no_grad during training
        with torch.set_grad_enabled(not training):
            snap_dense = self.cls_encoder(obs_flat, mask_flat)

        # Path 2: Sparse (BidFormer), no_grad during training
        macro = base_flat[:, :self.macro_dim]
        with torch.set_grad_enabled(not training):
            snap_sparse = self.bidformer_encoder(obs_flat, mask_flat, macro)

        # ── Gate ──
        if training:
            # Random gate from Beta distribution → actor sees full spectrum
            g = torch.distributions.Beta(
                self.gate_beta_alpha, self.gate_beta_alpha
            ).sample((BxT, 1)).to(device=obs_flat.device, dtype=obs_flat.dtype)
        else:
            # Deterministic heuristic gate
            g = compute_gate(base_flat, self._stat_mean_1, self._stat_std_1,
                             self.gate_temperature)

        # Soft blending
        snap_fused = g * snap_sparse + (1 - g) * snap_dense  # (BxT, hidden_size)

        # Reshape and concat
        snap_fused = snap_fused.reshape(B, T, self.snap_dim)
        return torch.cat([base_states, snap_fused], dim=-1)

    def step(self, batch):
        """Single training step. Identical interface to DGAB.step()."""
        # ── Encode obs ──
        if self.cls_encoder is not None and 'obs_padded' in batch:
            states = self._encode_obs(
                batch['obs_padded'], batch['obs_mask'],
                batch['states'],  # (B, T, base_state_dim)
                training=True)
        else:
            states = batch['states']

        rtg = batch['rtg']             # (B, T, 2)
        actions = batch['actions']     # (B, T, 1)
        timesteps = batch['timesteps'] # (B, T)
        mask = batch['mask']           # (B, T)
        past_v = batch['past_v']       # (B, T, 1)
        past_c = batch['past_c']       # (B, T, 1)
        rtg_v = batch['rtg_v']         # (B, T, 1)
        rtg_c = batch['rtg_c']         # (B, T, 1)
        cpa_target = batch['cpa_target']  # (B, T, 1)

        # ── Actor forward ──
        action_pred, beta_pred = self.actor(rtg, states, actions, timesteps, mask)
        explore_action = beta_pred * actions

        # ── Critic loss ──
        loss_critic, critic_metrics = self._critic_loss(
            rtg, states, actions, explore_action, timesteps, mask,
            rtg_v, rtg_c)

        # ── Advantage estimation (no grad) ──
        with torch.no_grad():
            v_orig, c_orig = self._critic_predict(
                rtg, states, actions, timesteps, mask)
            v_expl, c_expl = self._critic_predict(
                rtg, states, explore_action.detach(), timesteps, mask)

        # ── Global scores ──
        s_orig = self._global_score(past_v, v_orig, past_c, c_orig, cpa_target)
        s_expl = self._global_score(past_v, v_expl, past_c, c_expl, cpa_target)

        # ── Actor loss ──
        loss_actor, actor_metrics = self._actor_loss(
            action_pred, actions, explore_action, s_orig, s_expl, mask)

        # ── Total loss ──
        loss_total = self.lambda_critic * loss_critic + self.lambda_actor * loss_actor

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)
        self.optimizer.step()

        return {**critic_metrics, **actor_metrics,
                'loss_total': loss_total.item(),
                'loss_critic': loss_critic.item(),
                'loss_actor': loss_actor.item(),
                's_orig_mean': s_orig.mean().item(),
                's_expl_mean': s_expl.mean().item()}

    # ── Loss helpers (copied and simplified from DGAB) ──

    def _critic_loss(self, rtg, states, actions, explore_action,
                     timesteps, mask, rtg_v, rtg_c):
        """Dual-path critic loss: MSE on original + expectile on explore."""
        v_orig, c_orig = self.critic(rtg, states, actions, timesteps, mask)
        v_expl, c_expl = self.critic(
            rtg, states, explore_action.detach(), timesteps, mask)

        # MSE on original actions
        v_std = rtg_v[mask.bool()].std().clamp_min(EPS)
        c_std = rtg_c[mask.bool()].std().clamp_min(EPS)
        v_orig = v_orig.squeeze(-1)
        c_orig = c_orig.squeeze(-1)
        loss_v_mse = ((v_orig - rtg_v.squeeze(-1)) ** 2 * mask).sum() / (
            mask.sum() * v_std + EPS)
        loss_c_mse = ((c_orig - rtg_c.squeeze(-1)) ** 2 * mask).sum() / (
            mask.sum() * c_std + EPS)

        # Expectile regression on explore actions
        v_expl = v_expl.squeeze(-1)
        c_expl = c_expl.squeeze(-1)
        diff_v = rtg_v.squeeze(-1) - v_expl
        diff_c = rtg_c.squeeze(-1) - c_expl
        tau_v, tau_c = self.tau_v, self.tau_c
        loss_v_exp = (
            (torch.where(diff_v >= 0, tau_v, 1 - tau_v) * diff_v ** 2 * mask).sum()
            / (mask.sum() * v_std + EPS))
        loss_c_exp = (
            (torch.where(diff_c >= 0, tau_c, 1 - tau_c) * diff_c ** 2 * mask).sum()
            / (mask.sum() * c_std + EPS))

        loss = (loss_v_mse + loss_c_mse + loss_v_exp + loss_c_exp) * 0.25
        return loss, {
            'loss_v_mse': loss_v_mse.item(), 'loss_c_mse': loss_c_mse.item(),
            'loss_v_exp': loss_v_exp.item(), 'loss_c_exp': loss_c_exp.item(),
        }

    def _critic_predict(self, rtg, states, actions, timesteps, mask):
        v, c = self.critic(rtg, states, actions, timesteps, mask)
        return v.squeeze(-1), c.squeeze(-1)

    def _global_score(self, past_v, v_pred, past_c, c_pred, cpa_target):
        v_total = (past_v.squeeze(-1) + v_pred).clamp_min(0)
        c_total = (past_c.squeeze(-1) + c_pred).clamp_min(EPS)
        cpa = c_total / (v_total + EPS)
        penalty = (cpa_target.squeeze(-1) / (cpa + EPS)).clamp_max(1.0) ** CPA_PENALTY_BETA
        return v_total * penalty

    def _actor_loss(self, action_pred, actions, explore_action,
                    s_orig, s_expl, mask):
        # BC loss
        loss_bc = ((action_pred - actions) ** 2 * mask.unsqueeze(-1)).sum() / (
            mask.sum() * actions.std().clamp_min(EPS) + EPS)

        # Advantage weight
        adv = s_expl - s_orig
        adv_std = adv[mask.bool()].std().clamp_min(EPS)
        adv_normed = adv / adv_std
        w = torch.sigmoid(self.alpha * adv_normed).detach()

        # Explore imitation loss
        loss_expl = ((action_pred - explore_action.detach()) ** 2
                     * mask.unsqueeze(-1) * w.unsqueeze(-1)).sum() / (
            mask.sum() * EPS + EPS)

        loss = loss_bc + loss_expl
        return loss, {
            'w_mean': w[mask.bool()].mean().item(),
            'w_std': w[mask.bool()].std().item(),
            'adv_std': adv_std.item(),
            'beta_mean': 0.0,  # placeholder — beta_pred not tracked here
        }


# ═══════════════════════════════════════════════
# DGABFusionRollout — inference wrapper
# ═══════════════════════════════════════════════

class DGABFusionRollout(DGABRollout):
    """Identical to DGABRollout but uses DGABFusion._encode_obs internally.

    The parent class DGABRollout already calls model._encode_obs(), so as long
    as the model is a DGABFusion instance, the dual-path fusion path is used.
    """
    pass
