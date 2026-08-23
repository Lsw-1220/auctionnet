"""
DGAB (fully-observable) — Decision-Transformer actor/critic with dual (V, C) RTG.

Two-stage training (replaces the old single-optimizer joint step()):

  Stage 1 — DGABCritic.step(batch)
    Trained ONLY on real (s, a) transitions from the offline dataset.
    Two independent heads per channel (V, C):
      * _mse  head: plain MSE, unbiased point estimate of the real transition.
      * _opt  head: expectile regression (tau_v upper / tau_c lower), an
        optimistic-but-data-calibrated anchor for the real transition.
    No counterfactual/explore action is ever seen during critic training,
    so the critic cannot be pulled toward "wanting" high-value explore
    actions the way the old joint expectile-on-explore-action loss did.

  Stage 2 — DGABActorTrainer.step(batch), critic frozen (requires_grad_(False), eval())
    * action_pred is trained with advantage-weighted MSE behavior cloning.
      The advantage is computed from the frozen critic as S(Q(s,a))-S(V(s)),
      so the critic supplies sample weights but receives no actor gradients.

Only the actor is needed at inference time (see ActorRollout) — the critic
is a training-time-only value estimator.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block

EPS = 1e-8
LOSS_SCALE_FLOOR = 1e-3
CPA_PENALTY_BETA = 2.0   # exponent in the global score's CPA penalty


def global_score(past_v, past_c, v_remain, c_remain, cpa_target):
    """All inputs [B, T, 1] (or broadcastable). Returns S [*, 1].

    S = V_total * penalty, where penalty <= 1 shrinks the score once the
    achieved CPA exceeds cpa_target (soft constraint).
    """
    V_total = past_v + v_remain
    C_total = past_c + c_remain
    cpa_total = C_total / (V_total + EPS)
    penalty = torch.clamp((cpa_target / (cpa_total + EPS)) ** CPA_PENALTY_BETA, max=1.0)
    return V_total * penalty


# =====================================================================
# Dataset annotation — dual RTG + past cumulative V/C
# =====================================================================
def annotate_trajectory_with_dual_rtg(traj, sparse_v_credit=0.0):
    """
    在 EpisodeReplayBuffer 加工后的 trajectory dict 上原地添加 5 个数组:
        rtg_v[t]      = sum_{k>=t} v_k    本步及以后的剩余转化
        rtg_c[t]      = sum_{k>=t} c_k    本步及以后的剩余花费
        past_v[t]     = sum_{k<t}  v_k    本步之前的累计转化
        past_c[t]     = sum_{k<t}  c_k    本步之前的累计花费
        cpa_target[t] = traj['cpacons'][t]  逐步的 CPA 约束(原始量纲)

    依赖字段:
        traj['rewards']      [T, 1] -> 每步真实转化数 v_t
        traj['resourceleft'] [T, 2] -> 第 1 列 bgtleft, 用于反推 c_t
        traj['budget']       [T, 1] -> 反推 c_t 用的总预算
        traj['cpacons']      [T, 1] -> 每条 episode 的 CPA 约束

    Args:
        sparse_v_credit: float, default 0.0 (disabled).
            When > 0, adds a pseudo-V decrement for timesteps where cost was
            incurred but no conversion occurred, preventing RTG_v stagnation
            (token collapse) in sparse-reward environments.
            pseudo_v_t = c_t / cpa_target * sparse_v_credit for v_t==0 & c_t>0.
    """
    v = np.asarray(traj['rewards'], dtype=np.float32).reshape(-1)        # [T]
    bgtleft = np.asarray(traj['resourceleft'], dtype=np.float32)[:, 1]   # [T] step t 开始前的剩余预算比例
    budget = float(np.asarray(traj['budget']).reshape(-1)[0])
    realcost = float(np.asarray(traj['realcost']).reshape(-1)[0])
    # bgtleft[t] = step t 之前的剩余预算 → budget*(1-bgtleft[t]) = step t 之前的累计花费
    cum_cost = budget * (1.0 - bgtleft)                                  # [T] Σ_{k<t} c_k
    # 每步花费 c_t = cum_cost[t+1] − cum_cost[t]，cum_cost[T] = realcost（episode 结束累计花费）
    c = np.diff(np.concatenate([cum_cost, [realcost]])).astype(np.float32)   # [T] c_t

    past_v = np.concatenate([[0.0], np.cumsum(v)[:-1]]).astype(np.float32)
    # past_c_1 = Σ_{k<t} c_k = cum_cost（不能用 cumsum(c)，那样 past_c 会错位一格）
    past_c_1 = cum_cost.astype(np.float32)   # 第 t 步之前的花费
    #past_c_2 = budget * (1.0 - bgtleft).astype(np.float32) # alternative, same as above
    if sparse_v_credit > 0.0:
        cpa_target_arr = np.asarray(traj['cpacons'], dtype=np.float32).reshape(-1)
        pseudo_v = np.where(
            (v == 0.0) & (c > 0.0),
            c / (cpa_target_arr + 1e-8) * sparse_v_credit,
            0.0
        ).astype(np.float32)
        v_effective = v + pseudo_v
    else:
        v_effective = v

    rtg_v = (v_effective.sum() - np.concatenate([[0.0], np.cumsum(v_effective)[:-1]])).astype(np.float32)
    #rtg_c = (c.sum() - past_c).astype(np.float32)
    rtg_c = traj['realcost'] - past_c_1
    traj['rtg_v']      = rtg_v
    traj['rtg_c']      = rtg_c
    traj['past_v']     = past_v
    traj['past_c']     = past_c_1
    traj['costs']      = c.astype(np.float32)   # 每步花费 c_t，IQL 的 Q TD target 需要
    traj['cpa_target'] = np.asarray(traj['cpacons'], dtype=np.float32).reshape(-1)
    return traj


# =====================================================================
# Actor — Decision Transformer with 2D RTG input, 3-token/step stacking
#   token order per step: [rtg, state, action]
#   predicts a deterministic action from the state-token position
# =====================================================================
class DGABActor(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size=64, max_ep_len=96,
                 time_dim=8, block_config=None):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size

        self.embed_state  = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_rtg    = nn.Linear(2, hidden_size)            # dual RTG
        self.embed_time   = nn.Embedding(max_ep_len, time_dim)

        self.trans_state  = nn.Linear(hidden_size + time_dim, hidden_size)
        self.trans_action = nn.Linear(hidden_size + time_dim, hidden_size)
        self.trans_rtg    = nn.Linear(hidden_size + time_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.transformer = nn.ModuleList([Block(block_config)
                                          for _ in range(block_config['n_layer'])])

        self.predict_action = nn.Linear(hidden_size, act_dim)
        self.predict_rtg = nn.Linear(hidden_size, 2)  # dual RTG prediction head
    def forward(self, rtg, states, actions, timesteps, attention_mask=None):
        """
        rtg:        [B, T, 2]
        states:     [B, T, state_dim]
        actions:    [B, T, act_dim]
        timesteps:  [B, T]
        attention_mask: [B, T] (1 = valid)
        Returns: action_pred [B,T,act_dim]
        """
        B, T = states.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.long, device=states.device)

        time_emb   = self.embed_time(timesteps)
        rtg_emb    = self.trans_rtg(   torch.cat([self.embed_rtg(rtg),       time_emb], dim=-1))
        state_emb  = self.trans_state( torch.cat([self.embed_state(states),  time_emb], dim=-1))
        action_emb = self.trans_action(torch.cat([self.embed_action(actions),time_emb], dim=-1))

        stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=1)             # [B,3,T,H]
        stacked = stacked.permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_size)
        stacked = self.embed_ln(stacked)

        stacked_mask = (attention_mask.unsqueeze(1).expand(-1, 3, -1)
                        .reshape(B, 3 * T).to(stacked.dtype))

        x = stacked
        for block in self.transformer:
            x = block(x, stacked_mask)
        x = x.reshape(B, T, 3, self.hidden_size).permute(0, 2, 1, 3)               # [B,3,T,H]

        state_token = x[:, 1]                   
        action_token = x[:, 2]                                    # [B,T,H]
        action_pred = self.predict_action(state_token)                             # [B,T,act_dim]
        # Legacy deterministic actor API. The new shared experiment uses
        # DGABSharedModel in shared_model.py for action-token auxiliary heads.
        return action_pred


# =====================================================================
# Critic — (s, a, r) autoregressive, trained ONLY on real transitions.
#   Mirrors the Actor's (r, s, a) token order: a_t position sees s0,a0,r0,
#   ..., s_t, a_t but NOT r_t (causal mask) -> predicts r_t = (V_remain, C_remain).
#
#   Two heads per channel:
#     head_v_mse / head_c_mse — plain MSE, unbiased point estimate.
#       Used (a) as the critic's evaluation function for actor-loss advantage,
#       and (b) as the differentiable scorer for beta's counterfactual S(beta*a).
#     head_v_opt / head_c_opt — expectile regression (tau_v upper / tau_c
#       lower bound), trained ONLY on the real action. Used as beta's
#       regression target (data-calibrated optimistic anchor), never as a
#       gradient path back into the actor.
# =====================================================================
# =====================================================================
# Critic — (s, a, r) autoregressive, trained ONLY on real transitions.
#   Mirrors the Actor's (r, s, a) token order: a_t position sees s0,a0,r0,
#   ..., s_t, a_t but NOT r_t (causal mask) -> predicts r_t = (V_remain).
#
#   Two heads per critic:
#     head_v_mse / head_c_mse — plain MSE, unbiased point estimate.
#       Used as the critic's evaluation function for actor-loss advantage,
#       and as the differentiable scorer for beta's counterfactual S(beta*a).
#     head_v_opt / head_c_opt — expectile regression (tau_v upper / tau_c
#       lower bound), trained ONLY on the real action. Used as beta's
#       regression target (data-calibrated optimistic anchor).
# =====================================================================
class DGABCriticV(nn.Module):
    """V-channel critic predicting remaining conversions."""
    def __init__(self, state_dim, act_dim, hidden_size=64, max_ep_len=96,
                 time_dim=8, block_config=None, tau_v=0.99, tau_c=0.01,
                 learning_rate=1e-4, weight_decay=1e-4, device='cpu',
                 v_loss_scale=1.0, c_loss_scale=1.0, grad_clip=5.0):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.tau_v = tau_v
        self.tau_c = tau_c
        self.device = device
        self.v_loss_scale = float(v_loss_scale)
        self.c_loss_scale = float(c_loss_scale)
        self.grad_clip = float(grad_clip)
        self.register_buffer('v_target_scale', torch.tensor(float(v_loss_scale)), persistent=False)
        self.register_buffer('c_target_scale', torch.tensor(float(c_loss_scale)), persistent=False)

        self.embed_state  = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_rtg    = nn.Linear(2, hidden_size)
        self.embed_time   = nn.Embedding(max_ep_len, time_dim)

        self.trans_state  = nn.Linear(hidden_size + time_dim, hidden_size)
        self.trans_action = nn.Linear(hidden_size + time_dim, hidden_size)
        self.trans_rtg    = nn.Linear(hidden_size + time_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.transformer = nn.ModuleList([Block(block_config)
                                          for _ in range(block_config['n_layer'])])

        # self.head_v_mse = nn.Linear(hidden_size, 1)
        # self.head_v_opt = nn.Linear(hidden_size, 1)
        self.head_v = nn.Linear(hidden_size, 1)
        self.head_c = nn.Linear(hidden_size, 1)
        for head in (self.head_v, self.head_c):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)

        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.to(device)

    def forward(self, rtg, states, actions, timesteps, attention_mask=None):
        """
        V 网络 = 状态值 V(s)。清零所有 action token，使输出不条件化当前动作 a_t。
        Returns: v (剩余转化 expectile), c (剩余花费 expectile) — each [B, T, 1]
        """
        B, T = states.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.long, device=states.device)

        # IQL：V(s) 是状态值，不能看到当前动作 → 把 action 全部清零（信息被抹掉）
        actions = torch.zeros_like(actions)

        time_emb   = self.embed_time(timesteps)
        state_emb  = self.trans_state( torch.cat([self.embed_state(states),  time_emb], dim=-1))
        action_emb = self.trans_action(torch.cat([self.embed_action(actions), time_emb], dim=-1))
        rtg_emb    = self.trans_rtg(   torch.cat([self.embed_rtg(rtg),       time_emb], dim=-1))

        stacked = torch.stack([state_emb, action_emb, rtg_emb], dim=1)
        stacked = stacked.permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_size)
        stacked = self.embed_ln(stacked)

        stacked_mask = (attention_mask.unsqueeze(1).expand(-1, 3, -1)
                        .reshape(B, 3 * T).to(stacked.dtype))

        x = stacked
        for block in self.transformer:
            x = block(x, stacked_mask)
        x = x.reshape(B, T, 3, self.hidden_size).permute(0, 2, 1, 3)

        h = x[:, 1]
        # v_mse = self.head_v_mse(h) * self.v_target_scale
        # v_opt = self.head_v_opt(h) * self.v_target_scale
        # return v_mse, v_opt
        v = self.head_v(h) * self.v_target_scale
        c = self.head_c(h) * self.c_target_scale
        return v,c
    # ── 旧版 step：回归 rtg_v/rtg_c 的 expectile。
    #    已废弃——IQL 更新现在由 DGABCritic.step 统一编排
    #    （V 网络直接对观测 RTG 做 expectile 回归）。
    # def step(self, batch):
    #     """Train V-channel on real transitions."""
    #     m = batch['mask'] > 0
    #     v_real = batch['rtg_v']
    #     v_real_m = v_real[m]
    #     c_real = batch['rtg_c']
    #     c_real_m = c_real[m]
    #
    #     v_scale = v_real_m.new_tensor(self.v_loss_scale)
    #     c_scale = c_real_m.new_tensor(self.c_loss_scale)
    #     v_opt, c_opt = self.forward(
    #         batch['rtg'], batch['states'], batch['actions'],
    #         batch['timesteps'], batch['mask'])
    #     v_opt_m = v_opt[m]
    #     c_opt_m = c_opt[m]
    #
    #     diff_v = (v_real_m - v_opt_m) / v_scale
    #     w_v = torch.where(diff_v > 0,
    #                       torch.full_like(diff_v, self.tau_v),
    #                       torch.full_like(diff_v, 1.0 - self.tau_v))
    #     loss_v_opt = (w_v * diff_v.pow(2)).mean()
    #
    #     diff_c = (c_real_m - c_opt_m) / c_scale
    #     w_c = torch.where(diff_c > 0,
    #                       torch.full_like(diff_c, self.tau_c),
    #                       torch.full_like(diff_c, 1.0 - self.tau_c))
    #     loss_c_opt = (w_c * diff_c.pow(2)).mean()
    #     loss_total = loss_v_opt + loss_c_opt
    #     self.optimizer.zero_grad()
    #     loss_total.backward()
    #     grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
    #     self.optimizer.step()
    #
    #     return {
    #         'loss_total':  loss_total.item(),
    #         'loss_v_opt':  loss_v_opt.item(),
    #         'loss_c_opt':  loss_c_opt.item(),
    #         'v_scale': v_scale.item(),
    #         'v_target_mean': v_real_m.mean().item(),
    #         'v_pred_mean': v_opt_m.mean().item(),
    #         'c_scale': c_scale.item(),
    #         'c_target_mean': c_real_m.mean().item(),
    #         'c_pred_mean': c_opt_m.mean().item(),
    #         'grad_norm': float(grad_norm),
    #     }


class DGABCriticC(nn.Module):
    """Q 网络：看到 action 的 Q(s,a)，TD 更新（target = r_t + γ·V(s_{t+1})）。"""
    def __init__(self, state_dim, act_dim, hidden_size=64, max_ep_len=96,
                 time_dim=8, block_config=None, tau_c=0.05,
                 learning_rate=1e-4, weight_decay=1e-4, device='cpu',
                 v_loss_scale=1.0, c_loss_scale=1.0, grad_clip=5.0):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.tau_c = tau_c
        self.device = device
        self.c_loss_scale = float(c_loss_scale)
        self.v_loss_scale = float(v_loss_scale)
        self.grad_clip = float(grad_clip)
        self.register_buffer('c_target_scale', torch.tensor(float(c_loss_scale)), persistent=False)
        self.register_buffer('v_target_scale', torch.tensor(float(v_loss_scale)), persistent=False)
        self.embed_state  = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_rtg    = nn.Linear(2, hidden_size)
        self.embed_time   = nn.Embedding(max_ep_len, time_dim)

        self.trans_state  = nn.Linear(hidden_size + time_dim, hidden_size)
        self.trans_action = nn.Linear(hidden_size + time_dim, hidden_size)
        self.trans_rtg    = nn.Linear(hidden_size + time_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.transformer = nn.ModuleList([Block(block_config)
                                          for _ in range(block_config['n_layer'])])

        self.head_q_v = nn.Linear(hidden_size, 1)   # Q 转化值
        self.head_q_c = nn.Linear(hidden_size, 1)   # Q 花费值
        for head in (self.head_q_v, self.head_q_c):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)

        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.to(device)

    def forward(self, rtg, states, actions, timesteps, attention_mask=None):
        """
        Q 网络 = Q(s,a)，看到真实 action。
        Returns: q_v (转化值), q_c (花费值) — each [B, T, 1]
        """
        B, T = states.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.long, device=states.device)

        time_emb   = self.embed_time(timesteps)
        state_emb  = self.trans_state( torch.cat([self.embed_state(states),  time_emb], dim=-1))
        action_emb = self.trans_action(torch.cat([self.embed_action(actions), time_emb], dim=-1))
        rtg_emb    = self.trans_rtg(   torch.cat([self.embed_rtg(rtg),       time_emb], dim=-1))

        stacked = torch.stack([state_emb, action_emb, rtg_emb], dim=1)
        stacked = stacked.permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_size)
        stacked = self.embed_ln(stacked)

        stacked_mask = (attention_mask.unsqueeze(1).expand(-1, 3, -1)
                        .reshape(B, 3 * T).to(stacked.dtype))

        x = stacked
        for block in self.transformer:
            x = block(x, stacked_mask)
        x = x.reshape(B, T, 3, self.hidden_size).permute(0, 2, 1, 3)

        h = x[:, 1]
        q_v = self.head_q_v(h) * self.v_target_scale
        q_c = self.head_q_c(h) * self.c_target_scale
        return q_v, q_c

    # ── 旧版 step：直接回归 rtg_v/rtg_c 的 MSE。
    #    已废弃——IQL 更新现在由 DGABCritic.step 统一编排
    #    （Q 网络 TD target = r_t + γ·V(s_{t+1})，需要每步 reward）。
    # def step(self, batch):
    #     """Train C-channel on real transitions."""
    #     m = batch['mask'] > 0
    #     v_real = batch['rtg_v']
    #     c_real = batch['rtg_c']
    #     v_real_m = v_real[m]
    #     c_real_m = c_real[m]
    #     v_scale = v_real_m.new_tensor(self.v_loss_scale)
    #     c_scale = c_real_m.new_tensor(self.c_loss_scale)
    #
    #     v_mse, c_mse = self.forward(
    #         batch['rtg'], batch['states'], batch['actions'],
    #         batch['timesteps'], batch['mask'])
    #     v_mse_m = v_mse[m]
    #     c_mse_m = c_mse[m]
    #
    #     loss_v_mse = ((v_real_m - v_mse_m) / v_scale).pow(2).mean()
    #     loss_c_mse = ((c_real_m - c_mse_m) / c_scale).pow(2).mean()
    #
    #     loss_total = loss_v_mse + loss_c_mse
    #
    #     self.optimizer.zero_grad()
    #     loss_total.backward()
    #     grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
    #     self.optimizer.step()
    #
    #     return {
    #         'loss_total':  loss_total.item(),
    #         'loss_v_mse':  loss_v_mse.item(),
    #         'loss_c_mse':  loss_c_mse.item(),
    #         'v_scale': v_scale.item(),
    #         'c_scale': c_scale.item(),
    #         'v_target_mean': v_real_m.mean().item(),
    #         'c_target_mean': c_real_m.mean().item(),
    #         'v_pred_mean': v_mse_m.mean().item(),
    #         'c_pred_mean': c_mse_m.mean().item(),
    #         'grad_norm': float(grad_norm),
    #     }


class DGABCritic(nn.Module):
    """双网络 critic（Q 为 MSE，非 IQL TD）：
      critic_v = V 网络（状态值，action 内部清零），τ-expectile 直接回归观测 RTG
      critic_q = Q 网络（看到 action），MSE 回归向观测 RTG（无自举、无 γ）
    """
    def __init__(self, state_dim, act_dim, hidden_size=64, max_ep_len=96,
                 time_dim=8, block_config=None, tau_v=0.99, tau_c=0.05,
                 learning_rate=1e-4, weight_decay=1e-4, device='cpu',
                 v_loss_scale=1.0, c_loss_scale=1.0, grad_clip=5.0,
                 q_huber_beta=1.0):
        super().__init__()
        self.tau_v = float(tau_v)
        self.tau_c = float(tau_c)
        self.v_loss_scale = float(v_loss_scale)
        self.c_loss_scale = float(c_loss_scale)
        if q_huber_beta <= 0:
            raise ValueError("q_huber_beta must be positive")
        self.q_huber_beta = float(q_huber_beta)
        self.critic_v = DGABCriticV(
            state_dim=state_dim, act_dim=act_dim, hidden_size=hidden_size,
            max_ep_len=max_ep_len, time_dim=time_dim, block_config=block_config,
            tau_v=tau_v, tau_c=tau_c, learning_rate=learning_rate, weight_decay=weight_decay,
            device=device, v_loss_scale=v_loss_scale, c_loss_scale=c_loss_scale, grad_clip=grad_clip)
        self.critic_q = DGABCriticC(
            state_dim=state_dim, act_dim=act_dim, hidden_size=hidden_size,
            max_ep_len=max_ep_len, time_dim=time_dim, block_config=block_config,
            tau_c=tau_c, learning_rate=learning_rate, weight_decay=weight_decay,
            device=device, v_loss_scale=v_loss_scale, c_loss_scale=c_loss_scale, grad_clip=grad_clip)

    def forward(self, rtg, states, actions, timesteps, attention_mask=None):
        """Returns: (v_v, v_c) = V 状态值, (q_v, q_c) = Q(s,a) — each [B, T, 1]"""
        v_v, v_c = self.critic_v(rtg, states, actions, timesteps, attention_mask)
        q_v, q_c = self.critic_q(rtg, states, actions, timesteps, attention_mask)
        return v_v, v_c, q_v, q_c

    def _legacy_step_unused(self, batch):
        """简化的 critic 更新（Q 不再用 IQL TD 公式）：

        Q 网络:  MSE 回归向观测 RTG（rtg_v / rtg_c）→ 无自举、无 γ，天然稳定且非负
        V 网络:  对观测 RTG 做 τ-expectile 回归 → 仍作 actor 优势的乐观基线
        """
        m = batch['mask'] > 0

        # ── 前向：Q 用真实 action；V 内部清零 action（状态值）──
        q_v, q_c = self.critic_q(batch['rtg'], batch['states'], batch['actions'],
                                 batch['timesteps'], batch['mask'])
        v_v, v_c = self.critic_v(batch['rtg'], batch['states'], batch['actions'],
                                 batch['timesteps'], batch['mask'])

        v_scale = batch['states'].new_tensor(self.v_loss_scale)
        c_scale = batch['states'].new_tensor(self.c_loss_scale)

        # ── Q 更新：MSE 回归向观测 RTG（未折扣，和 actor 的 RTG token 同语义）──
        loss_q_v = ((q_v[m] - batch['rtg_v'][m]) / v_scale).pow(2).mean()
        loss_q_c = ((q_c[m] - batch['rtg_c'][m]) / c_scale).pow(2).mean()
        loss_total_q = loss_q_v + loss_q_c

        # ── V 更新：直接对观测 RTG 做 expectile 回归 ──
        diff_v = (batch['rtg_v'] - v_v)[m]
        w_v = torch.where(diff_v > 0,
                          torch.full_like(diff_v, self.tau_v),
                          torch.full_like(diff_v, 1.0 - self.tau_v))
        loss_v = (w_v * (diff_v / v_scale).pow(2)).mean()
        diff_c = (batch['rtg_c'] - v_c)[m]
        w_c = torch.where(diff_c > 0,
                          torch.full_like(diff_c, self.tau_c),
                          torch.full_like(diff_c, 1.0 - self.tau_c))
        loss_c = (w_c * (diff_c / c_scale).pow(2)).mean()
        loss_total_v = loss_v + loss_c

        # ── 分别优化两个网络 ──
        self.critic_v.optimizer.zero_grad()
        loss_total_v.backward()
        grad_norm_v = nn.utils.clip_grad_norm_(self.critic_v.parameters(),
                                               self.critic_v.grad_clip)
        self.critic_v.optimizer.step()

        self.critic_q.optimizer.zero_grad()
        loss_total_q.backward()
        grad_norm_q = nn.utils.clip_grad_norm_(self.critic_q.parameters(),
                                               self.critic_q.grad_clip)
        self.critic_q.optimizer.step()

        return {
            'loss_total':  (loss_total_v + loss_total_q).item(),
            'loss_v_opt':  loss_v.item(),    # V 网络 expectile（转化）
            'loss_c_opt':  loss_c.item(),    # V 网络 expectile（花费）
            'loss_v_mse':  loss_q_v.item(),  # Q 网络 MSE（转化）
            'loss_c_mse':  loss_q_c.item(),  # Q 网络 MSE（花费）
            'v_scale': v_scale.item(),
            'c_scale': c_scale.item(),
            'v_target_mean': batch['rtg_v'][m].mean().item(),
            'c_target_mean': batch['rtg_c'][m].mean().item(),
            'v_pred_mean': q_v[m].mean().item(),
            'c_pred_mean': q_c[m].mean().item(),
            'grad_norm': max(float(grad_norm_v), float(grad_norm_q)),
        }

    def _losses_and_metrics(self, batch):
        """Build Monte-Carlo Q/expectile V losses without updating parameters."""
        mask = batch['mask'] > 0
        if not mask.any():
            raise ValueError("critic batch contains no valid tokens")

        q_v, q_c = self.critic_q(
            batch['rtg'], batch['states'], batch['actions'],
            batch['timesteps'], batch['mask'])
        v_v, v_c = self.critic_v(
            batch['rtg'], batch['states'], batch['actions'],
            batch['timesteps'], batch['mask'])

        v_scale = batch['states'].new_tensor(self.v_loss_scale)
        c_scale = batch['states'].new_tensor(self.c_loss_scale)
        q_v_error = (q_v[mask] - batch['rtg_v'][mask]) / v_scale
        q_c_error = (q_c[mask] - batch['rtg_c'][mask]) / c_scale
        # Huber loss (disabled; kept here for easy experiment rollback):
        # loss_q_v = F.smooth_l1_loss(
        #     q_v_error, torch.zeros_like(q_v_error), beta=self.q_huber_beta)
        # loss_q_c = F.smooth_l1_loss(
        #     q_c_error, torch.zeros_like(q_c_error), beta=self.q_huber_beta)
        loss_q_v = q_v_error.square().mean()
        loss_q_c = q_c_error.square().mean()

        # V uses the observed Monte-Carlo returns directly.  It does not use
        # the learned Q prediction as an expectile target.
        diff_v = (batch['rtg_v'] - v_v)[mask] / v_scale
        weight_v = torch.where(
            diff_v > 0, torch.full_like(diff_v, self.tau_v),
            torch.full_like(diff_v, 1.0 - self.tau_v))
        loss_v = (weight_v * diff_v.square()).mean()

        diff_c = (batch['rtg_c'] - v_c)[mask] / c_scale
        weight_c = torch.where(
            diff_c > 0, torch.full_like(diff_c, self.tau_c),
            torch.full_like(diff_c, 1.0 - self.tau_c))
        loss_c = (weight_c * diff_c.square()).mean()

        loss_v_total = loss_v + loss_c
        loss_q_total = loss_q_v + loss_q_c
        metrics = {
            'loss_total': float((loss_v_total + loss_q_total).detach()),
            'loss_v_opt': float(loss_v.detach()),
            'loss_c_opt': float(loss_c.detach()),
            'loss_v_mse': float(loss_q_v.detach()),
            'loss_c_mse': float(loss_q_c.detach()),
            'v_scale': float(v_scale),
            'c_scale': float(c_scale),
            'v_target_mean': float(batch['rtg_v'][mask].mean()),
            'c_target_mean': float(batch['rtg_c'][mask].mean()),
            'v_pred_mean': float(q_v[mask].mean().detach()),
            'c_pred_mean': float(q_c[mask].mean().detach()),
            'valid_count': int(mask.sum()),
        }
        return loss_v_total, loss_q_total, metrics

    def zero_grad(self):
        self.critic_v.optimizer.zero_grad(set_to_none=True)
        self.critic_q.optimizer.zero_grad(set_to_none=True)

    def accumulate(self, batch, loss_divisor=1.0):
        """Accumulate one micro-batch; caller controls optimizer-step cadence."""
        if loss_divisor <= 0:
            raise ValueError("loss_divisor must be positive")
        loss_v, loss_q, metrics = self._losses_and_metrics(batch)
        (loss_v / loss_divisor).backward()
        (loss_q / loss_divisor).backward()
        return metrics

    def optimizer_step(self):
        grad_norm_v = nn.utils.clip_grad_norm_(
            self.critic_v.parameters(), self.critic_v.grad_clip)
        grad_norm_q = nn.utils.clip_grad_norm_(
            self.critic_q.parameters(), self.critic_q.grad_clip)
        self.critic_v.optimizer.step()
        self.critic_q.optimizer.step()
        return max(float(grad_norm_v), float(grad_norm_q))

    def step(self, batch):
        """Backward-compatible single-batch optimizer update."""
        self.zero_grad()
        metrics = self.accumulate(batch)
        metrics['grad_norm'] = self.optimizer_step()
        self.zero_grad()
        return metrics

    def train(self):
        self.critic_v.train()
        self.critic_q.train()
        return self

    def eval(self):
        self.critic_v.eval()
        self.critic_q.eval()
        return self

    def to(self, device):
        self.critic_v.to(device)
        self.critic_q.to(device)
        return self

    def state_dict(self):
        return {
            'critic_v': self.critic_v.state_dict(),
            'critic_q': self.critic_q.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and 'critic_v' in state_dict and \
                ('critic_q' in state_dict or 'critic_c' in state_dict):
            # New split checkpoint format
            self.critic_v.load_state_dict(state_dict['critic_v'])
            q_key = 'critic_q' if 'critic_q' in state_dict else 'critic_c'
            self.critic_q.load_state_dict(state_dict[q_key])
        else:
            # Backward compatibility: try to load as combined checkpoint
            # Split V and C heads into separate critics
            try:
                v_state = {}
                c_state = {}
                for key, val in state_dict.items():
                    if 'head_v' in key or 'embed' in key or 'trans' in key:
                        v_state[key.replace('critic_v.', '')] = val
                        c_state[key.replace('critic_c.', '')] = val
                    elif 'head_c' in key:
                        c_state[key.replace('critic_c.', '')] = val

                # Fill in the individual head params
                for key, val in state_dict.items():
                    if 'head_v_mse' in key or 'head_v_opt' in key:
                        v_state[key] = val
                    elif 'head_c_mse' in key or 'head_c_opt' in key:
                        c_state[key] = val

                self.critic_v.load_state_dict(v_state, strict=False)
                self.critic_q.load_state_dict(c_state, strict=False)
            except Exception as e:
                raise ValueError(f"Could not load state_dict: {e}")

    def parameters(self):
        return list(self.critic_v.parameters()) + list(self.critic_q.parameters())

    def requires_grad_(self, requires_grad):
        self.critic_v.requires_grad_(requires_grad)
        self.critic_q.requires_grad_(requires_grad)
        return self


# =====================================================================
# Stage 2 — Actor trainer, critic frozen.
#
#   action_pred: advantage-weighted BC. The advantage is a pure evaluation
#     of the frozen critic's _mse head at (states, actions) vs
#     (states, beta*actions), computed under no_grad — it never trains the
#     critic and never asks the critic to extrapolate a target value.
#
#   beta_pred: GAVE-style self-distillation. S(beta*a) is computed through
#     the frozen-but-differentiable _mse head (gradient flows to beta_pred,
#     not to the critic's parameters, since those have requires_grad=False).
#     It is regressed (MSE, not maximized) toward a detached, data-calibrated
#     optimistic anchor S_opt(a_real) from the critic's _opt head. Because
#     the target is bounded and anchored to the real action's distribution,
#     beta cannot be pushed toward states the critic only extrapolates —
#     the failure mode of plain policy-gradient ascent on a frozen critic
#     (loss_beta = -S(beta*a)) in an offline setting.
# =====================================================================
class DGABActorTrainer:
    def __init__(self, actor: DGABActor, critic: DGABCritic,
                 alpha=3.0, lambda_beta=1.0,
                 learning_rate=1e-4, weight_decay=1e-4, device='cpu',
                 action_mean=0.0, action_std=1.0, grad_clip=1.0,
                 max_adv_weight=5.0):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic.eval()
        for p in self.critic.parameters():
            p.requires_grad_(False)

        self.alpha = alpha
        self.lambda_beta = lambda_beta
        self.device = device
        self.action_mean = float(action_mean)
        self.action_std = float(action_std)
        self.grad_clip = float(grad_clip)
        if max_adv_weight < 1.0:
            raise ValueError("max_adv_weight must be at least 1")
        self.max_adv_weight = float(max_adv_weight)
        self.action_upper = None

        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _legacy_step_unused(self, batch):
        """
        batch requires: rtg[B,T,2], states[B,T,S], actions[B,T,1],
                        timesteps[B,T], past_v[B,T,1], past_c[B,T,1],
                        cpa_target[B,T,1], mask[B,T]
        """
        self.actor.train()
        m = batch['mask'] > 0

        action_pred = self.actor(
            batch['rtg'], batch['states'], batch['actions'],
            batch['timesteps'], batch['mask'])

        # ── action_pred: advantage-weighted BC (advantage = pure critic eval) ──
        # if self.action_upper is not None:
        #     explore_real = torch.clamp(beta_pred * action_real, 0.0, self.action_upper)
        # else:
        #     explore_real = beta_pred * action_real
        # explore_action_detached = ((explore_real - self.action_mean) /
        #                            self.action_std).detach()
        with torch.no_grad():
            v, c, q_v, q_c = self.critic(
                batch['rtg'], batch['states'], batch['actions'],
                batch['timesteps'], batch['mask'])
            # v_expl, c_expl, _, _ = self.critic(
            #     batch['rtg'], batch['states'], explore_action_detached,
            #     batch['timesteps'], batch['mask'])

            # _, _, v_expl, c_expl = self.critic(
            #     batch['rtg'], batch['states'], batch['actions'],
            #     batch['timesteps'], batch['mask'])
            s_v = global_score(batch['past_v'], batch['past_c'], v, c, batch['cpa_target'])
            s_a = global_score(batch['past_v'], batch['past_c'], q_v, q_c, batch['cpa_target'])

        a_pred = action_pred[m]
        a_target = batch['actions'][m]
        #a_expl   = explore_action_detached[m]

        adv = (s_a - s_v)[m]
        #adv = (s_orig - s_expl)[m]  # maximize S(beta*a) -> minimize S(a) - S(beta*a)
        #adv = (q_v - v)[m]
        #adv_std = adv.detach().std(unbiased=False).clamp(min=LOSS_SCALE_FLOOR)
        adv_std = adv.detach().std(unbiased=False).clamp(min=LOSS_SCALE_FLOOR, max=100.0)
        adv_normed = torch.clamp(adv / adv_std, min=-10.0, max=10.0)
        weight = torch.exp(adv_normed).clamp(max=50.0)   # 封顶，防离群样本主导梯度
        a_scale = a_target.new_tensor(1.0)
        squared_error = ((a_pred - a_target) / a_scale).pow(2)
        # loss_action = ((1.0 - w) * ((a_pred - a_target) / a_scale).pow(2)
        #                + w        * ((a_pred - a_expl)   / a_scale).pow(2)).mean()
        #loss_action = ((a_pred - a_target) / a_scale).pow(2).mean()
        # ── beta_pred: GAVE-style self-distillation (bounded, data-calibrated) ──
        # NOT wrapped in no_grad: gradient must flow explore_action -> beta_pred.
        # Critic params are frozen (requires_grad=False) so no gradient reaches them.
        # if self.action_upper is not None:
        #     explore_real = torch.clamp(beta_pred * action_real, 0.0, self.action_upper)
        # else:
        #     explore_real = beta_pred * action_real
        # explore_action = (explore_real - self.action_mean) / self.action_std
        # v_cf, c_cf, _, _ = self.critic(
        #     batch['rtg'], batch['states'], explore_action,
        #     batch['timesteps'], batch['mask'])
        # s_counterfactual = global_score(batch['past_v'], batch['past_c'], v_cf, c_cf, batch['cpa_target'])

        # with torch.no_grad():
        #     _, _, v_opt_real, c_opt_real = self.critic(
        #         batch['rtg'], batch['states'], batch['actions'],
        #         batch['timesteps'], batch['mask'])
        #     s_anchor = global_score(batch['past_v'], batch['past_c'], v_opt_real, c_opt_real, batch['cpa_target'])

        # s_scale = s_anchor.detach()[m].std(unbiased=False).clamp(min=LOSS_SCALE_FLOOR)
        # loss_beta = ((s_counterfactual[m] - s_anchor[m]) / s_scale).pow(2).mean()

        #loss_total = loss_action + self.lambda_beta * loss_beta
        # Advantage-weighted behavior cloning with a deterministic MSE objective.
        loss_total = (weight * squared_error).mean()
        self.optimizer.zero_grad()
        loss_total.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.optimizer.step()

        # return {
        #     'loss_total':  loss_total.item(),
        #     'loss_action': loss_action.item(),
        #     'loss_beta':   loss_beta.item(),
        #     'w_mean':      w.mean().item(),
        #     'w_std':       w.std().item(),
        #     'adv_std':     adv_std.item(),
        #     's_orig_mean': s_orig.mean().item(),
        #     's_expl_mean': s_expl.mean().item(),
        #     's_anchor_mean': s_anchor.mean().item(),
        #     'beta_mean':   beta_pred.mean().item(),
        #     'action_scale': a_scale.item(),
        #     'score_scale': s_scale.item(),
        #     'grad_norm': float(grad_norm),
        #     'delta_v_abs': (v_expl[m] - v_orig[m]).abs().mean().item(),
        #     'delta_c_abs': (c_expl[m] - c_orig[m]).abs().mean().item(),
        #     'delta_s_abs': adv.abs().mean().item(),
        #     'adv_pos_rate': (adv > 0).float().mean().item(),
        #     'adv_neg_rate': (adv < 0).float().mean().item(),
        #     'adv_zero_rate': (adv.abs() < 1e-6).float().mean().item(),
        #     'beta_low_rate': (beta_pred[m] < 0.51).float().mean().item(),
        #     'beta_high_rate': (beta_pred[m] > 1.49).float().mean().item(),
        # }
        return {
            'loss_total':  loss_total.item(),
            'w_mean':      weight.mean().item(),
            'w_std':       weight.std().item(),
            'adv_std':     adv_std.item(),
            'q_v_mean':    q_v[m].mean().item(),
            'v_mean':      v[m].mean().item(),
            'action_scale': a_scale.item(),
            'grad_norm': float(grad_norm),
            'delta_s_abs': adv.abs().mean().item(),
            'adv_pos_rate': (adv > 0).float().mean().item(),
            'adv_neg_rate': (adv < 0).float().mean().item(),
            'adv_zero_rate': (adv.abs() < 1e-6).float().mean().item(),
        }

    def _loss_and_metrics(self, batch):
        """Build the pure AWBC loss without changing optimizer state."""
        self.actor.train()
        mask = batch['mask'] > 0
        if not mask.any():
            raise ValueError("actor batch contains no valid tokens")

        action_pred = self.actor(
            batch['rtg'], batch['states'], batch['actions'],
            batch['timesteps'], batch['mask'])
        with torch.no_grad():
            value_v, value_c, q_v, q_c = self.critic(
                batch['rtg'], batch['states'], batch['actions'],
                batch['timesteps'], batch['mask'])
            score_v = global_score(
                batch['past_v'], batch['past_c'], value_v, value_c,
                batch['cpa_target'])
            score_q = global_score(
                batch['past_v'], batch['past_c'], q_v, q_c,
                batch['cpa_target'])

        advantage = (score_q - score_v)[mask]
        advantage_std = advantage.std(unbiased=False).clamp(
            min=LOSS_SCALE_FLOOR, max=100.0)
        normalized_advantage = torch.clamp(
            self.alpha * advantage / advantage_std, min=-10.0, max=10.0)
        weight = torch.exp(normalized_advantage).clamp(max=self.max_adv_weight)
        squared_error = (action_pred[mask] - batch['actions'][mask]).square()
        loss = (weight * squared_error).mean()
        metrics = {
            'loss_total': float(loss.detach()),
            'w_mean': float(weight.mean()),
            'w_std': float(weight.std(unbiased=False)),
            'w_max': float(weight.max()),
            'adv_std': float(advantage_std),
            'q_v_mean': float(q_v[mask].mean()),
            'v_mean': float(value_v[mask].mean()),
            'action_scale': 1.0,
            'delta_s_abs': float(advantage.abs().mean()),
            'adv_pos_rate': float((advantage > 0).float().mean()),
            'adv_neg_rate': float((advantage < 0).float().mean()),
            'adv_zero_rate': float((advantage.abs() < 1e-6).float().mean()),
            'valid_count': int(mask.sum()),
        }
        return loss, metrics

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def accumulate(self, batch, loss_divisor=1.0):
        if loss_divisor <= 0:
            raise ValueError("loss_divisor must be positive")
        loss, metrics = self._loss_and_metrics(batch)
        (loss / loss_divisor).backward()
        return metrics

    def optimizer_step(self):
        grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(grad_norm)

    def step(self, batch):
        """Backward-compatible single-batch optimizer update."""
        self.zero_grad()
        metrics = self.accumulate(batch)
        metrics['grad_norm'] = self.optimizer_step()
        self.zero_grad()
        return metrics


# =====================================================================
# Online rollout — actor only (critic is training-time-only).
# =====================================================================
class DGABRollout:
    """
    Online rollout wrapper. rtg_v/c use a fixed scale divisor (rtg_scale),
    matching training-time preprocessing (see data_fo.MDPCsvReplayBuffer).
    state: z-scored 16-dim FO state.
    """
    def __init__(self, actor: DGABActor, V_goal, C_goal, K=20,
                 rtg_scale=1.0, device='cpu'):
        self.actor = actor
        self.K = K
        self.rtg_scale = rtg_scale
        self.device = device
        self.act_dim = actor.act_dim
        self.state_dim = actor.state_dim

        v_init = V_goal / rtg_scale
        c_init = C_goal / rtg_scale
        self.rtg = torch.tensor([v_init, c_init], dtype=torch.float32, device=self.device)
        self.rtgs, self.states, self.actions, self.timesteps = [], [], [], []
        self.t = 0

    @torch.no_grad()
    def act(self, state):
        """state: np.array [state_dim]"""
        self.states.append(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        self.rtgs.append(self.rtg.clone())
        self.actions.append(torch.zeros(self.act_dim, device=self.device))
        self.timesteps.append(torch.tensor(self.t, dtype=torch.long, device=self.device))

        rtg_seq    = self._pad(self.rtgs,    (2,))
        state_seq  = self._pad(self.states,  (self.state_dim,))
        action_seq = self._pad(self.actions, (self.act_dim,))
        time_seq   = self._pad(self.timesteps, ())
        mask       = self._mask(len(self.rtgs))

        # ── 旧版（decision-transformer 直接输出 action_pred，无高斯头）──
        # action_pred, _ = self.actor(
        #     rtg_seq.unsqueeze(0), state_seq.unsqueeze(0),
        #     action_seq.unsqueeze(0), time_seq.long().unsqueeze(0),
        #     attention_mask=mask.unsqueeze(0))
        # action = action_pred[0, -1]
        # self.actions[-1] = action
        # return action.cpu().numpy()

        action_seq_pred = self.actor(
            rtg_seq.unsqueeze(0), state_seq.unsqueeze(0),
            action_seq.unsqueeze(0), time_seq.long().unsqueeze(0),
            attention_mask=mask.unsqueeze(0))
        action = action_seq_pred[0, -1]
        self.actions[-1] = action
        return action.cpu().numpy()

    def update_rtg(self, v_t, c_t):
        dv = v_t / self.rtg_scale
        dc = c_t / self.rtg_scale
        delta = torch.tensor([dv, dc], dtype=torch.float32, device=self.device)
        self.rtg = torch.clamp(self.rtg - delta, min=0.0)
        self.t += 1

    def _pad(self, seq, item_shape):
        K = self.K
        cur = torch.stack(seq[-K:], dim=0)
        if cur.shape[0] < K:
            pad = torch.zeros((K - cur.shape[0],) + item_shape,
                              dtype=cur.dtype, device=self.device)
            cur = torch.cat([pad, cur], dim=0)
        return cur

    def _mask(self, n):
        K = self.K
        n = min(n, K)
        return torch.cat([torch.zeros(K - n, dtype=torch.long, device=self.device),
                          torch.ones(n,     dtype=torch.long, device=self.device)])
