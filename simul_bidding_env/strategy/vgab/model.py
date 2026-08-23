"""
VGAB — user-trained DGABActor (Gaussian head) + DGABRollout for AuctionNet.

Extracted verbatim from the user's project (bidding_train_env/baseline/dgab/model.py)
so that the trained actor.pt checkpoint loads exactly. Inference only needs the
actor — the critic/trainer code is intentionally not copied.
"""
import numpy as np
import torch
import torch.nn as nn

from simul_bidding_env.strategy.vgab.blocks import Block

EPS = 1e-8


# =====================================================================
# Actor — Decision Transformer with 2D RTG input, 3-token/step stacking
#   token order per step: [rtg, state, action]
#   predicts from state-token position: (mu, sigma) Gaussian policy
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
        self.predict_beta = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

        self.gauss_action = nn.Linear(hidden_size, 2)

    def forward(self, rtg, states, actions, timesteps, attention_mask=None):
        """
        rtg:        [B, T, 2]
        states:     [B, T, state_dim]
        actions:    [B, T, act_dim]
        timesteps:  [B, T]
        attention_mask: [B, T] (1 = valid)
        Returns: (mu, sigma) — each [B, T, act_dim]/[B, T, 1]
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

        state_token = x[:, 1]                                                       # [B,T,H]
        mu, sigma = self.gauss_action(state_token).split(1, dim=-1)   # 沿最后一维切，各 [B,T,1]
        return mu, sigma


# =====================================================================
# Online rollout — actor only (deterministic mean evaluation).
#   RTG initialized as [V_goal, C_goal] / rtg_scale, decremented per tick.
# =====================================================================
class DGABRollout:
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
        """state: np.array [state_dim] → normalized action (mu)."""
        self.states.append(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        self.rtgs.append(self.rtg.clone())
        self.actions.append(torch.zeros(self.act_dim, device=self.device))
        self.timesteps.append(torch.tensor(self.t, dtype=torch.long, device=self.device))

        rtg_seq    = self._pad(self.rtgs,    (2,))
        state_seq  = self._pad(self.states,  (self.state_dim,))
        action_seq = self._pad(self.actions, (self.act_dim,))
        time_seq   = self._pad(self.timesteps, ())
        mask       = self._mask(len(self.rtgs))

        mu, sigma = self.actor(
            rtg_seq.unsqueeze(0), state_seq.unsqueeze(0),
            action_seq.unsqueeze(0), time_seq.long().unsqueeze(0),
            attention_mask=mask.unsqueeze(0))
        # sigma 必须与训练侧做同一个正值化变换 (softplus + 1e-6)
        mu    = mu[0, -1]                                               # [act_dim]
        sigma = torch.nn.functional.softplus(sigma[0, -1]) + 1e-6       # 正值
        # 确定性均值评估
        action = mu
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
