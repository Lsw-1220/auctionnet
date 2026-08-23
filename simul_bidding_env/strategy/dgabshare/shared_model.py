"""Shared causal Transformer with deterministic actor, V, and Q heads."""

import torch
import torch.nn as nn

from .blocks import Block
from .model import global_score


LOSS_FLOOR = 1e-4


class DGABSharedModel(nn.Module):
    """Deterministic policy/V read state tokens; Q reads action tokens."""

    def __init__(self, state_dim, act_dim=1, hidden_size=128, max_ep_len=48,
                 time_dim=8, block_config=None):
        super().__init__()
        self.state_dim, self.act_dim = state_dim, act_dim
        self.hidden_size = hidden_size

        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_rtg = nn.Linear(2, hidden_size)
        self.embed_time = nn.Embedding(max_ep_len, time_dim)
        self.trans_state = nn.Linear(hidden_size + time_dim, hidden_size)
        self.trans_action = nn.Linear(hidden_size + time_dim, hidden_size)
        self.trans_rtg = nn.Linear(hidden_size + time_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.transformer = nn.ModuleList(
            [Block(block_config) for _ in range(block_config['n_layer'])])

        # All task heads are deliberately linear for this experiment.
        self.action_head = nn.Linear(hidden_size, act_dim)
        self.value_head = nn.Linear(hidden_size, 2)  # V(next conversion/cost RTG | state)
        self.q_head = nn.Linear(hidden_size, 2)      # Q(next conversion/cost RTG | state, action)

    def encode(self, rtg, states, actions, timesteps, attention_mask=None):
        batch_size, length = states.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, length, dtype=torch.long,
                                        device=states.device)
        time = self.embed_time(timesteps)
        tokens = torch.stack([
            self.trans_rtg(torch.cat([self.embed_rtg(rtg), time], dim=-1)),
            self.trans_state(torch.cat([self.embed_state(states), time], dim=-1)),
            self.trans_action(torch.cat([self.embed_action(actions), time], dim=-1)),
        ], dim=2)
        hidden = self.embed_ln(tokens.reshape(batch_size, 3 * length, self.hidden_size))
        token_mask = attention_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(
            batch_size, 3 * length).to(hidden.dtype)
        for block in self.transformer:
            hidden = block(hidden, token_mask)
        hidden = hidden.reshape(batch_size, length, 3, self.hidden_size)
        return hidden[:, :, 1], hidden[:, :, 2]

    def forward(self, rtg, states, actions, timesteps, attention_mask=None):
        state_token, action_token = self.encode(
            rtg, states, actions, timesteps, attention_mask)
        return dict(action=self.action_head(state_token),
                    value=self.value_head(state_token), q=self.q_head(action_token))


class DGABAWBCTrainer:
    """Jointly train Q, expectile V, and advantage-weighted MSE BC."""

    def __init__(self, model, learning_rate=1e-4, weight_decay=1e-4,
                 grad_clip=1.0, q_coef=1.0, v_coef=1.0, actor_coef=1.0,
                 tau_v=0.9, tau_c=0.1, alpha=1.0, max_adv_weight=20.0,
                 v_loss_scale=1.0, c_loss_scale=1.0):
        if not 0.5 < tau_v < 1.0 or not 0.0 < tau_c < 0.5:
            raise ValueError('require 0.5 < tau_v < 1 and 0 < tau_c < 0.5')
        if max_adv_weight < 1.0:
            raise ValueError('max_adv_weight must be >= 1')
        self.model = model
        self.grad_clip = float(grad_clip)
        self.q_coef, self.v_coef, self.actor_coef = q_coef, v_coef, actor_coef
        self.tau_v, self.tau_c = float(tau_v), float(tau_c)
        self.alpha, self.max_adv_weight = float(alpha), float(max_adv_weight)
        self.v_loss_scale = max(float(v_loss_scale), 1e-3)
        self.c_loss_scale = max(float(c_loss_scale), 1e-3)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def loss_and_metrics(self, batch):
        output = self.model(batch['rtg'], batch['states'], batch['actions'],
                            batch['timesteps'], batch['mask'])
        mask = batch['mask'].bool()
        target = torch.cat([batch['next_rtg_v'], batch['next_rtg_c']], dim=-1)
        scales = target.new_tensor([self.v_loss_scale, self.c_loss_scale])

        # Q(s,a): supervised next-RTG regression on dataset actions.
        q_error = (output['q'] - target) / scales
        q_loss = q_error[mask].square().mean()

        # V(s): IQL-style asymmetric regression to the detached paired Q output.
        value_error = (output['q'].detach() - output['value']) / scales
        value_weight_v = torch.where(
            value_error[..., 0] > 0, self.tau_v, 1.0 - self.tau_v)
        value_weight_c = torch.where(
            value_error[..., 1] > 0, self.tau_c, 1.0 - self.tau_c)
        v_loss_v = (value_weight_v[mask] * value_error[..., 0][mask].square()).mean()
        v_loss_c = (value_weight_c[mask] * value_error[..., 1][mask].square()).mean()
        value_loss = v_loss_v + v_loss_c

        # Actor: detached score advantage supplies only a sample weight.
        with torch.no_grad():
            q_v, q_c = output['q'][..., 0:1].clamp_min(0), output['q'][..., 1:2].clamp_min(0)
            v_v = output['value'][..., 0:1].clamp_min(0)
            v_c = output['value'][..., 1:2].clamp_min(0)
            score_q = global_score(batch['past_v'], batch['past_c'], q_v, q_c,
                                   batch['cpa_target'])
            score_v = global_score(batch['past_v'], batch['past_c'], v_v, v_c,
                                   batch['cpa_target'])
            advantage = (score_q - score_v)[mask]
            advantage_std = advantage.std(unbiased=False).clamp_min(LOSS_FLOOR)
            weight = torch.exp(torch.clamp(
                self.alpha * advantage / advantage_std, -10.0, 10.0)).clamp(
                    max=self.max_adv_weight)
        action_squared_error = (output['action'][mask] - batch['actions'][mask]).square()
        actor_loss = (weight * action_squared_error).mean()

        loss = (self.q_coef * q_loss + self.v_coef * value_loss
                + self.actor_coef * actor_loss)
        return loss, dict(
            loss_total=float(loss.detach()), loss_actor=float(actor_loss.detach()),
            loss_q=float(q_loss.detach()), loss_v=float(value_loss.detach()),
            loss_v_conversion=float(v_loss_v.detach()),
            loss_v_cost=float(v_loss_c.detach()),
            advantage_mean=float(advantage.mean()), advantage_std=float(advantage_std),
            weight_mean=float(weight.mean()), weight_max=float(weight.max()),
            q_v_mean=float(output['q'][..., 0][mask].mean().detach()),
            q_c_mean=float(output['q'][..., 1][mask].mean().detach()),
            v_v_mean=float(output['value'][..., 0][mask].mean().detach()),
            v_c_mean=float(output['value'][..., 1][mask].mean().detach()),
            valid_count=int(mask.sum()))

    def accumulate(self, batch, loss_divisor=1.0):
        loss, metrics = self.loss_and_metrics(batch)
        (loss / loss_divisor).backward()
        return metrics

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self):
        norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(norm)


class DGABSharedRollout:
    """Autoregressively emit the deterministic policy action."""

    def __init__(self, model, V_goal, C_goal, K=20, rtg_scale=1.0, device='cpu',
                 **_ignored):
        self.model = self.actor = model
        self.K, self.rtg_scale, self.device = K, rtg_scale, device
        self.state_dim, self.act_dim = model.state_dim, model.act_dim
        self.rtg = torch.tensor([V_goal / rtg_scale, C_goal / rtg_scale],
                                dtype=torch.float32, device=device)
        self.rtgs, self.states, self.actions, self.timesteps = [], [], [], []
        self.t = 0

    @torch.no_grad()
    def act(self, state):
        self.states.append(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        self.rtgs.append(self.rtg.clone())
        self.actions.append(torch.zeros(self.act_dim, device=self.device))
        self.timesteps.append(torch.tensor(self.t, dtype=torch.long, device=self.device))
        rtg, states = self._pad(self.rtgs, (2,)), self._pad(self.states, (self.state_dim,))
        actions, times = self._pad(self.actions, (self.act_dim,)), self._pad(self.timesteps, ())
        mask = self._mask(len(self.rtgs))
        output = self.model(
            rtg[None], states[None], actions[None], times.long()[None], mask[None])
        action = output['action'][0, -1]
        self.actions[-1] = action
        return action.cpu().numpy()

    def update_rtg(self, v_t, c_t):
        self.rtg = torch.clamp(self.rtg - torch.tensor(
            [v_t, c_t], dtype=torch.float32, device=self.device) / self.rtg_scale, min=0)
        self.t += 1

    def _pad(self, sequence, shape):
        current = torch.stack(sequence[-self.K:])
        if len(current) < self.K:
            current = torch.cat([torch.zeros((self.K - len(current),) + shape,
                                             dtype=current.dtype, device=self.device), current])
        return current

    def _mask(self, length):
        length = min(length, self.K)
        return torch.cat([torch.zeros(self.K - length, dtype=torch.long, device=self.device),
                          torch.ones(length, dtype=torch.long, device=self.device)])
