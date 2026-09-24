import torch
import torch.nn as nn

from simul_bidding_env.strategy.dgabshare.blocks import Block


class ConversionRTGModel(nn.Module):
    """DGAB actor conditioned on conversion-to-go only."""

    def __init__(self, state_dim, act_dim=1, hidden_size=128, max_ep_len=48,
                 time_dim=8, block_config=None):
        super().__init__()
        self.state_dim, self.act_dim = state_dim, act_dim
        self.hidden_size = hidden_size
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_time = nn.Embedding(max_ep_len, time_dim)
        self.trans_state = nn.Linear(hidden_size + time_dim, hidden_size)
        self.trans_action = nn.Linear(hidden_size + time_dim, hidden_size)
        self.trans_rtg = nn.Linear(hidden_size + time_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.transformer = nn.ModuleList(
            [Block(block_config) for _ in range(block_config["n_layer"])])
        self.action_head = nn.Linear(hidden_size, act_dim)
        self.value_head = nn.Linear(hidden_size, 2)
        self.q_head = nn.Linear(hidden_size, 2)

    def forward(self, rtg, states, actions, timesteps, attention_mask=None):
        batch, length = states.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(batch, length, dtype=torch.long,
                                        device=states.device)
        time = self.embed_time(timesteps)
        tokens = torch.stack([
            self.trans_rtg(torch.cat([self.embed_rtg(rtg), time], dim=-1)),
            self.trans_state(torch.cat([self.embed_state(states), time], dim=-1)),
            self.trans_action(torch.cat([self.embed_action(actions), time], dim=-1)),
        ], dim=2)
        hidden = self.embed_ln(tokens.reshape(batch, 3 * length, self.hidden_size))
        mask = attention_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(
            batch, 3 * length).to(hidden.dtype)
        for block in self.transformer:
            hidden = block(hidden, mask)
        hidden = hidden.reshape(batch, length, 3, self.hidden_size)
        return {"action": self.action_head(hidden[:, :, 1]),
                "value": self.value_head(hidden[:, :, 1]),
                "q": self.q_head(hidden[:, :, 2])}


class ConversionRTGRollout:
    def __init__(self, actor, V_goal, C_goal=None, K=20, rtg_scale=1.0,
                 device="cpu"):
        self.actor = actor
        self.K, self.rtg_scale, self.device = K, rtg_scale, device
        self.state_dim, self.act_dim = actor.state_dim, actor.act_dim
        self.rtg = torch.tensor([V_goal], dtype=torch.float32,
                                device=device) / rtg_scale
        self.rtgs, self.states, self.actions, self.timesteps = [], [], [], []
        self.t = 0

    def _pad(self, sequence, shape):
        current = torch.stack(sequence[-self.K:])
        if len(current) < self.K:
            current = torch.cat([torch.zeros(
                (self.K - len(current),) + shape, dtype=current.dtype,
                device=self.device), current])
        return current

    def _mask(self, length):
        length = min(length, self.K)
        return torch.cat([torch.zeros(self.K - length, dtype=torch.long,
                                      device=self.device),
                          torch.ones(length, dtype=torch.long,
                                     device=self.device)])

    @torch.no_grad()
    def act(self, state):
        self.states.append(torch.as_tensor(state, dtype=torch.float32,
                                           device=self.device))
        self.rtgs.append(self.rtg.clone())
        self.actions.append(torch.zeros(self.act_dim, device=self.device))
        self.timesteps.append(torch.tensor(self.t, dtype=torch.long,
                                           device=self.device))
        out = self.actor(
            self._pad(self.rtgs, (1,))[None],
            self._pad(self.states, (self.state_dim,))[None],
            self._pad(self.actions, (self.act_dim,))[None],
            self._pad(self.timesteps, ()).long()[None],
            self._mask(len(self.rtgs))[None])
        action = out["action"][0, -1]
        self.actions[-1] = action
        return action.cpu().numpy()

    def update_rtg(self, conversion, cost=None):
        delta = torch.tensor([conversion], dtype=torch.float32,
                             device=self.device) / self.rtg_scale
        self.rtg = torch.clamp(self.rtg - delta, min=0)
        self.t += 1
