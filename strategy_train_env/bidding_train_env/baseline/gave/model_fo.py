"""
gave/model_fo.py — GAVE/GAoVE models for Fully Observable (state=16) setting.
Ported from baseline/dt/dt.py. score_target_mode: 'next'|'prev'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from gin import current_scope_str


def getScore(budget, cpa_cons, states, all_reward):
    beta = 2
    curr_cost = budget * (1 - states[1])
    curr_all_reward = all_reward
    curr_cpa = curr_cost / (curr_all_reward + 1e-10)
    curr_coef = cpa_cons / (curr_cpa + 1e-10)
    curr_penalty = pow(curr_coef, beta)
    curr_penalty = 1.0 if curr_penalty > 1.0 else curr_coef
    curr_score = curr_penalty * curr_all_reward
    return curr_score


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.key   = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])
        self.attn_drop  = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])
        self.register_buffer("bias",
            torch.tril(torch.ones(config['n_ctx'], config['n_ctx']))
                .view(1, 1, config['n_ctx'], config['n_ctx']))
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.proj   = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']

    def forward(self, x, mask):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        mask = mask.view(B, -1)[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.where(self.bias[:, :, :T, :T].bool(), att, self.masked_bias.to(att.dtype))
        att = att + mask
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1  = nn.LayerNorm(config['n_embd'])
        self.ln2  = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.mlp  = nn.Sequential(
            nn.Linear(config['n_embd'], config['n_inner']),
            nn.GELU(),
            nn.Dropout(config['resid_pdrop']),
            nn.Linear(config['n_inner'], config['n_embd']),
        )

    def forward(self, inputs_embeds, attention_mask):
        x = inputs_embeds + self.attn(self.ln1(inputs_embeds), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GAVE(nn.Module):
    def __init__(self, state_dim, act_dim, state_mean, state_std,
                 hidden_size=64, action_tanh=False, K=20,
                 max_ep_len=96, scale=2000, warmup_steps=10000,
                 weight_decay=0.0001, learning_rate=0.0001, time_dim=8,
                 target_return=4, device="cpu", expectile=0.99,
                 score_target_mode="prev",
                 block_config={
                     "n_ctx": 1024, "n_embd": 64, "n_layer": 3, "n_head": 1,
                     "n_inner": 512, "activation_function": "relu",
                     "n_position": 1024, "resid_pdrop": 0.1, "attn_pdrop": 0.1,
                 }):
        super(GAVE, self).__init__()
        self.device = device
        self.train_mode = 'update'
        self.length_times  = 3
        self.hidden_size   = hidden_size
        self.state_mean    = state_mean
        self.state_std     = state_std
        self.max_length    = K
        self.max_ep_len    = max_ep_len
        self.state_dim     = state_dim
        self.act_dim       = act_dim
        self.scale         = scale
        self.target_return = target_return
        self.warmup_steps  = warmup_steps
        self.weight_decay  = weight_decay
        self.learning_rate = learning_rate
        self.time_dim      = time_dim
        self.expectile     = expectile
        self.score_target_mode = score_target_mode

        self.transformer   = nn.ModuleList([Block(block_config) for _ in range(block_config['n_layer'])])
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.time_dim)
        self.embed_return  = nn.Linear(1, self.hidden_size)
        self.embed_reward  = nn.Linear(1, self.hidden_size)
        self.embed_state   = nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action  = nn.Linear(self.act_dim, self.hidden_size)
        self.trans_return  = nn.Linear(self.time_dim + self.hidden_size, self.hidden_size)
        self.trans_reward  = nn.Linear(self.time_dim + self.hidden_size, self.hidden_size)
        self.trans_state   = nn.Linear(self.time_dim + self.hidden_size, self.hidden_size)
        self.trans_action  = nn.Linear(self.time_dim + self.hidden_size, self.hidden_size)
        self.embed_ln      = nn.LayerNorm(self.hidden_size)
        self.predict_state  = nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else [])))
        self.predict_beta   = nn.Sequential(
            nn.Linear(self.hidden_size, 16), nn.GELU(),
            nn.Linear(16, 8), nn.GELU(),
            nn.Linear(8, 1), nn.Sigmoid())
        self.predict_return = nn.Sequential(
            nn.Linear(self.hidden_size, 128), nn.GELU(),
            nn.Linear(128, 16), nn.GELU(),
            nn.Linear(16, 1))
        self.predict_value  = nn.Sequential(
            nn.Linear(self.hidden_size, 128), nn.GELU(),
            nn.Linear(128, 16), nn.GELU(),
            nn.Linear(16, 1))

        self.optimizer = torch.optim.AdamW(self.parameters(),
            lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / self.warmup_steps, 1))
        self.init_eval()
        self.to(self.device)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        B, T = states.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.long)

        s_emb = self.embed_state(states)
        a_emb = self.embed_action(actions)
        r_emb = self.embed_return(returns_to_go)
        rw_emb = self.embed_reward(rewards)
        t_emb = self.embed_timestep(timesteps)

        s_emb  = self.trans_state( torch.cat([s_emb,  t_emb], dim=-1))
        a_emb  = self.trans_action(torch.cat([a_emb,  t_emb], dim=-1))
        r_emb  = self.trans_return(torch.cat([r_emb,  t_emb], dim=-1))
        rw_emb = self.trans_reward(torch.cat([rw_emb, t_emb], dim=-1))

        stacked = torch.stack((r_emb, s_emb, a_emb), dim=1).permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_size)
        stacked = self.embed_ln(stacked)
        stacked_mask = torch.stack([attention_mask] * 3, dim=1).permute(0, 2, 1).reshape(B, 3 * T).to(stacked.dtype)

        x = stacked
        for block in self.transformer:
            x = block(x, stacked_mask)
        x = x.reshape(-1, T, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])
        state_preds  = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])

        if self.training:
            value_preds = self.predict_value(x[:, 1])
            beta_preds  = self.predict_beta(x[:, 1]) + 0.5
            actions_1   = actions.clone().detach() * beta_preds
            a1_emb = self.trans_action(torch.cat([self.embed_action(actions_1), t_emb], dim=-1))
            stacked1 = torch.stack((r_emb, s_emb, a1_emb), dim=1).permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_size)
            stacked1 = self.embed_ln(stacked1)
            x1 = stacked1
            for block in self.transformer:
                x1 = block(x1, stacked_mask)
            x1 = x1.reshape(-1, T, 3, self.hidden_size).permute(0, 2, 1, 3)
            return_preds_1 = self.predict_return(x1[:, 2])
            return state_preds, action_preds, return_preds, None, return_preds_1, actions_1, value_preds
        return None, action_preds, None, None, None, None, None

    def get_action(self, states, actions, rewards, curr_score, timesteps, **kwargs):
        states     = states.reshape(1, -1, self.state_dim)
        actions    = actions.reshape(1, -1, self.act_dim)
        curr_score = curr_score.reshape(1, -1, 1)
        rewards    = rewards.reshape(1, -1, 1)
        timesteps  = timesteps.reshape(1, -1)
        if self.max_length is not None:
            states     = states[:, -self.max_length:]
            actions    = actions[:, -self.max_length:]
            curr_score = curr_score[:, -self.max_length:]
            rewards    = rewards[:, -self.max_length:]
            timesteps  = timesteps[:, -self.max_length:]
            n = states.shape[1]
            pad = self.max_length - n
            attn = torch.cat([torch.zeros(pad), torch.ones(n)]).to(dtype=torch.long, device=states.device).reshape(1, -1)
            def lpad(t, fill=0.0):
                shape = list(t.shape); shape[1] = pad
                return torch.cat([torch.full(shape, fill, dtype=t.dtype, device=t.device), t], dim=1)
            states     = lpad(states)
            actions    = lpad(actions, -10.0)
            curr_score = lpad(curr_score)
            rewards    = lpad(rewards)
            timesteps  = lpad(timesteps).to(torch.long)
        else:
            attn = None
        _, action_preds, *_ = self.forward(states, actions, rewards, curr_score, timesteps, attention_mask=attn, **kwargs)
        return action_preds[0, -1]

    def step(self, states, actions, rewards, dones, all_reward, curr_score, timesteps, attention_mask, next_states=None):
        action_target     = torch.clone(actions).detach()
        curr_score_target = torch.clone(curr_score).detach()
        if self.score_target_mode == "next":
            curr_score_target = curr_score_target[:, 1:]
        else:
            curr_score_target = curr_score_target[:, :-1]

        _, action_preds, curr_score_preds, _, curr_score_preds_1, action_1, value_preds = self.forward(
            states, actions, rewards, curr_score[:, :-1], timesteps, attention_mask=attention_mask)

        m = attention_mask.reshape(-1) > 0
        action_preds      = action_preds.reshape(-1, self.act_dim)[m]
        action_target     = action_target.reshape(-1, self.act_dim)[m]
        action_1          = action_1.reshape(-1, self.act_dim)[m]
        action_1_frozen   = action_1.clone().detach()

        cs_dim            = curr_score_preds.shape[2]
        curr_score_preds  = curr_score_preds.reshape(-1, cs_dim)[m]
        curr_score_preds_1= curr_score_preds_1.reshape(-1, cs_dim)[m]
        curr_score_target = curr_score_target.reshape(-1, cs_dim)[m]
        value_preds       = value_preds.reshape(-1, cs_dim)[m]
        value_preds_frozen= value_preds.clone().detach()

        wo       = torch.sigmoid(100 * (curr_score_preds_1 - curr_score_preds))
        wo_frozen= wo.clone().detach()
        diff     = curr_score_target - value_preds
        weight   = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        loss1    = torch.mean((1 - wo_frozen) * (action_preds - action_target) ** 2
                             + wo_frozen       * (action_preds - action_1_frozen) ** 2)
        loss2    = torch.mean((curr_score_preds - curr_score_target) ** 2) * 200
        loss3    = torch.mean(weight * diff ** 2) * 100
        loss4    = torch.mean((curr_score_preds_1 - value_preds_frozen) ** 2) * 100
        loss     = loss1 + loss2 + loss3 + loss4

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)
        self.optimizer.step()
        return (loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(),
                torch.mean(wo_frozen.squeeze()).item(),
                torch.mean(curr_score_target).item(),
                torch.mean(curr_score_preds).item(),
                torch.mean(curr_score_preds_1).item())

    def take_actions(self, state, target_return=None, pre_reward=None, budget=100, cpa=2):
        self.eval()
        target_return = target_return.to(self.device) if target_return is not None else self.target_return
        if self.eval_states is None:
            self.eval_states     = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            all_reward           = torch.zeros(1).to(self.device)
            self.eval_all_reward = torch.tensor(all_reward, dtype=torch.float32).reshape(1, 1).to(self.device)
            self.eval_curr_score = torch.tensor(target_return, dtype=torch.float32).reshape(1, 1).to(self.device)
        else:
            assert pre_reward is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0)
            self.eval_rewards[-1] = pre_reward
            pred_all_reward = self.eval_all_reward[0, -1] + pre_reward
            self.eval_all_reward = torch.cat([self.eval_all_reward, pred_all_reward.reshape(1, 1)], dim=1)
            curr_score = target_return - getScore(budget, cpa, self.eval_states[-1], pred_all_reward) / self.scale
            self.eval_curr_score = torch.cat([self.eval_curr_score, curr_score.reshape(1, 1)], dim=1)
            self.eval_timesteps  = torch.cat([self.eval_timesteps,
                torch.ones((1, 1), dtype=torch.long).to(self.device) * self.eval_timesteps[:, -1] + 1], dim=1)
        self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim).to(self.device)], dim=0)
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1).to(self.device)])
        action = self.get_action(
            (self.eval_states.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(self.device))
            / torch.tensor(self.state_std).to(self.device),
            self.eval_actions.to(dtype=torch.float32),
            self.eval_rewards.to(dtype=torch.float32),
            self.eval_curr_score.to(dtype=torch.float32),
            self.eval_timesteps.to(dtype=torch.long))
        self.eval_actions[-1] = action
        return action.detach().cpu().numpy()

    def init_eval(self):
        self.eval_states       = None
        self.eval_actions      = torch.zeros((0, self.act_dim), dtype=torch.float32).to(self.device)
        self.eval_rewards      = torch.zeros(0, dtype=torch.float32).to(self.device)
        self.eval_target_return= None
        self.eval_timesteps    = torch.tensor(0, dtype=torch.long).reshape(1, 1).to(self.device)
        self.eval_episode_return= 0
        self.eval_episode_length= 0

    def save_net(self, save_path, name):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_path, name))

    def load_net(self, load_path, device='cpu'):
        self.load_state_dict(torch.load(load_path, map_location=device))
