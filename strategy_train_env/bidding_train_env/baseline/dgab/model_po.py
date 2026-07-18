"""
GAVE refactored:
- 2D RTG (V_remain, C_remain), no scalar subtraction
- Dual-head value Critic (V upper bound, C lower bound) via expectile regression
- Actor uses DT with 2D RTG; produces (a_hat, beta_explore) where a_tilde = beta * a
- Actor loss combines BC and explore imitation, weighted by global-score advantage
  (advantage scaled by batch std so alpha becomes a dimensionless temperature)
"""
import numpy as np
import torch
import torch.nn as nn

from bidding_train_env.baseline.dgab.blocks import Block

EPS = 1e-8
CPA_PENALTY_BETA = 2.0   # 全局分数公式里的指数 β,与 actor 的 beta_explore 无关


# =====================================================================
# Module 0: 步内曝光集编码器 (R2=ClsToken / R3/R5=BidFormer)
# =====================================================================
class ClsTokenObsEncoder(nn.Module):
    """
    R2: 无条件 cls-token + 双向自注意力池化.
    obs_padded : [N, max_imp, obs_dim]
    obs_mask   : [N, max_imp] float, 1=valid 0=pad
    returns    : [N, hidden_size]
    """
    def __init__(self, obs_dim=9, hidden_size=64, n_heads=4):
        super().__init__()
        self.embed     = nn.Linear(obs_dim, hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.attn      = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True,
                                               dropout=0.1)
        self.ln        = nn.LayerNorm(hidden_size)

    def forward(self, obs_padded, obs_mask):
        N = obs_padded.shape[0]
        x   = self.embed(obs_padded)                                   # [N, M, H]
        cls = self.cls_token.expand(N, -1, -1)                        # [N, 1, H]
        x   = torch.cat([cls, x], dim=1)                              # [N, M+1, H]
        # key_padding_mask: True=ignore; cls slot always valid
        kpm = torch.cat([
            torch.zeros(N, 1, device=obs_padded.device, dtype=torch.bool),
            (obs_mask < 0.5)], dim=1)                                  # [N, M+1]
        out, _ = self.attn(x, x, x, key_padding_mask=kpm)
        return self.ln(out[:, 0])                                      # [N, H]


class BidFormerEncoder(nn.Module):
    """
    R3/R5: 宏观条件化 context-token (BidFormer 最小变体).
    三域划分 (9维 obs, 索引见 data_v2.py):
      value:       [pValue(0), pValueSigma(1)]
      competition: [xi(2), cost(4), bid(7), lwc(8)]
      outcome:     [adSlot(3), isExposed(5), conversionAction(6)]
    context_token 由 norm(rl) (macro) 投影初始化,
    cross-attend 曝光集后输出为步快照.
    obs_padded : [N, max_imp, obs_dim]
    obs_mask   : [N, max_imp] float
    macro      : [N, macro_dim]  = norm(rl) for this step
    returns    : [N, hidden_size]
    """
    _VAL_IDX  = [0, 1]
    _COMP_IDX = [2, 4, 7, 8]
    _OUT_IDX  = [3, 5, 6]

    def __init__(self, hidden_size=64, macro_dim=2, n_heads=4, n_layers=2):
        super().__init__()
        self.embed_val  = nn.Linear(len(self._VAL_IDX),  hidden_size)
        self.embed_comp = nn.Linear(len(self._COMP_IDX), hidden_size)
        self.embed_out  = nn.Linear(len(self._OUT_IDX),  hidden_size)
        self.imp_ln     = nn.LayerNorm(hidden_size)
        self.macro_proj = nn.Linear(macro_dim, hidden_size)
        self.cross_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, n_heads, batch_first=True, dropout=0.1)
            for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_size),
                          nn.Linear(hidden_size, hidden_size * 4), nn.GELU(),
                          nn.Linear(hidden_size * 4, hidden_size))
            for _ in range(n_layers)])
        self.ln_layers  = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_layers)])
        self.out_ln     = nn.LayerNorm(hidden_size)

    def forward(self, obs_padded, obs_mask, macro):
        # three-domain impression embedding
        val  = self.embed_val( obs_padded[:, :, self._VAL_IDX])
        comp = self.embed_comp(obs_padded[:, :, self._COMP_IDX])
        out  = self.embed_out( obs_padded[:, :, self._OUT_IDX])
        imp  = self.imp_ln(val + comp + out)                          # [N, M, H]

        ctx = self.macro_proj(macro).unsqueeze(1)                     # [N, 1, H]
        kpm = (obs_mask < 0.5)                                        # [N, M] True=pad

        # ctx 自身作为兜底 key: 当某步曝光集全为 pad 时(左侧 padding 或当步无曝光),
        # 保证每行至少一个非 mask key,避免 MultiheadAttention 整行 -inf 导致 softmax 出 NaN。
        # 此时该步退化为只 attend 自己 == 纯宏观状态 macro_proj(norm(rl)),与 use_obs=False 语义一致。
        self_kpm = torch.zeros(kpm.shape[0], 1, device=kpm.device, dtype=torch.bool)

        for ca, ffn, ln in zip(self.cross_layers, self.ffn_layers, self.ln_layers):
            q = ln(ctx)
            kv = torch.cat([q, imp], dim=1)                           # [N, 1+M, H]
            full_kpm = torch.cat([self_kpm, kpm], dim=1)               # [N, 1+M]
            attn_out, _ = ca(q, kv, kv, key_padding_mask=full_kpm)
            ctx = ctx + attn_out
            ctx = ctx + ffn(ctx)

        return self.out_ln(ctx[:, 0])                                 # [N, H]


# =====================================================================
# Module 1: Dataset annotation —— 在 trajectory 上挂 2D RTG 与历史累计量
# =====================================================================
def annotate_trajectory_with_dual_rtg(traj, sparse_v_credit=0.0):
    """
    在 EpisodeReplayBuffer 加工后的 trajectory dict 上原地添加 5 个数组:
        rtg_v[t]      = sum_{k>=t} v_k    本步及以后的剩余转化
        rtg_c[t]      = sum_{k>=t} c_k    本步及以后的剩余花费
        past_v[t]     = sum_{k<t}  v_k    本步之前的累计转化
        past_c[t]     = sum_{k<t}  c_k    本步之前的累计花费
        cpa_target[t] = traj['cpacons'][t]  逐步的 CPA 约束(原始量纲)

    依赖字段(EpisodeReplayBuffer 现有口径):
        traj['rewards']      [T, 1] -> 每步真实转化数 v_t
        traj['resourceleft'] [T, 2] -> 第 1 列 bgtleft, 用于反推 c_t
        traj['budget']       [T, 1] -> 反推 c_t 用的总预算
        traj['cpacons']      [T, 1] -> 每条 episode 的 CPA 约束
    所有产物保持原始量纲, 未做归一化.

    Args:
        sparse_v_credit: float, default 0.0 (disabled).
            When > 0, adds a pseudo-V decrement for timesteps where cost was
            incurred but no conversion occurred.  This prevents RTG_v from
            stagnating in sparse-reward environments (e.g. pvalue_mean < 0.001).
            The pseudo-V for a zero-conversion step is:
                pseudo_v_t = c_t / cpa_target * sparse_v_credit
            Recommended range: 0.1 ~ 0.3.  0.0 = original behavior.
    """
    v = np.asarray(traj['rewards'], dtype=np.float32).reshape(-1)        # [T]
    bgtleft = np.asarray(traj['resourceleft'], dtype=np.float32)[:, 1]   # [T]
    budget = float(np.asarray(traj['budget']).reshape(-1)[0])
    cum_cost = budget * (1.0 - bgtleft)                                  # [T]
    c = np.diff(cum_cost, prepend=0.0).astype(np.float32)                # [T]

    past_v = np.concatenate([[0.0], np.cumsum(v)[:-1]]).astype(np.float32)
    past_c = np.concatenate([[0.0], np.cumsum(c)[:-1]]).astype(np.float32)

    # ── Sparse V credit: pseudo-V for zero-conversion timesteps ──
    # In sparse environments, many timesteps have v_t=0 but c_t>0 (budget
    # spent without conversions).  The open-loop rtg_v doesn't decrease on
    # these steps, causing all RTG tokens to look identical (token collapse).
    # Adding a pseudo-V credit makes rtg_v decrease proportionally to the
    # "wasted" cost, so the transformer can distinguish early vs late timesteps.
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
    rtg_c = (c.sum() - past_c).astype(np.float32)

    traj['rtg_v']      = rtg_v
    traj['rtg_c']      = rtg_c
    traj['past_v']     = past_v
    traj['past_c']     = past_c
    traj['cpa_target'] = np.asarray(traj['cpacons'], dtype=np.float32).reshape(-1)
    return traj


# =====================================================================
# Module 2.1: Actor —— Decision Transformer with 2D RTG input
# =====================================================================
class DGABStackActor(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size=64, max_ep_len=96,
                 time_dim=8, block_config=None):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size

        self.embed_state  = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_rtg    = nn.Linear(2, hidden_size)            # 关键: 2D RTG
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

    def forward(self, rtg, states, actions, timesteps, attention_mask=None):
        """
        rtg:        [B, T, 2]
        states:     [B, T, state_dim]
        actions:    [B, T, act_dim]
        timesteps:  [B, T]
        attention_mask: [B, T] (1 = valid)
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

        # state-token 位置预测 action 和 beta
        action_pred = self.predict_action(x[:, 1])                                 # [B,T,act_dim]
        beta_pred = self.predict_beta(x[:, 1]) + 0.5                               # [B,T,1] ∈ (0.5, 1.5)
        return action_pred, beta_pred


# =====================================================================
# Module 2.1b: DGABCrossAttnActor —— encoder-decoder 版 Actor
#   memory  = (s_t, a_{t-1}) 错位配对, 因果编码
#   query   = 单 2D-RTG query (可升级为双 V/C query)
#   a_t     = head([cross_attn_out ; s_t 残差])
# =====================================================================
class CausalCrossAttention(nn.Module):
    """单头因果 cross-attention: query 只能 attend memory 的 ≤t 位置."""
    def __init__(self, hidden_size, n_head):
        super().__init__()
        assert hidden_size % n_head == 0
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(0.1)

    def forward(self, query, memory, memory_mask=None):
        """
        query   : [B, T, H]
        memory  : [B, T, H]
        memory_mask: [B, T] float, 1=valid 0=pad
        Returns : [B, T, H]
        """
        B, T, H = query.shape
        nh, dh = self.n_head, self.head_dim

        Q = self.q_proj(query).view(B, T, nh, dh).transpose(1, 2)   # [B,nh,T,dh]
        K = self.k_proj(memory).view(B, T, nh, dh).transpose(1, 2)
        V = self.v_proj(memory).view(B, T, nh, dh).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (dh ** 0.5)            # [B,nh,T,T]

        # 因果掩码: query_t 只看 memory_{<=t}
        causal = torch.tril(torch.ones(T, T, device=query.device)).bool()
        scores = scores.masked_fill(~causal, -1e4)

        # padding mask
        if memory_mask is not None:
            pad = (1.0 - memory_mask).bool()                         # [B, T]
            scores = scores.masked_fill(pad[:, None, None, :], -1e4)

        attn = self.drop(torch.softmax(scores, dim=-1))
        out  = (attn @ V).transpose(1, 2).reshape(B, T, H)
        return self.out_proj(out)


class DGABCrossAttnActor(nn.Module):
    """
    encoder-decoder DT Actor — 双 V/C query 版本.
    memory_t  = embed(s_t) + embed(a_{t-1}) + time
    q^V_t     = embed_V(rtg_v_t) + embed_state(s_t) + time   ← state 锚定防退化
    q^C_t     = embed_C(rtg_c_t) + embed_state(s_t) + time
    a_t       = head([attn^V ; attn^C ; s_res])
    """
    def __init__(self, state_dim, act_dim, hidden_size=64, max_ep_len=96,
                 time_dim=8, n_head=4, n_layer=3, block_config=None):
        super().__init__()
        self.state_dim   = state_dim
        self.act_dim     = act_dim
        self.hidden_size = hidden_size

        # memory stream
        self.embed_state  = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim,   hidden_size)
        self.embed_time   = nn.Embedding(max_ep_len, time_dim)
        self.trans_memory = nn.Linear(hidden_size * 2 + time_dim, hidden_size)

        # 双 query: V query 和 C query 各自独立
        self.embed_rtg_v  = nn.Linear(1, hidden_size)
        self.embed_rtg_c  = nn.Linear(1, hidden_size)
        self.embed_state_v = nn.Linear(state_dim, hidden_size)  # state 锚定
        self.embed_state_c = nn.Linear(state_dim, hidden_size)
        self.trans_query_v = nn.Linear(hidden_size + time_dim, hidden_size)
        self.trans_query_c = nn.Linear(hidden_size + time_dim, hidden_size)

        # memory self-attention encoder
        self.mem_encoder = nn.ModuleList(
            [Block(block_config) for _ in range(n_layer)])
        self.mem_ln = nn.LayerNorm(hidden_size)

        # 双路 cross-attention
        self.cross_attn_v = nn.ModuleList(
            [CausalCrossAttention(hidden_size, n_head) for _ in range(n_layer)])
        self.cross_attn_c = nn.ModuleList(
            [CausalCrossAttention(hidden_size, n_head) for _ in range(n_layer)])
        self.ln_v = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_layer)])
        self.ln_c = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_layer)])
        self.ffn_v = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_size),
                          nn.Linear(hidden_size, hidden_size * 4),
                          nn.GELU(),
                          nn.Linear(hidden_size * 4, hidden_size))
            for _ in range(n_layer)])
        self.ffn_c = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_size),
                          nn.Linear(hidden_size, hidden_size * 4),
                          nn.GELU(),
                          nn.Linear(hidden_size * 4, hidden_size))
            for _ in range(n_layer)])

        # state residual
        self.state_proj = nn.Linear(state_dim, hidden_size)

        # prediction heads: 输入 = [attn^V ; attn^C ; s_res] = 3H
        self.predict_action = nn.Linear(hidden_size * 3, act_dim)
        self.predict_beta   = nn.Sequential(
            nn.Linear(hidden_size * 3, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, rtg, states, actions, timesteps, attention_mask=None):
        """
        rtg   : [B, T, 2]  — [:,:,0]=rtg_v, [:,:,1]=rtg_c (已 z-score 归一化)
        """
        B, T = states.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=states.device)

        time_emb = self.embed_time(timesteps)   # [B,T,time_dim]

        # memory
        a_shifted = torch.cat([
            torch.zeros(B, 1, self.act_dim, device=states.device),
            actions[:, :-1]], dim=1)
        mem = self.trans_memory(torch.cat([
            self.embed_state(states),
            self.embed_action(a_shifted),
            time_emb], dim=-1))
        mem = self.mem_ln(mem)
        for block in self.mem_encoder:
            mem = block(mem, attention_mask)

        # V query: rtg_v + state + time
        rtg_v = rtg[:, :, 0:1]   # [B,T,1]
        rtg_c = rtg[:, :, 1:2]
        qv = self.trans_query_v(torch.cat([
            self.embed_rtg_v(rtg_v) + self.embed_state_v(states),
            time_emb], dim=-1))
        qc = self.trans_query_c(torch.cat([
            self.embed_rtg_c(rtg_c) + self.embed_state_c(states),
            time_emb], dim=-1))

        # 双路 cross-attn
        for cav, cac, lnv, lnc, ffnv, ffnc in zip(
                self.cross_attn_v, self.cross_attn_c,
                self.ln_v, self.ln_c,
                self.ffn_v, self.ffn_c):
            qv = qv + cav(lnv(qv), mem, attention_mask)
            qv = qv + ffnv(qv)
            qc = qc + cac(lnc(qc), mem, attention_mask)
            qc = qc + ffnc(qc)

        s_res = self.state_proj(states)
        fused = torch.cat([qv, qc, s_res], dim=-1)   # [B,T,3H]

        action_pred = self.predict_action(fused)
        beta_pred   = self.predict_beta(fused) + 0.5
        return action_pred, beta_pred


# =====================================================================
# Module 2.2a: SequenceCritic —— (s, a, r) autoregressive
#   与 Actor 的 (r, s, a) 对称: a_t 位置因因果mask看不到r_t, 预测 r_t
#   Actor:  "给定目标 r, 我该怎么做?" → 预测 a
#   Critic: "给定动作 a, 我能得到什么?" → 预测 r
# =====================================================================
class SequenceCritic(nn.Module):
    """
    Sequence Critic with (s, a, r) interleaving — mirrors Actor's (r, s, a).
    At a_t position the model sees s₀,a₀,r₀,...,s_t,a_t (but NOT r_t
    due to causal mask), which is exactly the context needed to predict r_t.

    r = dual RTG = (rtg_v, rtg_c), same 2D vector as Actor's input.
    Past r_{<t} are visible as legitimate historical context.
    """
    def __init__(self, state_dim, act_dim, hidden_size=64, max_ep_len=96,
                 time_dim=8, block_config=None):
        super().__init__()
        self.state_dim   = state_dim
        self.act_dim     = act_dim
        self.hidden_size = hidden_size

        # embeddings — mirrors DGABStackActor, order changed to (s, a, r)
        self.embed_state  = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_rtg    = nn.Linear(2, hidden_size)          # dual RTG
        self.embed_time   = nn.Embedding(max_ep_len, time_dim)

        self.trans_state  = nn.Linear(hidden_size + time_dim, hidden_size)
        self.trans_action = nn.Linear(hidden_size + time_dim, hidden_size)
        self.trans_rtg    = nn.Linear(hidden_size + time_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.transformer = nn.ModuleList([Block(block_config)
                                          for _ in range(block_config['n_layer'])])

        # prediction heads at action token position → predict V_remain / C_remain
        self.head_v = nn.Linear(hidden_size, 1)
        self.head_c = nn.Linear(hidden_size, 1)

    def forward(self, rtg, states, actions, timesteps, attention_mask=None):
        """
        rtg:            [B, T, 2]  — (rtg_v, rtg_c) for teacher forcing context
        states:         [B, T, state_dim]
        actions:        [B, T, act_dim]
        timesteps:      [B, T] long
        attention_mask: [B, T] float, 1=valid 0=pad
        Returns: (v_pred, c_pred) each [B, T, 1]
        """
        B, T = states.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.long, device=states.device)

        time_emb   = self.embed_time(timesteps)
        state_emb  = self.trans_state( torch.cat([self.embed_state(states),  time_emb], dim=-1))
        action_emb = self.trans_action(torch.cat([self.embed_action(actions), time_emb], dim=-1))
        rtg_emb    = self.trans_rtg(   torch.cat([self.embed_rtg(rtg),       time_emb], dim=-1))

        # 3 token/step: [s₀, a₀, r₀, s₁, a₁, r₁, ...]  ← (s, a, r) order
        stacked = torch.stack([state_emb, action_emb, rtg_emb], dim=1)  # [B,3,T,H]
        stacked = stacked.permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_size)
        stacked = self.embed_ln(stacked)

        stacked_mask = (attention_mask.unsqueeze(1).expand(-1, 3, -1)
                        .reshape(B, 3 * T).to(stacked.dtype))

        x = stacked
        for block in self.transformer:
            x = block(x, stacked_mask)

        x = x.reshape(B, T, 3, self.hidden_size).permute(0, 2, 1, 3)    # [B,3,T,H]

        # predict from action token position (index 1) — has seen s₀,a₀,r₀,...,s_t,a_t
        # but NOT r_t (causal mask blocks it) → predict r_t = (V_remain, C_remain)
        h = x[:, 1]                                                       # [B,T,H]
        v_pred = self.head_v(h)                                           # [B,T,1]
        c_pred = self.head_c(h)
        return v_pred, c_pred


# =====================================================================
# Module 2.2b: Legacy Dual-head per-step MLP Critic
# =====================================================================
class DualHeadCritic(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size),         nn.GELU(),
        )
        self.head_v = nn.Linear(hidden_size, 1)
        self.head_c = nn.Linear(hidden_size, 1)

    def forward(self, states, actions):
        h = self.trunk(torch.cat([states, actions], dim=-1))
        return self.head_v(h), self.head_c(h)               # 各 [B,T,1]


# =====================================================================
# Module 3+4: 训练主体 —— forward + 全局分数 + critic/actor loss
# =====================================================================
class DGAB(nn.Module):
    def __init__(self, base_state_dim, act_dim,
                 hidden_size=64, max_ep_len=96, time_dim=8, block_config=None,
                 tau_v=0.99, tau_c=0.05,
                 alpha=3.0,
                 lambda_critic=1.0, lambda_actor=1.0,
                 rtg_dropout=0.0,
                 learning_rate=1e-4, weight_decay=1e-4,
                 actor_type='stack',
                 critic_type='sequence',    # 'sequence'|'mlp'
                 obs_encoder_type='none',   # 'none'|'cls'|'bidformer'
                 obs_dim=9, macro_dim=2,    # obs列数; macro=norm(rl)维数
                 device='cpu'):
        super().__init__()
        # obs encoder → snapshot_dim=hidden_size; no encoder → snapshot_dim=0
        if obs_encoder_type == 'cls':
            self.obs_encoder = ClsTokenObsEncoder(obs_dim=obs_dim,
                                                  hidden_size=hidden_size)
            snap_dim = hidden_size
        elif obs_encoder_type == 'bidformer':
            self.obs_encoder = BidFormerEncoder(hidden_size=hidden_size,
                                                macro_dim=macro_dim)
            snap_dim = hidden_size
        else:
            self.obs_encoder = None
            snap_dim = 0
        self.obs_encoder_type = obs_encoder_type
        self.snap_dim   = snap_dim
        self.macro_dim  = macro_dim

        state_dim = base_state_dim + snap_dim
        self.state_dim = state_dim
        self.base_state_dim = base_state_dim
        self.act_dim = act_dim
        self.tau_v = tau_v
        self.tau_c = tau_c
        self.alpha = alpha
        self.lambda_critic = lambda_critic
        self.lambda_actor = lambda_actor
        self.rtg_dropout = rtg_dropout
        self.device = device

        if actor_type == 'cross_attn':
            n_layer = block_config['n_layer'] if block_config else 3
            n_head  = block_config['n_head']  if block_config else 4
            self.actor = DGABCrossAttnActor(
                state_dim, act_dim, hidden_size=hidden_size,
                max_ep_len=max_ep_len, time_dim=time_dim,
                n_head=n_head, n_layer=n_layer, block_config=block_config)
        else:
            self.actor = DGABStackActor(state_dim, act_dim, hidden_size=hidden_size,
                                 max_ep_len=max_ep_len, time_dim=time_dim,
                                 block_config=block_config)
        if critic_type == 'sequence':
            self.critic = SequenceCritic(
                state_dim, act_dim,
                hidden_size=hidden_size, max_ep_len=max_ep_len,
                time_dim=time_dim, block_config=block_config)
        else:
            self.critic = DualHeadCritic(state_dim, act_dim)
        self.critic_type = critic_type

        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.to(device)

    def _encode_obs(self, obs_padded, obs_mask, base_states):
        """
        Encode per-step impression sets and concatenate to base state.
        obs_padded  : [B, T, max_imp, obs_dim]
        obs_mask    : [B, T, max_imp]
        base_states : [B, T, base_state_dim]
        returns     : [B, T, state_dim]
        """
        if self.obs_encoder is None:
            return base_states
        B, T, M, _ = obs_padded.shape
        obs_flat  = obs_padded.reshape(B * T, M, -1)
        mask_flat = obs_mask.reshape(B * T, M)
        if self.obs_encoder_type == 'bidformer':
            macro = base_states[:, :, :self.macro_dim].reshape(B * T, self.macro_dim)
            snap  = self.obs_encoder(obs_flat, mask_flat, macro)
        else:
            snap  = self.obs_encoder(obs_flat, mask_flat)
        snap = snap.reshape(B, T, self.snap_dim)
        return torch.cat([base_states, snap], dim=-1)

    # ----------- Module 3: 全局分数 -----------
    def global_score(self, past_v, past_c, v_remain, c_remain, cpa_target):
        """所有入参形状 [B, T, 1] -> 返回 S [B, T, 1]
        cpa_target 是 per-episode (or per-step) 的 CPA 约束,从 batch 传入,
        允许不同 episode 使用不同 CPA.
        """
        V_total = past_v + v_remain
        C_total = past_c + c_remain
        cpa_total = C_total / (V_total + EPS)
        penalty = torch.clamp((cpa_target / (cpa_total + EPS)) ** CPA_PENALTY_BETA,
                              max=1.0)
        return V_total * penalty

    def forward(self, rtg, states, actions, timesteps, past_v, past_c, cpa_target,
                attention_mask=None, obs_padded=None, obs_mask=None):
        states = self._encode_obs(obs_padded, obs_mask, states) if obs_padded is not None else states
        action_pred, beta_pred = self.actor(rtg, states, actions, timesteps, attention_mask)
        explore_action = beta_pred * actions                            # ã = β · a (a 为离线真值)

        # critic 评估都 no_grad,避免 actor 反传时影响 critic
        with torch.no_grad():
            if self.critic_type == 'sequence':
                v_orig, c_orig = self.critic(rtg, states, actions, timesteps, attention_mask)
                v_expl, c_expl = self.critic(rtg, states, explore_action.detach(), timesteps, attention_mask)
            else:
                v_orig, c_orig = self.critic(states, actions)
                v_expl, c_expl = self.critic(states, explore_action.detach())

        s_orig = self.global_score(past_v, past_c, v_orig, c_orig, cpa_target)
        s_expl = self.global_score(past_v, past_c, v_expl, c_expl, cpa_target)

        return {'action_pred': action_pred, 'beta_pred': beta_pred,
                'explore_action': explore_action,
                's_orig': s_orig, 's_expl': s_expl}

    # ----------- Module 4.1: critic 损失 -----------
    def critic_loss(self, states, actions, explore_action, v_real, c_real, mask,
                    rtg=None, timesteps=None,
                    obs_padded=None, obs_mask=None):
        """
        双路 Critic 损失 (自动归一化):
          原始动作:  MSE —— 无偏点估计,精确预测实际动作的价值
          探索动作:  Expectile regression —— V 上界 (τ_V) / C 下界 (τ_C),
                     学乐观探索估计

        自动用 target 的 batch std 缩放,使 V/C 量级差异不影响梯度分配:
          loss = mean( (diff / σ)² )  — σ = std(target), detach
          与 Actor 的 adv_std 缩放同理,让各分量都变成无量纲 ≈ O(1).

        约定 diff = real - pred:
          V 上界 (τ_V=0.99): 欠估(diff>0) 重罚, 过估(diff<0) 轻罚 → 把 V 推高
          C 下界 (τ_C=0.05): 欠估(diff>0) 轻罚, 过估(diff<0) 重罚 → 把 C 压低
        """
        states = self._encode_obs(obs_padded, obs_mask, states) if obs_padded is not None else states
        m = mask > 0
        v_real_m = v_real[m]; c_real_m = c_real[m]

        # batch std 缩放因子 (detach, 不影响梯度方向)
        v_scale = v_real_m.detach().std().clamp(min=EPS)
        c_scale = c_real_m.detach().std().clamp(min=EPS)

        # RTG dropout: 随机 mask 历史 RTG token, 迫使 critic 从 (s,a) 推断价值
        if self.training and self.rtg_dropout > 0 and rtg is not None:
            B, T = rtg.shape[:2]
            keep_mask = (torch.rand(B, T, 1, device=rtg.device) > self.rtg_dropout).float()
            rtg = rtg * keep_mask

        # --- 原始动作: MSE ---
        if self.critic_type == 'sequence':
            v_pred, c_pred = self.critic(rtg, states, actions, timesteps, mask)
        else:
            v_pred, c_pred = self.critic(states, actions)
        v_pred_m = v_pred[m]; c_pred_m = c_pred[m]
        loss_v_mse = ((v_real_m - v_pred_m) / v_scale).pow(2).mean()
        loss_c_mse = ((c_real_m - c_pred_m) / c_scale).pow(2).mean()

        # --- 探索动作: expectile regression (V 上界 + C 下界) ---
        # explore_action.detach() 防止 Critic 梯度回传到 Actor
        if self.critic_type == 'sequence':
            v_pred_e, c_pred_e = self.critic(rtg, states, explore_action.detach(),
                                             timesteps, mask)
        else:
            v_pred_e, c_pred_e = self.critic(states, explore_action.detach())
        v_pred_e = v_pred_e[m]; c_pred_e = c_pred_e[m]

        diff_v = (v_real_m - v_pred_e) / v_scale
        w_v = torch.where(diff_v > 0,
                          torch.full_like(diff_v, self.tau_v),
                          torch.full_like(diff_v, 1.0 - self.tau_v))
        loss_v_expectile = (w_v * diff_v.pow(2)).mean()

        diff_c = (c_real_m - c_pred_e) / c_scale
        w_c = torch.where(diff_c > 0,
                          torch.full_like(diff_c, self.tau_c),
                          torch.full_like(diff_c, 1.0 - self.tau_c))
        loss_c_expectile = (w_c * diff_c.pow(2)).mean()

        loss_mse      = loss_v_mse + loss_c_mse
        loss_expectile = loss_v_expectile + loss_c_expectile
        return (loss_mse + loss_expectile,
                loss_v_mse.detach(), loss_c_mse.detach(),
                loss_v_expectile.detach(), loss_c_expectile.detach())

    # ----------- Module 4.2: actor 损失 -----------
    def actor_loss(self, action_pred, action_target, explore_action,
                   s_orig, s_expl, mask):
        m = mask > 0
        a_pred   = action_pred[m]
        a_target = action_target[m]
        a_expl   = explore_action.detach()[m]

        adv = (s_expl - s_orig)[m]                              # [N, 1]
        # 用 batch 内 std 做自适应缩放,让 alpha 成为无量纲温度
        adv_std = adv.detach().std().clamp(min=EPS)
        adv_normed = torch.clamp(adv / adv_std, min=-10.0, max=10.0)
        w = torch.sigmoid(self.alpha * adv_normed)

        # actor loss 也做归一化 (与 critic 一致: diff / std(target), detach)
        a_scale = a_target.detach().std().clamp(min=EPS)
        loss = ((1.0 - w) * ((a_pred - a_target) / a_scale).pow(2)
                +  w      * ((a_pred - a_expl)   / a_scale).pow(2)).mean()
        return loss, w.mean().detach(), w.std().detach(), adv_std.detach()

    # ----------- 训练 step -----------
    def step(self, batch):
        """
        batch 要求 (除 mask 外都在 device 上的 tensor):
            rtg:        [B,T,2]   = stack(rtg_v, rtg_c)
            states:     [B,T,base_state_dim]
            actions:    [B,T,act_dim]
            timesteps:  [B,T] long
            past_v:     [B,T,1]
            past_c:     [B,T,1]
            rtg_v:      [B,T,1]
            rtg_c:      [B,T,1]
            cpa_target: [B,T,1]
            mask:       [B,T]
            obs_padded: [B,T,max_imp,obs_dim]  (optional, R2/R3/R5)
            obs_mask:   [B,T,max_imp]          (optional)
        """
        obs_p = batch.get('obs_padded', None)
        obs_m = batch.get('obs_mask',   None)

        # 1) 编码 obs + 运行 Actor 获取 explore_action
        states = self._encode_obs(obs_p, obs_m, batch['states']) if obs_p is not None else batch['states']
        action_pred, beta_pred = self.actor(
            batch['rtg'], states, batch['actions'], batch['timesteps'], batch['mask'])
        explore_action = beta_pred * batch['actions']

        # 2) Critic 损失: 原始动作 MSE + 探索动作 expectile
        loss_critic, loss_v_mse, loss_c_mse, loss_v_exp, loss_c_exp = self.critic_loss(
            states, batch['actions'], explore_action,
            batch['rtg_v'], batch['rtg_c'], batch['mask'],
            rtg=batch['rtg'], timesteps=batch['timesteps'],
            obs_padded=None, obs_mask=None)  # obs 已在上面编码进 states

        # 3) Critic 评估 advantage (no_grad, 不影响 Critic 训练)
        with torch.no_grad():
            # RTG dropout 同样应用于 advantage 估计, 让 actor 在稀疏上下文下也能学
            rtg_adv = batch['rtg']
            if self.training and self.rtg_dropout > 0:
                B, T = rtg_adv.shape[:2]
                keep_mask = (torch.rand(B, T, 1, device=rtg_adv.device) > self.rtg_dropout).float()
                rtg_adv = rtg_adv * keep_mask
            if self.critic_type == 'sequence':
                v_orig, c_orig = self.critic(rtg_adv, states, batch['actions'],
                                             batch['timesteps'], batch['mask'])
                v_expl, c_expl = self.critic(rtg_adv, states, explore_action.detach(),
                                             batch['timesteps'], batch['mask'])
            else:
                v_orig, c_orig = self.critic(states, batch['actions'])
                v_expl, c_expl = self.critic(states, explore_action.detach())

        s_orig = self.global_score(batch['past_v'], batch['past_c'], v_orig, c_orig, batch['cpa_target'])
        s_expl = self.global_score(batch['past_v'], batch['past_c'], v_expl, c_expl, batch['cpa_target'])

        # 4) Actor 损失
        loss_actor, w_mean, w_std, adv_std = self.actor_loss(
            action_pred, batch['actions'], explore_action,
            s_orig, s_expl, batch['mask'])

        loss_total = self.lambda_critic * loss_critic + self.lambda_actor * loss_actor

        self.optimizer.zero_grad()
        loss_total.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.25)
        self.optimizer.step()

        return {
            'loss_total':     loss_total.item(),
            'loss_critic':    loss_critic.item(),
            'loss_v_mse':     loss_v_mse.item(),
            'loss_c_mse':     loss_c_mse.item(),
            'loss_v_exp':     loss_v_exp.item(),
            'loss_c_exp':     loss_c_exp.item(),
            'loss_actor':     loss_actor.item(),
            'w_mean':         w_mean.item(),
            'w_std':          w_std.item(),
            'adv_std':        adv_std.item(),
            's_orig_mean':    s_orig.mean().item(),
            's_expl_mean':    s_expl.mean().item(),
            'beta_mean':      beta_pred.mean().item(),
        }


# =====================================================================
# Module 5: 在线推理 —— RTG 自减
# =====================================================================
class DGABRollout:
    """
    Online rollout wrapper.
    rtg_v/c 用 / rtg_scale 固定缩放 (不再用 z-score), 与训练时量纲一致。
    base_state: norm(rl) [+ norm(stat)] — obs snapshot 由模型内部 _encode_obs 处理。
    """
    def __init__(self, model: DGAB, V_goal, C_target, K=20, scale=2000,
                 rtg_scale=1.0):
        self.model = model
        self.K = K
        self.scale = scale
        self.rtg_scale = rtg_scale
        self.device = model.device
        self.act_dim       = model.act_dim
        self.base_state_dim = model.base_state_dim

        v_init = V_goal / rtg_scale
        c_init = C_target * V_goal / rtg_scale
        self.rtg = torch.tensor([v_init, c_init],
                                dtype=torch.float32, device=self.device)
        self.rtgs, self.states, self.actions, self.timesteps = [], [], [], []
        self.obs_list = []   # per-step padded obs tensors [1, max_imp, obs_dim]
        self.t = 0

    @torch.no_grad()
    def act(self, state, obs_padded=None, obs_mask=None):
        """
        state      : np.array [base_state_dim]
        obs_padded : np.array [max_imp, obs_dim] or None
        obs_mask   : np.array [max_imp] or None
        """
        self.states.append(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        self.rtgs.append(self.rtg.clone())
        self.actions.append(torch.zeros(self.act_dim, device=self.device))
        self.timesteps.append(torch.tensor(self.t, dtype=torch.long, device=self.device))
        if obs_padded is not None:
            self.obs_list.append(torch.as_tensor(obs_padded, dtype=torch.float32,
                                                 device=self.device))

        rtg_seq    = self._pad(self.rtgs,    (2,))
        state_seq  = self._pad(self.states,  (self.base_state_dim,))
        action_seq = self._pad(self.actions, (self.act_dim,))
        time_seq   = self._pad(self.timesteps, ())
        mask       = self._mask(len(self.rtgs))

        # build obs batch if encoder is active
        if self.model.obs_encoder is not None and len(self.obs_list) > 0:
            M  = self.obs_list[-1].shape[0]
            od = self.obs_list[-1].shape[1]
            K  = self.K
            obs_seq  = torch.zeros(K, M, od, device=self.device)
            omask_seq = torch.zeros(K, M,    device=self.device)
            n = min(len(self.obs_list), K)
            for i, o in enumerate(self.obs_list[-n:]):
                t_idx = K - n + i
                m_i   = o.shape[0]
                m_use = min(m_i, M)
                obs_seq[t_idx, :m_use]   = o[:m_use]
                if obs_mask is not None and i == n - 1:
                    omask_seq[t_idx, :m_use] = torch.as_tensor(
                        obs_mask[:m_use], dtype=torch.float32, device=self.device)
                else:
                    omask_seq[t_idx, :m_use] = 1.0
            obs_b  = obs_seq.unsqueeze(0)    # [1, K, M, od]
            omask_b = omask_seq.unsqueeze(0) # [1, K, M]
        else:
            obs_b = omask_b = None

        # encode obs into state inline
        state_in = state_seq.unsqueeze(0)    # [1, K, base_state_dim]
        if obs_b is not None:
            state_in = self.model._encode_obs(obs_b, omask_b, state_in)

        action_pred, _ = self.model.actor(
            rtg_seq.unsqueeze(0), state_in,
            action_seq.unsqueeze(0), time_seq.long().unsqueeze(0),
            attention_mask=mask.unsqueeze(0))
        action = action_pred[0, -1]
        self.actions[-1] = action
        return action.cpu().numpy()

    def update_rtg(self, v_t, c_t):
        dv = v_t / self.rtg_scale
        dc = c_t / self.rtg_scale
        self.rtg = self.rtg - torch.tensor([dv, dc],
                                           dtype=torch.float32, device=self.device)
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
