"""
GAVE / DGAB-FO agent wrappers for AuctionNet online testing.

Each agent implements the AuctionNet BaseBiddingStrategy interface but
internally loads and runs an autobidding model checkpoint.

Usage in Controller (modify agent_list):
    from simul_bidding_env.strategy.autobidding_agents import (
        GAVEAuctionNetAgent, DGABFOAuctionNetAgent,
    )
    agent = GAVEAuctionNetAgent(
        budget=..., cpa=..., category=...,
        save_dir='saved_model/GAVE',
        device='cuda:0',
    )
"""
import os, sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Add autobidding project to path (adjust if needed)
_autobidding_root = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'autobidding'))
if os.path.isdir(_autobidding_root):
    sys.path.insert(0, _autobidding_root)

from simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy as AuctionNetBase

EPS = 1e-8

# ──────────────────────────────────────────────
# State builder (same 16-dim as train_data_generator)
# ──────────────────────────────────────────────

def _build_state_16(timeStepIndex, remaining_budget, budget,
                    pValues, historyBid, historyAuctionResult,
                    historyImpressionResult, historyLeastWinningCost,
                    historyPValueInfo):
    """
    Build the 16-dim FO state from AuctionNet-format per-PV history.

    AuctionNet history format (per advertiser, per tick):
      historyPValueInfo[t]      : (n_pv, 2)  [pValue, pValueSigma]
      historyBid[t]             : (n_pv,)    bids
      historyAuctionResult[t]   : (n_pv, 3)  [xi, slot, cost]
      historyImpressionResult[t]: (n_pv, 2)  [isExposed, conversionAction]
      historyLeastWinningCost[t]: (n_pv,)    lwc
    """
    num_steps = 48
    time_left   = (num_steps - timeStepIndex) / num_steps
    budget_left = remaining_budget / (budget + EPS)

    # Extract: AuctionNet has xi at col 0, not col 2
    history_xi   = [np.asarray(r)[:, 0] for r in historyAuctionResult]
    history_conv = [np.asarray(r)[:, 1] for r in historyImpressionResult]
    history_pv   = [np.asarray(r)[:, 0] for r in historyPValueInfo]

    def _mean(lst):
        return float(np.mean([np.mean(x) for x in lst])) if lst else 0.0

    def _tail(lst, n=3):
        t = lst[-n:]
        return float(np.mean([np.mean(x) for x in t])) if t else 0.0

    bid_mean  = _mean(historyBid)
    bid_tail  = _tail(historyBid)
    lwc_mean  = _mean(historyLeastWinningCost)
    pv_mean   = _mean(history_pv)
    conv_mean = _mean(history_conv)
    xi_mean   = _mean(history_xi)
    lwc_tail  = _tail(historyLeastWinningCost)
    pv_tail   = _tail(history_pv)
    conv_tail = _tail(history_conv)
    xi_tail   = _tail(history_xi)

    cur_pv_mean = float(np.mean(pValues)) if len(pValues) > 0 else 0.0
    cur_vol     = float(len(pValues))
    hist_vol    = float(sum(len(b) for b in historyBid)) if historyBid else 0.0
    tail_vol    = float(sum(len(historyBid[i])
                           for i in range(max(0, len(historyBid) - 3), len(historyBid)))
                       ) if historyBid else 0.0

    return np.array([
        time_left, budget_left,
        bid_mean, bid_tail,
        lwc_mean, pv_mean, conv_mean, xi_mean,
        lwc_tail, pv_tail, conv_tail, xi_tail,
        cur_pv_mean, cur_vol, tail_vol, hist_vol,
    ], dtype=np.float32)


# ──────────────────────────────────────────────
# GAVE Agent
# ──────────────────────────────────────────────

class GAVEAuctionNetAgent(AuctionNetBase):
    """
    GAVE (score-target DT) agent for AuctionNet online testing.

    model_param keys:
        save_dir:          path to model directory (contains complete_train.pt + normalize_dict.pkl)
        hidden_size:       int (default 512)
        time_dim:          int (default 8)
        block_config:      dict
        device:            str (default 'cpu')
        expectile:         float (default 0.99)
        score_target_mode: 'next' | 'prev' (default 'prev')
    """

    def __init__(self, budget=100, name="GAVE-AuctionNet", cpa=2, category=1,
                 model_param=None):
        super().__init__(budget=budget, name=name, cpa=cpa, category=category)
        if model_param is None:
            model_param = {}

        device = model_param.get('device', 'cpu')

        import torch
        import pickle
        from bidding_train_env.baseline.gave.model_fo import GAVE

        pkl_path = os.path.join(model_param['save_dir'], 'normalize_dict.pkl')
        with open(pkl_path, 'rb') as f:
            nd = pickle.load(f)
        state_mean = np.asarray(nd['state_mean'], dtype=np.float32)
        state_std  = np.asarray(nd['state_std'],  dtype=np.float32)

        self._gave_model = GAVE(
            state_dim=16, act_dim=1,
            hidden_size=model_param.get('hidden_size', 512),
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            learning_rate=model_param.get('learning_rate', 1e-4),
            time_dim=model_param.get('time_dim', 8),
            block_config=model_param['block_config'],
            expectile=model_param.get('expectile', 0.99),
            score_target_mode=model_param.get('score_target_mode', 'prev'),
        )
        ckpt = os.path.join(model_param['save_dir'],
                            model_param.get('ckpt_name', 'complete_train.pt'))
        self._gave_model.load_state_dict(torch.load(ckpt, map_location=device))
        self._gave_model.to(device)
        self._gave_model.eval()

        self._device = device
        self._prev_conv = None

    def reset(self):
        self.remaining_budget = self.budget
        self._prev_conv = None
        self._gave_model.init_eval()

    def bidding(self, timeStepIndex, pValues, pValueSigmas,
                historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult,
                historyLeastWinningCost):
        import torch

        if timeStepIndex == 0:
            self._gave_model.init_eval()
            self._prev_conv = None

        state = _build_state_16(
            timeStepIndex, self.remaining_budget, self.budget,
            pValues, historyBid, historyAuctionResult,
            historyImpressionResult, historyLeastWinningCost,
            historyPValueInfo)

        pre_reward = self._prev_conv

        alpha = float(np.asarray(
            self._gave_model.take_actions(
                state, budget=self.budget, cpa=self.cpa, pre_reward=pre_reward)
        ).reshape(-1)[0])

        # track prev conversion for next tick
        if historyImpressionResult:
            last = np.asarray(historyImpressionResult[-1], dtype=np.float32)
            self._prev_conv = float(last[:, 1].sum())  # col 1 = conversionAction
        else:
            self._prev_conv = 0.0

        return alpha * np.asarray(pValues, dtype=np.float64)


# ──────────────────────────────────────────────
# DGAB-FO Agent
# ──────────────────────────────────────────────

class DGABFOAuctionNetAgent(AuctionNetBase):
    """
    DGAB-FO agent for AuctionNet online testing.
    Uses StateBuilder16 pattern — state = 16-dim fully observable.

    model_param keys:
        save_dir:        path to model directory
        hidden_size:     int (default 512)
        max_ep_len:      int (default 96)
        time_dim:        int (default 8)
        block_config:    dict
        device:          str (default 'cpu')
        actor_type:      'stack' | 'cross_attn' (default 'stack')
        critic_type:     'sequence' | 'mlp' (default 'mlp')
        K:               int (default 20)
        critic_alpha:    float (default 1.0)
            1.0 = pure open-loop (original), 0.0 = pure Critic guidance
        rtg_v_cap:       float (default None)
            Cap V_goal to this value.  When None, uses budget/cpa (the
            original formula).  Set to a small number (e.g. 10.0) in
            sparse-PV environments so the RTG signal is realistic.
        pvalue_mean_base: float (default None)
            When provided, used to auto-cap V_goal proportionally to
            the PV density:  V_goal = min(budget/cpa, budget/cpa * sqrt(ratio))

    The following are read automatically from normalize_dict.pkl and
    do NOT need to be passed via model_param:
        rtg_scale:       RTG fixed divisor (replaces z-score normalization)
        log1p_sparse_dims / sparse_dims / log1p_scale:
                         log1p transform on PV/Conv dims before z-score
        sparse_v_credit: pseudo-V factor for zero-conversion steps
    """

    def __init__(self, budget=100, name="DGAB-FO-AuctionNet", cpa=2, category=1,
                 model_param=None) :
        super().__init__(budget=budget, name=name, cpa=cpa, category=category)
        if model_param is None:
            model_param = {}

        device = model_param.get('device', 'cpu')

        import torch
        import pickle
        from bidding_train_env.baseline.dgab.model_po import DGAB, DGABRollout

        pkl_path = os.path.join(model_param['save_dir'], 'normalize_dict.pkl')
        with open(pkl_path, 'rb') as f:
            nd = pickle.load(f)
        self._state_mean = np.asarray(nd['state_mean'], dtype=np.float32)
        self._state_std  = np.asarray(nd['state_std'],  dtype=np.float32)

        # ── Sparse state log1p transform (must match training preprocessing) ──
        # Training applies log1p(x * log1p_scale) BEFORE computing mean/std,
        # so saved state_mean/state_std are already in log1p space.
        self._log1p_sparse_dims = nd.get('log1p_sparse_dims', False)
        self._sparse_dims = nd.get('sparse_dims', [5, 6, 9, 10, 12])
        self._log1p_scale = float(nd.get('log1p_scale', 1000.0))

        # ── sparse_v_credit from training config ──
        self._sparse_v_credit = float(nd.get('sparse_v_credit', 0.0))

        # ── RTG scale (replaces z-score mean/std) ──
        self._rtg_scale = float(nd.get('rtg_scale', 1.0))

        block_config = model_param['block_config']
        critic_type = nd.get('critic_type', model_param.get('critic_type', 'mlp'))
        actor_type  = nd.get('actor_type',  model_param.get('actor_type', 'stack'))

        model = DGAB(
            base_state_dim=16, act_dim=1,
            hidden_size=model_param.get('hidden_size', 512),
            max_ep_len=model_param.get('max_ep_len', 96),
            time_dim=model_param.get('time_dim', 8),
            block_config=block_config,
            actor_type=actor_type,
            critic_type=critic_type,
            tau_v=model_param.get('tau_v', 0.99),
            tau_c=model_param.get('tau_c', 0.05),
            alpha=model_param.get('alpha', 3.0),
            lambda_critic=model_param.get('lambda_critic', 1.0),
            lambda_actor=model_param.get('lambda_actor', 1.0),
            learning_rate=model_param.get('learning_rate', 1e-4),
            weight_decay=model_param.get('weight_decay', 1e-4),
            device=device,
        )
        ckpt = os.path.join(model_param['save_dir'],
                            model_param.get('ckpt_name', 'complete_train.pt'))
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        model.eval()
        self._rollout = DGABRollout(
            model,
            V_goal=budget / (cpa + EPS),
            C_target=cpa,
            K=model_param.get('K', 20),
            scale=model_param.get('scale', 2000),
            rtg_scale=self._rtg_scale,
        )
        self._device = device
        self._budget = budget
        self._cpa = cpa
        self._critic_alpha = model_param.get('critic_alpha', 1.0)  # 1.0=off
        self._capture_attention = model_param.get('capture_attention', False)
        self._attn_records = [] if self._capture_attention else None

        # ── Sparse state normalization ──
        # Training applies log1p(x * log1p_scale) BEFORE computing mean/std,
        # so the saved state_mean/state_std already reflect log1p-space stats.
        # At inference we apply log1p to the raw state first, then z-score
        # with the saved mean/std — matching training exactly.
        self._log1p_enabled = self._log1p_sparse_dims  # always on if trained with it
        if self._log1p_enabled:
            logger.info(f'[DGAB log1p] enabled on dims {self._sparse_dims} '
                        f'(log1p_scale={self._log1p_scale})')
        else:
            logger.info('[DGAB log1p] disabled (model was not trained with log1p)')

        # ── Sparse-PV improvements ──
        # RTG V_goal capping: prevent wildly over-optimistic RTG in sparse env
        self._rtg_v_cap = model_param.get('rtg_v_cap', None)
        self._pvalue_mean_base = model_param.get('pvalue_mean_base', None)

        self._V_goal = self._compute_v_goal(budget, cpa)

        # Re-init rollout with capped V_goal
        self._rollout.__init__(
            self._rollout.model,
            V_goal=self._V_goal,
            C_target=self._cpa,
            K=self._rollout.K,
            scale=self._rollout.scale,
            rtg_scale=self._rtg_scale,
        )

        # Per-tick state tracking (for StateBuilder16 pattern)
        self._bid_means = []
        self._lwc_means = []
        self._conv_means = []
        self._xi_means = []
        self._pv_means = []
        self._volumes = []

    def reset(self):
        self.remaining_budget = self.budget
        self._bid_means = []
        self._lwc_means = []
        self._conv_means = []
        self._xi_means = []
        self._pv_means = []
        self._volumes = []
        self._rollout.__init__(
            self._rollout.model,
            V_goal=self._V_goal,
            C_target=self._cpa,
            K=self._rollout.K,
            scale=self._rollout.scale,
            rtg_scale=self._rtg_scale,
        )

    def _compute_v_goal(self, budget, cpa):
        """Compute a PV-density-aware V_goal instead of naive budget/cpa.

        In sparse environments (low pvalue_mean_base), budget/cpa yields
        an unrealistically high V_goal (e.g. 150 conversions when only ~0.4
        are achievable).  This causes RTG to stagnate at the initial value
        for the entire episode, leading to token collapse in the transformer.

        Strategy: only reduce V_goal when PV density is significantly below
        training density.  For PV at or above training levels, keep the
        original formula (the model works well there already).
        """
        naive_vgoal = budget / (cpa + EPS)

        # 1) Explicit cap from model_param
        if self._rtg_v_cap is not None:
            return min(naive_vgoal, self._rtg_v_cap)

        # 2) Auto-cap from pvalue_mean_base
        if self._pvalue_mean_base is not None:
            # state_mean[5] may be in log1p space if log1p was applied
            # during training — invert to get the raw training pv_mean.
            if self._log1p_enabled and 5 in self._sparse_dims:
                training_pv = (np.exp(self._state_mean[5]) - 1) / self._log1p_scale
            else:
                training_pv = float(self._state_mean[5])
            pv_ratio = self._pvalue_mean_base / (training_pv + EPS)

            if pv_ratio >= 1.0:
                # PV at or above training density — keep original V_goal
                # (the model already works well in this regime)
                return naive_vgoal
            else:
                # PV below training — scale V_goal down with sqrt ratio
                # e.g. pv_ratio=0.2 → scale=0.447 → V_goal=67
                #      pv_ratio=0.6 → scale=0.774 → V_goal=116
                #      pv_ratio=1.0 → scale=1.000 → V_goal=150 (unchanged)
                scale_factor = pv_ratio ** 0.5
                v_goal = max(naive_vgoal * scale_factor, 20.0)  # floor at 20
                v_goal = min(v_goal, naive_vgoal)
                logger.info(
                    f'[DGAB V_goal] pv={self._pvalue_mean_base:.6f} '
                    f'training_pv={training_pv:.6f} ratio={pv_ratio:.2f} '
                    f'naive={naive_vgoal:.1f} capped={v_goal:.1f}')
                return v_goal

        # 3) No cap — use original formula
        return naive_vgoal

    @staticmethod
    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    @staticmethod
    def _tail(lst, n=3):
        return float(np.mean(lst[-n:])) if lst else 0.0

    def _build_state(self, timeStepIndex, pValues):
        """Build 16-dim state like StateBuilder16.build()."""
        num_steps = 48
        return np.array([
            (num_steps - timeStepIndex) / num_steps,
            self.remaining_budget / (self._budget + EPS),
            self._mean(self._bid_means),  self._tail(self._bid_means),
            self._mean(self._lwc_means),  self._mean(self._pv_means),
            self._mean(self._conv_means), self._mean(self._xi_means),
            self._tail(self._lwc_means),  self._tail(self._pv_means),
            self._tail(self._conv_means), self._tail(self._xi_means),
            float(np.mean(pValues)) if len(pValues) > 0 else 0.0,
            float(len(pValues)),
            float(sum(self._volumes[-3:])),
            float(sum(self._volumes)),
        ], dtype=np.float32)

    def _update_state(self, historyBid, historyLeastWinningCost,
                      historyAuctionResult, historyImpressionResult,
                      historyPValueInfo):
        """Update per-tick aggregates from the latest history entry."""
        if not historyBid:
            return
        last_bid = np.asarray(historyBid[-1], dtype=np.float32)
        last_lwc = np.asarray(historyLeastWinningCost[-1], dtype=np.float32) \
                   if historyLeastWinningCost else np.zeros_like(last_bid)
        # xi at col 0 in AuctionNet format
        last_auc = np.asarray(historyAuctionResult[-1], dtype=np.float32)
        tick_status = last_auc[:, 0]   # xi
        last_imp = np.asarray(historyImpressionResult[-1], dtype=np.float32)
        tick_conv = last_imp[:, 1]     # conversionAction
        last_pv = np.asarray(historyPValueInfo[-1], dtype=np.float32)
        pv_vals = last_pv[:, 0]        # pValue

        self._bid_means.append(float(np.mean(last_bid)) if len(last_bid) > 0 else 0.0)
        self._lwc_means.append(float(np.mean(last_lwc)) if len(last_lwc) > 0 else 0.0)
        self._conv_means.append(float(np.mean(tick_conv)) if len(tick_conv) > 0 else 0.0)
        self._xi_means.append(float(np.mean(tick_status)) if len(tick_status) > 0 else 0.0)
        self._pv_means.append(float(np.mean(pv_vals)) if len(pv_vals) > 0 else 0.0)
        self._volumes.append(len(pv_vals))

    def _critic_rtg(self):
        """Run Critic to estimate remaining RTG from current step to end."""
        import torch
        if self._rollout.model.critic_type != 'sequence':
            return None
        rtg_list = self._rollout.rtgs
        if len(rtg_list) < 2:
            return None
        K = self._rollout.K
        rtgs = rtg_list[-K:]
        states = self._rollout.states[-K:]
        actions = self._rollout.actions[-K:]
        timesteps = self._rollout.timesteps[-K:]
        n = len(rtgs); pad = K - n
        if pad > 0:
            z2 = torch.zeros(2, device=self._device); z16 = torch.zeros(16, device=self._device)
            z1 = torch.zeros(1, device=self._device); zt = torch.tensor(0, device=self._device)
            rtgs = [z2.clone()] * pad + rtgs
            states = [z16.clone()] * pad + states
            actions = [z1.clone()] * pad + actions
            timesteps = [zt.clone()] * pad + timesteps
        rtg_seq = torch.stack(rtgs).unsqueeze(0)
        state_seq = torch.stack(states).unsqueeze(0)
        action_seq = torch.stack(actions).unsqueeze(0)
        time_seq = torch.tensor([int(t.item()) for t in timesteps], device=self._device).unsqueeze(0)
        mask = torch.cat([torch.zeros(pad, device=self._device),
                          torch.ones(n, device=self._device)]).unsqueeze(0)
        with torch.no_grad():
            v_pred, c_pred = self._rollout.model.critic(
                rtg_seq, state_seq, action_seq, time_seq, attention_mask=mask)
        return float(v_pred[0, -1, 0].item()), float(c_pred[0, -1, 0].item())

    def save_attention(self, filepath=None):
        """Save captured attention maps and print summary."""
        import torch as _t
        if not self._attn_records:
            logger.warning('No attention data captured.')
            return
        import os as _os
        path = filepath or _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                          '..', '..', 'exp_data', 'dgab_attention.pt')
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        _t.save(self._attn_records, path)
        n_layers = len(self._attn_records[0]['layers'])
        n_heads = self._attn_records[0]['layers'][0].shape[1]
        logger.info(f'Attention saved: {len(self._attn_records)} ticks × {n_layers} layers × {n_heads} heads → {path}')

    def bidding(self, timeStepIndex, pValues, pValueSigmas,
                historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult,
                historyLeastWinningCost):
        import torch

        if timeStepIndex == 0:
            self.reset()

        # Build 16-dim state
        state_raw = self._build_state(timeStepIndex, pValues)

        # ── State normalization (must match training pipeline) ──
        # Training: build raw state → log1p on sparse dims → z-score with saved mean/std
        # So saved mean/std are already in log1p space; we just apply log1p first.
        if self._log1p_enabled:
            for d in self._sparse_dims:
                state_raw[d] = np.log1p(state_raw[d] * self._log1p_scale)
        state_norm = (state_raw - self._state_mean) / self._state_std

        # Update RTG from previous tick
        v_prev, c_prev = 0.0, 0.0
        if historyImpressionResult:
            last_imp = np.asarray(historyImpressionResult[-1], dtype=np.float32)
            v_prev = float(last_imp[:, 1].sum())
            last_auc = np.asarray(historyAuctionResult[-1], dtype=np.float32)
            c_prev = float((last_auc[:, 2] * last_imp[:, 0]).sum())

            # ── RTG debug logging ──
            rtg_before = self._rollout.rtg.cpu().numpy()

            # ── Pseudo-V correction (sparse_v_credit from training) ──
            # NOTE: The reference DGABFOBiddingStrategy does NOT apply pseudo-V
            # at inference — it only uses actual v/c for open-loop RTG.
            # Pseudo-V was a training-time feature to create more informative
            # RTG labels.  At inference, the open-loop RTG self-corrects via
            # actual rewards.  Adding pseudo-V at inference can cause RTG_v
            # to decrease too slowly (pseudo_v << actual_v), keeping RTG_v
            # high and encouraging overspending in sparse environments.
            v_effective = v_prev

            # Critic guidance: blend RTG with Critic estimate FIRST, THEN subtract actual
            if self._critic_alpha < 1.0 and timeStepIndex >= 2:
                critic_result = self._critic_rtg()
                if critic_result is not None:
                    v_critic, c_critic = critic_result
                    a = self._critic_alpha
                    # 1) Blend: correct the accumulated RTG with Critic's re-estimate
                    rtg_cur = self._rollout.rtg.cpu().numpy()
                    v_blend = a * rtg_cur[0] + (1 - a) * v_critic
                    c_blend = a * rtg_cur[1] + (1 - a) * c_critic
                    self._rollout.rtg = torch.tensor(
                        [v_blend, c_blend], dtype=torch.float32, device=self._device)
                    # 2) THEN subtract actual (using v_effective for sparse correction)
                    self._rollout.update_rtg(v_effective, c_prev)
                    rtg_after = self._rollout.rtg.cpu().numpy()
                    logger.info(
                        f'[DGAB RTG tick={timeStepIndex:02d}] '
                        f'v_real={v_prev:.2f} v_eff={v_effective:.4f} c={c_prev:.2f} | '
                        f'critic=[{v_critic:.4f},{c_critic:.4f}] '
                        f'blend=[{v_blend:.4f},{c_blend:.4f}] '
                        f'sub=[{rtg_after[0]:.4f},{rtg_after[1]:.4f}] '
                        f'(α={a:.2f})')
                else:
                    self._rollout.update_rtg(v_effective, c_prev)
            else:
                # Pure open-loop: subtract with sparse correction
                self._rollout.update_rtg(v_effective, c_prev)
                rtg_after = self._rollout.rtg.cpu().numpy()
                logger.info(
                    f'[DGAB RTG tick={timeStepIndex:02d}] '
                    f'v_real={v_prev:.2f} v_eff={v_effective:.4f} c={c_prev:.2f} | '
                    f'rtg_before=[{rtg_before[0]:.4f},{rtg_before[1]:.4f}] '
                    f'rtg_after=[{rtg_after[0]:.4f},{rtg_after[1]:.4f}]')
        else:
            self._rollout.update_rtg(v_prev, c_prev)

        # Get alpha from model
        alpha = float(np.asarray(self._rollout.act(state_norm)).reshape(-1)[0])

        # Update state tracker for next tick
        self._update_state(historyBid, historyLeastWinningCost,
                          historyAuctionResult, historyImpressionResult,
                          historyPValueInfo)

        # ── Capture attention maps ──
        if self._capture_attention:
            try:
                layer_attns = {}
                for i, block in enumerate(self._rollout.model.actor.transformer):
                    if hasattr(block.attn, '_attn_map'):
                        attn = block.attn._attn_map.detach().cpu()  # [1, n_head, T, T]
                        layer_attns[i] = attn
                self._attn_records.append({'tick': timeStepIndex, 'layers': layer_attns})
            except Exception:
                pass

        return alpha * np.asarray(pValues, dtype=np.float64)


# ──────────────────────────────────────────────
# Decision Transformer Agent
# ──────────────────────────────────────────────

class DTAuctionNetAgent(AuctionNetBase):
    """
    Decision Transformer agent for AuctionNet online testing.

    model_param keys:
        save_dir:          path to model directory (contains dt.pt + normalize_dict.pkl)
        device:            str (default 'cpu')
        target_return:     float (default 4.0)
        scale:             float (default 2000)
        K:                 int (default 10)
        max_ep_len:        int (default 96)
    """

    def __init__(self, budget=100, name="DT-AuctionNet", cpa=2, category=1,
                 model_param=None):
        super().__init__(budget=budget, name=name, cpa=cpa, category=category)
        if model_param is None:
            model_param = {}

        device = model_param.get('device', 'cpu')

        import torch
        import pickle
        import importlib.util
        # DT lives in AuctionNet's strategy_train_env, not autobidding.
        # Load by absolute path to avoid sys.path shadowing from autobidding root.
        _auctionnet_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..'))
        _dt_path = os.path.join(_auctionnet_root, 'strategy_train_env',
                                'bidding_train_env', 'baseline', 'dt', 'dt.py')
        _spec = importlib.util.spec_from_file_location('_dt_module', _dt_path)
        _dt_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_dt_module)
        DecisionTransformer = _dt_module.DecisionTransformer

        pkl_path = os.path.join(model_param['save_dir'], 'normalize_dict.pkl')
        with open(pkl_path, 'rb') as f:
            nd = pickle.load(f)
        state_mean = np.asarray(nd['state_mean'], dtype=np.float32)
        state_std  = np.asarray(nd['state_std'],  dtype=np.float32)

        # Convert to tensors on target device so that init_eval() can read
        # .device from them.  (Plain numpy arrays are not moved by .to().)
        state_mean_t = torch.tensor(state_mean, device=device)
        state_std_t  = torch.tensor(state_std,  device=device)

        self._dt_model = DecisionTransformer(
            state_dim=16, act_dim=1,
            state_mean=state_mean_t,
            state_std=state_std_t,
            action_tanh=False,
            K=model_param.get('K', 10),
            max_ep_len=model_param.get('max_ep_len', 96),
            scale=model_param.get('scale', 2000),
            target_return=model_param.get('target_return', 4),
        )
        ckpt = os.path.join(model_param['save_dir'],
                            model_param.get('ckpt_name', 'dt.pt'))
        self._dt_model.load_net(ckpt, device=device)
        self._dt_model.to(device)
        self._dt_model.device = device

        self._device = device
        self._target_return = model_param.get('target_return', 4)

    def reset(self):
        self.remaining_budget = self.budget
        self._dt_model.init_eval()

    def bidding(self, timeStepIndex, pValues, pValueSigmas,
                historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult,
                historyLeastWinningCost):
        if timeStepIndex == 0:
            self._dt_model.init_eval()

        state = _build_state_16(
            timeStepIndex, self.remaining_budget, self.budget,
            pValues, historyBid, historyAuctionResult,
            historyImpressionResult, historyLeastWinningCost,
            historyPValueInfo)

        pre_reward = None
        if historyImpressionResult:
            last = np.asarray(historyImpressionResult[-1], dtype=np.float32)
            pre_reward = float(last[:, 1].sum())

        alpha = float(np.asarray(
            self._dt_model.take_actions(
                state,
                target_return=self._target_return,
                pre_reward=pre_reward)
        ).reshape(-1)[0])

        return alpha * np.asarray(pValues, dtype=np.float64)
