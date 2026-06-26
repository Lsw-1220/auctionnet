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
import numpy as np

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
        save_dir:     path to model directory
        hidden_size:  int (default 512)
        max_ep_len:   int (default 96)
        time_dim:     int (default 8)
        block_config: dict
        device:       str (default 'cpu')
        actor_type:   'stack' | 'cross_attn' (default 'stack')
        critic_type:  'sequence' | 'mlp' (default 'mlp')
        K:            int (default 20)
    """

    def __init__(self, budget=100, name="DGAB-FO-AuctionNet", cpa=2, category=1,
                 model_param=None):
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
            V_goal=budget / (cpa + EPS) * 0.5,
            C_target=cpa,
            K=model_param.get('K', 20),
            scale=model_param.get('scale', 2000),
        )
        self._device = device
        self._budget = budget
        self._cpa = cpa

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
            V_goal=self._budget / (self._cpa + EPS) ,
            C_target=self._cpa,
            K=self._rollout.K,
            scale=self._rollout.scale,
        )

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

    def bidding(self, timeStepIndex, pValues, pValueSigmas,
                historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult,
                historyLeastWinningCost):
        import torch

        if timeStepIndex == 0:
            self.reset()

        # Build 16-dim state
        state_raw = self._build_state(timeStepIndex, pValues)
        state_norm = (state_raw - self._state_mean) / self._state_std

        # Update RTG from previous tick
        v_prev, c_prev = 0.0, 0.0
        if historyImpressionResult:
            last_imp = np.asarray(historyImpressionResult[-1], dtype=np.float32)
            v_prev = float(last_imp[:, 1].sum())  # conversionAction
            last_auc = np.asarray(historyAuctionResult[-1], dtype=np.float32)
            # cost_pit * is_exposed_pit = 实际扣费 (只有曝光的广告位才真正扣钱)
            c_prev = float((last_auc[:, 2] * last_imp[:, 0]).sum())
            if v_prev == 0.0 and c_prev > 0:
                expected_v_loss = c_prev / (self.cpa + EPS)
                # 扣减一半的期望转化，平滑过渡，防止网络崩溃
                v_prev = expected_v_loss * 0.5
            self._rollout.update_rtg(v_prev, c_prev)

        # Get alpha from model
        alpha = float(np.asarray(self._rollout.act(state_norm)).reshape(-1)[0])

        # Update state tracker for next tick
        self._update_state(historyBid, historyLeastWinningCost,
                          historyAuctionResult, historyImpressionResult,
                          historyPValueInfo)

        return alpha * np.asarray(pValues, dtype=np.float64)
