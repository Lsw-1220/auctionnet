"""
VGABStrategy — user-trained DGABActor (Gaussian head) plugged into AuctionNet.

Runs the Decision-Transformer actor inside AuctionNet's simulation using the
AuctionNet history format (xi at col 0, cost at col 2, conversion at col 1 of
impressionResult).  Pure open-loop RTG (no critic guidance) — the actor is a
goal-conditioned policy conditioned on the RTG tokens.

model_param keys:
    save_dir:      model dir containing actor.pt + normalize_dict.pkl
    hidden_size:   int (default from normalize_dict)
    max_ep_len:    int (default from normalize_dict)
    time_dim:      int (default from normalize_dict)
    block_config:  dict (default from normalize_dict)
    K:             context window (default 20)
    device:        str
    rtg_v_cap:     explicit cap on V_goal (optional)
    pvalue_mean_base: float — PV-density-aware V_goal scaling
"""
import os
import pickle
import logging
import numpy as np
import torch

from simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from simul_bidding_env.strategy.vgab.model import DGABActor, DGABRollout

logger = logging.getLogger(__name__)
EPS = 1e-8


class VGABStrategy(BaseBiddingStrategy):
    def __init__(self, budget=100, name="VGAB-PlayerStrategy", cpa=2, category=1,
                 model_param=None):
        super().__init__(budget, name, cpa, category)
        if model_param is None:
            model_param = {}

        device = model_param.get('device', 'cpu')

        pkl_path = os.path.join(model_param['save_dir'], 'normalize_dict.pkl')
        with open(pkl_path, 'rb') as f:
            nd = pickle.load(f)
        self._state_mean = np.asarray(nd['state_mean'], dtype=np.float32)
        self._state_std  = np.asarray(nd['state_std'],  dtype=np.float32)
        self._action_mean = float(nd.get('action_mean', 0.0))
        self._action_std  = float(nd.get('action_std', 1.0))
        self._action_upper = float(nd.get('action_upper', np.inf))
        self._rtg_scale = float(nd.get('rtg_scale', 1.0))

        block_config = nd.get('block_config') or model_param.get('block_config')
        hidden_size  = nd.get('hidden_size', model_param.get('hidden_size', 512))
        max_ep_len   = nd.get('max_ep_len', model_param.get('max_ep_len', 96))
        time_dim     = nd.get('time_dim', model_param.get('time_dim', 8))

        actor = DGABActor(
            state_dim=16, act_dim=1,
            hidden_size=hidden_size,
            max_ep_len=max_ep_len,
            time_dim=time_dim,
            block_config=block_config,
        )
        ckpt = os.path.join(model_param['save_dir'],
                            model_param.get('ckpt_name', 'actor.pt'))
        actor.load_state_dict(torch.load(ckpt, map_location=device))
        actor.to(device)
        actor.eval()

        self._device = device
        self._budget = budget
        self._cpa = cpa
        self._rtg_v_cap = model_param.get('rtg_v_cap', None)
        self._pvalue_mean_base = model_param.get('pvalue_mean_base', None)
        self._V_goal = self._compute_v_goal(budget, cpa)

        self._rollout = DGABRollout(
            actor,
            V_goal=self._V_goal,
            C_goal=budget,
            K=model_param.get('K', 20),
            rtg_scale=self._rtg_scale,
            device=device,
        )

        # Per-tick state tracking (StateBuilder16 pattern)
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
            self._rollout.actor,
            V_goal=self._V_goal,
            C_goal=self._budget,
            K=self._rollout.K,
            rtg_scale=self._rtg_scale,
            device=self._device,
        )

    def _compute_v_goal(self, budget, cpa):
        """PV-density-aware V_goal (same logic as DGABFOAuctionNetAgent)."""
        naive_vgoal = budget / (cpa + EPS)

        if self._rtg_v_cap is not None:
            return min(naive_vgoal, self._rtg_v_cap)

        if self._pvalue_mean_base is not None:
            training_pv = float(self._state_mean[5])   # no log1p in this model
            pv_ratio = self._pvalue_mean_base / (training_pv + EPS)
            if pv_ratio >= 1.0:
                return naive_vgoal
            scale_factor = pv_ratio ** 0.5
            v_goal = max(naive_vgoal * scale_factor, 20.0)
            v_goal = min(v_goal, naive_vgoal)
            logger.info(f'[VGAB V_goal] pv={self._pvalue_mean_base:.6f} '
                        f'training_pv={training_pv:.6f} ratio={pv_ratio:.2f} '
                        f'naive={naive_vgoal:.1f} capped={v_goal:.1f}')
            return v_goal

        return naive_vgoal

    @staticmethod
    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    @staticmethod
    def _tail(lst, n=3):
        return float(np.mean(lst[-n:])) if lst else 0.0

    def _build_state(self, timeStepIndex, pValues):
        """16-dim FO state (StateBuilder16 pattern)."""
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
        """Accumulate per-tick aggregates from AuctionNet-format history."""
        if not historyBid:
            return
        last_bid = np.asarray(historyBid[-1], dtype=np.float32)
        last_lwc = np.asarray(historyLeastWinningCost[-1], dtype=np.float32) \
                   if historyLeastWinningCost else np.zeros_like(last_bid)
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

        # 1) Build raw 16-dim state and z-score it
        state_raw = self._build_state(timeStepIndex, pValues)
        state_norm = (state_raw - self._state_mean) / self._state_std

        # 2) Update RTG from the previous tick (open-loop).
        #    只在有历史时调用（tick>=1）：update_rtg 会让时间步 +1，若在 tick 0 也调用，
        #    tick 47 时 self.t 会到 48，超出 embed_time 的 max_ep_len=48 → index out of range。
        if historyImpressionResult:
            last_imp = np.asarray(historyImpressionResult[-1], dtype=np.float32)
            v_prev = float(last_imp[:, 1].sum())                       # conversions
            last_auc = np.asarray(historyAuctionResult[-1], dtype=np.float32)
            c_prev = float((last_auc[:, 2] * last_imp[:, 0]).sum())    # cost × isExposed
            self._rollout.update_rtg(v_prev, c_prev)

        # 3) Actor inference → normalized action, then de-normalize to alpha
        action_norm = float(np.asarray(self._rollout.act(state_norm)).reshape(-1)[0])
        alpha = float(np.clip(action_norm * self._action_std + self._action_mean,
                              0.0, self._action_upper))

        # 4) Update state tracker for the next tick
        self._update_state(historyBid, historyLeastWinningCost,
                          historyAuctionResult, historyImpressionResult,
                          historyPValueInfo)

        return alpha * np.asarray(pValues, dtype=np.float64)
