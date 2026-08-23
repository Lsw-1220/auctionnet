"""AuctionNet adapter for the current DGAB shared actor checkpoint.

The checkpoint directory must contain actor.pt and normalize_dict.pkl.
"""
import os
import pickle
import numpy as np
import torch

from simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from .dgabshare.model import DGABActor, DGABRollout
from .dgabshare.shared_model import DGABSharedModel, DGABSharedRollout

EPS = 1e-10


class StateBuilder16:
    def __init__(self, num_steps=48):
        self.num_steps = num_steps
        self.bid_means, self.lwc_means = [], []
        self.conv_means, self.xi_means, self.pv_means, self.volumes = [], [], [], []

    @staticmethod
    def _mean(lst):      return float(np.mean(lst))      if lst else 0.0
    @staticmethod
    def _tail(lst, n=3): return float(np.mean(lst[-n:])) if lst else 0.0

    def build(self, t, remaining_budget, budget, pV):
        return np.array([
            (self.num_steps - t) / self.num_steps,
            remaining_budget / (budget + EPS),
            self._mean(self.bid_means),  self._tail(self.bid_means),
            self._mean(self.lwc_means),  self._mean(self.pv_means),
            self._mean(self.conv_means), self._mean(self.xi_means),
            self._tail(self.lwc_means),  self._tail(self.pv_means),
            self._tail(self.conv_means), self._tail(self.xi_means),
            float(np.mean(pV)) if len(pV) > 0 else 0.0,
            float(len(pV)),
            float(sum(self.volumes[-3:])),
            float(sum(self.volumes)),
        ], dtype=np.float32)

    def update(self, bids, lwc, tick_status, tick_conv, pV):
        self.bid_means.append(float(np.mean(bids))       if len(bids)       > 0 else 0.0)
        self.lwc_means.append(float(np.mean(lwc))        if len(lwc)        > 0 else 0.0)
        self.conv_means.append(float(np.mean(tick_conv)) if len(tick_conv)  > 0 else 0.0)
        self.xi_means.append(float(np.mean(tick_status)) if len(tick_status)> 0 else 0.0)
        self.pv_means.append(float(np.mean(pV))          if len(pV)         > 0 else 0.0)
        self.volumes.append(len(pV))


class DGABShareStrategy(BaseBiddingStrategy):

    def __init__(self, budget=100, cpa=2, category=1,
                 name="DGABShare", model_param=None):
        super().__init__(budget, name, cpa, category)
        if model_param is None:
            model_param = {}

        device = model_param.get('device', 'cpu')
        self.v_goal_multiplier = float(model_param.get('v_goal_multiplier', 1.0))
        if not np.isfinite(self.v_goal_multiplier) or self.v_goal_multiplier <= 0:
            raise ValueError("v_goal_multiplier must be finite and > 0")

        pkl_path = os.path.join(model_param['save_dir'], 'normalize_dict.pkl')
        with open(pkl_path, 'rb') as f:
            nd = pickle.load(f)
        self.state_mean = np.asarray(nd['state_mean'], dtype=np.float32)
        self.state_std  = np.asarray(nd['state_std'],  dtype=np.float32)
        self.action_mean = float(nd.get('action_mean', 0.0))
        self.action_std = float(nd.get('action_std', 1.0))
        self.action_upper = float(nd.get('action_upper', np.inf))
        if self.state_mean.ndim != 1 or self.state_std.shape != self.state_mean.shape:
            raise ValueError("invalid state normalization statistic shapes")
        if (not np.all(np.isfinite(self.state_mean)) or
                not np.all(np.isfinite(self.state_std)) or np.any(self.state_std <= 0)):
            raise ValueError("invalid state normalization statistics")

        self.log1p_sparse_dims = nd.get('log1p_sparse_dims', False)
        self.sparse_dims = nd.get('sparse_dims', [])
        self.log1p_scale = float(nd.get('log1p_scale', 1000.0))

        rtg_scale = float(nd.get('rtg_scale', 1.0))
        if not np.isfinite(rtg_scale) or rtg_scale <= 0:
            raise ValueError(f"invalid rtg_scale: {rtg_scale}")

        block_config = nd.get('block_config') or model_param.get('block_config')
        if block_config is None:
            raise ValueError('block_config missing from normalize_dict.pkl and model_param')
        state_dim = int(self.state_mean.shape[0])

        is_shared = nd.get('model_type') == 'shared_deterministic_linear_qv_awbc'
        model_class = DGABSharedModel if is_shared else DGABActor
        actor = model_class(
            state_dim=state_dim, act_dim=1,
            hidden_size=nd.get('hidden_size', model_param.get('hidden_size', 512)),
            max_ep_len=nd.get('max_ep_len', model_param.get('max_ep_len', 96)),
            time_dim=nd.get('time_dim', model_param.get('time_dim', 8)),
            block_config=block_config)
        ckpt = os.path.join(model_param['save_dir'],
                            model_param.get('ckpt_name', 'actor.pt'))
        actor.load_state_dict(torch.load(ckpt, map_location=device))
        actor.to(device)
        actor.eval()

        rollout_class = DGABSharedRollout if is_shared else DGABRollout
        self.rollout = rollout_class(
            actor,
            V_goal=self.v_goal_multiplier * budget / (cpa + EPS),
            C_goal=budget,
            K=model_param.get('K', 20),
            rtg_scale=rtg_scale,
            device=device,
        )
        self._builder = StateBuilder16(num_steps=48)

    def reset(self):
        self.remaining_budget = self.budget
        self._builder = StateBuilder16(num_steps=48)
        self.rollout.__init__(
            self.rollout.actor,
            V_goal=self.v_goal_multiplier * self.budget / (self.cpa + EPS),
            C_goal=self.budget,
            K=self.rollout.K,
            rtg_scale=self.rollout.rtg_scale,
            device=self.rollout.device,
        )

    def bidding(self, timeStepIndex, pValues, pValueSigmas,
                historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult,
                historyLeastWinningCost):

        if timeStepIndex == 0:
            self.reset()

        state_raw = self._builder.build(
            timeStepIndex, self.remaining_budget, self.budget, pValues)
        if self.log1p_sparse_dims:
            for d in self.sparse_dims:
                if state_raw[d] < 0:
                    raise ValueError(f"state dimension {d} is negative before log1p")
                state_raw[d] = np.log1p(state_raw[d] * self.log1p_scale)
        state_norm = (state_raw - self.state_mean) / self.state_std

        if historyImpressionResult:
            last_imp = np.asarray(historyImpressionResult[-1], dtype=np.float32)
            if last_imp.ndim == 3:
                last_imp = last_imp[0]
            v_prev = float(last_imp[:, -1].sum()) if last_imp.shape[1] > 0 else 0.0
            last_auc = np.asarray(historyAuctionResult[-1], dtype=np.float32)
            if last_auc.ndim == 3:
                last_auc = last_auc[0]
            c_prev = float((last_auc[:, 2] * last_imp[:, 0]).sum())
            self.rollout.update_rtg(v_prev, c_prev)

        action_norm = float(np.asarray(self.rollout.act(state_norm)).reshape(-1)[0])
        alpha = float(np.clip(action_norm * self.action_std + self.action_mean,
                              0.0, self.action_upper))
        rtg_now = self.rollout.rtg.detach().cpu().numpy()
        abs_state = np.abs(state_norm)
        self.last_diagnostics = {
            'alpha': alpha,
            'v_goal_multiplier': self.v_goal_multiplier,
            'rtg_v': float(rtg_now[0]),
            'rtg_c': float(rtg_now[1]),
            'state_z_abs_max': float(abs_state.max()),
            'state_z_abs_mean': float(abs_state.mean()),
            'state_z_gt5_count': int((abs_state > 5.0).sum()),
        }
        if historyBid:
            last_bid = np.asarray(historyBid[-1], dtype=np.float32)
            last_lwc = np.asarray(historyLeastWinningCost[-1], dtype=np.float32) \
                       if historyLeastWinningCost else np.zeros_like(last_bid)
            last_auc = np.asarray(historyAuctionResult[-1], dtype=np.float32)
            if last_auc.ndim == 3:
                last_auc = last_auc[0]
            tick_status = last_auc[:, 0]
            last_imp    = np.asarray(historyImpressionResult[-1], dtype=np.float32)
            if last_imp.ndim == 3:
                last_imp = last_imp[0]
            tick_conv = last_imp[:, -1]
            last_pv   = np.asarray(historyPValueInfo[-1], dtype=np.float32)
            if last_pv.ndim == 3:
                last_pv = last_pv[0]
            pv_vals = last_pv[:, 0]
            self._builder.update(last_bid, last_lwc, tick_status, tick_conv, pv_vals)

        return alpha * pValues


