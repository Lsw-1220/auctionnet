"""
DGABFusionBiddingStrategy — Plan B dual-encoder ensemble inference.

Loads a trained DGABFusion checkpoint (complete_train.pt + normalize_dict.pkl).
At inference time, the heuristic gate automatically selects the blending ratio
based on the current window's pv_mean.

model_param fields:
    save_dir          : model directory (contains complete_train.pt and normalize_dict.pkl)
    hidden_size       : int (default 512)
    max_ep_len        : int (default 96)
    time_dim          : int (default 8)
    block_config      : dict
    device            : str (default 'cpu')
    max_imp           : int (default 512)
"""
import os
import pickle
import numpy as np
import torch

from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from bidding_train_env.baseline.dgab.model_fusion import DGABFusion, DGABFusionRollout

EPS = 1e-8

_OBS_PV, _OBS_SIGMA, _OBS_XI = 0, 1, 2
_OBS_COST, _OBS_EXP, _OBS_CONV = 4, 5, 6
_OBS_BID, _OBS_LWC = 7, 8
STAT_DIM = 16
RL_DIM = 2


def _compute_stat_single(obs: np.ndarray) -> np.ndarray:
    """16-dim window statistics (same as dgab_bidding_strategy.py)."""
    stat = np.zeros(STAT_DIM, dtype=np.float32)
    n = obs.shape[0]
    if n == 0:
        return stat
    has_bid_lwc = obs.shape[1] >= 9
    pv = obs[:, _OBS_PV]
    sigma = obs[:, _OBS_SIGMA]
    xi = obs[:, _OBS_XI]
    cost = obs[:, _OBS_COST]
    exp = obs[:, _OBS_EXP]
    conv = obs[:, _OBS_CONV]
    bid = obs[:, _OBS_BID] if has_bid_lwc else np.zeros(n, np.float32)
    lwc = obs[:, _OBS_LWC] if has_bid_lwc else np.zeros(n, np.float32)
    won = xi > 0.5
    cost_won = cost[won]
    stat[0] = np.log(n + 1.0)
    stat[1] = pv.mean()
    stat[2] = pv.std()
    stat[3] = np.percentile(pv, 75)
    stat[4] = np.percentile(pv, 90)
    stat[5] = sigma.mean()
    stat[6] = xi.mean()
    stat[7] = exp.mean()
    stat[8] = cost_won.mean() if won.any() else 0.0
    stat[9] = np.percentile(cost_won, 90) if won.any() else 0.0
    stat[10] = lwc.mean()
    stat[11] = np.percentile(lwc, 90)
    stat[12] = np.mean(bid / (lwc + EPS))
    stat[13] = conv.sum() / (exp.sum() + EPS)
    return stat


class DGABFusionBiddingStrategy(BaseBiddingStrategy):
    """DGAB-Fusion (Plan B) bidding strategy for offline evaluation."""

    def __init__(self, budget=100, cpa=2, category=1,
                 name="DGAB-Fusion-Strategy", model_param=None):
        super().__init__(budget, name, cpa, category)
        if model_param is None:
            model_param = {}

        device = model_param.get('device', 'cpu')

        # ── Load normalize dict ──
        pkl_path = os.path.join(model_param['save_dir'], 'normalize_dict.pkl')
        with open(pkl_path, 'rb') as f:
            nd = pickle.load(f)

        # Base state config
        base_state_dim = nd.get('base_state_dim', RL_DIM + STAT_DIM)
        obs_dim = nd.get('obs_dim', 9)
        critic_type = nd.get('critic_type', 'sequence')

        self.rl_mean = np.asarray(nd['resourceleft_mean'], dtype=np.float32)
        self.rl_std = np.asarray(nd['resourceleft_std'], dtype=np.float32)
        self.stat_mean = np.asarray(nd.get('stat_mean', np.zeros(STAT_DIM)),
                                    dtype=np.float32)
        self.stat_std = np.asarray(nd.get('stat_std', np.ones(STAT_DIM)),
                                   dtype=np.float32)
        self.use_stat = True  # Fusion always uses stat
        self.max_imp = model_param.get('max_imp', 512)
        self.obs_dim = obs_dim

        # Log1p config (from training)
        self.log1p_stat_dims = nd.get('log1p_stat_dims', False)
        self.sparse_stat_dims = nd.get('sparse_stat_dims', [])
        self.log1p_scale = float(nd.get('log1p_scale', 1000.0))

        scale = int(nd.get('scale', 2000))
        rtg_scale = float(nd.get('rtg_scale', scale))

        # ── Build model ──
        block_config = model_param['block_config']

        # Gate config (from training normalize_dict or model_param)
        gate_temperature = model_param.get('gate_temperature',
                                           nd.get('gate_temperature', 1.0))

        model = DGABFusion(
            base_state_dim=base_state_dim, act_dim=1,
            hidden_size=model_param['hidden_size'],
            max_ep_len=model_param.get('max_ep_len', 96),
            time_dim=model_param.get('time_dim', 8),
            block_config=block_config,
            critic_type=critic_type,
            obs_dim=obs_dim,
            macro_dim=RL_DIM,
            device=device,
            gate_temperature=gate_temperature,
        )

        ckpt = os.path.join(model_param['save_dir'], 'complete_train.pt')
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        # Set gate parameters from training (needed for inference)
        model._stat_mean_1 = float(nd.get('stat_mean_1',
                                          float(self.stat_mean[1])))
        model._stat_std_1 = float(nd.get('stat_std_1',
                                         float(self.stat_std[1])))

        V_goal = model_param.get('V_goal', budget / (cpa + EPS)) * 2
        C_target = cpa
        self.rollout = DGABFusionRollout(
            model, V_goal=V_goal, C_target=C_target,
            K=model_param.get('K', 20), scale=scale,
            rtg_scale=rtg_scale)

        self._prev_stat = np.zeros(STAT_DIM, dtype=np.float32)

    def reset(self):
        self.remaining_budget = self.budget
        self._prev_stat = np.zeros(STAT_DIM, dtype=np.float32)

    def _build_obs_array(self, historyAuctionResult, historyBid,
                         historyLeastWinningCost):
        if not historyAuctionResult:
            return np.zeros((1, 9), dtype=np.float32)
        auc = np.asarray(historyAuctionResult[-1], dtype=np.float32)
        if auc.ndim == 3:
            auc = auc[0]
        n = auc.shape[0]
        bid = (np.asarray(historyBid[-1], dtype=np.float32).reshape(-1)
               if historyBid else np.zeros(n, np.float32))
        lwc = (np.asarray(historyLeastWinningCost[-1], dtype=np.float32).reshape(-1)
               if historyLeastWinningCost else np.zeros(n, np.float32))
        obs = np.concatenate([auc, bid.reshape(-1, 1), lwc.reshape(-1, 1)], axis=1)
        return obs.astype(np.float32)

    def _build_state(self, timeStepIndex, historyAuctionResult,
                     historyBid, historyLeastWinningCost):
        time_left = (48 - timeStepIndex) / 48.0
        budget_left = self.remaining_budget / (self.budget + EPS)
        rl = np.array([time_left, budget_left], dtype=np.float32)
        rl_norm = (rl - self.rl_mean) / self.rl_std

        obs = self._build_obs_array(historyAuctionResult, historyBid,
                                    historyLeastWinningCost)
        stat = _compute_stat_single(obs)
        stat[14] = stat[1] - self._prev_stat[1]
        stat[15] = stat[6] - self._prev_stat[6]
        self._prev_stat = stat.copy()

        if self.log1p_stat_dims:
            for d in self.sparse_stat_dims:
                stat[d] = np.log1p(stat[d] * self.log1p_scale)

        stat_norm = (stat - self.stat_mean) / self.stat_std
        return np.concatenate([rl_norm, stat_norm])

    def _build_obs_padded(self, historyAuctionResult, historyBid,
                          historyLeastWinningCost):
        obs = self._build_obs_array(historyAuctionResult, historyBid,
                                    historyLeastWinningCost)
        M = self.max_imp
        od = self.obs_dim
        if obs.shape[1] < od:
            obs = np.pad(obs, ((0, 0), (0, od - obs.shape[1])))
        n_use = min(obs.shape[0], M)
        obs_pad = np.zeros((M, od), dtype=np.float32)
        obs_mask = np.zeros(M, dtype=np.float32)
        obs_pad[:n_use] = obs[:n_use]
        obs_mask[:n_use] = 1.0
        return obs_pad, obs_mask

    def bidding(self, timeStepIndex, pValues, pValueSigmas,
                historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult,
                historyLeastWinningCost):
        if timeStepIndex == 0:
            self.rollout.__init__(
                self.rollout.model,
                V_goal=self.budget / (self.cpa + EPS),
                C_target=self.cpa,
                K=self.rollout.K,
                scale=self.rollout.scale,
                rtg_scale=self.rollout.rtg_scale,
            )
            self._prev_stat = np.zeros(STAT_DIM, dtype=np.float32)

        state = self._build_state(timeStepIndex, historyAuctionResult,
                                  historyBid, historyLeastWinningCost)

        if historyImpressionResult:
            last_imp = np.asarray(historyImpressionResult[-1], dtype=np.float32)
            if last_imp.ndim == 3:
                last_imp = last_imp[0]
            v_prev = float(last_imp[:, -1].sum()) if last_imp.shape[1] > 0 else 0.0
            last_auc = np.asarray(historyAuctionResult[-1], dtype=np.float32)
            if last_auc.ndim == 3:
                last_auc = last_auc[0]
            c_prev = float(last_auc[:, _OBS_COST].sum())
            self.rollout.update_rtg(v_prev, c_prev)

        # Fusion always uses obs encoder (both cls and bidformer need obs)
        obs_pad, obs_mask = self._build_obs_padded(
            historyAuctionResult, historyBid, historyLeastWinningCost)
        alpha = float(self.rollout.act(state, obs_padded=obs_pad, obs_mask=obs_mask))

        return alpha * pValues
