import json
import os
import pickle

import numpy as np
import torch

from simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from simul_bidding_env.strategy.dgabshare.shared_model import (
    DGABSharedModel, DGABSharedRollout)
from .model import ConversionRTGModel, ConversionRTGRollout

EPS = 1e-10


class StateBuilder16:
    def __init__(self, num_steps=48):
        self.num_steps = num_steps
        self.bid_means, self.lwc_means = [], []
        self.conv_means, self.xi_means, self.pv_means, self.volumes = [], [], [], []

    @staticmethod
    def mean(values):
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def tail(values, n=3):
        return float(np.mean(values[-n:])) if values else 0.0

    def build(self, tick, remaining, budget, pvalues):
        return np.asarray([
            (self.num_steps - tick) / self.num_steps, remaining / (budget + EPS),
            self.mean(self.bid_means), self.tail(self.bid_means),
            self.mean(self.lwc_means), self.mean(self.pv_means),
            self.mean(self.conv_means), self.mean(self.xi_means),
            self.tail(self.lwc_means), self.tail(self.pv_means),
            self.tail(self.conv_means), self.tail(self.xi_means),
            float(np.mean(pvalues)) if len(pvalues) else 0.0, float(len(pvalues)),
            float(sum(self.volumes[-3:])), float(sum(self.volumes)),
        ], dtype=np.float32)

    def update(self, bids, lwc, status, conversion, pvalues):
        for target, values in ((self.bid_means, bids), (self.lwc_means, lwc),
                               (self.conv_means, conversion),
                               (self.xi_means, status),
                               (self.pv_means, pvalues)):
            target.append(float(np.mean(values)) if len(values) else 0.0)
        self.volumes.append(len(pvalues))


class DGABAblationStrategy(BaseBiddingStrategy):
    """Adapter for v0-v5 DGAB ablation checkpoints."""

    def __init__(self, budget=100, cpa=2, category=1, name="DGAB-Ablation",
                 model_dir=None, device="cpu", model_param=None):
        params = dict(model_param or {})
        model_dir = model_dir or params.get("save_dir")
        device = params.get("device", device)
        if not model_dir:
            raise ValueError("model_dir is required")
        super().__init__(budget, name, cpa, category)
        with open(os.path.join(model_dir, "model_spec.json"), encoding="utf-8-sig") as f:
            spec = json.load(f)
        with open(os.path.join(model_dir, "normalize_dict.pkl"), "rb") as f:
            norm = pickle.load(f)
        self.variant = spec["variant"]
        if self.variant not in {"v0", "v1", "v2", "v3", "v4", "v5"}:
            raise ValueError(f"unsupported variant {self.variant}")
        self.device = device
        self.K = int(params.get("K", norm.get("K", 20)))
        self.rtg_scale = float(norm["rtg_scale"])
        self.v_goal_multiplier = float(params.get("v_goal_multiplier", 1.0))
        self.state_mean = np.asarray(norm["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(norm["state_std"], dtype=np.float32)
        self.action_mean = float(norm["action_mean"])
        self.action_std = float(norm["action_std"])
        self.action_upper = float(norm["action_upper"])
        config = norm["block_config"]
        kwargs = dict(state_dim=len(self.state_mean), act_dim=1,
                      hidden_size=int(norm["hidden_size"]),
                      max_ep_len=int(norm["max_ep_len"]),
                      time_dim=int(norm["time_dim"]), block_config=config)
        if self.variant == "v5":
            self.actor = ConversionRTGModel(**kwargs)
            self.rollout_class = ConversionRTGRollout
        else:
            self.actor = DGABSharedModel(**kwargs)
            self.rollout_class = DGABSharedRollout
        state = torch.load(os.path.join(model_dir, "best.pt"), map_location=device)
        incompatible = self.actor.load_state_dict(state, strict=self.variant != "v4")
        if self.variant == "v4":
            allowed = {"value_head.weight", "value_head.bias",
                       "q_head.weight", "q_head.bias"}
            if set(incompatible.missing_keys) != allowed or incompatible.unexpected_keys:
                raise RuntimeError(f"invalid v4 checkpoint: {incompatible}")
        self.actor.to(device).eval()
        self.builder = StateBuilder16()
        self._new_rollout()

    def _new_rollout(self):
        self.rollout = self.rollout_class(
            self.actor, V_goal=self.v_goal_multiplier * self.budget / (self.cpa + EPS),
            C_goal=self.budget, K=self.K, rtg_scale=self.rtg_scale,
            device=self.device)

    def reset(self):
        self.remaining_budget = self.budget
        self.builder = StateBuilder16()
        self._new_rollout()

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo,
                historyBid, historyAuctionResult, historyImpressionResult,
                historyLeastWinningCost):
        if timeStepIndex == 0:
            self.reset()
        state = self.builder.build(timeStepIndex, self.remaining_budget,
                                   self.budget, pValues)
        state = (state - self.state_mean) / self.state_std
        if historyImpressionResult:
            imp = np.asarray(historyImpressionResult[-1])
            auc = np.asarray(historyAuctionResult[-1])
            if imp.ndim == 3: imp = imp[0]
            if auc.ndim == 3: auc = auc[0]
            if auc.shape[1] >= 5:
                previous_cost = float(auc[:, 4].sum())
            elif auc.shape[1] == 3:
                previous_cost = float((auc[:, 2] * imp[:, 0]).sum())
            else:
                raise ValueError(f"unsupported auction history shape: {auc.shape}")
            self.rollout.update_rtg(float(imp[:, -1].sum()), previous_cost)
        action = float(np.asarray(self.rollout.act(state)).reshape(-1)[0])
        alpha = float(np.clip(action * self.action_std + self.action_mean,
                              0, self.action_upper))
        self.last_diagnostics = {"alpha": alpha, "variant": self.variant}
        if historyBid:
            bids = np.asarray(historyBid[-1])
            lwc = np.asarray(historyLeastWinningCost[-1])
            auc = np.asarray(historyAuctionResult[-1])
            imp = np.asarray(historyImpressionResult[-1])
            pv = np.asarray(historyPValueInfo[-1])
            if auc.ndim == 3: auc = auc[0]
            if imp.ndim == 3: imp = imp[0]
            if pv.ndim == 3: pv = pv[0]
            if auc.shape[1] >= 5:
                tick_status = auc[:, 2]
            elif auc.shape[1] == 3:
                tick_status = auc[:, 0]
            else:
                raise ValueError(f"unsupported auction history shape: {auc.shape}")
            self.builder.update(bids, lwc, tick_status, imp[:, -1], pv[:, 0])
        return alpha * pValues
