"""AuctionNet adapter for the GAS Decision Transformer policy."""

import os
import pickle

import numpy as np
import torch

from simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from .gas.dt_baselines import DecisionTransformer


class GASBiddingStrategy(BaseBiddingStrategy):
    """GAS base policy backed by the 400k-run step-280000 checkpoint."""

    def __init__(self, budget=100, name="GAS-Strategy", cpa=2, category=1,
                 model_dir=None, device="cpu", reweight_w=0.2):
        super().__init__(budget, name, cpa, category)

        if model_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.realpath(__file__))))
            model_dir = os.path.join(
                project_root, "saved_model", "GAS", "step_280000")

        model_path = os.path.join(model_dir, "dt.pt")
        normalize_path = os.path.join(model_dir, "normalize_dict.pkl")
        self.device = device if torch.cuda.is_available() else "cpu"

        with open(normalize_path, "rb") as f:
            normalize_dict = pickle.load(f)

        self.model = DecisionTransformer(
            state_dim=16,
            act_dim=1,
            state_mean=normalize_dict["state_mean"],
            state_std=normalize_dict["state_std"],
            target_return=1.0 + reweight_w,
            target_ctg=1.0,
            baseline_method="dt_reweight",
        )
        self.model.load_net(model_path, device=self.device)
        self.model.to(self.device)
        self.remaining_budget_last = self.budget

    def reset(self):
        self.remaining_budget = self.budget
        self.remaining_budget_last = self.budget

    @staticmethod
    def _history_mean(history):
        return np.mean([np.mean(item) for item in history]) if history else 0

    @staticmethod
    def _last_mean(history, n=3):
        return np.mean([np.mean(item) for item in history[-n:]]) if history else 0

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo,
                historyBid, historyAuctionResult, historyImpressionResult,
                historyLeastWinningCost):
        cost_cur = self.remaining_budget_last - self.remaining_budget
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pvalue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        state = np.array([
            (48 - timeStepIndex) / 48,
            self.remaining_budget / self.budget if self.budget > 0 else 0,
            self._history_mean(historyBid),
            self._last_mean(historyBid),
            self._history_mean(historyLeastWinningCost),
            self._history_mean(history_pvalue),
            self._history_mean(history_conversion),
            self._history_mean(history_xi),
            self._last_mean(historyLeastWinningCost),
            self._last_mean(history_pvalue),
            self._last_mean(history_conversion),
            self._last_mean(history_xi),
            np.mean(pValues) if len(pValues) else 0,
            len(pValues),
            sum(len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)),
            sum(len(bids) for bids in historyBid),
        ], dtype=np.float32)

        if timeStepIndex == 0:
            self.model.init_eval()

        alpha = self.model.take_actions(
            state,
            actual_excuted_action=None,
            pre_reward=sum(history_conversion[-1]) if history_conversion else None,
            pre_cost=cost_cur if history_conversion else None,
            cpa_constrain=self.cpa,
        )
        self.remaining_budget_last = self.remaining_budget
        return alpha * pValues
