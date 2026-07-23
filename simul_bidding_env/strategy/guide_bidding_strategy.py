import numpy as np
import os
import sys
import torch
import pickle

from simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy

# Add strategy_train_env to path so we can import GUIDE model
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
_strategy_train_env = os.path.join(_project_root, "strategy_train_env")
if _strategy_train_env not in sys.path:
    sys.path.insert(0, _strategy_train_env)

from bidding_train_env.baseline.GUIDE.dt_baselines import DecisionTransformer


class GUIDEStrategy(BaseBiddingStrategy):
    """
    GUIDE Strategy for AuctionNet simulation environment.
    Uses Decision Transformer + Inverse Dynamics Module + Critic for adaptive bidding.
    """

    def __init__(self, budget=100, name="GUIDE-Strategy", cpa=2, category=1,
                 model_dir=None, device='cpu'):
        super().__init__(budget, name, cpa, category)

        if model_dir is not None:
            model_path = os.path.join(model_dir, "GUIDE.pt")
            critic_path = os.path.join(model_dir, "GUIDE_critic_inverse.pt")
            idm_path = os.path.join(model_dir, "GUIDE_idm.pt")
            picklePath = os.path.join(model_dir, "normalize_dict.pkl")
        else:
            # Default: saved_model/GUIDE/ under strategy_train_env
            file_name = os.path.dirname(os.path.realpath(__file__))
            # simul_bidding_env/strategy/ → simul_bidding_env/ → project_root/
            proj_root = os.path.dirname(os.path.dirname(file_name))
            model_path = os.path.join(proj_root, "strategy_train_env", "saved_model", "GUIDE", "GUIDE.pt")
            critic_path = os.path.join(proj_root, "strategy_train_env", "saved_model", "GUIDE", "GUIDE_critic_inverse.pt")
            idm_path = os.path.join(proj_root, "strategy_train_env", "saved_model", "GUIDE", "GUIDE_idm.pt")
            picklePath = os.path.join(proj_root, "strategy_train_env", "saved_model", "GUIDE", "normalize_dict.pkl")

        device = device if torch.cuda.is_available() else "cpu"

        with open(picklePath, 'rb') as f:
            normalize_dict = pickle.load(f)

        self.model = DecisionTransformer(
            state_dim=16, act_dim=1,
            state_mean=normalize_dict["state_mean"],
            state_std=normalize_dict["state_std"],
            target_return=16., target_ctg=16.,
        )
        self.model.load_net(load_path=model_path,
                            critic_path=critic_path,
                            idm_path=idm_path,
                            device=device)
        self.model.to(device)
        self._device = device

        self.remaining_budget_last = self.budget

    def reset(self):
        self.remaining_budget = self.budget
        self.remaining_budget_last = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """
        Bids for all the opportunities in a delivery period.
        Uses DT + IDM with Critic-based action selection.
        """
        self.cost_cur = self.remaining_budget_last - self.remaining_budget

        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0
        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else 0
        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0
        historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0
        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            l = len(history)
            last_n_data = history[max(0, l - n):l]
            if len(last_n_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_n_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        if timeStepIndex == 0:
            self.model.init_eval()

        cost_constraint = self.cost_cur if len(history_conversion) != 0 else None

        alpha = self.model.take_action_inverse(
            test_state,
            actual_executed_action=None,
            pre_reward=sum(history_conversion[-1]) if len(history_conversion) != 0 else None,
            pre_cost=cost_constraint if len(history_conversion) != 0 else None,
            cpa_constrain=self.cpa
        )

        self.remaining_budget_last = self.remaining_budget
        bids = alpha * pValues
        return bids
