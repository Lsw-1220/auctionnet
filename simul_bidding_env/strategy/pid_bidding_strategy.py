import numpy as np
from simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy


class PidBiddingStrategy(BaseBiddingStrategy):
    def __init__(self, budget=100, name="PidBiddingStrategy", cpa=1, category=0, exp_tempral_ratio=np.ones(48)):
        super().__init__()
        self.budget = budget
        self.remaining_budget = budget
        self.name = name
        self.exp_budget_ratio = exp_tempral_ratio
        self.alpha = None
        self.base_action = 15
        self.last_remaining_budget = self.remaining_budget
        self.cpa = cpa
        self.category = category

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        if timeStepIndex == 0:
            self.alpha = self.base_action
        else:
            last_tick_cost = self.last_remaining_budget - self.remaining_budget
            self.last_remaining_budget -= last_tick_cost
            if last_tick_cost * self.exp_budget_ratio[timeStepIndex:].sum() / self.exp_budget_ratio[timeStepIndex - 1] / self.remaining_budget < 0.7:
                self.alpha *= 1.2
            elif last_tick_cost * (48 - timeStepIndex) / self.remaining_budget > 1.1:
                self.alpha *= 0.7
        bids = self.alpha * pValues
        return bids

