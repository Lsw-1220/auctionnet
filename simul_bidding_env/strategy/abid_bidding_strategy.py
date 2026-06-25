import numpy as np
from simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy


class AbidBiddingStrategy(BaseBiddingStrategy):
    def __init__(self, budget=100, name="AbidBiddingStrategy", cpa=1 / 1.5, category=0, exp_tempral_ratio=np.ones(48)):
        super().__init__()
        self.budget = budget
        self.remaining_budget = budget
        self.base_actions = exp_tempral_ratio
        self.name = name
        self.cpa = cpa
        self.category = category

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        alpha = self.base_actions[timeStepIndex] * self.cpa / pValues.mean()
        bids = alpha * pValues * pValues
        bids[bids < 0] = 0
        return bids
