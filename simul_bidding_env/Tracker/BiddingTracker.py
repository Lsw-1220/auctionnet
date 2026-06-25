import gin
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


@gin.configurable
class BiddingTracker:
    def __init__(self, name="BiddingTracker"):
        self.name = name
        self.deliveryPeriodIndexs = []
        self.advertiserNumbers = []
        self.advertiserCategoryIndexs = []
        self.budgets = []
        self.CPAConstraints = []
        self.timeStepIndexs = []
        self.remainingBudgets = []
        self.pvIndexs = []
        self.pValues = []
        self.pValueSigmas = []
        self.bids = []
        self.xis = []
        self.adslots = []
        self.costs = []
        self.isExposeds = []
        self.conversionActions = []
        self.leastWinningCosts = []
        self.isEnds = []

    def reset(self):
        self.deliveryPeriodIndexs = []
        self.advertiserNumbers = []
        self.advertiserCategoryIndexs = []
        self.budgets = []
        self.CPAConstraints = []
        self.timeStepIndexs = []
        self.remainingBudgets = []
        self.pvIndexs = []
        self.pValues = []
        self.pValueSigmas = []
        self.bids = []
        self.xis = []
        self.adslots = []
        self.costs = []
        self.isExposeds = []
        self.conversionActions = []
        self.leastWinningCosts = []
        self.isEnds = []

    def train_logging(self, episode, tickIndex, pValues, budgets, agentCpa, agentCategory, remaining_budgets,
                      totalPvNum, pValueSigma, bids, xis, adslots, costs, isExposeds, conversionActions,
                      leastWinningCosts, done):
        """Log data for each tick."""
        num_pv, num_agent = pValues.shape
        self.deliveryPeriodIndexs.append(np.ones(num_agent * num_pv) * episode)
        self.timeStepIndexs.append(np.ones(num_agent * num_pv) * tickIndex)
        agent_index = np.tile(np.arange(num_agent), num_pv).flatten()
        self.advertiserNumbers.append(agent_index)
        self.budgets.append(np.tile(budgets, num_pv).flatten())
        self.CPAConstraints.append(np.tile(agentCpa, num_pv).flatten())
        self.advertiserCategoryIndexs.append(np.tile(agentCategory, num_pv).flatten())
        self.remainingBudgets.append(np.tile(remaining_budgets, num_pv).flatten())
        self.pvIndexs.append(np.repeat(np.arange(totalPvNum, totalPvNum + num_pv), num_agent))
        self.pValues.append(pValues.flatten())
        self.pValueSigmas.append(pValueSigma.flatten())
        self.bids.append(bids.flatten())
        self.xis.append(xis.transpose().flatten())
        self.adslots.append(adslots.transpose().flatten())
        self.costs.append(costs.transpose().flatten())
        self.isExposeds.append(isExposeds.transpose().flatten())
        self.conversionActions.append(conversionActions.transpose().flatten())
        self.leastWinningCosts.append(np.repeat(leastWinningCosts, num_agent).flatten())
        self.isEnds.append(np.tile(done, num_pv).flatten())

    def generate_train_data(self, dataPath="data/log.csv"):
        """generate_train_data"""
        totaldeliveryPeriodIndexs = np.concatenate(self.deliveryPeriodIndexs)
        totaltimeStepIndexs = np.concatenate(self.timeStepIndexs)
        totaladvertiserNumbers = np.concatenate(self.advertiserNumbers)
        totaladvertiserCategoryIndexs = np.concatenate(self.advertiserCategoryIndexs)
        totalCPAConstraints = np.concatenate(self.CPAConstraints)
        totalbudgets = np.concatenate(self.budgets)
        totalremainingBudgets = np.concatenate(self.remainingBudgets)
        totalpValues = np.concatenate(self.pValues)
        totalpValueSigmas = np.concatenate(self.pValueSigmas)
        totalisEnds = np.concatenate(self.isEnds)
        totalpvIndexs = np.concatenate(self.pvIndexs)
        totalbids = np.concatenate(self.bids)
        totalisExposeds = np.concatenate(self.isExposeds)
        totalxis = np.concatenate(self.xis)
        totaladslots = np.concatenate(self.adslots)
        totalcosts = np.concatenate(self.costs)
        totalleastWinningCosts = np.concatenate(self.leastWinningCosts)
        totalconversionActions = np.concatenate(self.conversionActions)

        columns = ["deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex", "budget", "CPAConstraint",
                   "timeStepIndex",
                   "remainingBudget", "pvIndex", "pValue", "pValueSigma", "bid", "xi", "adSlot", "cost", "isExposed",
                   "conversionAction", "leastWinningCost", "isEnd"]
        FullTable = pd.DataFrame(np.array(
            [totaldeliveryPeriodIndexs, totaladvertiserNumbers, totaladvertiserCategoryIndexs,
             totalbudgets, totalCPAConstraints, totaltimeStepIndexs, totalremainingBudgets, totalpvIndexs,
             totalpValues, totalpValueSigmas, totalbids, totalxis, totaladslots, totalcosts, totalisExposeds,
             totalconversionActions, totalleastWinningCosts, totalisEnds]).transpose(), columns=columns)
        file_path = Path(dataPath)
        # 如果文件夹不存在则创建
        file_path.parent.mkdir(parents=True, exist_ok=True)
        FullTable.to_csv(dataPath, index=False)
        print(f"数据生成成功；保存到{dataPath}")


if __name__ == '__main__':
    pass
