import numpy as np
import math
import logging
from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from bidding_train_env.offline_eval.offline_env import OfflineEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def getScore_neurips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def run_test():
    """
    offline evaluation
    """

    data_loader = TestDataLoader(file_path='strategy_train_env/data/traffic/period-12.csv')
    env = OfflineEnv()
    agent = PlayerBiddingStrategy()
    print(agent.name)

    keys, test_dict = data_loader.keys, data_loader.test_dict
    all_scores, all_rewards, all_costs, all_cpas = [], [], [], []

    for key in keys:
        period, advertiser = int(key[0]), int(key[1])
        agent.reset()
        num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key)
        rewards = np.zeros(num_timeStepIndex)
        history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': [],
            'historyLeastWinningCost': [],
            'historyPValueInfo': []
        }

        for timeStep_index in range(num_timeStepIndex):
            pValue = pValues[timeStep_index]
            pValueSigma = pValueSigmas[timeStep_index]
            leastWinningCost = leastWinningCosts[timeStep_index]

            if agent.remaining_budget < env.min_remaining_budget:
                bid = np.zeros(pValue.shape[0])
            else:
                bid = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                    history["historyBids"],
                                    history["historyAuctionResult"], history["historyImpressionResult"],
                                    history["historyLeastWinningCost"])

            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                          leastWinningCost)

            # Handling over-cost
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                if len(pv_index) == 0:
                    break
                dropped_pv_index = np.random.choice(pv_index, min(int(math.ceil(pv_index.shape[0] * over_cost_ratio)), len(pv_index)),
                                                    replace=False)
                bid[dropped_pv_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                              leastWinningCost)
                over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            agent.remaining_budget -= np.sum(tick_cost)
            rewards[timeStep_index] = np.sum(tick_conversion)
            temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
            history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
            history["historyBids"].append(bid)
            history["historyLeastWinningCost"].append(leastWinningCost)
            temAuctionResult = np.array(
                [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
            history["historyAuctionResult"].append(temAuctionResult)
            temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
            history["historyImpressionResult"].append(temImpressionResult)

        all_reward = np.sum(rewards)
        all_cost = agent.budget - agent.remaining_budget
        cpa_real = all_cost / (all_reward + 1e-10)
        cpa_constraint = agent.cpa
        score = getScore_neurips(all_reward, cpa_real, cpa_constraint)

        all_scores.append(score)
        all_rewards.append(all_reward)
        all_costs.append(all_cost)
        all_cpas.append(cpa_real)
        logger.info(f'Period {period} Adv {advertiser}: score={score:.2f} reward={all_reward} '
                    f'cpa={cpa_real:.2f} budget_used={all_cost/agent.budget:.0%}')

    # ── Summary ──
    n = len(keys)
    logger.info(f'\n{"="*50}')
    logger.info(f'OFFLINE EVAL SUMMARY — {agent.name} over {n} (period, advertiser) pairs')
    logger.info(f'  Avg Score:  {np.mean(all_scores):.2f}')
    logger.info(f'  Avg Reward: {np.mean(all_rewards):.1f}')
    logger.info(f'  Avg CPA:    {np.mean(all_cpas):.2f}')


if __name__ == '__main__':
    run_test()
