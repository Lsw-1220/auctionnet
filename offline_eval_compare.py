"""
Offline evaluation comparison: GAVE vs DGAB-FO using AuctionNet's offline test pipeline.

Uses TestDataLoader + OfflineEnv — compares bids against historical leastWinningCost.
No live competition; opponent behavior is frozen from the logged data.

Usage:
    python offline_eval_compare.py
    python offline_eval_compare.py --data ./strategy_train_env/data/traffic/period-8.csv
    python offline_eval_compare.py --data ./data/log/0.csv --advertisers 0,4,24
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategy_train_env'))

import numpy as np
import logging
import time
import argparse

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def getScore_neurips(reward, cpa, cpa_constraint):
    """AuctionNet scoring function."""
    beta = 2
    if cpa > cpa_constraint:
        penalty = (cpa_constraint / (cpa + 1e-10)) ** beta
    else:
        penalty = 1.0
    return penalty * reward


def run_offline_eval(agent, data_path, advertiser_key):
    """Run offline evaluation for one advertiser with the given agent."""
    from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
    from bidding_train_env.offline_eval.offline_env import OfflineEnv

    data_loader = TestDataLoader(file_path=data_path)
    env = OfflineEnv()

    num_ticks, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(advertiser_key)

    history = {
        'historyBids': [],
        'historyAuctionResult': [],
        'historyImpressionResult': [],
        'historyLeastWinningCost': [],
        'historyPValueInfo': []
    }
    rewards = np.zeros(num_ticks)
    total_cost = 0.0

    for tick in range(num_ticks):
        pv = pValues[tick]
        sigma = pValueSigmas[tick]
        lwc = leastWinningCosts[tick]

        if agent.remaining_budget < env.min_remaining_budget:
            bid = np.zeros(len(pv))
        else:
            bid = agent.bidding(
                tick, pv, sigma,
                history["historyPValueInfo"],
                history["historyBids"],
                history["historyAuctionResult"],
                history["historyImpressionResult"],
                history["historyLeastWinningCost"])

        # Over-cost handling (same as original offline eval)
        over_cost_ratio = max((np.sum(bid > lwc) * np.mean(lwc) - agent.remaining_budget) /
                              (np.sum(bid > lwc) * np.mean(lwc) + 1e-4), 0)
        loop = 0
        while over_cost_ratio > 0 and loop < 5:
            pv_index = np.where(bid >= lwc)[0]
            if len(pv_index) == 0:
                break
            drop_n = max(1, int(np.ceil(len(pv_index) * over_cost_ratio)))
            dropped = np.random.choice(pv_index, min(drop_n, len(pv_index)), replace=False)
            bid[dropped] = 0
            over_cost_ratio = max((np.sum(bid > lwc) * np.mean(lwc) - agent.remaining_budget) /
                                  (np.sum(bid > lwc) * np.mean(lwc) + 1e-4), 0)
            loop += 1

        tick_value, tick_cost_vec, tick_status, tick_conversion = env.simulate_ad_bidding(
            pv, sigma, bid, lwc)

        tick_cost = np.sum(tick_cost_vec)
        agent.remaining_budget -= tick_cost
        total_cost += tick_cost
        rewards[tick] = np.sum(tick_conversion)

        history["historyPValueInfo"].append(
            np.array([(pv[i], sigma[i]) for i in range(len(pv))]))
        history["historyBids"].append(bid)
        history["historyLeastWinningCost"].append(lwc)
        history["historyAuctionResult"].append(
            np.array([(tick_status[i], tick_status[i], tick_cost_vec[i]) for i in range(len(tick_status))]))
        history["historyImpressionResult"].append(
            np.array([(tick_conversion[i], tick_conversion[i]) for i in range(len(tick_conversion))]))

    all_reward = np.sum(rewards)
    cpa_real = total_cost / (all_reward + 1e-10)
    score = getScore_neurips(all_reward, cpa_real, agent.cpa)

    return {
        'reward': int(all_reward),
        'cost': total_cost,
        'cpa_real': cpa_real,
        'cpa_target': agent.cpa,
        'score': score,
        'budget_used': total_cost / agent.budget if agent.budget > 0 else 0,
    }


def build_agent(name, budget, cpa, category):
    """Build an agent instance by name."""
    if name == 'PID':
        from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
        return PidBiddingStrategy(budget=budget, cpa=cpa, category=category,
                                  name='PID', exp_tempral_ratio=np.ones(48))
    elif name == 'IQL':
        from simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
        return IqlBiddingStrategy(budget=budget, cpa=cpa, category=category, name='IQL')
    elif name == 'OnlineLP':
        from simul_bidding_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy
        return OnlineLpBiddingStrategy(budget=budget, cpa=cpa, category=category,
                                       name='OnlineLP', episode=0)
    elif name == 'GAVE':
        from simul_bidding_env.strategy.autobidding_agents import GAVEAuctionNetAgent
        BLOCK_CONFIG = {
            'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
            'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
            'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
        }
        GAVE_DIR = 'D:/research/Experiment/autobidding/saved_model/gave_400k_sparse'
        DEVICE = 'cpu'
        return GAVEAuctionNetAgent(
            budget=budget, cpa=cpa, category=category, name='GAVE',
            model_param=dict(
                save_dir=GAVE_DIR, hidden_size=512, time_dim=8,
                block_config=BLOCK_CONFIG, device=DEVICE,
                expectile=0.99, score_target_mode='prev'))
    elif name == 'DGAB':
        from simul_bidding_env.strategy.autobidding_agents import DGABFOAuctionNetAgent
        BLOCK_CONFIG = {
            'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
            'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
            'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
        }
        DGAB_DIR = 'D:/research/Experiment/autobidding/saved_model/dgab_400k_sparse'
        DEVICE = 'cpu'
        return DGABFOAuctionNetAgent(
            budget=budget, cpa=cpa, category=category, name='DGAB-FO',
            model_param=dict(
                save_dir=DGAB_DIR, hidden_size=512, max_ep_len=96,
                time_dim=8, block_config=BLOCK_CONFIG, device=DEVICE,
                actor_type='stack', critic_type='sequence'))
    else:
        raise ValueError(f'Unknown agent: {name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='./strategy_train_env/data/traffic/period-8.csv')
    parser.add_argument('--advertisers', type=str, default='all',
                        help='Comma-separated advertiser indices, or "all"')
    parser.add_argument('--agents', type=str, default='PID,IQL,GAVE,DGAB',
                        help='Comma-separated agent names')
    args = parser.parse_args()

    from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
    loader = TestDataLoader(file_path=args.data)
    all_keys = loader.keys
    logger.info(f'Data: {args.data} — {len(all_keys)} (period, advertiser) pairs')

    agent_names = [s.strip() for s in args.agents.split(',')]

    if args.advertisers == 'all':
        # Collect unique advertiser numbers
        adv_set = sorted(set(k[1] for k in all_keys))
    else:
        adv_set = [int(x) for x in args.advertisers.split(',')]

    logger.info(f'Advertisers: {adv_set}')
    logger.info(f'Agents: {agent_names}')

    # Get budget/CPA from data
    import pandas as pd
    raw = pd.read_csv(args.data)
    adv_info = {}
    for adv in adv_set:
        row = raw[raw['advertiserNumber'] == adv].iloc[0]
        adv_info[adv] = {
            'budget': row['budget'],
            'cpa': row['CPAConstraint'],
            'category': int(row['advertiserCategoryIndex']),
        }

    all_results = {name: [] for name in agent_names}
    t0 = time.time()

    for adv in adv_set:
        info = adv_info[adv]
        # Use the first period key for this advertiser
        key = next(k for k in all_keys if k[1] == adv)
        period = int(key[0])

        logger.info(f'\nAdvertiser #{adv} (budget={info["budget"]}, cpa={info["cpa"]}, period={period})')

        for agent_name in agent_names:
            agent = build_agent(agent_name, info['budget'], info['cpa'], info['category'])
            agent.remaining_budget = agent.budget
            result = run_offline_eval(agent, args.data, key)
            all_results[agent_name].append(result)
            logger.info(f'  {agent_name:10s}: score={result["score"]:.2f} '
                        f'reward={result["reward"]} cpa={result["cpa_real"]:.2f} '
                        f'budget={result["budget_used"]:.0%}')

    elapsed = time.time() - t0
    n_runs = len(adv_set) * len(agent_names)
    logger.info(f'\nTotal: {n_runs} runs in {elapsed:.0f}s')

    # ── Summary ──
    print(f'\n{"="*100}')
    print(f'OFFLINE EVALUATION — avg over {len(adv_set)} advertisers × {len(agent_names)} strategies')
    print(f'{"="*100}')
    header = f'{"Metric":<25}' + ''.join(f'{name:>15}' for name in agent_names)
    print(header)
    print('-' * len(header))
    for metric, key, fmt in [
        ('Score', 'score', '.2f'),
        ('Reward', 'reward', '.0f'),
        ('Cost', 'cost', '.0f'),
        ('CPA real', 'cpa_real', '.2f'),
        ('Budget used %', 'budget_used', '.1%'),
    ]:
        row = f'{metric:<25}'
        for name in agent_names:
            v = np.mean([r[key] for r in all_results[name]])
            row += f'{v:>15{fmt}}'
        print(row)
    print()


if __name__ == '__main__':
    main()
