import numpy as np
import torch
import os
import sys
import argparse
import glob
import csv
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run.run_evaluate import run_test
from bidding_train_env.offline_eval.test_dataloader import TestDataLoader

torch.manual_seed(1)
np.random.seed(1)

STRATEGIES = {
    'iql': ('bidding_train_env.strategy.iql_bidding_strategy', 'IqlBiddingStrategy'),
    'bc': ('bidding_train_env.strategy.bc_bidding_strategy', 'BcBiddingStrategy'),
    'bcq': ('bidding_train_env.strategy.bcq_bidding_strategy', 'BcqBiddingStrategy'),
    'cql': ('bidding_train_env.strategy.cql_bidding_strategy', 'CqlBiddingStrategy'),
    'td3_bc': ('bidding_train_env.strategy.td3_bc_bidding_strategy', 'TD3_BCBiddingStrategy'),
    'dt': ('bidding_train_env.strategy.dt_bidding_strategy', 'DtBiddingStrategy'),
    'onlinelp': ('bidding_train_env.strategy.onlinelp_bidding_strategy', 'OnlineLpBiddingStrategy'),
}


def test_checkpoints(strategy, model_dir, test_data):
    import importlib
    module_name, class_name = STRATEGIES[strategy]
    strategy_class = getattr(importlib.import_module(module_name), class_name)
    checkpoints = sorted(path for path in glob.glob(os.path.join(model_dir, 'checkpoint_*'))
                         if os.path.isdir(path))
    if not checkpoints:
        checkpoints = [model_dir]

    data_loader = TestDataLoader(file_path=test_data)
    if len(data_loader.keys) != 48:
        raise ValueError(f'Expected 48 advertiser episodes, found {len(data_loader.keys)}')
    advertisers = {int(key[1]) for key in data_loader.keys}
    tick_counts = {key: data_loader.mock_data(key)[0] for key in data_loader.keys}
    if len(advertisers) != 48 or any(count != 48 for count in tick_counts.values()):
        raise ValueError('Test data must contain 48 advertisers with exactly 48 ticks each')
    print(f'Validated test coverage: 48 advertisers x 48 ticks ({test_data})')

    results, detail_results = [], []
    for checkpoint in checkpoints:
        # Keep stochastic conversion simulation identical across checkpoints.
        np.random.seed(1)
        torch.manual_seed(1)
        print(f'\nTesting {strategy} checkpoint: {checkpoint}')
        metrics, details = run_test(
            test_data=test_data, agent=strategy_class(model_dir=checkpoint),
            data_loader=data_loader, return_details=True)
        metrics['checkpoint'] = checkpoint
        results.append(metrics)
        for row in details:
            detail_results.append({'checkpoint': checkpoint, **row})
    result_path = os.path.join(model_dir, 'checkpoint_evaluation.csv')
    with open(result_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['checkpoint', 'score', 'reward', 'cost', 'cpa'])
        writer.writeheader()
        writer.writerows(results)
    print(f'Checkpoint evaluation saved to {result_path}')
    detail_path = os.path.join(model_dir, 'checkpoint_evaluation_details.csv')
    with open(detail_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'checkpoint', 'period', 'advertiser', 'score', 'reward', 'cost',
            'cpa', 'cpa_constraint', 'budget', 'num_ticks'])
        writer.writeheader()
        writer.writerows(detail_results)
    best = max(results, key=lambda row: row['score'])
    best_path = os.path.join(model_dir, 'best_checkpoint.json')
    with open(best_path, 'w', encoding='utf-8') as file:
        json.dump(best, file, ensure_ascii=False, indent=2)
    print(f'Best checkpoint: {best["checkpoint"]} (score={best["score"]:.4f})')
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Offline-test one model or all checkpoints')
    parser.add_argument('--test_data',
                        default='D:/research/Experiment/dgab/data/MDP/traffic/period-7.csv',
                        help='Offline evaluation CSV path')
    parser.add_argument('--strategy', choices=STRATEGIES, default=None)
    parser.add_argument('--model_dir', default=None,
                        help='Model directory containing checkpoint_* subdirectories')
    args = parser.parse_args()
    if args.strategy or args.model_dir:
        if not (args.strategy and args.model_dir):
            parser.error('--strategy and --model_dir must be specified together')
        test_checkpoints(args.strategy, args.model_dir, args.test_data)
    else:
        run_test(test_data=args.test_data)
