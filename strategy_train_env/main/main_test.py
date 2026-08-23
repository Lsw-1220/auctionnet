import numpy as np
import torch
import os
import sys
import argparse
import glob
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run.run_evaluate import run_test

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

    results = []
    for checkpoint in checkpoints:
        # Keep stochastic conversion simulation identical across checkpoints.
        np.random.seed(1)
        torch.manual_seed(1)
        print(f'\nTesting {strategy} checkpoint: {checkpoint}')
        metrics = run_test(test_data=test_data, agent=strategy_class(model_dir=checkpoint))
        metrics['checkpoint'] = checkpoint
        results.append(metrics)
    result_path = os.path.join(model_dir, 'checkpoint_evaluation.csv')
    with open(result_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['checkpoint', 'score', 'reward', 'cost', 'cpa'])
        writer.writeheader()
        writer.writerows(results)
    print(f'Checkpoint evaluation saved to {result_path}')
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Offline-test one model or all checkpoints')
    parser.add_argument('--test_data', default='data/traffic/period-12.csv',
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
