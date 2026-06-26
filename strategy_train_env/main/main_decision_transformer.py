import numpy as np
import torch
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run.run_decision_transformer import run_dt


torch.manual_seed(1)
np.random.seed(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Decision Transformer bidding strategy")
    parser.add_argument("--data", type=str, default=None, help="Path to training CSV")
    parser.add_argument("--steps", type=int, default=None, help="Training steps")
    parser.add_argument("--save", type=str, default=None, help="Model save directory")
    args = parser.parse_args()
    run_dt(train_data_path=args.data, step_num=args.steps, save_dir=args.save)
