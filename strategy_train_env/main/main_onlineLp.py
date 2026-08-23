import numpy as np
import torch
import os
import sys
import argparse
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)
print(sys.path)
from run.run_onlinelp import run_onlineLp
from main.main_test import test_checkpoints

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OnlineLP bidding strategy")
    parser.add_argument("--data", type=str, default=None, help="CSV, directory, comma-separated CSVs, or glob")
    parser.add_argument("--save", type=str, default=None, help="Model save directory")
    parser.add_argument("--test_data", default="data/traffic/period-12.csv", help="Offline test CSV")
    args = parser.parse_args()
    run_onlineLp(train_data_path=args.data, save_dir=args.save)
    test_checkpoints('onlinelp', args.save or 'saved_model/onlineLpTest', args.test_data)
