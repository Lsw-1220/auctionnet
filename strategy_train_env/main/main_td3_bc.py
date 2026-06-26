import numpy as np
import torch
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run.run_td3_bc import run_td3_bc

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TD3_BC bidding strategy")
    parser.add_argument("--data", type=str, default=None, help="Path to training CSV")
    parser.add_argument("--steps", type=int, default=None, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--save", type=str, default=None, help="Model save directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--multi_gpu", action="store_true", help="Use DataParallel on all GPUs")
    args = parser.parse_args()
    run_td3_bc(train_data_path=args.data, step_num=args.steps, batch_size=args.batch_size,
               save_dir=args.save, device=args.device, multi_gpu=args.multi_gpu)
