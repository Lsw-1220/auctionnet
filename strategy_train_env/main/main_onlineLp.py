import numpy as np
import torch
import os
import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)
print(sys.path)
from run.run_onlinelp import run_onlineLp

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    run_onlineLp()
