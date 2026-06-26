import numpy as np
import torch
import logging
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.onlineLp.onlineLp import OnlineLp

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_onlineLpModel(train_data_path="./data/traffic/", save_dir="saved_model/onlineLpTest"):
    onlineLp = OnlineLp(train_data_path)
    onlineLp.train(save_dir)


def run_onlineLp(train_data_path=None, save_dir=None):
    """
    Run onlinelp model training and evaluation.
    """
    kwargs = {}
    if train_data_path: kwargs['train_data_path'] = train_data_path
    if save_dir: kwargs['save_dir'] = save_dir
    train_onlineLpModel(**kwargs)


if __name__ == '__main__':
    run_onlineLp()
