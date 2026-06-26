import numpy as np
import torch
import torch.nn as nn
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.dt.dt import DecisionTransformer
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_dt(train_data_path=None, step_num=None, batch_size=None, save_dir=None,
          device=None, multi_gpu=False):
    kwargs = {}
    if train_data_path: kwargs['train_data_path'] = train_data_path
    if step_num: kwargs['step_num'] = step_num
    if batch_size: kwargs['batch_size'] = batch_size
    if save_dir: kwargs['save_dir'] = save_dir
    if device: kwargs['device'] = device
    kwargs['multi_gpu'] = multi_gpu
    train_dt_model(**kwargs)


def train_dt_model(train_data_path="./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv",
                  step_num=10000, batch_size=32, save_dir="saved_model/DTtest",
                  device="cuda", multi_gpu=False):
    state_dim = 16

    # replay_buffer = EpisodeReplayBuffer(16, 1, "./data/trajectory/trajectory_data.csv")
    replay_buffer = EpisodeReplayBuffer(16, 1, train_data_path)
    save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                        save_dir)
    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")

    model = DecisionTransformer(state_dim=state_dim, act_dim=1, state_mean=replay_buffer.state_mean,
                                state_std=replay_buffer.state_std)
    if device != "cpu" and torch.cuda.is_available():
        model.to(device)
        if multi_gpu and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f'Using {torch.cuda.device_count()} GPUs (DataParallel)')
        else:
            logger.info(f'Using device: {device}')
    else:
        logger.info('Using CPU')
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)

    model.train()
    i = 0
    for states, actions, rewards, dones, rtg, timesteps, attention_mask in dataloader:
        train_loss = model.step(states, actions, rewards, dones, rtg, timesteps, attention_mask)
        i += 1
        logger.info(f"Step: {i} Action loss: {np.mean(train_loss)}")
        model.scheduler.step()

    model.save_net(save_dir)
    test_state = np.ones(state_dim, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


def load_model():
    """
    加载模型。
    """
    with open('./Model/DT/saved_model/normalize_dict.pkl', 'rb') as f:
        normalize_dict = pickle.load(f)
    model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                state_std=normalize_dict["state_std"])
    model.load_net("Model/DT/saved_model")
    test_state = np.ones(16, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


if __name__ == "__main__":
    run_dt()
