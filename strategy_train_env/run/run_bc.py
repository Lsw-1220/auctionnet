import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from bidding_train_env.common.utils import (load_compact_transition_arrays, save_normalize_dict,
                                            save_training_checkpoint)
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.bc.behavior_clone import BC
import logging
import ast

np.set_printoptions(suppress=True, precision=4)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_bc(train_data_path=None, step_num=None, batch_size=None, save_dir=None,
           device=None, multi_gpu=False):
    """
    Run bc model training and evaluation.
    """
    kwargs = {}
    if train_data_path: kwargs['train_data_path'] = train_data_path
    if step_num: kwargs['step_num'] = step_num
    if batch_size: kwargs['batch_size'] = batch_size
    if save_dir: kwargs['save_dir'] = save_dir
    if device: kwargs['device'] = device
    kwargs['multi_gpu'] = multi_gpu
    train_bc_model(**kwargs)
    # load_model()


def train_bc_model(train_data_path="./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv",
                     step_num=20000, batch_size=100, save_dir="saved_model/BCtest",
                     device="cuda", multi_gpu=False):
    """
    train BC model
    """

    arrays, normalize_dic, paths = load_compact_transition_arrays(train_data_path, state_dim=16)
    logger.info(f'Loading {len(paths)} data file(s): {paths}')
    logger.info(f'Total training samples: {len(arrays[0])}')

    state_dim = 16
    save_normalize_dict(normalize_dic, save_dir)

    replay_buffer = ReplayBuffer(arrays)
    logger.info(f"Replay buffer size: {len(replay_buffer)}")

    model = BC(dim_obs=state_dim)
    if device != "cpu" and torch.cuda.is_available():
        model.to(device)
        if multi_gpu and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f'Using {torch.cuda.device_count()} GPUs (DataParallel)')
        else:
            logger.info(f'Using device: {device}')
    else:
        logger.info('Using CPU')
    for i in range(step_num):
        states, actions, _, _, _ = replay_buffer.sample(batch_size)
        if device != "cpu":
            states, actions = states.to(device), actions.to(device)
        a_loss = model.step(states, actions)
        logger.info(f"Step: {i} Action loss: {np.mean(a_loss)}")
        step = i + 1
        if step % 1000 == 0 or step == step_num:
            save_training_checkpoint(model, save_dir, step, normalize_dic)

    test_trained_model(model, replay_buffer)


def load_model():
    """
    load model
    """
    model = BC(dim_obs=16)
    model.load_net("saved_model/BCtest")
    test_state = np.ones(16, dtype=np.float32)
    test_state_tensor = torch.tensor(test_state, dtype=torch.float)
    logger.info(f"Test action: {model.take_actions(test_state_tensor)}")


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = row.state if not is_normalize else row.normalize_state, row.action, row.reward if not is_normalize else row.normalize_reward, row.next_state if not is_normalize else row.normalize_nextstate, row.done
        # ! 去掉了所有的done==1的数据
        if done != 1:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.array(next_state),
                               np.array([done]))
        else:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state),
                               np.array([done]))


def test_trained_model(model, replay_buffer):
    states, actions, rewards, next_states, terminals = replay_buffer.sample(100)
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred_action:", tem)


if __name__ == "__main__":
    run_bc()
