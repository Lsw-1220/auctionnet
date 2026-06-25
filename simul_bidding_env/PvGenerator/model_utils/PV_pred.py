


import os
import pickle
import torch
import pandas as pd
import numpy as np
from torch import nn
from typing import List, Tuple
from simul_bidding_env.PvGenerator.model_utils.build_data import build_data_from_df
from simul_bidding_env.PvGenerator.model_utils.PvModel import PvModel

# Constant definitions
DEFAULT_BATCH_SIZE = 1000
DEFAULT_MODEL_NAME = 'PV_model'
DEFAULT_PV_DATA_NAME = 'generated_105K_pv_data.csv'
DEFAULT_USE_CUDA = True
NON_ZERO_BIAS = 0.001

# Get the current file path
current_file_path = os.path.abspath(__file__)
current_directory_path = os.path.dirname(current_file_path)

class PvPredictor(nn.Module):
    """
    The PvPredictor class is used for traffic prediction tasks.

    Args:
        num_tick (int): Number of time ticks.
        num_agent_category (int): Number of agent categories.
        select_category (List[int]): List of selected categories.
        batch_size (int): Batch size.
        model_name (str): Model name.
        pv_data_name (str): Traffic data file name.
        use_cuda (bool): Whether to use CUDA.
    """
    def __init__(self,
                 num_tick: int = 24,
                 num_agent_category: int = 6,
                 select_category: List[int] = [1, 15, 21, 30, 50],
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 model_name: str = DEFAULT_MODEL_NAME,
                 pv_data_name: str = DEFAULT_PV_DATA_NAME,
                 use_cuda: bool = DEFAULT_USE_CUDA) -> None:
        super(PvPredictor, self).__init__()
        self.batch_size = batch_size
        self.num_tick = num_tick
        self.num_agent_category = num_agent_category
        self.select_category = select_category
        self.num_category = len(select_category)
        self.num_all_agent = num_agent_category * self.num_category
        self.data_prefix = current_directory_path
        self.base_num_tick = 96
        self.tick_multi = self.base_num_tick // self.num_tick

        self.load_base_time_distribution()
        self.load_time_category_relation()

        self.model_name = model_name
        self.model_type = 'Pv'
        self.pv_data_name = pv_data_name
        self.model_path = f'check_point/{model_name}'
        self.use_cuda = use_cuda
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

        self.load_model_args()
        self.load_generate_info_dict()

        self.all_feature_dim = 340
        self.model = PvModel(input_dim=self.all_feature_dim, info_dict=self.generate_info_dict,
                             device=self.device, args=self.model_args).to(device=self.device, dtype=torch.float64)

        self.load_model_state_dict()
        self.update_feature_info_dict()
        self.load_pv_data()

        self.time_dim = self.model.time_input_dim
        self.category_dim = self.model.category_input_dim
        self.time_eye = np.eye(self.time_dim)
        self.category_eye = np.eye(self.category_dim)

    def load_base_time_distribution(self):
        """
        Load the base time distribution data.
        """
        with open(f'{self.data_prefix}/data/PV_time_distribution_data.pkl', 'rb') as f:
            self.base_time_dict = pickle.load(f)

        self.base_time_dist = np.zeros(self.base_num_tick)
        for i in range(self.base_num_tick):
            idx = self.base_time_dict['key_list'][i]
            prob = self.base_time_dict['value_list'][i]
            self.base_time_dist[idx] = prob

        self.base_time_dist += NON_ZERO_BIAS
        self.time_dist = np.zeros(self.num_tick)
        for i in range(self.base_num_tick):
            idx = i // self.tick_multi
            self.time_dist[idx] += self.base_time_dist[i]
        self.time_dist /= self.time_dist.sum()

    def load_time_category_relation(self):
        """
        Load the time category relation data.
        """
        with open(f'{self.data_prefix}/data/PV_time_category_relation_data.pkl', 'rb') as f:
            self.base_tc_relation_dict = pickle.load(f)

        tc_counts_list = self.base_tc_relation_dict['counts']
        self.tc_dist = np.zeros([self.num_tick, self.num_category])
        for i in range(self.base_num_tick):
            t_id = i // self.tick_multi
            for j in range(self.num_category):
                c_id = self.select_category[j]
                self.tc_dist[t_id, j] += tc_counts_list[i, c_id]

        for i in range(self.num_tick):
            sum_i = self.tc_dist[i, :].sum()
            self.tc_dist[i, :] /= sum_i

    def load_model_args(self):
        """
        Load the model arguments.
        """
        with open(f'{self.data_prefix}/{self.model_path}/{self.model_type}_args.pkl', 'rb') as f:
            self.model_args = pickle.load(f)

    def load_generate_info_dict(self):
        """
        Load the generate information dictionary.
        """
        with open(f'{self.data_prefix}/{self.model_path}/{self.model_type}_info_dict.pkl', 'rb') as f:
            self.generate_info_dict = pickle.load(f)

    def load_model_state_dict(self):
        """
        Load the model state dictionary.
        """
        state_dict = torch.load(f'{self.data_prefix}/{self.model_path}/{self.model_type}_latest.pth',
                                map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict)

    def update_feature_info_dict(self):
        """
        Update the feature information dictionary.
        """
        self.feature_info_dict = self.model.feature_info_dict
        for key in self.feature_info_dict:
            if self.feature_info_dict[key]['dtype'] == 'scalar':
                self.feature_info_dict[key]['scale'] = self.generate_info_dict[key]['scale']
                self.feature_info_dict[key]['shift'] = self.generate_info_dict[key]['shift']

    def load_pv_data(self):
        """
        Load the traffic data.
        """
        self.pv_df = pd.read_csv(f'{self.data_prefix}/data/{self.pv_data_name}')
        self.pv_np_data, _, _ = build_data_from_df(self.pv_df,
                                                  key_map_dict_name='str_key_map_dict.pkl',
                                                  additional_name=self.model_args.additional_name,
                                                  key_type='user',
                                                  normalize=self.model_args.build_data_normalize,
                                                  ori_info_dict=self.feature_info_dict,
                                                  args=self.model_args)

    def generate_and_pred(self, pv_num: int = 5001) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Generate and predict traffic data.

        Args:
            pv_num (int): Number of traffic data points.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]: Traffic test data, predicted values, category indices, time category counts, time counts, test size.
        """
        test_size, epoch_num = self.calculate_test_size_and_epoch_num(pv_num)
        # Allow sampling with replacement when pv_num > available data
        if test_size > len(self.pv_np_data):
            indices = np.random.choice(len(self.pv_np_data), size=test_size, replace=True)
            pv_np_test_data = self.pv_np_data[indices]
        else:
            np.random.shuffle(self.pv_np_data)
            pv_np_test_data = self.pv_np_data[:test_size]

        time_counts_list, time_idx, time_onehot = self.generate_time_data(pv_num)
        tc_counts_list, category_idx, category_onehot = self.generate_category_data(time_counts_list)

        pctr_key = 'pctr_clbr'
        pctr_shift, pctr_scale = self.generate_info_dict[pctr_key]['shift'], self.generate_info_dict[pctr_key]['scale']
        pcvr_key = 'pcvr_clbr'
        pcvr_shift, pcvr_scale = self.generate_info_dict[pcvr_key]['shift'], self.generate_info_dict[pcvr_key]['scale']

        value_all = self.predict_values(pv_np_test_data, time_onehot, category_onehot, pctr_shift, pctr_scale, pcvr_shift, pcvr_scale, epoch_num)

        self.denormalize_test_data(pv_np_test_data)

        return pv_np_test_data, value_all, category_idx, tc_counts_list, time_counts_list, test_size

    def calculate_test_size_and_epoch_num(self, pv_num: int) -> Tuple[int, int]:
        """
        Calculate the test size and epoch number.

        Args:
            pv_num (int): Number of traffic data points.

        Returns:
            Tuple[int, int]: Test size and epoch number.
        """
        test_size = pv_num - 1 if pv_num % self.batch_size == 1 else pv_num
        epoch_num = test_size // self.batch_size + 1 if test_size % self.batch_size != 0 else test_size // self.batch_size
        return test_size, epoch_num

    def generate_time_data(self, pv_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate time data.

        Args:
            pv_num (int): Number of traffic data points.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Time counts list, time indices, time one-hot encoding.
        """
        time_counts_list = (self.time_dist * pv_num).astype(int)
        time_int_check = time_counts_list.sum()
        time_residual = pv_num - time_int_check
        time_random_idx = np.random.randint(low=0, high=self.num_tick)
        time_counts_list[time_random_idx] += time_residual

        time_idx = np.concatenate([np.ones(count, dtype=int) * i * self.tick_multi for i, count in enumerate(time_counts_list)])
        time_onehot = self.time_eye[time_idx]
        return time_counts_list, time_idx, time_onehot

    def generate_category_data(self, time_counts_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate category data.

        Args:
            time_counts_list (np.ndarray): Time counts list.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Time category counts list, category indices, category one-hot encoding.
        """
        tc_counts_list = np.expand_dims(time_counts_list, axis=1).repeat(self.num_category, axis=1)
        tc_counts_list = (tc_counts_list * self.tc_dist).astype(int)

        for i in range(self.num_tick):
            correct_cnt = time_counts_list[i]
            trans_cnt = tc_counts_list[i].sum()
            tc_i_residual = correct_cnt - trans_cnt
            tc_random_idx = np.random.randint(low=0, high=self.num_category)
            tc_counts_list[i, tc_random_idx] += tc_i_residual

        category_idx = np.ones([tc_counts_list.sum(), self.num_category], dtype=int)
        for j in range(self.num_category):
            category_idx[:, j] = category_idx[:, j] * self.select_category[j]
        category_onehot = self.category_eye[category_idx]
        return tc_counts_list, category_idx, category_onehot

    def predict_values(self, pv_np_test_data: np.ndarray, time_onehot: np.ndarray, category_onehot: np.ndarray,
                       pctr_shift: float, pctr_scale: float, pcvr_shift: float, pcvr_scale: float, epoch_num: int) -> np.ndarray:
        """
        Predict values.

        Args:
            pv_np_test_data (np.ndarray): Traffic test data.
            time_onehot (np.ndarray): Time one-hot encoding.
            category_onehot (np.ndarray): Category one-hot encoding.
            pctr_shift (float): pctr shift.
            pctr_scale (float): pctr scale.
            pcvr_shift (float): pcvr shift.
            pcvr_scale (float): pcvr scale.
            epoch_num (int): Epoch number.

        Returns:
            np.ndarray: Predicted values.
        """
        value_all = []
        for i in range(epoch_num):
            f_data_i = pv_np_test_data[i * self.batch_size: (i + 1) * self.batch_size]
            t_onehot_i = time_onehot[i * self.batch_size: (i + 1) * self.batch_size]

            f_data_i = torch.from_numpy(f_data_i).to(device=self.device)
            t_onehot_i = torch.from_numpy(t_onehot_i).to(device=self.device)

            value_i = []
            for j in range(self.num_category):
                c_onehot_ij = category_onehot[i * self.batch_size: (i + 1) * self.batch_size, j]
                c_onehot_ij = torch.from_numpy(c_onehot_ij).to(device=self.device)

                y_pred_ij = self.model.pred(f_data_i, c_onehot_ij, t_onehot_i)
                y_pred_ij = y_pred_ij.detach().cpu().numpy()

                pctr_ij = y_pred_ij[:, 3] * pctr_scale + pctr_shift
                pcvr_ij = y_pred_ij[:, 5] * pcvr_scale + pcvr_shift

                value_ij = pctr_ij * pcvr_ij
                value_i.append(value_ij)

            value_i = np.stack(value_i, axis=1)
            value_all.append(value_i)

        return np.concatenate(value_all, axis=0)

    def denormalize_test_data(self, pv_np_test_data: np.ndarray):
        """
        Denormalize test data.

        Args:
            pv_np_test_data (np.ndarray): Traffic test data.
        """
        for key in self.feature_info_dict:
            if self.feature_info_dict[key]['dtype'] == 'scalar':
                key_start = self.feature_info_dict[key]['pos'][0]
                key_end = self.feature_info_dict[key]['pos'][1]
                key_shift, key_scale = self.feature_info_dict[key]['shift'], self.feature_info_dict[key]['scale']
                pv_np_test_data[:, key_start:key_end] = pv_np_test_data[:, key_start:key_end] * key_scale + key_shift