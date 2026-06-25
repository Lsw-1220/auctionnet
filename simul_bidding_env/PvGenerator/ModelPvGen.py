

import torch
import numpy as np
from typing import List, Tuple
from simul_bidding_env.PvGenerator.model_utils.PV_pred import PvPredictor

# Constant definitions

DEVICE_DEFAULT = 'cpu'

# Set random seeds

torch.manual_seed(1)
np.random.seed(1)


class ModelPvGenerator():
    """
    The ModelPvGenerator class is used to generate traffic data.

    Args:        num_tick (int): Number of time ticks.
        num_agent_category (int): Number of agent categories.
        select_category (List[int]): List of selected categories.
        batch_size (int): Batch size.
        pv_num (int): Number of traffic data points.
        episodic_std (float): Standard deviation between days.
        episode (int): Current episode number.
        merge_category_agent_pv (bool): Whether to merge category agent traffic.
    """

    def __init__(self,
                 num_tick: int = 48,
                 num_agent_category: int = 8,
                 select_category: List[int] = [1, 2, 3, 4, 5, 6],
                 batch_size: int = 1000,
                 pv_num: int = 105000,
                 episodic_std: float = 0,
                 episode: int = 0,
                 merge_category_agent_pv: bool = True):
        super().__init__()

        self.num_tick = num_tick
        self.num_agent_category = num_agent_category
        self.num_category = len(select_category)
        self.num_agent = self.num_category * self.num_agent_category
        self.pv_num = int(pv_num * np.clip(np.random.normal(1, episodic_std), 0.5, 1.5))
        self.merge_category_agent_pv = merge_category_agent_pv
        self.pvalue_sigma_mean = 0.15
        self.pvalue_sigma_std = 0.06
        self.episode = episode
        self.value_noise_scale = 2e-5
        self.unique_noise_scale = 0.1

        self.predictor = PvPredictor(num_tick=num_tick, num_agent_category=num_agent_category,
                                     select_category=select_category, batch_size=batch_size)
        self.pv_features, self.pv_values, self.pValueSigmas = self.new_generate(self.episode)

    def reset(self, episode: int = 0):
        """
        Reset the model state.

        Args:            episode (int): Current episode number.
        """
        self.episode = episode
        self.pv_features, self.pv_values, self.pValueSigmas = self.new_generate(self.episode)

    def new_generate(self, episode: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Generate new traffic data.

        Args:            episode (int): Current episode number.

        Returns:            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]: List of features, values, and standard deviations.
        """
        self.episode = episode
        pv_feature_all, pv_values_all, pv_categorys_all, tc_counts_list, time_counts_list, actual_pv_num = self.predictor.generate_and_pred(
            self.pv_num)

        pv_feature_list, pv_values_list = [], []
        start_pos = 0
        for i in range(self.num_tick):
            t_cnt_i = time_counts_list[i]
            pv_feature_i = pv_feature_all[start_pos:start_pos + t_cnt_i]
            pv_feature_list.append(pv_feature_i)
            pv_values_i_agent = pv_values_all[start_pos:start_pos + t_cnt_i]
            pv_values_i_agent = np.expand_dims(pv_values_i_agent, axis=2)
            pv_values_i_agent = np.repeat(pv_values_i_agent, axis=2, repeats=self.num_agent_category)

            agent_scale_noise_i = self.scale_noise(scale=self.unique_noise_scale, shape=np.shape(pv_values_i_agent))
            agent_scale_noise_i = agent_scale_noise_i.detach().cpu().numpy()

            agent_shift_noise_i = self.shift_noise(scale=self.value_noise_scale, shape=np.shape(pv_values_i_agent))
            agent_shift_noise_i = agent_shift_noise_i.detach().cpu().numpy()

            pv_values_i_agent = pv_values_i_agent * agent_scale_noise_i + agent_shift_noise_i
            pv_values_i_agent = np.reshape(pv_values_i_agent, [t_cnt_i, -1])
            pv_values_list.append(pv_values_i_agent)
            start_pos += t_cnt_i

        pValueSigmas = self.generate_pvalue_sigma(pv_values_list)
        return pv_feature_list, pv_values_list, pValueSigmas

    def generate_pvalue_sigma(self, pvalues: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate standard deviations.

        Args:            pvalues (List[np.ndarray]): List of values.

        Returns:            List[np.ndarray]: List of standard deviations.
        """
        pValueSigmas = [np.empty_like(array) for array in pvalues]
        rg = np.random.default_rng(seed=self.episode)
        sigma_ratio_mean_agent = rg.normal(self.pvalue_sigma_mean, self.pvalue_sigma_std, self.num_agent)
        sigma_ratio_mean_agent[sigma_ratio_mean_agent < 0] *= -1
        sigma_ratio_mean_agent[sigma_ratio_mean_agent > 0.3] = 0.2
        means = sigma_ratio_mean_agent[np.newaxis, :]
        stds = (sigma_ratio_mean_agent / 2)[np.newaxis, :]

        for i, array in enumerate(pvalues):
            shape = array.shape
            rg = np.random.default_rng(seed=self.episode * 120000 + i)
            sampled_array = rg.normal(means, stds, shape)
            sampled_array[sampled_array < 0] *= -1
            sampled_array[sampled_array > 0.3] = 0.3
            pValueSigmas[i] = sampled_array * array
        return pValueSigmas

    def scale_noise(self, scale: float, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Generate scale noise.

        Args:            scale (float): Scale factor.
            shape (Tuple[int, ...]): Shape.

        Returns:            torch.Tensor: Scale noise.
        """
        unique_noise = torch.rand(shape)
        unique_noise = (unique_noise - 0.5) * 2 * scale + 1.0
        return unique_noise

    def shift_noise(self, scale: float, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Generate shift noise.

        Args:            scale (float): Scale factor.
            shape (Tuple[int, ...]): Shape.

        Returns:            torch.Tensor: Shift noise.
        """
        unique_noise = torch.rand(shape)
        return unique_noise * scale

