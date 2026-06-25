

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Dict, Optional

from simul_bidding_env.PvGenerator.model_utils.model_utils import TransformerBlock

# Constant definitions
DEVICE_DEFAULT = 'cpu'

# Class definitions

class PvModel(nn.Module):
    """
    The PvModel class is used for traffic value prediction tasks.

    Args:
        input_dim (int): Input dimension.
        info_dict (dict): Information dictionary.
        args (Optional[Any]): Arguments object.
        device (str): Device type, default is 'cpu'.
        hidden_dims (Optional[List[int]]): List of hidden layer dimensions.
    """

    def __init__(self,
                 input_dim: int,
                 info_dict: Dict,
                 args=None,
                 device: str = DEVICE_DEFAULT,
                 hidden_dims: Optional[List[int]] = None) -> None:
        super(PvModel, self).__init__()
        self.input_dim = input_dim
        self.info_dict = deepcopy(info_dict)
        self.args = args
        self.device = device
        self.timesteps = args.diffusion_timesteps
        self.loss_type = args.diffusion_loss_type
        self.denoise_depth = args.diffusion_denoise_depth
        self.with_vae = args.diffusion_with_vae
        self.seq_input = args.diffusion_with_seq
        self.pv_arch = args.pv_arch
        self.pv_multi = eval(args.pv_multi)
        self.pv_init_dim = args.pv_init_dim

        self.condition_key = ['id_age', 'time_stamp']
        self.y_label_key_list = ['pctr_clbr_ratio', 'pcfr_clbr_ratio', 'ppcvr_clbr_ratio',
                                 'pctr_clbr', 'pcfr_clbr', 'pcvr_clbr', 'win_price', 'win_cost']

        if self.args.pv_select_pred:
            self.setup_pv_select_mask()

        self.setup_for_parse()
        self.setup_model_layers(hidden_dims)

    def setup_pv_select_mask(self):
        """
        Set up pv_select_mask.
        """
        self.pv_select_key_list = self.args.pv_select_key_list.split(',')
        self.pv_select_mask = [1.0 if key in self.pv_select_key_list else 0.0 for key in self.y_label_key_list]
        self.pv_select_mask = torch.tensor(np.array(self.pv_select_mask)).to(self.device).detach()

    def setup_for_parse(self):
        """
        Parse input data and set up related parameters.
        """
        bias = 0
        feature_info_dict = {}
        feature_key_cnt = 0
        self.dis_start_list = []
        self.dis_end_list = []
        dis_curr_start = -1
        dis_curr_end = -1
        self.time_start_pos = -1
        self.time_end_pos = -1
        self.category_start_pos = -1
        self.category_end_pos = -1
        self.scalar_start_pos = -1
        self.scalar_end_pos = -1
        self.y_label_start_pos = -1
        self.y_label_end_pos = -1

        for key in self.info_dict:
            if key in self.condition_key or key in self.y_label_key_list:
                if dis_curr_start > -1:
                    self.dis_start_list.append(dis_curr_start)
                    self.dis_end_list.append(dis_curr_end)
                    dis_curr_start, dis_curr_end = -1, -1

                if key == 'time_stamp':
                    self.time_start_pos, self.time_end_pos = self.info_dict[key]['pos']
                if key == 'id_age':
                    self.category_start_pos, self.category_end_pos = self.info_dict[key]['pos']

                if key in self.y_label_key_list:
                    if self.y_label_start_pos == -1:
                        self.y_label_start_pos, self.y_label_end_pos = self.info_dict[key]['pos']
                    else:
                        self.y_label_end_pos = self.info_dict[key]['pos'][1]

                bias += (self.info_dict[key]['pos'][1] - self.info_dict[key]['pos'][0])
                continue
            else:
                key_dtype = self.info_dict[key]['dtype']
                if key_dtype == 'onehot':
                    if dis_curr_start == -1:
                        dis_curr_start, dis_curr_end = self.info_dict[key]['pos']
                    else:
                        dis_curr_end = self.info_dict[key]['pos'][1]
                elif key_dtype == 'scalar':
                    if dis_curr_start > -1:
                        self.dis_start_list.append(dis_curr_start)
                        self.dis_end_list.append(dis_curr_end)
                        dis_curr_start, dis_curr_end = -1, -1

                    if self.scalar_start_pos == -1:
                        self.scalar_start_pos, self.scalar_end_pos = self.info_dict[key]['pos']
                    else:
                        self.scalar_end_pos = self.info_dict[key]['pos'][1]

                feature_info_dict[key] = deepcopy(self.info_dict[key])
                feature_info_dict[key]['pos'][0] -= bias
                feature_info_dict[key]['pos'][1] -= bias
                if key == 'zip_code':
                    feature_key_cnt += 6
                else:
                    feature_key_cnt += 1

        self.feature_info_dict = feature_info_dict
        self.feature_input_dim = self.input_dim - bias
        self.feature_seq_length = feature_key_cnt
        self.y_label_dim = self.y_label_end_pos - self.y_label_start_pos
        self.time_input_dim = self.time_end_pos - self.time_start_pos
        self.category_input_dim = self.category_end_pos - self.category_start_pos

    def setup_model_layers(self, hidden_dims: Optional[List[int]]):
        """
        Set up model layers.

        Args:
            hidden_dims (Optional[List[int]]): List of hidden layer dimensions.
        """
        if hidden_dims is None:
            if self.pv_arch == 'vector':
                hidden_dims = [256, 512, 256]

        self.embed_dim = hidden_dims[0]

        if self.pv_arch == 'vector':
            self.time_mlp = None
            self.vec_input_dim = self.feature_input_dim + self.category_input_dim + self.time_input_dim
            self.vec_first_layer = nn.Sequential(
                nn.Linear(in_features=self.vec_input_dim, out_features=self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.LeakyReLU()).to(self.device)

            vec_encoder = []
            for i in range(len(hidden_dims) - 1):
                bn_layer_i_1 = nn.BatchNorm1d(hidden_dims[i])
                bn_layer_i_2 = nn.BatchNorm1d(hidden_dims[i + 1])
                trans_block = TransformerBlock(input_dim=hidden_dims[i], hidden_dim=hidden_dims[i], num_heads=self.args.attn_heads)
                layer_test = nn.Sequential(
                    trans_block,
                    bn_layer_i_1,
                    nn.LeakyReLU(),
                    nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1]),
                    bn_layer_i_2,
                    nn.LeakyReLU()).to(self.device)
                vec_encoder.append(layer_test)

            self.vec_encoder = nn.Sequential(*vec_encoder).to(self.device)
            self.pred_layer = nn.Sequential(
                nn.Linear(in_features=hidden_dims[-1], out_features=self.args.Pv_pred_hidden_dim),
                nn.BatchNorm1d(self.args.Pv_pred_hidden_dim),
                nn.GELU(),
                nn.Linear(in_features=self.args.Pv_pred_hidden_dim, out_features=self.y_label_dim),
            ).to(self.device)

    def parse_data(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Parse input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): Parsed feature, time, category, and label.
        """
        time_onehot = x[:, self.time_start_pos:self.time_end_pos]
        y_label = x[:, self.y_label_start_pos:self.y_label_end_pos]
        category_onehot = x[:, self.category_start_pos:self.category_end_pos]
        feature_list = []
        for start, end in zip(self.dis_start_list, self.dis_end_list):
            feature_list.append(x[:, start:end])
        feature_list.append(x[:, self.scalar_start_pos:self.scalar_end_pos])
        feature = torch.cat(feature_list, dim=-1)
        return feature, time_onehot, category_onehot, y_label

    def pred(self, feature: torch.Tensor, category_onehot: torch.Tensor, time_onehot: torch.Tensor) -> torch.Tensor:
        """
        Predict output.

        Args:
            feature (torch.Tensor): Feature.
            category_onehot (torch.Tensor): Category one-hot encoding.
            time_onehot (torch.Tensor): Time one-hot encoding.

        Returns:
            torch.Tensor: Prediction result.
        """
        if self.pv_arch == 'vector':
            all_feature = torch.cat([feature, category_onehot, time_onehot], dim=-1)
            x = self.vec_first_layer(all_feature)
            x = self.vec_encoder(x)
            x = self.pred_layer(x)
            y_pred = self.apply_last_func(x)

        if self.args.pv_select_pred:
            y_pred = y_pred * self.pv_select_mask.unsqueeze(0).to(y_pred.device)
            for i in range(len(self.pv_select_mask)):
                if self.pv_select_mask[i] < 1.0:
                    y_pred[:, i] = y_pred[:, i].detach()

        return y_pred

    def apply_last_func(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the activation function of the last layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying the activation function.
        """
        if self.args.pv_last_func == 'linear':
            y_pred = x
        elif self.args.pv_last_func == 'sigmoid':
            y_pred = torch.sigmoid(x)
        elif self.args.pv_last_func == 'relu':
            y_pred = torch.relu(x)
        elif self.args.pv_last_func == 'tanh':
            y_pred = torch.tanh(x)

            if self.args.data_normalize_scale:
                y_pred = y_pred * self.args.data_normalize_scale_value

        return y_pred

    def process_and_pred(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Process input data and make predictions.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            (torch.Tensor, torch.Tensor): Prediction result and label.
        """
        feature, time_onehot, category_onehot, y_label = self.parse_data(x)
        category_int = torch.argmax(category_onehot, dim=-1)
        time_int = torch.argmax(time_onehot, dim=-1)
        y_pred = self.pred(feature, category_int, time_int, category_onehot, time_onehot)

        if self.args.pv_select_pred:
            y_label = y_label * self.pv_select_mask.unsqueeze(0).to(y_pred.device)

        return y_pred, y_label

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            *args: Input arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dict[str, torch.Tensor]: Loss dictionary.
        """
        x = args[0]
        y_pred, y_label = self.process_and_pred(x)
        loss = F.mse_loss(y_pred, y_label)
        return {'loss': loss, 'Reconstruction_Loss': loss.detach(), 'KLD': loss.detach()}

    def generate(self, x: torch.Tensor) -> (np.ndarray, np.ndarray, torch.Tensor):
        """
        Generate prediction data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            (np.ndarray, np.ndarray, torch.Tensor): Time data, category data, and prediction result.
        """
        feature, time_onehot, category_onehot, y_label = self.parse_data(x)
        time_int = torch.argmax(time_onehot, dim=-1)
        category_int = torch.argmax(category_onehot, dim=-1)
        y_pred = self.pred(feature, category_int, time_int)

        bs = feature.shape[0]
        time_data_list = []
        for t in range(self.time_input_dim):
            t_int_input = torch.ones_like(time_int) * t
            t_y_pred = self.pred(feature, category_int, t_int_input)
            time_data_list.append(t_y_pred)

        time_data = torch.stack(time_data_list, dim=1).detach().cpu().numpy()

        category_data_list = []
        for c in range(self.category_input_dim):
            c_int_input = torch.ones_like(category_int) * c
            c_y_pred = self.pred(feature, c_int_input, time_int)
            category_data_list.append(c_y_pred)

        category_data = torch.stack(category_data_list, dim=1).detach().cpu().numpy()

        return time_data, category_data, y_pred