import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.diffusion import DiffusionModel
from models.PvModel import PvModel
from utils import seed_everything, draw_stats, process_numpy_data, build_data_from_df

# Constants
DEFAULT_SEED = 1
DEFAULT_BATCH_SIZE = 128
DEFAULT_TEST_BATCH_SIZE = 32
DEFAULT_DATA_LOAD_WORKERS = 0
DEFAULT_LATENT_DIM = 64
DEFAULT_KLD_WEIGHT = 2.5e-6
DEFAULT_LR = 5e-4
DEFAULT_EPOCHS = 1000
DEFAULT_TRAIN_LOG_INTERVAL = 1
DEFAULT_DATA_LOAD_INTERVAL = 1
DEFAULT_MAX_GRAD_NORM = 20
DEFAULT_LOSS_MODE = 'direct'
DEFAULT_DATA_PATH = 'data/diffusion_startup.xlsx'
DEFAULT_USE_CUDA = True
DEFAULT_GENERATE_MODE = 'argmax'
DEFAULT_DATA_SAMPLE_EPOCH = 5
DEFAULT_DATA_SAMPLE_BATCH_SIZE = 2000
DEFAULT_DATA_SAMPLE_INIT = 0
DEFAULT_DIFFUSION_TIMESTEPS = 100
DEFAULT_DIFFUSION_DENOISE_DEPTH = 100
DEFAULT_DIFFUSION_LOSS_TYPE = 'huber'
DEFAULT_DIFFUSION_MULTI = '[1,2,4,8]'
DEFAULT_DIFFUSION_INIT_DIM = 256
DEFAULT_DIFFUSION_VAE_HIDDEN = '[512,512,256]'
DEFAULT_PV_MULTI = '[1,2,4,8]'
DEFAULT_PV_INIT_DIM = 64
DEFAULT_PV_ARCH = 'vector'
DEFAULT_ATTN_HEADS = 8
DEFAULT_EXPERIMENT_NAME = 'id_age_level_test'
DEFAULT_ALGO_NAME = 'Pv' #diffusion or Pv
DEFAULT_PV_PRED_HIDDEN=256
DEFAULT_PV_SELECT_PRED = False
DEFAULT_PV_LABEL_MINMAX = False
DEFAULT_PV_SELECT_KEY_LIST = 'pctr_clbr,pcfr_clbr,pcvr_clbr'
DEFAULT_PV_LAST_FUNC = 'sigmoid'
DEFAULT_LASTLAYER_FUNC = 'tanh'
DEFAULT_DATA_NORMALIZE_SCALE_VALUE = 5.0
DEFAULT_DATA_NORMALIZE_SCALE = False
DEFAULT_SPECIAL_NORMALIZE = False
DEFAULT_SPECIAL_NORMALIZE_KEY_LIST = 'id_birthyear'
DEFAULT_SPECIAL_NORMALIZE_TYPE = 'minmax'
DEFAULT_BUILD_DATA_NORMALIZE = 'mean'

# Logging setup
logging.basicConfig(level=logging.INFO)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Model training and testing script.")
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help="Random seed for reproducibility.")
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training.")
    parser.add_argument('--test_batch_size', type=int, default=DEFAULT_TEST_BATCH_SIZE, help="Batch size for testing.")
    parser.add_argument('--data_load_workers', type=int, default=DEFAULT_DATA_LOAD_WORKERS, help="Number of data loader workers.")
    parser.add_argument('--latent_dim', type=int, default=DEFAULT_LATENT_DIM, help="Latent dimension for the model.")
    parser.add_argument('--kld_weight', type=float, default=DEFAULT_KLD_WEIGHT, help="Weight for KLD loss.")
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help="Learning rate for optimizer.")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument('--train_log_interval', type=int, default=DEFAULT_TRAIN_LOG_INTERVAL, help="Interval for training log outputs.")
    parser.add_argument('--data_load_interval', type=int, default=DEFAULT_DATA_LOAD_INTERVAL, help="Interval for data loading during training.")
    parser.add_argument('--max_grad_norm', type=int, default=DEFAULT_MAX_GRAD_NORM, help="Max gradient norm for clipping.")
    parser.add_argument('--loss_mode', type=str, default=DEFAULT_LOSS_MODE, help="Loss mode for the model.")
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help="Path to the data file.")
    parser.add_argument('--use_cuda', type=bool, default=DEFAULT_USE_CUDA, help="Flag to use CUDA for computations.")
    parser.add_argument('--generate_mode', type=str, default=DEFAULT_GENERATE_MODE, help="Generation mode for the model.")
    parser.add_argument('--data_sample_epoch', type=int, default=DEFAULT_DATA_SAMPLE_EPOCH, help="Epoch interval for data sampling.")
    parser.add_argument('--data_sample_batch_size', type=int, default=DEFAULT_DATA_SAMPLE_BATCH_SIZE, help="Batch size for data sampling.")
    parser.add_argument('--data_sample_init', type=int, default=DEFAULT_DATA_SAMPLE_INIT, help="Initial value for data sampling.")
    parser.add_argument('--diffusion_timesteps', type=int, default=DEFAULT_DIFFUSION_TIMESTEPS, help="Number of diffusion timesteps.")
    parser.add_argument('--diffusion_denoise_depth', type=int, default=DEFAULT_DIFFUSION_DENOISE_DEPTH, help="Denoising depth for diffusion model.")
    parser.add_argument('--diffusion_loss_type', type=str, default=DEFAULT_DIFFUSION_LOSS_TYPE, help="Loss type for diffusion model.")
    parser.add_argument('--diffusion_multi', type=str, default=DEFAULT_DIFFUSION_MULTI, help="Multipliers for diffusion model.")
    parser.add_argument('--diffusion_init_dim', type=int, default=DEFAULT_DIFFUSION_INIT_DIM, help="Initial dimension for diffusion model.")
    parser.add_argument('--diffusion_vae_hidden', type=str, default=DEFAULT_DIFFUSION_VAE_HIDDEN, help="Hidden layers for diffusion VAE.")
    parser.add_argument('--pv_multi', type=str, default=DEFAULT_PV_MULTI, help="Multipliers for Pv model.")
    parser.add_argument('--pv_init_dim', type=int, default=DEFAULT_PV_INIT_DIM, help="Initial dimension for Pv model.")
    parser.add_argument('--pv_arch', type=str, default=DEFAULT_PV_ARCH, help="Architecture for Pv model.")
    parser.add_argument('--Pv_pred_hidden_dim', type=int, default=DEFAULT_PV_PRED_HIDDEN)
    parser.add_argument('--attn_heads', type=int, default=DEFAULT_ATTN_HEADS, help="Number of attention heads.")
    parser.add_argument('--experiment_name', type=str, default=DEFAULT_EXPERIMENT_NAME, help="Name of the experiment.")
    parser.add_argument('--algo_name', type=str, default=DEFAULT_ALGO_NAME, help="Algorithm name: 'diffusion' or 'Pv'.")
    parser.add_argument('--pv_select_pred', type=bool, default=DEFAULT_PV_SELECT_PRED, help="Flag to select prediction for Pv model.")
    parser.add_argument('--pv_label_minmax', type=bool, default=DEFAULT_PV_LABEL_MINMAX, help="Flag to use min-max normalization for Pv labels.")
    parser.add_argument('--pv_select_key_list', type=str, default=DEFAULT_PV_SELECT_KEY_LIST, help="Key list for Pv selection.")
    parser.add_argument('--pv_last_func', type=str, default=DEFAULT_PV_LAST_FUNC, help="Last function for Pv model.")
    parser.add_argument('--lastlayer_func', type=str, default=DEFAULT_LASTLAYER_FUNC, help="Last layer function for the model.")
    parser.add_argument('--data_normalize_scale_value', type=float, default=DEFAULT_DATA_NORMALIZE_SCALE_VALUE, help="Scale value for data normalization.")
    parser.add_argument('--data_normalize_scale', type=bool, default=DEFAULT_DATA_NORMALIZE_SCALE, help="Flag to normalize data scale.")
    parser.add_argument('--special_normalize', type=bool, default=DEFAULT_SPECIAL_NORMALIZE, help="Flag to use special normalization.")
    parser.add_argument('--special_normalize_key_list', type=str, default=DEFAULT_SPECIAL_NORMALIZE_KEY_LIST, help="Key list for special normalization.")
    parser.add_argument('--special_normalize_type', type=str, default=DEFAULT_SPECIAL_NORMALIZE_TYPE, help="Type of special normalization.")
    parser.add_argument('--build_data_normalize', type=str, default=DEFAULT_BUILD_DATA_NORMALIZE, help="Normalization method for building data.")
    return parser.parse_args()

def load_data(args: argparse.Namespace,  pv_model: bool) -> Tuple[pd.DataFrame, str]:
    """Load data based on the epoch and model type."""

    if pv_model:
        data_file = 'data/pv_startup.xlsx'
        key_type = 'all'
    else:
        data_file = args.data_path
        key_type = 'user'
    epoch_load_data = pd.read_excel(data_file)

    return epoch_load_data, key_type

def initialize_model(args: argparse.Namespace, feature_dim: int, info_dict: Dict, device: str) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Initialize the model and optimizer based on the algorithm name."""
    if args.algo_name == 'diffusion':
        model = DiffusionModel(input_dim=feature_dim, latent_dim=args.latent_dim, info_dict=info_dict, device=device, args=args).to(device=device, dtype=torch.float64)
    elif args.algo_name == 'Pv':
        model = PvModel(input_dim=feature_dim, info_dict=info_dict, device=device, args=args).to(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer

def train_epoch(args: argparse.Namespace, model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader, data_dict_all: Dict, epoch: int, device: str) -> float:
    """Train the model for one epoch."""
    model.train()
    epoch_train_loss = 0.0
    loss_batch_count = 0
    start_time = time.time()

    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()

        if args.algo_name == 'diffusion':
            loss_dict = model(data, noise=None, mode=args.loss_mode, M_N=args.kld_weight)
        elif args.algo_name == 'Pv':
            loss_dict = model(data)

        batch_loss = loss_dict['loss']
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=np.inf)
        optimizer.step()

        loss_value = loss_dict['loss'].item()
        recon_loss = loss_dict['Reconstruction_Loss'].item()
        kld_loss = loss_dict['KLD'].item()

        data_dict_all['train_loss'].append(loss_value / len(data))
        data_dict_all['train_recon_loss'].append(recon_loss / len(data))
        data_dict_all['train_KLD'].append(kld_loss / len(data))

        epoch_train_loss += loss_value
        loss_batch_count += 1

        if batch_idx % args.train_log_interval == 0:
            logging.info(f'Epoch {epoch} batch {batch_idx} ({100. * (batch_idx + 1) * len(data) / len(train_loader.dataset):.0f}%): Loss = {loss_value / len(data):.4f}')

    logging.info(f'Training time for epoch {epoch}: {time.time() - start_time:.2f} seconds')

    if loss_batch_count == 0:
        return 0.0

    avg_train_loss = epoch_train_loss / loss_batch_count
    logging.info(f'Average Train Loss for Epoch {epoch}: {avg_train_loss:.4f}')
    return avg_train_loss

def evaluate_model(args: argparse.Namespace, model: torch.nn.Module, test_loader: DataLoader, device: str) -> Tuple[float, float, float]:
    """Evaluate the model on the test dataset."""
    model.eval()
    total_test_loss = 0.0
    total_recon_loss = 0.0
    total_kld_loss = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data[0].to(device)
            logging.info(f'Test batch {batch_idx} with {data.shape[0]} data points.')

            if args.algo_name == 'diffusion':
                loss_dict = model(data, noise=None, mode=args.loss_mode, M_N=args.kld_weight)
            elif args.algo_name == 'Pv':
                loss_dict = model(data)

            total_test_loss += loss_dict['loss'].item()
            total_recon_loss += loss_dict['Reconstruction_Loss'].item()
            total_kld_loss += loss_dict['KLD'].item()

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    avg_recon_loss = total_recon_loss / len(test_loader.dataset)
    avg_kld_loss = total_kld_loss / len(test_loader.dataset)

    return avg_test_loss, avg_recon_loss, avg_kld_loss

def save_model_and_logs(args: argparse.Namespace, model: torch.nn.Module, data_dict_all: Dict, epoch: int, best_test_loss: float, checkpoint_path: str, plot_path: str, info_dict: Dict, name_dict: Dict) -> float:
    """Save the model and logs, and update the best test loss if necessary."""
    if data_dict_all['test_loss'][-1] < best_test_loss:
        best_test_loss = data_dict_all['test_loss'][-1]
        torch.save(model.state_dict(), f'{checkpoint_path}/{args.algo_name}_best.pth')
        logging.info(f'Updated best model in Epoch {epoch}')

    torch.save(model.state_dict(), f'{checkpoint_path}/{args.algo_name}_latest.pth')
    with open(f'{checkpoint_path}/{args.algo_name}_args.pkl', 'wb') as f:
        pickle.dump(args, f)
    with open(f'{checkpoint_path}/{args.algo_name}_name_dict.pkl', 'wb') as f:
        pickle.dump(name_dict, f)
    with open(f'{checkpoint_path}/{args.algo_name}_info_dict.pkl', 'wb') as f:
        pickle.dump(info_dict, f)
    with open(f'{checkpoint_path}/{args.algo_name}_args.txt', 'w') as f:
        f.write(str(args))

    for key in data_dict_all:
        draw_stats(plot_path, key, data_dict_all[key])
    with open(f'{plot_path}/data_dict.pkl', 'wb') as f:
        pickle.dump(data_dict_all, f)

    return best_test_loss

def model_training(device: str, experiment_name: str, name_dict: Dict, args: argparse.Namespace) -> None:
    """Train and evaluate the model."""
    plot_path = f'plot/{experiment_name}'
    checkpoint_path = f'checkpoint/{experiment_name}'
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    data_dict_all = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_KLD': [],
        'test_loss': [],
        'test_recon_loss': [],
        'test_KLD': []
    }
    best_test_loss = float('inf')
    sample_epoch_count = args.data_sample_init // args.data_sample_batch_size

    stats_table = None
    ori_info_dict = None

    for epoch in range(args.epochs):
        if epoch % args.data_load_interval == 0:
            epoch_load_data, key_type = load_data(args,  args.algo_name == 'Pv')
            sample_epoch_count += args.data_sample_epoch

            np_data, feature_dim, info_dict = build_data_from_df(
                epoch_load_data,
                key_map_dict_name=key_map_dict_name,
                additional_name=additional_name,
                key_type=key_type,
                normalize=args.build_data_normalize,
                ori_info_dict=ori_info_dict,
                stats_table=stats_table,
                args=args
            )

            train_loader, test_loader, _ = process_numpy_data(
                np_data,
                args.batch_size,
                args.test_batch_size,
                num_workers=args.data_load_workers
            )

            model, optimizer = initialize_model(args, feature_dim, info_dict, device)

        avg_train_loss = train_epoch(args, model, optimizer, train_loader, data_dict_all, epoch, device)

        if avg_train_loss == 0.0:
            continue

        avg_test_loss, avg_recon_loss, avg_kld_loss = evaluate_model(args, model, test_loader, device)
        data_dict_all['test_loss'].append(avg_test_loss)
        data_dict_all['test_recon_loss'].append(avg_recon_loss)
        data_dict_all['test_KLD'].append(avg_kld_loss)

        logging.info(f'Average Test Loss for Epoch {epoch}: {avg_test_loss:.4f}')

        best_test_loss = save_model_and_logs(args, model, data_dict_all, epoch, best_test_loss, checkpoint_path, plot_path, info_dict, name_dict)

if __name__ == "__main__":
    args = parse_arguments()

    seed_everything(args.seed)

    device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device: {device}')

    args.data_load_interval = 999999999999

    additional_name = None
    if args.algo_name == 'Pv':
        key_map_dict_name = 'all_str_key_map_dict.pkl'
    else:
        key_map_dict_name = 'str_key_map_dict.pkl'

    with open(f'stats_useful/{key_map_dict_name}', 'rb') as f:
        name_dict = pickle.load(f)

    model_training(device, args.experiment_name, name_dict, args)