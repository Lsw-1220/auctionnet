"""
Training entry for DGAB under the POMDP (partially observable) setting.
R0-R5 ablation configurations.

单卡:
    python -m main.train_dgab_po --use_stat --device cuda:0 --batch_size 8 --step_num 40000

多卡 DDP (4 卡):
    torchrun --nproc_per_node=4 -m main.train_dgab_po --ddp --batch_size 32 --step_num 20000
"""
import sys
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import random
import argparse
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler

from logger.logger import Logger
from bidding_train_env.baseline.common.utils_po import save_normalize_dict
from bidding_train_env.baseline.dgab.data_po import (
    DGABReplayBuffer, dgab_collate_fn,
)
from bidding_train_env.baseline.dgab.sharded_data_po import ShardedDGABReplayBuffer
from bidding_train_env.baseline.dgab.model_po import DGAB
from bidding_train_env.baseline.common.param_saver import save_params


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_ddp():
    """初始化 DDP 进程组, 返回 (rank, local_rank, world_size)."""
    dist.init_process_group(backend='nccl')
    rank       = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def train(args, logger):
    # ── DDP 设置 ──
    ddp = args.ddp
    if ddp:
        rank, local_rank, world_size = setup_ddp()
        device = f"cuda:{local_rank}"
    else:
        rank = 0
        world_size = 1
        device = args.device

    is_main = (rank == 0)

    g = torch.Generator()
    g.manual_seed(42 + rank)

    if is_main:
        param_path = save_params(args, save_dir="log", tag="po")
        logger.info(f"Parameters saved to {param_path}")

    # Auto-detect: directory of chunked pkls  →  ShardedDGABReplayBuffer
    #              single .pkl file           →  DGABReplayBuffer (original)
    if os.path.isdir(args.dir):
        if is_main:
            logger.info(f"Directory detected — using ShardedDGABReplayBuffer")
        replay_buffer = ShardedDGABReplayBuffer(
            device=device, data_dir=args.dir, use_stat=args.use_stat,
            use_obs=(args.obs_encoder_type != 'none'),
            max_imp=args.max_imp,
            sparse_v_credit=args.sparse_v_credit,
            log1p_stat_dims=args.log1p_stat_dims,
            reward_col=args.reward_col)
    else:
        replay_buffer = DGABReplayBuffer(
            device=device, data_path=args.dir, use_stat=args.use_stat,
            use_obs=(args.obs_encoder_type != 'none'),
            max_imp=args.max_imp,
            sparse_v_credit=args.sparse_v_credit,
            log1p_stat_dims=args.log1p_stat_dims,
            reward_col=args.reward_col)
    if is_main:
        n_traj = len(replay_buffer)
        logger.info(f"Replay buffer trajectories: {n_traj}, "
                    f"state_dim={replay_buffer.STATE_DIM}, "
                    f"obs_encoder={args.obs_encoder_type}  world_size={world_size}")

    sampler = WeightedRandomSampler(replay_buffer.p_sample,
                                    num_samples=args.step_num * args.batch_size,
                                    replacement=True, generator=g)
    collate_fn = partial(dgab_collate_fn, device=device)
    dataloader = DataLoader(replay_buffer, sampler=sampler,
                            batch_size=args.batch_size, collate_fn=collate_fn,
                            generator=g)

    block_config = {
        'n_ctx':              args.n_ctx,
        'n_embd':             args.hidden_size,
        'n_layer':            args.n_layer,
        'n_head':             args.n_head,
        'n_inner':            args.n_inner,
        'activation_function': args.activation_function,
        'n_position':         args.n_position,
        'resid_pdrop':        args.resid_pdrop,
        'attn_pdrop':         args.attn_pdrop,
    }

    state_dim = replay_buffer.STATE_DIM   # base state dim (without obs snapshot)
    model = DGAB(
        base_state_dim=state_dim, act_dim=1,
        hidden_size=args.hidden_size,
        max_ep_len=args.max_ep_len,
        time_dim=args.time_dim,
        block_config=block_config,
        tau_v=args.tau_v, tau_c=args.tau_c,
        alpha=args.alpha,
        rtg_dropout=args.rtg_dropout,
        lambda_critic=args.lambda_critic, lambda_actor=args.lambda_actor,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        actor_type=args.actor_type,
        critic_type=args.critic_type,
        obs_encoder_type=args.obs_encoder_type,
        obs_dim=replay_buffer.OBS_DIM,
        macro_dim=2,
        device=device,
    )
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        _model = model.module
    else:
        _model = model

    _model.train()

    effective_batch = args.batch_size * world_size
    for i, batch in enumerate(dataloader, start=1):
        if ddp:
            m = model.module.step(batch)
        else:
            m = model.step(batch)

        if i % args.loss_report == 0 or i == 1:
            if is_main:
                logger.info(
                    "step={i} total={total:.4f} critic={critic:.4f} "
                    "(V_mse={v_mse:.4f} C_mse={c_mse:.4f} V_exp={v_exp:.4f} C_exp={c_exp:.4f}) "
                    "actor={actor:.4f} "
                    "w_mean={w_mean:.3f} w_std={w_std:.3f} adv_std={adv_std:.4f} "
                    "S_orig={s_o:.3f} S_expl={s_e:.3f} beta={beta:.3f} "
                    "eff_bs={eff_bs}".format(
                        i=i, total=m['loss_total'], critic=m['loss_critic'],
                        v_mse=m['loss_v_mse'], c_mse=m['loss_c_mse'],
                        v_exp=m['loss_v_exp'], c_exp=m['loss_c_exp'],
                        actor=m['loss_actor'],
                        w_mean=m['w_mean'], w_std=m['w_std'], adv_std=m['adv_std'],
                        s_o=m['s_orig_mean'], s_e=m['s_expl_mean'], beta=m['beta_mean'],
                        eff_bs=effective_batch))

    # ── 保存 (仅主进程) ──
    if is_main:
        time_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = args.save_dir + "_" + time_now
        os.makedirs(save_dir, exist_ok=True)
        torch.save(_model.state_dict(), os.path.join(save_dir, "complete_train.pt"))
        save_normalize_dict({
            'resourceleft_mean': replay_buffer.rl_mean,
            'resourceleft_std':  replay_buffer.rl_std,
            'observation_mean':  replay_buffer.ob_mean,
            'observation_std':   replay_buffer.ob_std,
            'stat_mean':         replay_buffer.stat_mean,
            'stat_std':          replay_buffer.stat_std,
            'scale':             replay_buffer.scale,
            'rtg_scale':         replay_buffer.scale,  # PO 管线用 scale 作为 RTG 缩放因子
            'obs_encoder_type':  args.obs_encoder_type,
            'actor_type':        args.actor_type,
            'critic_type':       args.critic_type,
            'base_state_dim':    replay_buffer.STATE_DIM,
            'obs_dim':           replay_buffer.OBS_DIM,
            'sparse_v_credit':   args.sparse_v_credit,
            'log1p_stat_dims':   args.log1p_stat_dims,
            'sparse_stat_dims':  DGABReplayBuffer.SPARSE_STAT_DIMS,
            'log1p_scale':       DGABReplayBuffer.LOG1P_SCALE,
        }, save_dir)
        logger.info(f"Model and normalize dict saved to {save_dir}")

    if ddp:
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data / runtime
    parser.add_argument('--step_num', type=int, default=40000)
    parser.add_argument('--dir', type=str,
                        default="../data/data/training_data_all-Series.pkl")
    parser.add_argument('--save_dir', type=str, default="./saved_model/GAVEv2")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--loss_report', type=int, default=100)
    parser.add_argument('--ddp', action='store_true',
                        help="启用 DistributedDataParallel 多卡训练 (通过 torchrun 启动)")
    # actor (DT) hyperparams  (state_dim 从 buffer.STATE_DIM 自动推断, 无需传入)
    parser.add_argument('--use_stat', action=argparse.BooleanOptionalAction, default=True,
                        help="--use_stat=R1/R4/R5 加市场统计量(default); --no-use_stat=R0/R2/R3 仅 resourceleft")
    parser.add_argument('--actor_type', type=str, default='stack',
                        choices=['stack', 'cross_attn', 'cross_attn_r7'],
                        help="stack (R0-R3); cross_attn (R4-R6); cross_attn_r7 (R7: stat_t in memory only)")
    parser.add_argument('--critic_type', type=str, default='sequence',
                        choices=['sequence', 'mlp'],
                        help="sequence=SequenceCritic (s,a,r) autoregressive (default); mlp=legacy DualHeadCritic")
    parser.add_argument('--obs_encoder_type', type=str, default='none',
                        choices=['none', 'cls', 'bidformer'],
                        help="none=R0/R1/R4; cls=R2; bidformer=R3/R5")
    parser.add_argument('--max_imp', type=int, default=512,
                        help="每步曝光集 padding 上限 (use_obs=True 时生效)")
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--max_ep_len', type=int, default=96)
    parser.add_argument('--time_dim', type=int, default=8)
    parser.add_argument('--n_ctx', type=int, default=1024)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--n_inner', type=int, default=1024)
    parser.add_argument('--activation_function', type=str, default="relu")
    parser.add_argument('--n_position', type=int, default=1024)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    # losses / optim
    parser.add_argument('--tau_v', type=float, default=0.99)
    parser.add_argument('--tau_c', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--rtg_dropout', type=float, default=0.0,
                        help="RTG token dropout rate for critic (0.0=off, 0.1-0.2 recommended for sparse signals)")
    parser.add_argument('--lambda_critic', type=float, default=1.0)
    parser.add_argument('--lambda_actor',  type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--sparse_v_credit', type=float, default=0.0,
                        help="V channel pseudo-V credit factor for sparse environments. "
                             "0.0=disabled; 0.1~0.3 recommended.")
    parser.add_argument('--log1p_stat_dims', action='store_true',
                        help="Apply log1p(x*1000) to PV/Conv sparse stat dims before z-score, "
                             "preventing signal collapse in near-zero-variance dimensions.")
    parser.add_argument('--reward_col', type=str, default='reward',
                        help="reward column name: 'reward' (sparse int) or 'reward_continous' (dense float)")

    args = parser.parse_args()

    # torchrun 设置的环境变量优先级高于 --device
    if 'LOCAL_RANK' in os.environ and not args.ddp:
        args.ddp = True  # torchrun 启动时自动识别

    if not args.ddp:
        set_global_seed(42)

    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger_obj = Logger(save_path="log",
                        file_name=f"DGAB_train_{time_stamp}.log")
    logger = logger_obj.get_logger()

    train(args, logger)
