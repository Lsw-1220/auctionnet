
import math
from typing import Optional, Tuple, List, Dict
import argparse
import numpy as np
import torch
from einops import rearrange
from torch import nn, einsum
from torch.nn import functional as F, MultiheadAttention

# Constants
EPSILON = 1e-20

def sample_gumbel(shape: Tuple[int], device: torch.device, eps: float = EPSILON) -> torch.Tensor:
    """Sample Gumbel noise."""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits: torch.Tensor, temperature: float = 1) -> torch.Tensor:
    """Sample from the Gumbel-Softmax distribution."""
    y = logits + sample_gumbel(logits.size(), logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits: torch.Tensor, temperature: float = 1, hard: bool = False) -> torch.Tensor:
    """Perform Gumbel-Softmax operation."""
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

def exists(val: Optional[torch.Tensor]) -> bool:
    """Check if a value exists."""
    return val is not None

def default(val: Optional[torch.Tensor], d: torch.Tensor) -> torch.Tensor:
    """Return default value if the given value does not exist."""
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]) -> torch.Tensor:
    """Extract values from a tensor based on indices."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule for beta values."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class Residual(nn.Module):
    """Residual connection module."""
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)

class PreNorm1d(nn.Module):
    """Pre-normalization module."""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x)

class LinearAttention1d(nn.Module):
    """Linear attention module for 1D data."""
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h=self.heads)
        return self.to_out(out)

class Attention1d(nn.Module):
    """Standard attention module for 1D data."""
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

class SelfLinearAttn1d(nn.Module):
    """Self-attention with linear attention for 1D data."""
    def __init__(self, input_dim: int, num_heads: int = 8, dim_heads: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.attn = LinearAttention1d(input_dim, heads=num_heads, dim_head=dim_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = x.unsqueeze(-1)
        ret = self.attn(attn_input)
        return ret.squeeze(-1)

class SelfAttn1d(nn.Module):
    """Self-attention for 1D data."""
    def __init__(self, input_dim: int, num_heads: int = 8, embed_multi: int = 1):
        super().__init__()
        self.embed_dim = embed_multi * num_heads
        self.seq_len = input_dim // self.embed_dim
        self.input_dim = input_dim
        self.attn = MultiheadAttention(self.embed_dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = x.reshape([-1, self.seq_len, self.embed_dim])
        attn_input = attn_input.permute(1, 0, 2)
        ret, _ = self.attn(attn_input, attn_input, attn_input)
        ret = ret.permute(1, 0, 2)
        return ret.reshape([-1, self.input_dim])

class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers."""
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8, dropout_p: float = 0.1, linear_attn: bool = False):
        super().__init__()
        if linear_attn:
            self.attention = SelfLinearAttn1d(input_dim, num_heads=num_heads)
        else:
            self.attention = SelfAttn1d(input_dim, num_heads=num_heads)

        self.norm1 = nn.BatchNorm1d(input_dim)
        self.norm2 = nn.BatchNorm1d(input_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.attention(x)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        return self.norm2(x + residual)

class LastLayer(nn.Module):
    """Final layer for the model."""
    def __init__(self, last_hidden_dim: int, output_dim: int, info_dict: Dict, mode: str, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.base_layer = nn.Sequential(
            nn.Linear(in_features=last_hidden_dim, out_features=last_hidden_dim),
            nn.BatchNorm1d(last_hidden_dim),
            nn.LeakyReLU()
        )
        self.head_list = nn.ModuleList()
        self.mode = mode
        self.special_key_flag = []
        if self.args.special_normalize:
            self.special_normalize_key_list = self.args.special_normalize_key_list.split(',')
        self.info_dict = info_dict
        self.onehot_key_list = []
        onehot_dim_all = 0
        for key in info_dict:
            if info_dict[key]['dtype'] == 'onehot':
                head_dim = info_dict[key]['pos'][1] - info_dict[key]['pos'][0]
                if key == 'zip_code':
                    sub_head_dim = head_dim // 6
                    for i in range(6):
                        sub_head_layer = nn.Linear(in_features=last_hidden_dim, out_features=sub_head_dim)
                        self.head_list.append(sub_head_layer)
                        self.onehot_key_list.append(f'zip_code_{i}')
                else:
                    head_layer = nn.Linear(in_features=last_hidden_dim, out_features=head_dim)
                    self.head_list.append(head_layer)
                    self.onehot_key_list.append(key)
                onehot_dim_all += head_dim
            else:
                if self.args.special_normalize:
                    if key in self.special_normalize_key_list:
                        self.special_key_flag.append(True)
                    else:
                        self.special_key_flag.append(False)
                else:
                    self.special_key_flag.append(False)
        scalar_dim = output_dim - onehot_dim_all
        scalar_layer = nn.Linear(in_features=last_hidden_dim, out_features=scalar_dim)
        self.head_list.append(scalar_layer)
        self.head_num = len(self.head_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_list = self.list_forward(x)
        return torch.cat(output_list, dim=-1)

    def list_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        base_x = self.base_layer(x)
        output_list = []
        for i in range(self.head_num):
            head_output = self.head_list[i](base_x)
            if i != self.head_num - 1:
                if self.mode == 'direct':
                    head_output = gumbel_softmax(head_output, hard=True)
                else:
                    head_output = F.softmax(head_output, dim=-1)
            else:
                if self.args.special_normalize:
                    dim_cnt = head_output.shape[-1]
                    for i in range(dim_cnt):
                        if self.special_key_flag[i] and self.args.special_normalize_type == 'minmax':
                            head_output[:, i] = torch.sigmoid(head_output[:, i])
                        else:
                            head_output[:, i] = torch.tanh(head_output[:, i])
                            if self.args.data_normalize_scale:
                                head_output[:, i] = head_output[:, i] * self.args.data_normalize_scale_value
                else:
                    if self.args.lastlayer_func == 'sigmoid':
                        head_output = torch.sigmoid(head_output)
                    elif self.args.lastlayer_func == 'tanh':
                        head_output = torch.tanh(head_output)
                    if self.args.data_normalize_scale:
                        head_output = head_output * self.args.data_normalize_scale_value
            output_list.append(head_output)
        return output_list

def Upsample1d(dim: int, dim_out: Optional[int] = None) -> nn.Sequential:
    """Upsample 1D data."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample1d(dim: int, dim_out: Optional[int] = None) -> nn.Conv1d:
    """Downsample 1D data."""
    return nn.Conv1d(dim, default(dim_out, dim), 3, 2, 1)

class Block1d(nn.Module):
    """Basic block for 1D data."""
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: Optional[Tuple[torch.Tensor]] = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)

class ResnetBlock1d(nn.Module):
    """Residual block for 1D data."""
    def __init__(self, dim: int, dim_out: int, time_emb_dim: Optional[int] = None, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block1d(dim, dim_out, groups=groups)
        self.block2 = Block1d(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)