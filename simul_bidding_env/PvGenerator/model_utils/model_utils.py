
import torch
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from einops import rearrange
from torch import nn, Tensor
from typing import Optional, Any

# Constant definitions
EPSILON = 1e-20

# Helper functions
def get_key_list(text_file: str, split_key: bool = False) -> Any:
    """
    Get key list from a text file.

    Args:
        text_file (str): Path to the text file.
        split_key (bool): Whether to split by key.

    Returns:
        Any: List of keys or dictionary.
    """
    with open(text_file, 'r') as f:
        if not split_key:
            all_data = f.read().split('\n')
            result = all_data
        else:
            all_data = f.readlines()
            result = {}
            curr_split_key = None
            for data in all_data:
                if data.startswith('**'):
                    curr_split_key = data.strip('*\n')
                    result[curr_split_key] = []
                else:
                    result[curr_split_key].append(data.strip())
    return result

class RMSNorm(nn.Module):
    """
    RMS normalization.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)

class LinearAttention1d(nn.Module):
    """
    Linear attention.
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x: Tensor) -> Tensor:
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

class SelfLinearAttn1d(nn.Module):
    """
    Self-linear attention.
    """
    def __init__(self, input_dim: int, num_heads: int = 8, dim_heads: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.attn = LinearAttention1d(input_dim, heads=num_heads, dim_head=self.dim_heads)

    def forward(self, x: Tensor) -> Tensor:
        attn_input = x.unsqueeze(-1)
        ret = self.attn(attn_input)
        return ret.squeeze(-1)

class SelfAttn1d(nn.Module):
    """
    Self-attention.
    """
    def __init__(self, input_dim: int, num_heads: int = 8, embed_multi: int = 1):
        super().__init__()
        self.embed_dim = embed_multi * num_heads
        self.seq_len = input_dim // self.embed_dim
        self.input_dim = input_dim
        self.attn = MultiheadAttention(self.embed_dim, num_heads=num_heads)

    def forward(self, x: Tensor) -> Tensor:
        attn_input = x.reshape([-1, self.seq_len, self.embed_dim])
        attn_input = attn_input.permute(1, 0, 2)
        ret, _ = self.attn(attn_input, attn_input, attn_input)
        ret = ret.permute(1, 0, 2)
        return ret.reshape([-1, self.input_dim])

class SelfAttnSeq(nn.Module):
    """
    Self-attention sequence.
    """
    def __init__(self, input_dim: int, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.attn = MultiheadAttention(self.input_dim, num_heads=num_heads)

    def forward(self, x: Tensor) -> Tensor:
        attn_input = x.permute(1, 0, 2)
        ret, _ = self.attn(attn_input, attn_input, attn_input)
        return ret.permute(1, 0, 2)

class TransformerBlock(nn.Module):
    """
    Transformer block.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8, dropout_p: float = 0.1, seq_length: Optional[int] = None, seq_input: bool = False, linear_attn: bool = False):
        super().__init__()
        self.seq_input = seq_input
        if seq_input:
            self.attention = SelfAttnSeq(input_dim, num_heads=num_heads)
        else:
            if linear_attn:
                self.attention = SelfLinearAttn1d(input_dim, num_heads=num_heads)
            else:
                self.attention = SelfAttn1d(input_dim, num_heads=num_heads)

        if seq_input:
            self.norm1 = nn.BatchNorm1d(seq_length)
            self.norm2 = nn.BatchNorm1d(seq_length)
        else:
            self.norm1 = nn.BatchNorm1d(input_dim)
            self.norm2 = nn.BatchNorm1d(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input: Tensor) -> Tensor:
        residual = input
        input = self.attention(input)
        input = self.dropout(input)
        input = self.norm1(input + residual)
        residual = input
        input = self.feed_forward(input)
        input = self.dropout(input)
        input = self.norm2(input + residual)
        return input