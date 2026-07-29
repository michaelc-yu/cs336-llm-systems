import torch
from torch import nn
import einops
from einops import rearrange, einsum
from math import sqrt, exp
import numpy

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        std = sqrt(2 / (in_features + out_features))
        init_weights = torch.nn.init.trunc_normal_(
            torch.zeros((out_features, in_features), dtype=dtype, device=device),
            mean = 0,
            std = std,
            a = -3 * std,
            b = 3 * std
        )
        self.W = nn.Parameter(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        init_weights = torch.nn.init.trunc_normal_(
            torch.zeros((num_embeddings, embedding_dim), dtype=dtype, device=device),
            mean = 0,
            std = 1,
            a = -3,
            b = 3
        )
        self.embedding_matrix = nn.Parameter(init_weights)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.long:
            token_ids = token_ids.to(torch.long)
        # token_ids: (batch_size, sequence_length)
        # advanced indexing into embedding matrix to produce output
        # output: (batch_size, sequence_length, embedding_dim)
        return self.embedding_matrix[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gi = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # x: (batch_size, sequence_length, d_model)
        x_square_mean = einops.reduce(x**2, "batch seq_len d_model -> batch seq_len", "mean")
        rms = torch.sqrt(x_square_mean + self.eps)
        rms = rearrange(rms, "batch seq_len -> batch seq_len 1")
        rms_norm = (x / rms) * self.gi

        return rms_norm.to(in_dtype)

def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation function."""
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        std = sqrt(2 / (d_model + d_ff))
        w1_init = torch.nn.init.trunc_normal_(
            torch.zeros(d_ff, d_model), mean=0, std=std, a=-3*std, b=3*std
        )
        self.W1 = nn.Parameter(w1_init)

        w2_init = torch.nn.init.trunc_normal_(
            torch.zeros(d_model, d_ff), mean=0, std=std, a=-3*std, b=3*std
        )
        self.W2 = nn.Parameter(w2_init)

        w3_init = torch.nn.init.trunc_normal_(
            torch.zeros(d_ff, d_model), mean=0, std=std, a=-3*std, b=3*std
        )
        self.W3 = nn.Parameter(w3_init)

    def forward(self, x):
        # x: ... d_model
        W1x = einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        W3x = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")

        z = silu(W1x)
        z = z * W3x
        res = einsum(z, self.W2, "... d_ff, d_model d_ff -> ... d_model")
        return res
    
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # compute cos and sin values
        # if max_seq_len = 4, this creates idx = [0, 1, 2, 3]
        idx = torch.arange(0, max_seq_len, device=device, dtype=dtype)
        denom = theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k)
        theta_i_k = idx.unsqueeze(1) / denom.unsqueeze(0)

        cos_cache = torch.cos(theta_i_k)
        sin_cache = torch.sin(theta_i_k)

        # makes tensors part of the module so caches move to GPU/CPU with the model
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        # return a tensor of same shape
        assert x.shape[-1] == self.d_k

        # get the cos and sin values for the given token positions
        cos_vals = self.cos_cache[token_positions]
        sin_vals = self.sin_cache[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rot = x_even * cos_vals - x_odd * sin_vals
        x_odd_rot = x_even * sin_vals + x_odd * cos_vals
        
        result = torch.zeros_like(x)
        result[..., 0::2] = x_even_rot  # Put rotated even dims back in positions 0,2,4,...
        result[..., 1::2] = x_odd_rot   # Put rotated odd dims back in positions 1,3,5,...
        
        return result

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Applies softmax to the i-th dimension of the input tensor"""
    # Numerical stability mathematical property: softmax(x) = softmax(x - c) for any constant c
    max_values, max_indices = torch.max(x, dim=dim, keepdim=True)
    x = x - max_values

    numerator = torch.exp(x)
    denom = torch.sum(numerator, dim=dim, keepdim=True)

    return numerator / denom

def scaled_dot_product_attention(Q, K, V, mask: torch.Tensor | None) -> torch.Tensor:
    """
    Args:
        Q: (... seq_len_q d_k)
        K: (... seq_len_k d_k)
        V: (... seq_len_k d_v)
        mask: (seq_len_q, seq_len_k), bool tensor.
    """
    assert Q.shape[-1] == K.shape[-1]
    assert K.shape[-2] == V.shape[-2]

    d_k = Q.shape[-1]

    attn_weights = einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k")
    attn_weights = attn_weights / sqrt(d_k)

    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

    attn = softmax(attn_weights, dim=-1)
    
    res = einsum(attn, V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    return res

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device = None,
        dtype = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        d = d_model // num_heads
        
        self.use_rope = False
        if theta and max_seq_len:
            self.use_rope = True
            self.rope = RoPE(theta=theta, d_k=d, max_seq_len=max_seq_len)
        
        std = sqrt(2 / (d_model + d_model))
        self.W_Q = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((d * num_heads, d_model)), mean=0, std=std, a=-3*std, b=3*std
            )
        )
        self.W_K = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((d * num_heads, d_model)), mean=0, std=std, a=-3*std, b=3*std
            )
        )
        self.W_V = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((d * num_heads, d_model)), mean=0, std=std, a=-3*std, b=3*std
            )
        )
        self.W_O = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((d_model, d * num_heads)), mean=0, std=std, a=-3*std, b=3*std
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: "... seq_len d_in"
        seq_len = x.shape[-2]
        # d_in should be equal to d_model
        d = self.d_model // self.num_heads

        # 1: project
        Q = einsum(self.W_Q, x, "d_out d_in, ... seq_len d_in -> ... seq_len d_out")
        K = einsum(self.W_K, x, "d_out d_in, ... seq_len d_in -> ... seq_len d_out")
        V = einsum(self.W_V, x, "d_out d_in, ... seq_len d_in -> ... seq_len d_out")

        # 2: reshape/split to (..., num_heads, seq_len, d)
        Q = rearrange(Q, "... seq_len (num_heads d_out) -> ... num_heads seq_len d_out", num_heads=self.num_heads, d_out=d)
        K = rearrange(K, "... seq_len (num_heads d_out) -> ... num_heads seq_len d_out", num_heads=self.num_heads, d_out=d)
        V = rearrange(V, "... seq_len (num_heads d_out) -> ... num_heads seq_len d_out", num_heads=self.num_heads, d_out=d)

        # 3: apply rope if needed
        if self.use_rope:
            # positions: (..., seq_len)
            positions = torch.arange(0, seq_len)
            Q = self.rope.forward(Q, positions)
            K = self.rope.forward(K, positions)
        
        # 4: scaled dot product attention
        # lower triangular mask to prevent attention to future tokens
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        attn_out = scaled_dot_product_attention(Q, K, V, mask)

        # 5: concat heads
        attn_out = rearrange(attn_out, "... num_heads seq_len d_out -> ... seq_len (num_heads d_out)", num_heads=self.num_heads, d_out=d)

        # 6: final linear projection
        result = einsum(attn_out, self.W_O, "... seq_len d_in, d_model d_in -> ... seq_len d_model")
        return result

