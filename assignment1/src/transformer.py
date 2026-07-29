import torch
from torch import nn
import einops
from einops import rearrange, einsum
from math import sqrt, exp
import numpy
from layers import Linear, Embedding, MultiHeadSelfAttention, RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=None, max_seq_len=None, **kwargs):
        super().__init__()
        self.ln1 = RMSNorm(d_model, **kwargs)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len, **kwargs)
        self.ln2 = RMSNorm(d_model, **kwargs)
        self.ffn = SwiGLU(d_model, d_ff, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sublayer1_out = x + self.attn(self.ln1(x))
        sublayer2_out = sublayer1_out + self.ffn(self.ln2(sublayer1_out))
        return sublayer2_out


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, num_heads: int, d_model: int, d_ff: int, rope_theta: float, **kwargs):
        super().__init__()
        self.input_embedding = Embedding(vocab_size, d_model, **kwargs)
        self.transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length)
                for _ in range(num_layers)]
        )
        self.rms_norm = RMSNorm(d_model, **kwargs)
        self.lm_head = Linear(d_model, vocab_size, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.rms_norm(x)
        x = self.lm_head(x)
        return x


