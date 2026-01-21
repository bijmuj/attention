import enum

import torch
from torch import nn
from torch.nn import functional as F


class ATTENTION_IMPLEMENTATIONS(enum.StrEnum):
    TORCH = enum.auto()
    EINSUM = enum.auto()
    MATMUL = enum.auto()


class Attention(nn.Module):
    def __init__(self, embedding_dims, attn_dims):
        super().__init__()
        self.attn_dims = attn_dims
        self.W_q = nn.Linear(embedding_dims, attn_dims)
        self.W_k = nn.Linear(embedding_dims, attn_dims)
        self.W_v = nn.Linear(embedding_dims, attn_dims)
        self.W_o = nn.Linear(attn_dims, embedding_dims)
        self.attn_dropout = nn.Dropout(0.2)
        self.residual_dropout = nn.Dropout(0.2)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.attn_dims**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn = attn_weights @ V
        return self.residual_dropout(self.W_o(attn))


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dims: int,
        attn_dims: int,
        num_heads: int,
        attn_impl: str,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_dims = attn_dims
        self.num_heads = num_heads
        self.W_qkv = nn.Linear(embedding_dims, num_heads * attn_dims * 3)
        self.W_o = nn.Linear(num_heads * attn_dims, embedding_dims)
        self.dropout = dropout

        assert attn_impl in ATTENTION_IMPLEMENTATIONS
        self.attn_impl = attn_impl

    def forward(self, x):
        B, L, _ = x.shape
        # Q, K, V shape: B, L, num_heads * attn_dims
        Q, K, V = self.W_qkv(x).split(self.num_heads * self.attn_dims, dim=2)

        # Q, K, V shape: B, num_heads, L, attn_dims
        Q = Q.view(B, L, self.num_heads, self.attn_dims).transpose(-2, -3)
        K = K.view(B, L, self.num_heads, self.attn_dims).transpose(-2, -3)
        V = V.view(B, L, self.num_heads, self.attn_dims).transpose(-2, -3)

        scale_factor = 1 / (self.attn_dims**0.5)

        if self.attn_impl == ATTENTION_IMPLEMENTATIONS.TORCH:
            attn_weight = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout, scale=scale_factor
            )
        elif self.attn_impl == ATTENTION_IMPLEMENTATIONS.EINSUM:
            attn_weight = forward_einsum(Q, K, V, self.dropout, scale_factor)
        elif self.attn_impl == ATTENTION_IMPLEMENTATIONS.MATMUL:
            attn_weight = forward_matmul(Q, K, V, self.dropout, scale_factor)
        else:
            raise NotImplementedError(
                f"self.attn_impl not set to a value of attention.ATTENTION_IMPLEMENTATIONS."
            )

        # attn_weight shape: B, num_heads, L, attn_dims ->  B, L, num_heads, attn_dims
        attn_weight = attn_weight.transpose(-2, -3).contiguous()

        attn_weight = attn_weight.view(B, L, -1)

        return F.dropout(self.W_o(attn_weight), self.dropout)


def forward_einsum(Q, K, V, dropout_p, scale_factor):
    # attn_weight shape: B, num_heads, L, L
    attn_weight = torch.einsum("bhld,bhmd->bhlm", Q, K) * scale_factor

    # attn_weight shape: B, num_heads, L
    attn_weight = F.softmax(attn_weight, dim=-1)
    attn_weight = F.dropout(attn_weight, dropout_p)

    return torch.einsum("bhld,bhdm->bhlm", attn_weight, V)


# @torch.compile(backend="cudagraphs")
def forward_matmul(Q, K, V, dropout_p, scale_factor):
    # attn_weight: B, num_heads, L, L
    attn_weight = (Q @ K.transpose(-2, -1)) * scale_factor

    # attn_weight: B, num_heads, L
    attn_weight = F.softmax(attn_weight, dim=-1)
    attn_weight = F.dropout(attn_weight, dropout_p)

    return attn_weight @ V
