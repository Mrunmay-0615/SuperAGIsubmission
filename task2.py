from typing import Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
TASK 2, Part 1: RoFormer
This is the code for implementing rotatary positional embeddings
Reference -> https://github.com/JunnYu/RoFormer_pytorch
Paper -> Su et. al. RoFormer, https://arxiv.org/pdf/2104.09864.pdf
"""

class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, positions):
        return super().forward(positions)


"""
TASK 2, Part 2: Grouped Query Attention
Paper -> https://arxiv.org/pdf/2305.13245v2
References ->  https://aliissa99.medium.com/-a596e4d86f79 
References -> https://arxiv.org/pdf/1911.02150
Here I have modified the MultiHead attention in the base architecture to GQA
The idea is to group the n_heads queries into G groups and each group will have a single head and value.
"""

class GroupQueryAttention(nn.Module):

    def __init__(self, embed_size, num_heads, dropout=0.0, block_size=1024, num_groups=4, bias=True):
        super().__init__()
        assert embed_size % num_heads == 0
        assert num_heads % num_groups == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_size, 3 * embed_size, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_size, embed_size, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = num_heads
        self.n_embd = embed_size
        self.n_groups = num_groups
        self.dropout = dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        G = self.n_head // self.n_groups # number of heads in one group
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2) # (B, T, C)
        # Split them into n_heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # split the keys and values in groups, No need to split the queries as the code will handle it
        k = torch.cat([k_.unsqueeze(1) for k_ in k.split(G, dim=1)], dim=1)  # (B, n_groups, G, T, hs)
        v = torch.cat([v_.unsqueeze(1) for v_ in v.split(G, dim=1)], dim=1) # (B, n_groups, G, T, hs)
        # Perform mean pooling so that for each group, we have a single head and value pair
        k = torch.mean(k, dim=2).unsqueeze(2).repeat(1, 1, G, 1, 1)
        v = torch.mean(v, dim=2).unsqueeze(2).repeat(1, 1, G, 1, 1)
        # Reshaping to remove the groups now that the keys and values have been mean-pooled
        k = k.view(B, self.n_head, T, C // self.n_head)
        v = v.view(B, self.n_head, T, C // self.n_head)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y






