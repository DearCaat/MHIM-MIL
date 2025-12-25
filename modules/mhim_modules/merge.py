"""
Merge module for MHIM model.
"""

import math
import torch
from torch import nn
from einops import rearrange
from functools import reduce
from operator import mul
from .masking import select_mask_fn


class MCA(nn.Module):
    """Multi-head Cross-Attention module."""
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """
        Args:
            dim: Input dimension
            heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Dropout rate
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, _q):
        """
        Args:
            x: Input tensor [b, n, d]
            _q: Query tensor [b, m, d]
        
        Returns:
            Output tensor [b, m, d]
        """
        kv = self.to_kv(x).chunk(2, dim=-1)
        q = self.to_q(_q)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Merge(nn.Module):
    """
    Merge module that combines patches using cross-attention.
    Supports both training (with masking) and inference modes.
    """
    
    def __init__(self, dim, heads=8, merge_h_dim=64, dropout=0.1, k=10, g_q_mm=1., 
                 merge_ratio=0.2, global_q_enable=True, no_merge=False, g_q_grad=False, 
                 mask_type='random', **kwargs):
        """
        Args:
            dim: Feature dimension
            heads: Number of attention heads
            merge_h_dim: Dimension per head in merge attention
            dropout: Dropout rate
            k: Number of global query tokens
            g_q_mm: Momentum for global query EMA update
            merge_ratio: Ratio of patches to keep during masking
            global_q_enable: Whether to use global query tokens
            no_merge: If True, skip merging and only use global queries
            g_q_grad: Whether global query requires gradient
            mask_type: Type of masking ('random' or 'low')
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MCA(dim, heads, merge_h_dim, dropout)
        self.merge_k = k 
        self.no_merge = no_merge
        self.mask_type = mask_type
        
        if global_q_enable:
            if g_q_grad:
                self.global_q_grad = nn.Parameter(torch.zeros(1, k, dim), requires_grad=True)
                # Initialize from VPT@Google
                val = math.sqrt(6. / float(3 * reduce(mul, (16, 16), 1) + dim))
                nn.init.uniform_(self.global_q_grad.data, -val, val)
            else:
                self.global_q_grad = torch.zeros(1, k, dim)
                
            if g_q_mm != 1.:
                self.global_q_mm = nn.Parameter(torch.zeros(1, k, dim), requires_grad=False)
                # Initialize from VPT@Google
                val = math.sqrt(6. / float(3 * reduce(mul, (16, 16), 1) + dim))
                nn.init.uniform_(self.global_q_mm.data, -val, val)
            else:
                self.global_q_mm = torch.zeros(1, k, dim)

            if g_q_grad and g_q_mm == 1.:
                self.global_q = self.global_q_grad
            elif not g_q_grad and g_q_mm != 1.:
                self.global_q = self.global_q_mm
        else:
            self.global_q = None

        self.g_q_mm = g_q_mm
        self.g_q_grad = g_q_grad
        self.merge_ratio = merge_ratio
        self.k = k

    def update_q_ema(self, new):
        """Update global query with exponential moving average."""
        self.global_q_mm.data.mul_(self.g_q_mm).add_(new, alpha=1. - self.g_q_mm)
    
    def merge(self, x):
        """
        Merge patches using cross-attention with global queries.
        
        Args:
            x: Input tensor [b, n, d]
        
        Returns:
            Merged tensor [b, k, d]
        """
        z = self.attn(self.norm(x), self.norm(self.global_q))
        if self.training and self.global_q is not None and self.g_q_mm != 1.:
            self.update_q_ema(z[:, :self.k])
        return z
    
    def masking(self, x, attn):
        """
        Mask patches based on attention scores or randomly.
        
        Args:
            x: Input tensor [b, n, d]
            attn: Attention scores
        
        Returns:
            x_keep: Kept patches
            x_masked: Masked patches
        """
        B, L, C = x.shape
        merge_ratio = self.merge_ratio

        if self.mask_type == 'random':
            # Random masking
            len_keep = int(L * merge_ratio)
            noise = torch.rand(L, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=0)
        elif self.mask_type == 'low':
            # Low-attention masking
            len_keep, ids_shuffle = select_mask_fn(L, attn, False, 1 - merge_ratio)
            ids_shuffle = ids_shuffle.squeeze(0)
            
        ids_keep = ids_shuffle[:len_keep]
        ids_random = ids_shuffle[len_keep:]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))
        x_masked = torch.gather(x, dim=1, index=ids_random.unsqueeze(-1).repeat(1, 1, C))

        return x_keep, x_masked

    def forward(self, x, attn=None):
        """
        Forward pass with different behavior for training and inference.
        
        Args:
            x: Input tensor [b, n, d]
            attn: Attention scores (optional)
        
        Returns:
            Output tensor with merged patches
        """
        if self.training:
            x_keep, x_masked = self.masking(x, attn)
            if self.no_merge:
                if self.global_q is not None:
                    x_keep = torch.cat((x_keep, self.global_q), dim=1)
                return x_keep
            else:
                return torch.cat((x_keep, self.merge(x_masked)), dim=1)
        else:
            if not self.no_merge:
                return torch.cat((x, self.merge(x)), dim=1) 
            else:
                if self.global_q is not None:
                    x = torch.cat((x, self.global_q), dim=1)
                return x
