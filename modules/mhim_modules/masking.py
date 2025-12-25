"""
Masking utilities for MHIM model.
"""

import torch
import numpy as np


def select_mask_fn(ps, attn, largest, mask_ratio, mask_ids_other=None, len_keep_other=None, 
                   cls_attn_topk_idx_other=None, random_ratio=1., select_inv=False, msa_fusion='vote'):
    """
    Select patches to mask based on attention scores.
    
    Args:
        ps: Total number of patches
        attn: Attention scores
        largest: Whether to select patches with largest attention scores
        mask_ratio: Ratio of patches to mask
        mask_ids_other: Previously masked indices (optional)
        len_keep_other: Number of patches kept in previous masking (optional)
        cls_attn_topk_idx_other: Top-k indices from previous masking (optional)
        random_ratio: Ratio for random selection within selected patches
        select_inv: Whether to invert the selection
        msa_fusion: Multi-head fusion strategy ('mean' or 'vote')
    
    Returns:
        len_keep: Number of patches to keep
        mask_ids: Tensor of patch indices (kept first, then masked)
    """
    ps_tmp = ps
    mask_ratio_ori = mask_ratio
    mask_ratio = mask_ratio / random_ratio
    if mask_ratio > 1:
        random_ratio = mask_ratio_ori
        mask_ratio = 1.
        
    if mask_ids_other is not None:
        if cls_attn_topk_idx_other is None:
            cls_attn_topk_idx_other = mask_ids_other[:, len_keep_other:].squeeze()
            ps_tmp = ps - cls_attn_topk_idx_other.size(0)
    
    # Multi-head attention fusion
    if len(attn.size()) > 2:
        if msa_fusion == 'mean':
            _, cls_attn_topk_idx = torch.topk(
                attn, int(np.ceil((ps_tmp * mask_ratio)) // attn.size(1)), largest=largest
            )
            cls_attn_topk_idx = torch.unique(cls_attn_topk_idx.flatten(-3, -1))
        elif msa_fusion == 'vote':
            vote = attn.clone()
            vote[:] = 0
            
            _, idx = torch.topk(attn, k=int(np.ceil((ps_tmp * mask_ratio))), sorted=False, largest=largest)
            mask = vote.clone() 
            mask = mask.scatter_(2, idx, 1) == 1
            vote[mask] = 1
            vote = vote.sum(dim=1)
            _, cls_attn_topk_idx = torch.topk(vote, k=int(np.ceil((ps_tmp * mask_ratio))), sorted=False)
            cls_attn_topk_idx = cls_attn_topk_idx[0]
    else:
        k = int(np.ceil((ps_tmp * mask_ratio)))
        _, cls_attn_topk_idx = torch.topk(attn, k, largest=largest)
        cls_attn_topk_idx = cls_attn_topk_idx.squeeze(0)
    
    # Randomly subsample selected patches
    if random_ratio < 1.:
        random_idx = torch.randperm(cls_attn_topk_idx.size(0), device=cls_attn_topk_idx.device)
        cls_attn_topk_idx = torch.gather(
            cls_attn_topk_idx, dim=0, 
            index=random_idx[:int(np.ceil((cls_attn_topk_idx.size(0) * random_ratio)))]
        )
    
    # Concatenate with other masking indices
    if mask_ids_other is not None:
        cls_attn_topk_idx = torch.cat([cls_attn_topk_idx, cls_attn_topk_idx_other]).unique()

    len_keep = ps - cls_attn_topk_idx.size(0)
    a = set(cls_attn_topk_idx.tolist())
    b = set(list(range(ps)))
    mask_ids = torch.tensor(list(b.difference(a)), device=attn.device)
    
    if select_inv:
        mask_ids = torch.cat([cls_attn_topk_idx, mask_ids]).unsqueeze(0)
        len_keep = ps - len_keep
    else:
        mask_ids = torch.cat([mask_ids, cls_attn_topk_idx]).unsqueeze(0)

    return len_keep, mask_ids


def mask_fn(x, ids_shuffle=None, len_keep=None):
    """
    Mask patches according to shuffle indices.
    
    Args:
        x: Input tensor of shape [N, L, D]
        ids_shuffle: Shuffle indices for masking
        len_keep: Number of patches to keep
    
    Returns:
        x_keep: Tensor with only kept patches
    """
    N, L, D = x.shape
    assert ids_shuffle is not None

    # Keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    return x_keep
