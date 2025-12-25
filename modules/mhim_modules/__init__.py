"""
This package provides components for the MHIM model:
- baseline: Baseline attention modules (SAttention, DAttention, DSMIL)
- masking: Patch masking utilities
- scoring: Pseudo-score computation from attention
- merge: Patch merging with cross-attention
- losses: Loss functions for training
- utils: Utility functions
"""

from .baseline import SAttention, DAttention, DSMIL
from .masking import select_mask_fn, mask_fn
from .scoring import get_pseudo_score, get_pseudo_score_trans
from .merge import Merge, MCA
from .losses import SoftTargetCrossEntropy
from .utils import initialize_weights

__all__ = [
    # Baseline modules
    'SAttention',
    'DAttention', 
    'DSMIL',
    # Masking functions
    'select_mask_fn',
    'mask_fn',
    # Scoring functions
    'get_pseudo_score',
    'get_pseudo_score_trans',
    # Merge modules
    'Merge',
    'MCA',
    # Loss functions
    'SoftTargetCrossEntropy',
    # Utilities
    'initialize_weights',
]
