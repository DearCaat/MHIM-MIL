"""
Utility functions for MHIM model.
"""

import torch.nn as nn


def initialize_weights(module):
    """
    Initialize weights for Linear and LayerNorm layers.
    
    Args:
        module: PyTorch module to initialize
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
