"""
Loss functions for MHIM model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    Soft target cross-entropy loss with temperature scaling.
    Used for knowledge distillation between teacher and student networks.
    """
    
    def __init__(self, temp_t=1., temp_s=1.):
        """
        Args:
            temp_t: Temperature for teacher (target) softmax
            temp_s: Temperature for student (prediction) softmax
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.temp_t = temp_t
        self.temp_s = temp_s

    def forward(self, x: torch.Tensor, target: torch.Tensor, mean: bool = True) -> torch.Tensor:
        """
        Compute soft target cross-entropy loss.
        
        Args:
            x: Student predictions
            target: Teacher predictions (soft targets)
            mean: Whether to return mean loss or per-sample loss
        
        Returns:
            Loss value
        """
        loss = torch.sum(
            -F.softmax(target / self.temp_t, dim=-1) * F.log_softmax(x / self.temp_s, dim=-1), 
            dim=-1
        )
        if mean:
            return loss.mean()
        else:
            return loss
