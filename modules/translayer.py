import torch.nn as nn
from .nystrom_attention import NystromAttention

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,head=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = head,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x, need_attn=False):
        if need_attn:
            z,attn = self.attn(self.norm(x),return_attn=need_attn)
            x = x+z
            return x,attn
        else:
            x = x + self.attn(self.norm(x))
            return x  