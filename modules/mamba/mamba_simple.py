import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.mamba.pscan import pscan, pscan_2d

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison. Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""


@dataclass
class MambaConfig:
    d_model: int = 256  # D: dim of input token
    n_layers: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  # hidden state dimension
    expand_factor: int = 2  # E in paper/comments, expand factor to extend the dim of input token
    d_conv: int = 4  # kernel size of depth-wise convolution

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    pscan: bool = True  # use parallel scan mode or sequential mode when training
    use_cuda: bool = False  # use official CUDA implementation when training (not compatible with (b)float16)

    mamba_2d: bool = True  # Mamba 1D or 2D 

    mamba_2d_max_w: int = 100000
    mamba_2d_max_h: int = 100000
    mamba_2d_patch_size: int = 256
    mamba_2d_pad_token: str = 'zero'

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.mamba_2d:
            print('Using 2D Mamba')
            self.HH = self.mamba_2d_max_h // self.mamba_2d_patch_size
            self.WW = self.mamba_2d_max_w // self.mamba_2d_patch_size
        else:
            print('Using 1D Mamba')

        if self.use_cuda:
            if self.mamba_2d:
                print('Using our 2D CUDA implementation')
            else:
                print('Using official 1D CUDA implementation')
        else:
            if self.pscan:
                print('Using Pytorch implemetation with parallel scan')
            else:
                print('Using Pytorch implemetation without parallel scan')


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        if config.mamba_2d_pad_token == 'zero':
            self.pad_token = nn.Parameter(torch.zeros(config.d_model), requires_grad=False)
        elif config.mamba_2d_pad_token == 'trainable':
            self.pad_token = nn.Parameter(torch.rand(config.d_model), requires_grad=True)

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x, coords=None, pos_embs=None):
        for layer in self.layers:
            if self.config.mamba_2d:
                x = layer(x, coords, self.pad_token, pos_embs)
            else:
                x = layer(x, coords)

        return x

    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x, coords=None, pad_token=None, pos_embs=None):
        if len(x.shape) == 3 and self.config.mamba_2d:
            x = reconstruct_2d_wsi(x, coords,
                                   self.config.mamba_2d_max_w,
                                   self.config.mamba_2d_max_h,
                                   self.config.mamba_2d_patch_size,
                                   pad_token)

            if self.config.use_cuda:
                B, W, H, ED = x.shape
                x = torch.reshape(x, (B, W * H, ED))

            if pos_embs is not None:
                pos_embs = F.interpolate(pos_embs.unsqueeze(0).permute(0, 3, 1, 2),
                                         size=(x.shape[1], x.shape[2]),
                                         mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                x = x + pos_embs

        elif len(x.shape) == 4 and self.config.mamba_2d and self.config.use_cuda:
            x = x.reshape(x.shape[0], -1, x.shape[-1])

        output = self.mixer(self.norm(x)) + x

        return output

    def step(self, x, cache):
        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)  # do not change the dim of expanded x

        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)

        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                if self.config.mamba_2d:
                    from modules.mamba.pscan_2d import selective_scan_fn as selective_scan_fn
                    self.selective_scan_cuda = selective_scan_fn
                else:
                    from modules.mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                    self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        if len(x.shape) == 4:
            BS, W, H, _ = x.shape
        else:
            _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)

        # x branch
        if len(x.shape) == 4:
            x = x.view(BS, W * H, -1).transpose(1, 2)  # (BS, ED, W*H)
            x = self.conv1d(x)[:, :, :W * H]  # depthwise convolution over time, with a short filter
            x = x.view(BS, -1, W, H).permute(0, 2, 3, 1)  # (B, L, ED)
        else:
            x = x.transpose(1, 2)  # (B, ED, L)
            x = self.conv1d(x)[:, :, :L]  # depthwise convolution over time, with a short filter
            x = x.transpose(1, 2)  # (B, L, ED)

        x = F.silu(x)
        if self.config.mamba_2d:
            y = self.ssm_2d(x, z)

        else:
            y = self.ssm(x, z)

        if self.config.use_cuda:
            output = self.out_proj(y)  # (B, L, D)
            return output  # the rest of the operations are done in the ssm function (fused with the CUDA pscan)

        # z branch
        z = F.silu(z)  # BS, W, H, ED
        output = y * z
        output = self.out_proj(output)  # (B, L, D)

        return output

    def ssm(self, x, z):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2)  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)

        # choose which selective_scan function to use, according to config
        if self.config.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True,
                                         delta_bias=self.dt_proj.bias.float())
            y = y.transpose(1, 2)  # (B, L, ED)

        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def ssm_2d(self, x, z=None):
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)

        if self.config.use_cuda:
            delta = self.dt_proj.weight @ delta.transpose(1, 2)  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True,
                                         delta_bias=self.dt_proj.bias.float(), HH=self.config.HH, WW=self.config.WW)
            y = y.transpose(1, 2)

        else:
            if self.config.mamba_2d:
                delta = F.softplus(self.dt_proj(delta))
            else:
                delta = delta.transpose(1, 2)
                delta = F.softplus(delta + self.dt_proj.bias)
            if self.config.pscan:
                y = self.selective_scan_2D(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq_2d(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)   # A_bar with ZOH discretization
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N) # B_bar with Euler discretization

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @(B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_2D(self, x, delta, A, B, C, D):
        # Reshape tensors from 1D to 2D

        # x : (BS, H, W, ED)
        # Δ : (BS, H, W, ED)
        # A : (ED, N)
        # B : (BS, H, W, N)
        # C : (BS, H, W, N)
        # D : (ED)
        # y : (BS, H, W, ED)

        # BS, H, W, ED = x.size()
        # _,_,_, N = B.size()

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (BS, H, W, ED, N)   # A_bar with ZOH discretization
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(3)  # (BS, H, W, ED, N) # B_bar with Euler discretization

        BX = deltaB * (x.unsqueeze(-1))  # (B, H, W, ED, N)

        hs = pscan_2d(deltaA, BX)
        y = (hs @ C.unsqueeze(-1)).squeeze(4)  # (B, L, ED, N) @(B, L, N, 1) -> (B, L, ED, 1)
        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        # Init the first hidden state as zero
        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @(B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq_2d(self, x, delta, A, B, C, D):
        # x : (B, L, ED) 
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape
        image_shape = int(math.sqrt(L))  # CIFAR size: 8x8; L = 64

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        # Init the first hidden state as zero
        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  # (B, ED, N)
        hs = []

        for t in range(0, L):
            col_id = int(t // image_shape)
            row_id = int(t % image_shape)
            if col_id == 0:
                h = deltaA[:, t] * h + BX[:, t]  # h = A*(state of upper patch) + B*(input of current patch)
            elif row_id == 0:
                h = deltaA[:, t] * hs[t - image_shape] + BX[:,
                                                         t]  # h = A*(state of left patch)  + B*(input of current patch)
            else:
                h = deltaA[:, t] * h + deltaA[:, t] * hs[t - image_shape] + BX[:, t]  # h = A*(state of upper patch) 
                #     + A*(state of left patch)
                #     + B*(input of current patch)
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @(B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs : (B, ED, d_conv-1)

        # y : (B, D)
        # cache : (h, inputs)

        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)      # deep-wise convolution
        x, z = xz.chunk(2, dim=1)  # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]  # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  # (B, ED, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(
            self.A_log.float())  # (ED, N) #todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))  # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  # (B, ED, N)

        h = deltaA * h + BX  # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  # (B, ED, N) @(B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h


def reconstruct_2d_wsi(feats, coords, max_w, max_h, patch_size, pad_token):
    """
    Transform a raster 1D sequence extract by CLAM tool into padded 2D feature
    """
    B, N_feat, C = feats.shape

    raw_max_w, raw_max_h = coords[0, 0].tolist()
    # max_w = raw_max_w // patch_size + 1
    # max_h = raw_max_h // patch_size + 1

    data_coords = coords[:, 1:, :]              # [B, N_feat, 2]
    normalized = data_coords // patch_size      # [B, N_feat, 2]
    w_idxs = normalized[:, :, 0].flatten().long()  # [B*N_feat]→[N_feat] if B=1
    h_idxs = normalized[:, :, 1].flatten().long()

    results = feats.new_empty((B, max_h, max_w, C))
    results[:] = pad_token.to(device=feats.device, dtype=feats.dtype)

    # feats_flat: [B, N_feat, C]
    feats_flat = feats.view(B, -1, C)
    results[:, h_idxs, w_idxs, :] = feats_flat

    return results


# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output