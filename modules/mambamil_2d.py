import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modules.mamba.mamba_simple import MambaConfig as SimpleMambaConfig
from modules.mamba.mamba_simple import Mamba as SimpleMamba


def split_tensor(data, batch_size):
    num_chk = int(np.ceil(data.shape[0] / batch_size))
    return torch.chunk(data, num_chk, dim=0)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MambaMIL_2D(nn.Module):
    def __init__(self, in_dim, n_classes, mambamil_dim, mambamil_layer, mambamil_state_dim, pscan, cuda_pscan,
                 mamba_2d_max_w, mamba_2d_max_h, mamba_2d_pad_token, mamba_2d_patch_size, pos_emb_type,
                 drop_out, pos_emb_dropout):
        super(MambaMIL_2D, self).__init__()
        # self.args = args
        self.pos_emb_type = pos_emb_type
        self._fc1 = [nn.Linear(in_dim, mambamil_dim)]
        self._fc1 += [nn.GELU()]
        if drop_out > 0:
            self._fc1 += [nn.Dropout(drop_out)]

        self._fc1 = nn.Sequential(*self._fc1)

        self.norm = nn.LayerNorm(mambamil_dim)

        self.layers = nn.ModuleList()
        # self.patch_encoder_batch_size = patch_encoder_batch_size
        config = SimpleMambaConfig(
            d_model=mambamil_dim,
            n_layers=mambamil_layer,
            d_state=mambamil_state_dim,
            inner_layernorms=mambamil_state_dim,
            pscan=pscan,
            use_cuda=cuda_pscan,
            mamba_2d=True,
            mamba_2d_max_w=mamba_2d_max_w,
            mamba_2d_max_h=mamba_2d_max_h,
            mamba_2d_pad_token=mamba_2d_pad_token,
            mamba_2d_patch_size=mamba_2d_patch_size
        )
        self.layers = SimpleMamba(config)
        self.config = config

        self.n_classes = n_classes

        self.attention = nn.Sequential(
            nn.Linear(mambamil_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Linear(mambamil_dim, self.n_classes)
        # self.survival = args.survival

        if pos_emb_type == 'linear':
            self.pos_embs = nn.Linear(2, mambamil_dim)
            self.norm_pe = nn.LayerNorm(mambamil_dim)
            self.pos_emb_dropout = nn.Dropout(pos_emb_dropout)
        else:
            self.pos_embs = None

        self.apply(initialize_weights)

    def forward(self, x, pos, **kwargs):

        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)  # (1, num_patch, feature_dim)

        B, N_feat, C = x.shape

        h = x.float()  # [1, num_patch, feature_dim]

        h = self._fc1(h)  # [1, num_patch, mamba_dim];   project from feature_dim -> mamba_dim

        # Add Pos_emb
        if self.pos_emb_type == 'linear':
            pos_embs = self.pos_embs(pos)
            h = h + pos_embs.unsqueeze(0)
            h = self.pos_emb_dropout(h)

        h = self.layers(h, pos, self.pos_embs)

        h = self.norm(h)  # LayerNorm
        A = self.attention(h)  # [1, W, H, 1]

        if len(A.shape) == 3:
            A = torch.transpose(A, 1, 2)
        else:
            A = A.permute(0, 3, 1, 2)
            A = A.view(1, 1, -1)
            h = h.view(1, -1, self.config.d_model)

        A = F.softmax(A, dim=-1)  # [1, 1, num_patch]  # A: attention weights of patches
        h = torch.bmm(A, h)  # [1, 1, 512] , weighted combination to obtain slide feature
        h = h.squeeze(0)  # [1, 512], 512 is the slide dim

        logits = self.classifier(h)  # [1, n_classes]
        # Y_prob = F.softmax(logits, dim=1)
        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # results_dict = None

        # if self.survival:
        #     hazards = torch.sigmoid(logits)
        #     S = torch.cumprod(1 - hazards, dim=1)
        #     return hazards, S, Y_hat, None, None  # same return as other models

        return logits

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers = self.layers.to(device)

        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)