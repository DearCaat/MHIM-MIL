import torch
from torch import nn
import numpy as np

class SINCOS(nn.Module):
    def __init__(self):
        super(SINCOS, self).__init__()

    def get_1d_sincos_pos_embed_from_grid(self,embed_dim, pos, device):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float, device=device)
        omega = omega / (embed_dim / 2.)
        omega = 1. / (10000**omega)  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out) # (M, D/2)
        emb_cos = torch.cos(out) # (M, D/2)

        emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb

    def get_2d_sincos_pos_embed_from_grid(self,embed_dim, grid, device):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], device)  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], device)  # (H*W, D/2)

        emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
        return emb

    def get_2d_sincos_pos_embed(self,embed_dim, grid_size_h, grid_size_w, device='cpu', cls_token=False):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """

        grid_h = torch.arange(grid_size_h, dtype=torch.float, device=device)
        grid_w = torch.arange(grid_size_w, dtype=torch.float, device=device)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij') 
        grid = torch.stack([grid[1],grid[0]], dim=0)

        grid = grid.reshape(2, 1, grid_size_h, grid_size_w)
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device)

        if cls_token:
            pos_embed = torch.cat([torch.zeros(1, embed_dim, device=device), pos_embed], dim=0)
        return pos_embed

    def forward_single(self,x,pos=None,C=None):
        pos_all,pos = pos[0],pos[1:]

        pos = torch.tensor([ _pos[1]*pos_all[0]+_pos[0] for _pos in pos],device=x.device)

        pos_embed = self.get_2d_sincos_pos_embed(C, int(pos_all[1]),int(pos_all[0]),device=x.device)
        
        pos_embed = pos_embed[pos]

        x = x + pos_embed.unsqueeze(0)

        return x

    def forward(self, x, pos=None):
        B,N,C = x.shape

        if pos.size(0) == 1:
            pos = pos[0]

        if len(pos.shape) == 3:
            for i,_pos in enumerate(pos):
                x[i] = self.forward_single(x[i],_pos,C)
        else:
            x = self.forward_single(x,pos,C)
        
        return x

class PPEG(nn.Module):
    def __init__(self, dim=512,k=7,conv_1d=False,bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5,1), 1, (5//2,0), groups=dim,bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3,1), 1, (3//2,0), groups=dim,bias=bias)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        
        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) 

        if H < 7:
            H,W = 7,7
            zero_pad = H * W - (N+add_length)
            x = torch.cat([x, torch.zeros((B,zero_pad,C),device=x.device,dtype=x.dtype)],dim = 1)
            add_length += zero_pad

        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W).contiguous()
        #cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # print(add_length)
        if add_length >0:
            x = x[:,:-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x