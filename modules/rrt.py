import torch, einops
from torch import nn
import torch.nn.functional as F
from .nystrom_attention import NystromAttention
from timm.models.layers import DropPath
import numpy as np
import math

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
def region_partition(x, region_size):
    """
    Args:
        x: (B, H, W, C)
        region_size (int): region size
    Returns:
        regions: (num_regions*B, region_size, region_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
    return regions

def region_reverse(regions, region_size, H, W):
    """
    Args:
        regions: (num_regions*B, region_size, region_size, C)
        region_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(regions.shape[0] / (H * W / region_size / region_size))
    x = regions.view(B, H // region_size, W // region_size, region_size, region_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class InnerAttention(nn.Module):
    def __init__(self, dim, head_dim=None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,epeg=True,epeg_k=15,epeg_2d=False,epeg_bias=True,epeg_type='attn',**kwargs):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.epeg_2d = epeg_2d
        self.epeg_type = epeg_type

        if epeg:
            padding = epeg_k // 2
            if epeg_2d:
                if epeg_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, epeg_k, padding = padding, groups = num_heads, bias = epeg_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, epeg_k, padding = padding, groups = head_dim * num_heads, bias = epeg_bias)
            else:
                if epeg_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, (epeg_k, 1), padding = (padding, 0), groups = num_heads, bias = epeg_bias)
                else:
                    self.pe = nn.Conv2d(head_dim *num_heads, head_dim *num_heads, (epeg_k, 1), padding = (padding, 0), groups = head_dim *num_heads, bias = epeg_bias)
        else:
            self.pe = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_regions*B, N, C)
        """
        B_, N, C = x.shape

        # x = self.pe(x)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.pe is not None and self.epeg_type == 'attn':
            pe = self.pe(attn)
            attn = attn+pe

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.pe is not None and self.epeg_type == 'value_bf':
            # B,H,N,C -> B,HC,N-0.5,N-0.5 
            pe = self.pe(v.permute(0,3,1,2).reshape(B_,C,int(np.ceil(np.sqrt(N))),int(np.ceil(np.sqrt(N)))))
            #pe = torch.einsum('ahbd->abhd',pe).flatten(-2,-1)
            v = v + pe.reshape(B_,self.num_heads, self.head_dim,N).permute(0,1,3,2)

        # print(v.size())

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads*self.head_dim)
        
        if self.pe is not None and self.epeg_type == 'value_af':
            #print(v.size())
            pe = self.pe(v.permute(0,3,1,2).reshape(B_,C,int(np.ceil(np.sqrt(N))),int(np.ceil(np.sqrt(N)))))
            # print(pe.size())
            # print(v.size())
            x = x + pe.reshape(B_,self.num_heads*self.head_dim,N).transpose(-1,-2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, region_size={self.region_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 region with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class RegionAttntion(nn.Module):
    def __init__(self, dim, head_dim=None,num_heads=8, region_size=0, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., region_num=8,epeg=False,min_region_num=0,min_region_ratio=0.,region_attn='native',**kawrgs):
        super().__init__()
 
        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio

        if region_attn == 'native':
            self.attn = InnerAttention(
                dim, head_dim=head_dim,num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,epeg=epeg,**kawrgs)
        elif region_attn == 'ntrans':
            self.attn = NystromAttention(
                dim=dim,
                dim_head=head_dim,
                heads=num_heads,
                dropout=drop
            )

    def padding(self,x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H+_n, W+_n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            _region_num = 16 if L > 100000 else self.region_num
            if not self.training and _region_num < 8:
                _region_num = 8
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % _region_num
            H, W = H+_n, W+_n
            region_size = int(H // _region_num)
            region_num = _region_num
        
        add_length = H * W - L

        # if padding much，i will give up region attention. only for ablation
        if (add_length > L / (self.min_region_ratio+1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H+_n, W+_n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B,add_length,C),device=x.device)],dim = 1)
        
        return x,H,W,add_length,region_num,region_size

    def forward(self,x,return_attn=False):
        B, L, C = x.shape
        
        # padding
        x,H,W,add_length,region_num,region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # partition regions
        x_regions = region_partition(x, region_size)  # nW*B, region_size, region_size, C

        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        # R-MSA
        attn_regions = self.attn(x_regions)  # nW*B, region_size*region_size, C

        # merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        if add_length >0:
            x = x[:,:-add_length]

        return x

class CrossRegionAttntion(nn.Module):
    def __init__(self, dim, head_dim=None,num_heads=8, region_size=0, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., region_num=8,epeg=False,min_region_num=0,min_region_ratio=0.,crmsa_k=3,crmsa_mlp=False,region_attn='native',**kawrgs):
        super().__init__()
 
        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio
        
        self.attn = InnerAttention(
            dim, head_dim=head_dim,num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,epeg=epeg,**kawrgs)

        self.crmsa_mlp = crmsa_mlp
        if crmsa_mlp:
            self.phi = [nn.Linear(self.dim, self.dim // 4,bias=False)]
            self.phi += [nn.Tanh()]
            self.phi += [nn.Linear(self.dim // 4, crmsa_k,bias=False)]
            self.phi = nn.Sequential(*self.phi)
        else:
            self.phi = nn.Parameter(
                torch.empty(
                    (self.dim, crmsa_k),
                )
            )
        nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def padding(self,x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H+_n, W+_n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H+_n, W+_n
            region_size = int(H // self.region_num)
            region_num = self.region_num
        
        add_length = H * W - L

        # if padding much，i will give up region attention. only for ablation
        if (add_length > L / (self.min_region_ratio+1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H+_n, W+_n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B,add_length,C),device=x.device)],dim = 1)
        
        return x,H,W,add_length,region_num,region_size

    def forward(self,x,return_attn=False):
        B, L, C = x.shape
        
        # padding
        x,H,W,add_length,region_num,region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # partition regions
        x_regions = region_partition(x, region_size)  # nW*B, region_size, region_size, C

        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        # CR-MSA
        if self.crmsa_mlp:
            logits = self.phi(x_regions).transpose(1,2) # W*B, sW, region_size*region_size
        else:
            logits = torch.einsum("w p c, c n -> w p n", x_regions, self.phi).transpose(1,2) # nW*B, sW, region_size*region_size

        combine_weights = logits.softmax(dim=-1)
        dispatch_weights = logits.softmax(dim=1)

        logits_min,_ = logits.min(dim=-1)
        logits_max,_ = logits.max(dim=-1)
        dispatch_weights_mm = (logits - logits_min.unsqueeze(-1)) / (logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8)

        attn_regions =  torch.einsum("w p c, w n p -> w n p c", x_regions,combine_weights).sum(dim=-2).transpose(0,1) # sW, nW, C

        if return_attn:
            attn_regions,_attn = self.attn(attn_regions,return_attn)  # sW, nW, C
            attn_regions = attn_regions.transpose(0,1) # nW, sW, C
        else:
            attn_regions = self.attn(attn_regions).transpose(0,1)  # nW, sW, C

        attn_regions = torch.einsum("w n c, w n p -> w n p c", attn_regions, dispatch_weights_mm) # nW, sW, region_size*region_size, C
        attn_regions = torch.einsum("w n p c, w n p -> w n p c", attn_regions, dispatch_weights).sum(dim=1) # nW, region_size*region_size, C

        # merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        if add_length >0:
            x = x[:,:-add_length]

        return x

class Attention(nn.Module):
    def __init__(self,input_dim=512,act='relu',bias=False,dropout=False):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention = [nn.Linear(self.L, self.D,bias=bias)]

        if act == 'gelu': 
            self.attention += [nn.GELU()]
        elif act == 'relu':
            self.attention += [nn.ReLU()]
        elif act == 'tanh':
            self.attention += [nn.Tanh()]

        if dropout:
            self.attention += [nn.Dropout(0.25)]

        self.attention += [nn.Linear(self.D, self.K,bias=bias)]

        self.attention = nn.Sequential(*self.attention)

    def forward(self,x,no_norm=False):
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)
        
        if no_norm:
            return x,A_ori
        else:
            return x,A

class AttentionGated(nn.Module):
    def __init__(self,input_dim=512,act='relu',bias=False,dropout=False):
        super(AttentionGated, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

    def forward(self, x,no_norm=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        if no_norm:
            return x,A_ori
        else:
            return x,A

class DAttention(nn.Module):
    def __init__(self,input_dim=512,act='relu',gated=False,bias=False,dropout=False):
        super(DAttention, self).__init__()
        self.gated = gated
        if gated:
            self.attention = AttentionGated(input_dim,act,bias,dropout)
        else:
            self.attention = Attention(input_dim,act,bias,dropout)

    def forward(self, x, return_attn=False,no_norm=False,**kwargs):

        x,attn = self.attention(x,no_norm)

        if return_attn:
            return x.squeeze(1),attn.squeeze(1)
        else:   
            return x.squeeze(1)

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,head=8,drop_out=0.1,drop_path=0.,ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,attn='rmsa',n_region=8,epeg=False,region_size=0,min_region_num=0,min_region_ratio=0,qkv_bias=True,crmsa_k=3,epeg_k=15,**kwargs):
        super().__init__()

        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()
        if attn == 'ntrans':
            self.attn = NystromAttention(
                dim = dim,
                dim_head = trans_dim,  # dim // 8
                heads = head,
                num_landmarks = 256,    # number of landmarks dim // 2
                pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
                residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
                dropout=drop_out
            )
        elif attn == 'rmsa':
            self.attn = RegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                epeg_k=epeg_k,
                **kwargs
            )
        elif attn == 'crmsa':
            self.attn = CrossRegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                **kwargs
            )
        else:
            raise NotImplementedError
        # elif attn == 'rrt1d':
        #     self.attn = RegionAttntion1D(
        #         dim=dim,
        #         num_heads=head,
        #         drop=drop_out,
        #         region_num=n_region,
        #         head_dim=trans_dim,
        #         conv=epeg,
        #         **kwargs
        #     )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop_out) if ffn else nn.Identity()

    def forward(self,x,need_attn=False):

        x,attn = self.forward_trans(x,need_attn=need_attn)
        
        if need_attn:
            return x,attn
        else:
            return x

    def forward_trans(self, x, need_attn=False):
        attn = None
        
        if need_attn:
            z,attn = self.attn(self.norm(x),return_attn=need_attn)
        else:
            z = self.attn(self.norm(x))

        x = x+self.drop_path(z)

        # FFN
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x,attn
    
class PositionEmbedding(nn.Module):
    def __init__(self, size, dim=512):
        super().__init__()
        self.size=size
        self.pe = nn.Embedding(size+1, dim, padding_idx=0)
        self.pos_ids = torch.arange(1, size+1, dtype=torch.long).cuda()
        
    def forward(self, emb):
        device = emb.device
        b, n, *_ = emb.shape
        pos_ids = self.pos_ids
        if n > self.size:
            zeros = torch.zeros(n-self.size, dtype=torch.long, device=device)
            pos_ids = torch.cat([pos_ids, zeros])
        pos_ids = einops.repeat(pos_ids, 'n -> b n', b=b)
        pos_emb = self.pe(pos_ids) # [b n pe_dim]
        embeddings = torch.cat([emb, pos_emb], dim=-1)
        return embeddings
        
class PPEG(nn.Module):
    def __init__(self, dim=512,k=7,conv_1d=False,bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5,1), 1, (5//2,0), groups=dim,bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3,1), 1, (3//2,0), groups=dim,bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        
        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) 

        if H < 7:
            H,W = 7,7
            zero_pad = H * W - (N+add_length)
            x = torch.cat([x, torch.zeros((B,zero_pad,C),device=x.device)],dim = 1)
            add_length += zero_pad

        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)

        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # print(add_length)
        if add_length >0:
            x = x[:,:-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class PEG(nn.Module):
    def __init__(self, dim=512,k=7,bias=True,conv_1d=False):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        x = torch.cat([x, x[:,:add_length,:]],dim = 1)

        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat

        x = x.flatten(2).transpose(1, 2)
        if add_length >0:
            x = x[:,:-add_length]

        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class RRTEncoder(nn.Module):
    def __init__(self,mlp_dim=512,pos_pos=0,pos='none',peg_k=7,attn='rmsa',region_num=8,drop_out=0.1,n_layers=2,n_heads=8,drop_path=0.,ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,epeg=True,epeg_k=15,region_size=0,min_region_num=0,min_region_ratio=0,qkv_bias=True,peg_bias=True,peg_1d=False,cr_msa=True,crmsa_k=3,all_shortcut=False,crmsa_mlp=False,crmsa_heads=8,need_init=False,**kwargs):
        super(RRTEncoder, self).__init__()
        
        self.final_dim = mlp_dim

        self.norm = nn.LayerNorm(self.final_dim)
        self.all_shortcut = all_shortcut

        self.layers = []
        for i in range(n_layers-1):
            self.layers += [TransLayer(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,attn=attn,n_region=region_num,epeg=epeg,epeg_k=epeg_k,region_size=region_size,min_region_num=min_region_num,min_region_ratio=min_region_ratio,qkv_bias=qkv_bias,**kwargs)]
        self.layers = nn.Sequential(*self.layers)
    
        # CR-MSA
        self.cr_msa = TransLayer(dim=mlp_dim,head=crmsa_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,attn='crmsa',qkv_bias=qkv_bias,crmsa_k=crmsa_k,crmsa_mlp=crmsa_mlp,**kwargs) if cr_msa else nn.Identity()

        # only for ablation
        self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

        if need_init:
            self.apply(initialize_weights)

    def forward(self, x):
        shape_len = 3
        # for N,C
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        # for B,C,H,W
        if len(x.shape) == 4:
            x = x.reshape(x.size(0),x.size(1),-1)
            x = x.transpose(1,2)
            shape_len = 4

        batch, num_patches, C = x.shape 
        x_shortcut = x

        # PEG/PPEG
        if self.pos_pos == -1:
            x = self.pos_embedding(x)
        
        # R-MSA within region
        for i,layer in enumerate(self.layers.children()):
            if i == 1 and self.pos_pos == 0:
                x = self.pos_embedding(x)
            x = layer(x)

        x = self.cr_msa(x)

        if self.all_shortcut:
            x = x+x_shortcut

        x = self.norm(x)

        if shape_len == 2:
            x = x.squeeze(0)
        elif shape_len == 4:
            x = x.transpose(1,2)
            x = x.reshape(batch,C,int(num_patches**0.5),int(num_patches**0.5))
        return x
    
class RRTMIL(nn.Module):
    def __init__(self, input_dim=1024,inner_dim=512,act='relu',n_classes=2,dropout=0.25,pos_pos=0,pos='none',peg_k=7,attn='rmsa',pool='attn',region_num=8,n_layers=2,n_heads=8,drop_path=0.,da_act='relu',trans_dropout=0.1,ffn=False,ffn_act='gelu',mlp_ratio=4.,da_gated=False,da_bias=False,da_dropout=False,trans_dim=64,epeg=True,min_region_num=0,qkv_bias=True,mil_norm=None,embed_norm_pos=0,mil_bias=True,**kwargs):
        super(RRTMIL, self).__init__()

        self.feature = [nn.Linear(input_dim, inner_dim,bias=mil_bias)]

        if act.lower() == 'relu':
            self.feature += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.feature += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.feature = nn.Sequential(*self.feature)

        self.mil_norm = mil_norm
        self.embed_norm_pos = embed_norm_pos

        if mil_norm == 'bn':
            self.norm = nn.BatchNorm1d(input_dim) if embed_norm_pos == 0 else nn.BatchNorm1d(inner_dim)
        elif mil_norm == 'ln':
            self.norm = nn.LayerNorm(input_dim,bias=mil_bias) if embed_norm_pos == 0 else nn.LayerNorm(inner_dim,bias=mil_bias)
        else:
            self.norm = nn.Identity()

        self.online_encoder = RRTEncoder(mlp_dim=inner_dim,pos_pos=pos_pos,pos=pos,peg_k=peg_k,attn=attn,region_num=region_num,n_layers=n_layers,n_heads=n_heads,drop_path=drop_path,drop_out=trans_dropout,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,epeg=epeg,min_region_num=min_region_num,qkv_bias=qkv_bias,**kwargs)

        self.pool_fn = DAttention(self.online_encoder.final_dim,da_act,gated=da_gated,bias=da_bias,dropout=da_dropout) if pool == 'attn' else nn.AdaptiveAvgPool1d(1)
        
        self.predictor = nn.Linear(self.online_encoder.final_dim,n_classes,bias=mil_bias)

        self.apply(initialize_weights)

    def forward(self, x, return_attn=False,no_norm=False,pos=None,**kwargs):
        if self.embed_norm_pos == 0:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)
            else:
                x = self.norm(x)

        x = self.feature(x) # n*512
        x = self.dp(x)

        # feature re-embedding
        x = self.online_encoder(x)

        if self.embed_norm_pos == 1:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)
            else:
                x = self.norm(x)
        
        # feature aggregation
        if return_attn:
            x,a = self.pool_fn(x,return_attn=True,no_norm=no_norm)
        else:
            x = self.pool_fn(x)

        # prediction
        logits = self.predictor(x)

        if return_attn:
            return logits,a
        else:
            return logits
        
if __name__ == "__main__":
    x = torch.rand(1,100,1024)
    x_rrt = torch.rand(1,100,512)

    # epeg_k，crmsa_k are the primary hyper-para, you can set crmsa_heads, all_shortcut and crmsa_mlp if you want.
    # C16-R50: input_dim=1024,epeg_k=15,crmsa_k=1,crmsa_heads=8,all_shortcut=True
    # C16-PLIP: input_dim=512,epeg_k=9,crmsa_k=3,crmsa_heads=8,all_shortcut=True
    # TCGA-LUAD&LUSC-R50: input_dim=1024,epeg_k=21,crmsa_k=5,crmsa_heads=8
    # TCGA-LUAD&LUSC-PLIP: input_dim=512,epeg_k=13,crmsa_k=3,crmsa_heads=1,all_shortcut=True,crmsa_mlp=True
    # TCGA-BRCA-R50:input_dim=1024,epeg_k=17,crmsa_k=3,crmsa_heads=1
    # TCGA-BRCA-PLIP: input_dim=512,epeg_k=15,crmsa_k=1,crmsa_heads=8,all_shortcut=True

    # rrt+abmil
    rrt_mil = RRTMIL(n_classes=2,epeg_k=15,crmsa_k=3)
    x = rrt_mil(x)  # 1,N,D -> 1,C

    # rrt. you should put the rrt_enc before aggregation module, after fc and dp
    # x_rrt = fc(x_rrt) # 1,N,1024 -> 1,N,512
    # x_rrt = dropout(x_rrt)
    rrt = RRTEncoder(mlp_dim=512,epeg_k=15,crmsa_k=3) 
    x_rrt = rrt(x_rrt) # 1,N,512 -> 1,N,512
    # x_rrt = mil_model(x_rrt) # 1,N,512 -> 1,N,C

    print(x.size())
    print(x_rrt.size())