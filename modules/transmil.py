import torch
import torch.nn as nn
import numpy as np
from .nystrom_attention import NystromAttention

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, n_heads=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//n_heads,
            heads = n_heads,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, need_attn=False, need_v=False, no_norm=False):
        if need_attn:
            z,attn,v = self.attn(self.norm(x),return_attn=need_attn,no_norm=no_norm)
            x = x+z
            if need_v:
                return x,attn,v
            else:
                return x,attn
        else:
            x = x + self.attn(self.norm(x))
            return x  

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    def __init__(self, input_dim,n_classes,dropout,act,mil_norm=None,mil_bias=True,inner_dim=512,embed_feat=True,pos='ppeg',n_heads=8,**kwargs):
        super(TransMIL, self).__init__()

        #self.pos_layer = PPEG(dim=inner_dim)
        self.pos=pos
        if pos == 'none':
            self.pos_layer = nn.Identity()
        else:
            self.pos_layer = PPEG(dim=inner_dim)
        # self.feature = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),nn.Dropout(0.25))
        self.feature = []
        self.mil_norm = mil_norm
        if mil_norm == 'bn':   
            # self.feature += [nn.LayerNorm(input_dim,bias=mil_bias)]
            self.norm1 = nn.BatchNorm1d(input_dim)
        elif mil_norm == 'ln':
            self.feature += [nn.LayerNorm(input_dim,bias=mil_bias)]
        else:
            self.norm1 = nn.Identity()

        if embed_feat:
            self.feature += [nn.Linear(input_dim, inner_dim,bias=mil_bias)]

            if act.lower() == 'relu':
                self.feature += [nn.ReLU()]
            elif act.lower() == 'gelu':
                self.feature += [nn.GELU()]

            if dropout:
                self.feature += [nn.Dropout(0.25)]
        
        self.feature = nn.Sequential(*self.feature) if len(self.feature) > 0 else nn.Identity()
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, inner_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=inner_dim,n_heads=n_heads)
        self.layer2 = TransLayer(dim=inner_dim,n_heads=n_heads)
        self.norm = nn.LayerNorm(inner_dim)
        self.classifier = nn.Linear(inner_dim, self.n_classes,bias=mil_bias)

        self.apply(initialize_weights)

    def forward(self, x,return_attn=False,return_act=False,**kwargs):

        #x = x.float() #[B, n, 1024]
        attn = []

        if self.mil_norm == 'bn':
            x = torch.transpose(x, -1, -2)
            x = self.norm1(x)
            x = torch.transpose(x, -1, -2)

        x = self.feature(x) #[B, n, 512]
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        #---->pad
        H = x.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(x.device)
        x = torch.cat((cls_tokens, x), dim=1)

        #---->Translayer x1
        if return_attn:
            x,_attn,v = self.layer1(x,need_attn=True,need_v=True)
            if add_length >0:
                _attn = _attn[:,:,:-add_length]
            attn.append(_attn.clone())
        else:
            x = self.layer1(x)  #[B, N, 512]
        # x = self.layer1(x) 

        #---->PPEG
        if self.pos != 'none':
            x = self.pos_layer(x, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        if return_attn:
            x,_attn = self.layer2(x,need_attn=True)
            if add_length >0:
                _attn = _attn[:,:,:-add_length]
            attn.append(_attn.clone())
        else:
            x = self.layer2(x)
        #x = self.layer2(x) #[B, N, 512]

        #---->cls_token
        x = self.norm(x)[:,0]

        #---->predict
        logits = self.classifier(x) #[B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        # return logits
        if return_attn:
            output = []
            output.append(logits)
            output.append(attn)
            if return_act:
                output.append(v)
            return output
        else:
            return logits

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024))
    model = TransMIL(n_classes=2,dropout=False,act='relu')
    for k, v in model.state_dict().items():
        print(k)
    # print(model.eval())
    # results_dict = model(data = data)
    # print(results_dict)