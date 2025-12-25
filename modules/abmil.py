import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .emb_position import SINCOS

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    for m in module.modules():
        if hasattr(m,'init_weights'):
            m.init_weights()

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()

        self.model = list(models.resnet50(pretrained = True).children())[:-1]
        self.features = nn.Sequential(*self.model)

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.classifier = nn.Linear(512,1)
        initialize_weights(self.feature_extractor_part2)
        initialize_weights(self.classifier)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x=self.feature_extractor_part2(x)
        # feat = torch.mean(x,dim=0)
        x1 = self.classifier(x)
        # x2 = torch.mean(x1, dim=0).view(1,-1)
        x2,_ = torch.max(x1, dim=0)
        x2=x2.view(1,-1)
        return x2,x
class AttentionGated(nn.Module):
    def __init__(self,input_dim,n_classes,act='relu',dropout=0.,mil_norm=None,mil_bias=True,mil_cls_bias=True,inner_dim=512,embed_feat=True,embed_norm_pos=0,pos=None,**kwargs):
        super(AttentionGated, self).__init__()
        self.L = inner_dim
        self.D = 384 #128  384 uni
        self.K = 1
        self.mil_norm = mil_norm
        self.embed_norm_pos = embed_norm_pos
        self.pos = pos

        if mil_norm == 'bn':
            self.norm = nn.BatchNorm1d(input_dim) if embed_norm_pos == 0 else nn.BatchNorm1d(inner_dim)
            self.norm1 = nn.BatchNorm1d(self.L*self.K)
        elif mil_norm == 'ln':
            if embed_norm_pos == 0:
                self.feature += [nn.LayerNorm(input_dim,bias=mil_bias)]
                self.norm1 = nn.LayerNorm(self.L*self.K,bias=mil_bias)
            else:
                self.norm = nn.LayerNorm(inner_dim,bias=mil_bias)
                self.norm1 = nn.LayerNorm(self.L*self.K,bias=mil_bias)
        else:
            self.norm1 = self.norm = nn.Identity()

        self.feature = [nn.Linear(input_dim, inner_dim,bias=mil_bias)]
        if act == 'gelu': 
            self.feature += [nn.GELU()]
        elif act == 'relu':
            self.feature += [nn.ReLU()]
        
        self.feature += [nn.Dropout(dropout)]
        self.feature = nn.Sequential(*self.feature)

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=mil_bias),
        ]
        # if act == 'gelu': 
        #     self.attention_a += [nn.GELU()]
        # elif act == 'relu':
        #     self.attention_a += [nn.ReLU()]
        # elif act == 'tanh':
        self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=mil_bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=mil_bias)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes,bias=mil_bias),
        )

        self.apply(initialize_weights)

    def forward(self, x,**kwargs):
        if len(x.size()) == 2:
            x.unsqueeze_(0)

        if self.embed_norm_pos == 0:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)

        x = self.feature(x)

        if self.embed_norm_pos == 1:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)
            else:
                x = self.norm(x)

        A = self.attention_a(x)
        b = self.attention_b(x)
        A = A.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        #x = torch.matmul(A,x)
        x = torch.einsum('b k n, b n d -> b k d', A,x)

        Y_prob = self.classifier(x.squeeze(1))

        return Y_prob

class DAttention(nn.Module):
    def __init__(self,input_dim,n_classes,dropout,act,mil_norm=None,mil_bias=True,mil_cls_bias=True,inner_dim=512,embed_feat=True,embed_norm_pos=0,pos=None,**kwargs):
        super(DAttention, self).__init__()
        self.L = inner_dim #512
        self.D = 128 #128
        self.K = 1
        self.feature = []
        self.mil_norm = mil_norm
        self.embed_norm_pos = embed_norm_pos
        self.pos = pos

        if mil_bias:
            mil_cls_bias = True

        assert pos in ('sincos','none',None)
        assert self.embed_norm_pos in (0,1)

        if pos == 'sincos':
            self.pos_embed = SINCOS()
        else:
            self.pos_embed = nn.Identity()

        if mil_norm == 'bn':
            self.norm = nn.BatchNorm1d(input_dim) if embed_norm_pos == 0 else nn.BatchNorm1d(inner_dim)
            self.norm1 = nn.BatchNorm1d(self.L*self.K)
        elif mil_norm == 'ln':
            if embed_norm_pos == 0:
                self.feature += [nn.LayerNorm(input_dim,bias=mil_bias)]
                self.norm1 = nn.LayerNorm(self.L*self.K,bias=mil_bias)
            else:
                self.norm = nn.LayerNorm(inner_dim,bias=mil_bias)
                self.norm1 = nn.LayerNorm(self.L*self.K,bias=mil_bias)
        else:
            self.norm1 = self.norm = nn.Identity()
        
        if embed_feat:
            self.feature += [nn.Linear(input_dim, inner_dim,bias=mil_bias)]
            
            if act.lower() == 'gelu':
                self.feature += [nn.GELU()]
            else:
                self.feature += [nn.ReLU()]

            if dropout:
                self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature) if len(self.feature) > 0 else nn.Identity()

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=mil_bias),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=mil_bias)
        )

        self.classifier = nn.Linear(self.L*self.K, n_classes,bias=mil_cls_bias)

        self.apply(initialize_weights)
        
    def forward(self, x, return_attn=False,no_norm=False,return_act=False,pos=None,return_img_feat=False,**kwargs):
        if len(x.size()) == 2:
            x.unsqueeze_(0)
        
        if self.embed_norm_pos == 0:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)

        x = self.feature(x)
        if self.pos == 'sincos':
            x = self.pos_embed(x,pos=pos)

        if self.embed_norm_pos == 1:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)
            else:
                x = self.norm(x)

        act = x.clone()

        # feature = group_shuffle(feature)
        #feature = feature
        A = self.attention(x)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        #M = torch.mm(A, feature)  # KxL
        x = torch.einsum('b k n, b n d -> b k d', A,x).squeeze(1)

        img_feat = x.clone()
        x = self.norm1(x)
        _logits = self.classifier(x)

        if return_img_feat:
            _logits = [_logits,img_feat]

        if return_attn:
            output = []
            output.append(_logits)
            output.append(A.squeeze(1))
            if return_act:
                output.append(act.squeeze(1))
            return output
        else:   
            return _logits

        # if return_attn:
        #     if no_norm:
        #         return Y_prob,A_ori
        #     else:
        #         return Y_prob,A
        # else:
        #     return Y_prob
if __name__ == "__main__":
    x=torch.rand(5,3,64,64).cuda()
    gcnnet=Resnet().cuda()
    Y_prob=gcnnet(x)
    criterion = nn.BCEWithLogitsLoss()
    # loss_max = criterion(Y_prob[1].view(1,-1), label.view(1,-1))
    print(Y_prob)

