import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.nystrom_attention import NystromAttention
from modules.emb_position import *
from einops import repeat

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

    def forward(self, x, return_attn=False,no_norm=False,return_act=False,**kwargs):

        act = x.clone()
        x,attn = self.attention(x,no_norm)

        if return_attn:
            output = []
            output.append(x.squeeze(1))
            output.append(attn.squeeze(1))
            if return_act:
                output.append(act.squeeze(1))
            return output
        else:   
            return x.squeeze(1)
        
class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=True): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c, no_norm=False): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device))
        if no_norm:
            _A = A
        A = F.softmax( A, 0) # normalize attention scores, A in shape N x C, 
        if not no_norm:
            _A = A
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, _A, B 
class DSMIL(nn.Module):
    def __init__(self,n_classes=2,mask_ratio=0.,mlp_dim=512,cls_attn=True,attn_index='max'):
        super(DSMIL, self).__init__()

        self.i_classifier = nn.Sequential(
            nn.Linear(mlp_dim, n_classes))
        self.b_classifier = BClassifier(mlp_dim,n_classes)

        self.cls_attn = cls_attn
        self.attn_index = attn_index

        self.mask_ratio = mask_ratio

    def attention(self,x,no_norm=False,return_attn=False,return_cam=False,**kwargs):
        ps = x.size(1)
        feats = x.squeeze(0)
        classes = self.i_classifier(feats)
        prediction_bag, A, B = self.b_classifier(feats, classes,no_norm)
        
        classes_bag,_ = torch.max(classes, 0) 

        if return_attn:
            if self.attn_index == 'max':
                attn,_ = torch.max(classes,-1) if self.cls_attn else torch.max(A,-1)
            else:
                attn = classes[:,int(self.attn_index)] if self.cls_attn else A[:,int(self.attn_index)]
            attn = attn.unsqueeze(0)
            if return_cam:
                _cam = classes
                attn = [attn,_cam.unsqueeze(0)]
        else:
            attn = None

        return prediction_bag,classes_bag.unsqueeze(0),attn,B

    def forward(self, x,return_attn=False,no_norm=False,**kwargs):

        logits, logits_ins,attn, B= self.attention(x, no_norm, return_attn=return_attn,**kwargs)
        if return_attn:
            return [logits, logits_ins],B, attn
        else:   
            return [logits, logits_ins],B

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

class SAttention(nn.Module):

    def __init__(self,mlp_dim=512,pos_pos=0,pos='ppeg',peg_k=7,head=8):
        super(SAttention, self).__init__()
        self.norm = nn.LayerNorm(mlp_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))

        self.layer1 = TransLayer(dim=mlp_dim,head=head)
        self.layer2 = TransLayer(dim=mlp_dim,head=head)

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(512,k=peg_k)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

    def forward(self, x, return_attn=False,return_act=False,no_norm=False,**kwargs):
        batch, num_patches, C = x.shape 
        attn = []

        if self.pos_pos == -2:
            x = self.pos_embedding(x)
        
        # cls_token
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = batch)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.pos_pos == -1:
            x = self.pos_embedding(x)
        # translayer1
        if return_attn:
            x,_attn,v = self.layer1(x,need_attn=True,need_v=True,no_norm=no_norm)
            attn.append(_attn.clone())
        else:
            x = self.layer1(x)

        # add pos embedding
        if self.pos_pos == 0:
            x[:,1:,:] = self.pos_embedding(x[:,1:,:])
        
        # translayer2
        if return_attn:
            x,_attn,_ = self.layer2(x,need_attn=True,need_v=True,no_norm=no_norm)
            attn.append(_attn.clone())
        else:
            x = self.layer2(x)

        #---->cls_token
        x = self.norm(x)

        logits = x[:,0,:]

        if return_attn:
            output = []
            output.append(logits)
            output.append(attn)
            if return_act:
                output.append(v)
            return output
        else:
            return logits