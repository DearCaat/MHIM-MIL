import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1,dropout=True,act='relu'):
        super(FCLayer, self).__init__()

        self.embed = [nn.Linear(1024, 512)]
    
        if act.lower() == 'gelu':
            self.embed += [nn.GELU()]
        else:
            self.embed += [nn.ReLU()]

        if dropout:
            self.embed += [nn.Dropout(0.25)]

        self.embed = nn.Sequential(*self.embed)
        self.fc = nn.Sequential(
            nn.Linear(in_size, out_size))
    def forward(self, feats):
        feats = self.embed(feats)
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        # feats = feats.squeeze()
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=True,bias=True,norm=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128,bias=bias), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128,bias=bias)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size,bias=bias),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.mil_norm = norm
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(input_size)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(input_size)
        else:
            self.norm = nn.Identity()
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size,bias=bias)  
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        #Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        Q = self.q(feats)
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 1, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.gather(feats, dim=1, index=m_indices[:,0, :].unsqueeze(-1).expand(-1, -1, feats.size(-1))) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.matmul(Q, q_max.transpose(-2, -1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32, device=device)), 1) # normalize attention scores, A in shape N x C, 
        B = torch.matmul(A.transpose(-2, -1), V) # compute bag representation, B in shape C x V
                
        #B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        if self.mil_norm == 'bn':
            B = torch.transpose(B, -1, -2)
            B = self.norm(B)
            B = torch.transpose(B, -1, -2)
        else:
            B = self.norm(B)

        C = self.fcc(B) # B x C x 1
        C = C.squeeze(-1)  # B x C

        return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self,n_classes,dropout,act,input_dim=1024,mil_norm=None,mil_bias=True,inner_dim=512,**kwargs):
        super(MILNet, self).__init__()

        self.feature = []
        self.mil_norm = mil_norm
        if mil_norm == 'bn':
            self.norm = nn.BatchNorm1d(input_dim)
            self.norm1 = nn.BatchNorm1d(inner_dim)
        elif mil_norm == 'ln':
            self.feature += [nn.LayerNorm(input_dim,bias=mil_bias)]
        else:
            self.norm1 = self.norm = nn.Identity()

        self.feature += [nn.Linear(input_dim, inner_dim,bias=mil_bias)]

        if act.lower() == 'relu':
            self.feature += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        self.feature= nn.Sequential(*self.feature)
        
        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        #self.i_classifier = FCLayer(512,n_classes,dropout,act,input_dim=input_dim)

        self.i_classifier = nn.Linear(inner_dim, n_classes,bias=mil_bias)
        self.b_classifier = BClassifier(inner_dim,n_classes,bias=mil_bias,norm=mil_norm)

        self.apply(initialize_weights)
        
    def forward(self, x,label=None,loss=None,pos=None,**kwargs):
        ps = x.size(1)
        bs = x.size(0)

        if self.mil_norm == 'bn':
            x = torch.transpose(x, -1, -2)
            x = self.norm(x)
            x = torch.transpose(x, -1, -2)

        x = self.feature(x)
        feats = self.dp(x)

        # feats = torch.transpose(feats, -1, -2)
        # feats = self.norm1(feats)
        # feats = torch.transpose(feats, -1, -2)

        classes = self.i_classifier(feats)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        classes,_ = torch.max(classes, 1)

        if self.training:
            if isinstance(loss,nn.CrossEntropyLoss):
                max_loss = loss(classes.view(bs,-1),label)
            elif isinstance(loss,nn.BCEWithLogitsLoss):
                max_loss = loss(classes.view(bs, -1), label.view(bs, -1).float())
            else:
                max_loss = loss(logits=classes.view(bs,-1),Y=label[0],c=label[1])
            return prediction_bag,max_loss,ps
        else:
            return prediction_bag,classes