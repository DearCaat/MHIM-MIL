import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # ref from meituan
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0,n_robust=0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)
        
        self.apply(initialize_weights)

        if n_robust>0:
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            self.apply(initialize_weights)
            torch.random.set_rng_state(rng)
            [torch.rand((1024,512)) for i in range(n_robust)]

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x

class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0,dropout=False,act='relu',n_robust=0,swin=False,swin_convk=15,swin_moek=3,swin_as=False,swin_md=False):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.swin = SwinEncoder(attn='swin',pool='none',n_layers=2,conv_k=swin_convk,moe_k=swin_moek,all_shortcut=swin_as,moe_mask_diag=swin_md,init=True) if swin else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True) if act.lower() == 'relu' else nn.GELU()
        self.numRes = numLayer_Res
        self.drop = nn.Dropout(0.25)
        self.dropout = dropout
        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

        
        self.apply(initialize_weights)

        if n_robust>0:
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            self.apply(initialize_weights)
            torch.random.set_rng_state(rng)
            [torch.rand((1024,512)) for i in range(n_robust)]
    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)
        if self.dropout:
            x = self.drop(x)
        if self.numRes > 0:
            x = self.resBlocks(x)
        
        x = self.swin(x)

        return x

class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0,n_robust=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention(L, D, K,n_robust=n_robust)
        self.classifier = Classifier_1fc(L, num_cls, droprate,n_robust=n_robust)
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred

class Attention(nn.Module):
    def __init__(self, L=512, D=128, K=1,n_robust=0):
        super(Attention, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.apply(initialize_weights)

        if n_robust>0:
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            self.apply(initialize_weights)
            torch.random.set_rng_state(rng)
            [torch.rand((1024,512)) for i in range(n_robust)]

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N

class DTFD(nn.Module):
    def __init__(self,device,lr,weight_decay,steps,input_dim=1024,inner_dim=512,n_classes=2,group=5,distill='AFS',**kwargs) -> None:
        super().__init__()
        self.classifier = Classifier_1fc(inner_dim, n_classes, 0.25)
        self.attention = Attention(inner_dim)
        self.dimReduction = DimReduction(input_dim, inner_dim, dropout=0.25)
        self.UClassifier = Attention_with_Classifier(L=inner_dim, num_cls=n_classes, droprate=0.25)
        self.group = group
        self.distill = distill
        self.ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        trainable_parameters = []
        trainable_parameters += list(self.classifier.parameters())
        trainable_parameters += list(self.attention.parameters())
        trainable_parameters += list(self.dimReduction.parameters())
        self.optimizer0 = torch.optim.Adam(trainable_parameters, lr=lr,  weight_decay=weight_decay)
        self.scheduler0 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer0, steps, 0)

    def train_forward(self,x,label):
        feat_index = list(range(x.shape[0]))
        
        index_chunk_list = np.array_split(np.array(feat_index), self.group)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]
        slide_pseudo_feat = []
        slide_sub_preds = []
        slide_sub_labels = []

        midFeat = self.dimReduction(x)

        for tindex in index_chunk_list:
            slide_sub_labels.append(label)
            #subFeat_tensor = torch.index_select(x, dim=0, index=torch.LongTensor(tindex))
            #tmidFeat = self.dimReduction(subFeat_tensor)
            device = midFeat.device
            tmidFeat = midFeat.index_select(dim=0, index=torch.LongTensor(tindex).to(device))
            tAA = self.attention(tmidFeat).squeeze(0)

            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = self.classifier(tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)

            patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
            topk_idx_max = sort_idx[:1].long()
            topk_idx_min = sort_idx[-1:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if self.distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif self.distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif self.distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### 
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
        # slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
        # loss0 = self.ce_cri(slide_sub_preds, slide_sub_labels).mean()
        # self.optimizer0.zero_grad()
        # loss0.backward(retain_graph=True)

        gSlidePred = self.UClassifier(slide_pseudo_feat)
        return gSlidePred

    def test_forward(self,x):
        tfeat = x
        midFeat = self.dimReduction(tfeat)
        AA = self.attention(midFeat, isNorm=False).squeeze(0)  ## N

        feat_index = list(range(tfeat.shape[0]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), self.group)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        slide_d_feat = []
        slide_sub_preds = []

        for tindex in index_chunk_list:
            idx_tensor = torch.LongTensor(tindex).to(x.device)
            tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

            tAA = AA.index_select(dim=0, index=idx_tensor)
            tAA = torch.softmax(tAA, dim=0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

            tPredict = self.classifier(tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)

            patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

            if self.distill == 'MaxMinS':
                topk_idx_max = sort_idx[:1].long()
                topk_idx_min = sort_idx[-1:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                slide_d_feat.append(d_inst_feat)
            elif self.distill == 'MaxS':
                topk_idx_max = sort_idx[:1].long()
                topk_idx = topk_idx_max
                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                slide_d_feat.append(d_inst_feat)
            elif self.distill == 'AFS':
                slide_d_feat.append(tattFeat_tensor)

        slide_d_feat = torch.cat(slide_d_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)

        gSlidePred = self.UClassifier(slide_d_feat)
        
        return gSlidePred

    def forward(self, x, label=None,**kwargs):
        x = x.squeeze(0)
        if self.training:
            return self.train_forward(x,label)
        else:
            return self.test_forward(x)
        

if __name__ == "__main__":
    x = torch.rand(100,1024)
    y = torch.ones(1,).long()
    dtfd = DTFD(x.device,1e-5,1e-5,100,)
    pred,loss = dtfd(x,y)
    print(pred.size())
    print(loss)
    dtfd.eval()
    x_test = dtfd(x)
    print(x_test.size())