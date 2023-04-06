import copy
import torch
import numpy as np
from torch import nn
from einops import repeat
import torchvision.models as models
from modules.emb_position import *
from modules.transformer import *
from modules.mlp import *
from modules.datten import *
from modules.swin_atten import *
import torch.nn.functional as F
from modules.translayer import *
import math

sys.path.append("..")
from utils import group_shuffle, patch_shuffle

def initialize_weights(module):
    for m in module.modules():
        # ref from https://github.com/Meituan-AutoML/Twins/blob/4700293a2d0a91826ab357fc5b9bc1468ae0e987/gvt.py#L356
        # if isinstance(m, nn.Conv2d):
        #     fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     fan_out //= m.groups
        #     m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        #     if m.bias is not None:
        #         m.bias.data.zero_()
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        # elif isinstance(m, nn.Conv2d):
        #     # ref from huggingface
        #     nn.init.xavier_normal_(m.weight)
        #     #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     if m.bias is not None:
        #         m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
class MAE(nn.Module):
    '''
    learnable mask token
    '''
    def __init__(self,mlp_dim=512,only_cls=False,mask_ratio=0.25,drop_in_encoder=True,zero_mask=False,mae_init=False,pos_pos=0,decoder_embed_dim=512,decoder_out_dim=512,norm_targ_loss='none',norm_pred_loss='none',mfm_loss='l2',mfm_enable=True,decoder='fc',pos='ppeg',peg_k=7,mlp_norm_last_layer=False,mlp_nlayers=2,mlp_hidden_dim=512,mlp_bottleneck_dim=256,temp_t=1.,temp_s=1.,mfm_no_targ_grad=False,mfm_feats='last_layer',multi_add_pos=False,multi_type='cat_all',attn='ntrans',pool='cls_token',region_num=8,head=8,shuffle_group=0,attn_layer=1):
        super(MAE, self).__init__()
        self.norm = nn.LayerNorm(mlp_dim)
        self.decoder=decoder
        self.pool = pool
        self.attn_layer = attn_layer
        # not mae recommendation
        if not mae_init:
            self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))
            nn.init.normal_(self.cls_token, std=1e-6)
            # print(self.cls_token.data)

            if not zero_mask:
                # self.mask_token = nn.Parameter(torch.zeros(1, 1, mlp_dim))
                # nn.init.trunc_normal_(self.mask_token,a=0.,b=1.,std=0.2)
                self.mask_token = nn.Parameter(torch.randn(1, 1, mlp_dim))
                nn.init.normal_(self.mask_token, std=0.2)
            else:
                self.mask_token = nn.Parameter(torch.zeros(1, 1, mlp_dim),requires_grad=False)
        else:
        # mae param
            self.cls_token = nn.Parameter(torch.zeros(1, 1, mlp_dim))
            nn.init.normal_(self.cls_token, std=.02)

            if not zero_mask:
                self.mask_token = nn.Parameter(torch.zeros(1, 1, mlp_dim))
                nn.init.normal_(self.mask_token, std=.02)
            else:
                self.mask_token = nn.Parameter(torch.zeros(1, 1, mlp_dim),requires_grad=False)

        if attn == 'ntrans':
            self.layer1 = TransLayer(dim=mlp_dim,head=head)
            self.layer2 = TransLayer(dim=mlp_dim,head=head)
        elif attn == 'dattn':
            self.layer1 = DAttention()
            self.layer2 = DAttention()
        elif attn == 'swin':
            self.layer1 = SwinTransformerBlock(dim=mlp_dim,num_heads=8,fused_window_process=True,drop_path=0.1,window_num=region_num,need_down=True,need_reduce=True)
            self.layer2 = SwinTransformerBlock(dim=mlp_dim,num_heads=8,fused_window_process=True,drop_path=0.1,window_num=region_num)

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(512,k=peg_k)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos
        self.only_cls = only_cls

        self.shuffle_group = shuffle_group
        self.mask_ratio = mask_ratio
        self.drop_in_encoder = drop_in_encoder
        self.norm_targ_loss = norm_targ_loss
        self.norm_pred_loss = norm_pred_loss
        self.mfm_no_targ_grad = mfm_no_targ_grad
        self.mfm_loss = mfm_loss
        self.mfm_feats = mfm_feats
        self.multi_add_pos = multi_add_pos
        self.multi_type = multi_type 
        if self.mfm_loss == 'ce':
            self._loss = SoftTargetCrossEntropy_v2(temp_t=temp_t,temp_s=temp_s)

        self.mfm_enable = mfm_enable
        self.zero_mask = zero_mask
        if mfm_enable:
            _num = 2 if multi_add_pos else 1

            if multi_type == 'u_net':
                self.fusion_layers = [ nn.Sequential(nn.Linear(1024,512),nn.ReLU()) for i in range(len(_num))]
            elif multi_type == 'cat_all':
                self.fusion_layers = nn.Sequential(nn.Linear(512*(_num+1),512),nn.ReLU())

            if decoder == 'fc':
                self.decoder =  nn.Sequential(nn.Linear(512,decoder_out_dim))
            elif decoder == 'mlp':
                self.decoder = MlpHead(hid_dim=mlp_hidden_dim,out_dim=decoder_out_dim)
            elif decoder == 'mlp_dino':
                self.decoder = MlpHeadDINO(in_dim=512,out_dim=decoder_out_dim,norm_last_layer=mlp_norm_last_layer,nlayers=mlp_nlayers,hidden_dim=mlp_hidden_dim,bottleneck_dim=mlp_bottleneck_dim)
            elif decoder == 'trans':
                self.decoder_layer1 = TransLayer(dim=mlp_dim)
                self.decoder_embed = nn.Linear(mlp_dim,decoder_embed_dim) if decoder_embed_dim > 0 else nn.Identity()
                self.decoder_pred = nn.Linear(mlp_dim,decoder_out_dim) if decoder_out_dim > 0 else nn.Identity()
                self.decoder_pos_embed = PEG(512,k=27)

        # MlpHead(out_dim=512)
       # copy from mae@facebook
    def random_masking(self, x, mask_ratio,ids_shuffle=None,len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if ids_shuffle is None:
            # sort noise for each sample
            len_keep = int(L * (1 - mask_ratio))
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
        else:
            _,ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if not self.drop_in_encoder:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x_masked.shape[1], 1)
            x_masked = torch.cat([x_masked, mask_tokens], dim=1)  # no cls token
            x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ids=None, len_keep=None, return_attn=False,mask_enable=False,shuffle=False):
        batch, num_patches, C = x.shape # 直接是特征
        
        _multi_feats = []
        attn = []

        if self.pos_pos == -2:
            x = self.pos_embedding(x)
        
        # masking
        if mask_enable and (self.mask_ratio > 0. or mask_ids is not None):
            x, mask, ids_restore = self.random_masking(x,self.mask_ratio,mask_ids,len_keep)
        else:
            mask,ids_restore = None,None

        if shuffle:
            x = patch_shuffle(x, self.shuffle_group)

        # cls_token
        if self.pool == 'cls_token':
            cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = batch)
            x = torch.cat((cls_tokens, x), dim=1)
        
        if self.pos_pos == -1:
            x = self.pos_embedding(x)

        # translayer1
        if return_attn:
            x,_attn = self.layer1(x,True)
            attn.append(_attn.clone())
        else:
            x = self.layer1(x)

        if self.mfm_feats == 'multi':
            _multi_feats.append(x.clone())

        #加位置编码
        if self.pos_pos == 0:
            x[:,1:,:] = self.pos_embedding(x[:,1:,:])
        
        if self.mfm_feats == 'multi' and self.multi_add_pos:
            _multi_feats.append(x.clone())

        # translayer2
        if return_attn:
            x,_attn = self.layer2(x,True)
            attn.append(_attn.clone())
        else:
            x = self.layer2(x)

        #---->cls_token
        x = self.norm(x)

        if self.mfm_feats == 'multi':
            _multi_feats.append(x.clone())

        if self.pool == 'cls_token':
            logits = x[:,0,:]
        elif self.pool == 'avg':
            logits = x.mean(dim=1)

        if return_attn:
            _a = attn
            # for i in range(1,len(attn)):
            #     if i == len(attn)-1:
            #         _a = attn[i]
            #     _a += attn[i]
            # _a /= len(attn)
            return logits ,x, mask, ids_restore,_multi_feats,_a
        else:
            return logits ,x, mask, ids_restore,_multi_feats
    
    def forward_decoder(self,x,ids_restore,multi_feats): 
        # 多层次特征的处理，借鉴RePre: Improving Self-Supervised Vision Transformer with Reconstructive Pre-training，采用u-net架构，从深到浅
        if self.mfm_feats == 'multi':
            x = multi_feats[-1]
            for i in range(2,multi_feats.size(0)+1):
                if self.multi_type == 'u_net':
                    x = torch.concat((x,multi_feats[-i]),dim=-1)
                    x = self.fusion_layers[i-2](x)
                elif self.multi_type == 'mm':
                    x = torch.matmul(multi_feats[-i],x)
                elif self.multi_type == 'cat_all':
                    x = torch.concat((x,multi_feats[-i]),dim=-1)
                    if i == multi_feats.size(0)-1:
                        x = self.fusion_layers(x)
            
        if self.decoder != 'trans':
            if self.drop_in_encoder:
                mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
                x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
                x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
                x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
            x = self.decoder(x)
            return x[:,1:,:]
        else:
            ids_restore = ids_restore if self.drop_in_encoder else None
            return self.forward_trans_decoder(x,ids_restore)

    def forward_trans_decoder(self,x,ids_restore):
        # embed tokens
        # x = self.decoder_embed(x)

        if ids_restore is not None:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = self.decoder_pos_embed(x)

        # Transformer
        x = self.decoder_layer1(x)
        x = self.norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self,pred,targ,mask):
        
        if self.norm_targ_loss == 'l2':
            targ = nn.functional.normalize(targ,dim=-1, p=2)
        elif self.norm_targ_loss == 'l1':
            targ = nn.functional.normalize(targ,dim=-1, p=1)

        if self.norm_pred_loss == 'l2':
            pred = nn.functional.normalize(pred,dim=-1, p=2)
            # _min,_ = pred.min(dim=-1,keepdim=True)
            # pred = pred - _min
        elif self.norm_pred_loss == 'l1':
            pred = nn.functional.normalize(pred,dim=-1, p=1)
        elif self.norm_pred_loss == 'minmax':
            _min,_ = pred.min(dim=-1,keepdim=True)
            _max,_ = pred.max(dim=-1,keepdim=True)
            pred = (pred - _min) / (_max - _min)

        if self.mfm_loss == 'l1':
            if self.mfm_no_targ_grad:
                loss = torch.abs((pred - targ.detach()))
            else:
                loss = torch.abs((pred - targ))
        elif self.mfm_loss == 's_l1':
            if self.mfm_no_targ_grad:
                loss = nn.functional.smooth_l1_loss(pred,targ.detach())
            else:
                loss = nn.functional.smooth_l1_loss(pred,targ)
            
        elif self.mfm_loss == 'l2':
            if self.mfm_no_targ_grad:
                loss = (pred-targ.detach()) ** 2
            else:
                loss = (pred-targ) ** 2
        elif self.mfm_loss == 'ce':
            if self.mfm_no_targ_grad:
                loss = self._loss(pred,targ.detach(),mean=False)
            else:
                loss = self._loss(pred,targ,mean=False)

        loss = loss.mean(dim=-1)
        loss = (loss*mask).sum() / mask.sum()

        return loss

    def forward(self,x,targ=None,mask_ids=None,len_keep=None,return_attn=False,mask_enable=False,shuffle=False):
        
        if return_attn:
            _logits,x,mask,ids_restore,multi_feats,attn = self.forward_encoder(x,mask_ids,len_keep,True,mask_enable=mask_enable,shuffle=shuffle)
        else:
            _logits,x,mask,ids_restore,multi_feats = self.forward_encoder(x,mask_ids,len_keep,mask_enable=mask_enable,shuffle=shuffle)
        
        logits = _logits.clone()

        if not self.zero_mask and self.mfm_enable and targ is not None:
            x = self.forward_decoder(x,ids_restore,multi_feats)
            loss = self.forward_loss(x,targ,mask)
        else:
            loss = 0

        if return_attn:
            return logits,loss,attn
        else:
            return logits,loss

class SoftTargetCrossEntropy_v2(nn.Module):

    def __init__(self,temp_t=1.,temp_s=1.):
        super(SoftTargetCrossEntropy_v2, self).__init__()
        self.temp_t = temp_t
        self.temp_s = temp_s

    def forward(self, x: torch.Tensor, target: torch.Tensor, mean: bool= True) -> torch.Tensor:
        loss = torch.sum(-F.softmax(target / self.temp_t,dim=-1) * F.log_softmax(x / self.temp_s, dim=-1), dim=-1)
        if mean:
            return loss.mean()
        else:
            return loss
        
class CLR(nn.Module):
    def __init__(self, mlp_dim=512,teacher_init='',mask_ratio=0,n_classes=2,temp_t=1.,temp_s=1.,mfm_decoder='fc',mfm_loss='l1',cl_loss='ce',cl_enable=False,mfm_enable=False,zero_mask=False,cl_type='feat',cl_out_dim=512,dropout=0.25,no_tea_mask=True,drop_in_encoder=True,mae_init=False,pos_pos=0,mfm_decoder_embed_dim=512,mfm_decoder_out_dim=512,mfm_norm_targ_loss='none',mfm_norm_pred_loss='none',n_robust=0,pos='ppeg',peg_k=7,mlp_norm_last_layer=False,mlp_nlayers=2,mlp_hidden_dim=512,mlp_bottleneck_dim=256,mfm_targ_ori=False,mfm_temp_t=0.1,mfm_temp_s=1.,mfm_no_targ_grad=False,act='relu',mfm_feats='last_layer',multi_add_pos=False,multi_type='cat_all',cl_pred_head='fc',cl_targ_no_pred=False,attn='ntrans',pool='cls_token',region_num=8,select_mask=False,head=8,select_inv=False,patch_shuffle=False,shuffle_group=0,msa_fusion='mean',mask_ratio_h=0.,mrh_sche=None,mrh_type='tea',mask_ratio_hr=1.,mask_ratio_l=0.,attn_layer=1):
        super(CLR, self).__init__()
        self.norm_target = False
        self.mfm_targ_ori = mfm_targ_ori
        self.mfm_enable = mfm_enable
        self.mask_ratio = mask_ratio
        self.mask_ratio_h = mask_ratio_h
        self.mask_ratio_hr = mask_ratio_hr
        self.mask_ratio_l = mask_ratio_l
        self.select_mask = select_mask
        self.select_inv = select_inv
        self.msa_fusion = msa_fusion
        self.mrh_sche = mrh_sche
        self.mrh_type = mrh_type
        self.test_type = 'ori'
        self.attn_layer = attn_layer
        # print(head)
        #shuffle
        self.patch_shuffle = patch_shuffle
        self.shuffle_group = shuffle_group

        if self.mfm_targ_ori:
            mfm_decoder_out_dim = 1024

        self.patch_to_emb = [nn.Linear(1024, 512)]

        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        self.online_encoder = MAE(mlp_dim=mlp_dim,mask_ratio=mask_ratio,drop_in_encoder=drop_in_encoder,zero_mask=zero_mask or (not mfm_enable and cl_enable),mae_init=mae_init,pos_pos=pos_pos,mfm_loss=mfm_loss,decoder_embed_dim=mfm_decoder_embed_dim,decoder_out_dim=mfm_decoder_out_dim,norm_targ_loss=mfm_norm_targ_loss,norm_pred_loss=mfm_norm_pred_loss,mfm_enable=mfm_enable,pos=pos,peg_k=peg_k,decoder=mfm_decoder,mlp_norm_last_layer=mlp_norm_last_layer,mlp_nlayers=mlp_nlayers,mlp_hidden_dim=mlp_hidden_dim,mlp_bottleneck_dim=mlp_bottleneck_dim,temp_t=mfm_temp_t,temp_s=mfm_temp_s,mfm_no_targ_grad=mfm_no_targ_grad,mfm_feats=mfm_feats,multi_add_pos=multi_add_pos,multi_type=multi_type,attn=attn,pool=pool,region_num=region_num,head=head,shuffle_group=shuffle_group,attn_layer=attn_layer)

        self.predictor = nn.Linear(mlp_dim,n_classes)

        self.cl_enable = cl_enable
        
        self.temp_t = temp_t
        self.temp_s = temp_s

        if cl_loss == 'l2':
            self.cl_loss = F.mse_loss
        elif cl_loss == 'ce':
            self.cl_loss = SoftTargetCrossEntropy_v2(self.temp_t,self.temp_s)
        elif cl_loss == 'l1':
            self.cl_loss = F.l1_loss

        if cl_enable:
            # 是在最后分类结果层面做对齐还是在特征层面做
            assert cl_type in ('cls','feat','feat_head')
            self.cl_type = cl_type 
            self.no_tea_mask = no_tea_mask
            self.cl_targ_no_pred = cl_targ_no_pred
            # 存不存在，老师又要mask，但又不在encoder丢掉mask部分，而stu却丢了的情况？
            self.target_encoder = MAE(mlp_dim=mlp_dim,mae_init=mae_init,zero_mask=zero_mask,mask_ratio=0. if no_tea_mask else mask_ratio,drop_in_encoder=drop_in_encoder,pos_pos=pos_pos,mfm_enable=False,pos=pos,peg_k=peg_k,attn=attn,pool=pool,region_num=region_num,head=head)
          
            if cl_type == 'cls':
                self.target_predictor = nn.Linear(mlp_dim, n_classes)
                self.predictor_cl = nn.Identity()
            elif cl_type == 'feat_head':
                if cl_pred_head == 'fc':
                    self.target_predictor = nn.Linear(mlp_dim, cl_out_dim)
                    self.predictor_cl = nn.Linear(mlp_dim, cl_out_dim)
                elif cl_pred_head == 'mlp':
                    self.target_predictor = MlpHead(hid_dim=mlp_hidden_dim,out_dim=cl_out_dim)
                    self.predictor_cl = MlpHead(hid_dim=mlp_hidden_dim,out_dim=cl_out_dim)
                elif cl_pred_head == 'mlp_dino':
                    self.target_predictor = MlpHeadDINO(in_dim=512,out_dim=cl_out_dim,nlayers=mlp_nlayers)
                    self.predictor_cl = MlpHeadDINO(in_dim=512,out_dim=cl_out_dim,nlayers=mlp_nlayers)
            else:
                self.predictor_cl = nn.Identity()
                self.target_predictor = nn.Identity()

            if cl_targ_no_pred:
                self.target_predictor = nn.Identity()

            for param in self.target_encoder.parameters():
                param.requires_grad = False
            for param in self.target_predictor.parameters():
                param.requires_grad = False
        else:
            self.target_predictor = nn.Identity()
            self.predictor_cl = nn.Identity()
        
        self.apply(initialize_weights)

        if n_robust>0:
            # 改变init
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            self.apply(initialize_weights)
            torch.random.set_rng_state(rng)
            # 改变后续的随机状态
            [torch.rand((1024,512)) for i in range(n_robust)]
            

        if teacher_init is not None and teacher_init != '' and cl_enable:
            try:
                pre_dict = torch.load(teacher_init)
                new_state_dict ={}
                target_dict = self.target_encoder.state_dict()
                # discard=['cls_token','norm.weight','norm.bias','pos_embedding.proj.weight','pos_embedding.proj.bias','pos_embedding.proj1.weight','pos_embedding.proj1.bias','pos_embedding.proj2.weight','pos_embedding.proj2.bias']
                discard = []
                for k,v in pre_dict.items():
                    k = k.replace('online_encoder.','') if 'online_encoder' in k else k
                    if k not in discard and k in target_dict:
                        new_state_dict[k]=v
                info = self.target_encoder.load_state_dict(new_state_dict,strict=False)
                print('Teacher Inited')
                print(info)
            except:
                pre_dict = self.online_encoder.state_dict()
                new_state_dict ={}
                target_dict = self.target_encoder.state_dict()
                # discard=['cls_token','norm.weight','norm.bias','pos_embedding.proj.weight','pos_embedding.proj.bias','pos_embedding.proj1.weight','pos_embedding.proj1.bias','pos_embedding.proj2.weight','pos_embedding.proj2.bias']
                discard = []
                for k,v in pre_dict.items():
                    k = k.replace('online_encoder.','') if 'online_encoder' in k else k
                    if k not in discard and k in target_dict:
                        new_state_dict[k]=v
                info = self.target_encoder.load_state_dict(new_state_dict,strict=False)
                print(info)

    # def trainable_parameters(self):
    #     r"""Returns the parameters that will be updated via an optimizer."""
    #     return list(self.online_encoder.parameters(), list(self.predictor.parameters()))

    @torch.no_grad()
    def update_target_network(self, mm=0.9999,mm_head=0.999):
        r"""Performs a momentum update of the target network's weights.
        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm

        mm_head = mm if mm_head == 0. else mm_head

        if not self.cl_enable:
            return
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm) # mm*k +(1-mm)*q

        # head 没有初始化，跟其它模型的更新率不应该一样
        if self.cl_type == 'cls':
            for param_q, param_k in zip(self.predictor.parameters(), self.target_predictor.parameters()):
                param_k.data.mul_(mm_head).add_(param_q.data, alpha=1. - mm_head) # mm*k +(1-mm)*q

        elif self.cl_type == 'feat_head' and not self.cl_targ_no_pred:
            for param_q, param_k in zip(self.predictor_cl.parameters(), self.target_predictor.parameters()):
                param_k.data.mul_(mm_head).add_(param_q.data, alpha=1. - mm_head) # mm*k +(1-mm)*q
    
    def select_mask_fn(self,ps,attn,largest,mask_ratio,mask_ids_other=None,len_keep_other=None,cls_attn_topk_idx_other=None,random_ratio=1.,select_inv=False):
        ps_tmp = ps
        mask_ratio_ori = mask_ratio
        mask_ratio = mask_ratio / random_ratio
        if mask_ratio > 1:
            random_ratio = mask_ratio_ori
            mask_ratio = 1.

        # print(attn.size())
        if mask_ids_other is not None:
            if cls_attn_topk_idx_other is None:
                cls_attn_topk_idx_other = mask_ids_other[:,len_keep_other:].squeeze()
                ps_tmp = ps - cls_attn_topk_idx_other.size(0)
        # 每个head 取一部分，这样做的坏处是有些head的attn没有任何差异性
        if self.msa_fusion == 'mean':
            _,cls_attn_topk_idx = torch.topk(attn,int(np.ceil((ps_tmp*mask_ratio)) // attn.size(1)),largest=largest)
            cls_attn_topk_idx = torch.unique(cls_attn_topk_idx.flatten(-3,-1))
        # 投票机制，这个的问题在于mask ratio 高了之后，选择也倾向于随机，因为大部分的票数还是1
        elif self.msa_fusion == 'vote':
            vote = attn.clone()
            vote[:] = 0
            
            _,idx = torch.topk(attn,k=int(np.ceil((ps_tmp*mask_ratio))),sorted=False,largest=largest)
            mask = vote.clone() 
            mask = mask.scatter_(2,idx,1) == 1
            vote[mask] = 1
            vote = vote.sum(dim=1)
            _,cls_attn_topk_idx = torch.topk(vote,k=int(np.ceil((ps_tmp*mask_ratio))),sorted=False)
            # print(cls_attn_topk_idx.size())
            cls_attn_topk_idx = cls_attn_topk_idx[0]
        
        cls_attn_keep_idx = None
        # 从中随机选取部分
        if random_ratio < 1.:
            random_idx = torch.randperm(cls_attn_topk_idx.size(0),device=cls_attn_topk_idx.device)
            # keep high
            #cls_attn_keep_idx = torch.gather(cls_attn_topk_idx,dim=0,index=random_idx[int(np.ceil((cls_attn_topk_idx.size(0)*random_ratio))):])
            cls_attn_topk_idx = torch.gather(cls_attn_topk_idx,dim=0,index=random_idx[:int(np.ceil((cls_attn_topk_idx.size(0)*random_ratio)))])
        
        # 合并其它的mask图块
        if mask_ids_other is not None:
            # print(attn.size())
            # print(cls_attn_topk_idx.size())
            # print(cls_attn_topk_idx_other.size())
            # if select_inv:
                 
            # else:
            cls_attn_topk_idx = torch.cat([cls_attn_topk_idx,cls_attn_topk_idx_other]).unique()
            #cls_attn_topk_idx = torch.cat([torch.gather(mask_ids_other[:,:len_keep_other],dim=1,index=cls_attn_topk_idx.unsqueeze(0))[0],cls_attn_topk_idx_other]).unique()

            
            # keep high
            #if largest and cls_attn_keep_idx is not None:
                #cls_attn_topk_idx = torch.tensor(list(set(cls_attn_topk_idx.tolist()).difference(set(cls_attn_keep_idx.tolist()))),device=attn.device)

        len_keep = ps - cls_attn_topk_idx.size(0)
        a = set(cls_attn_topk_idx.tolist())
        b = set(list(range(ps)))
        # 无论前面的idx是否有序，这里的difference操作会进行相应的排序，能够保证mask_ids的有序
        mask_ids =  torch.tensor(list(b.difference(a)),device=attn.device)

        if select_inv:
            mask_ids = torch.cat([cls_attn_topk_idx,mask_ids]).unsqueeze(0)
            len_keep = ps - len_keep
        else:
            mask_ids = torch.cat([mask_ids,cls_attn_topk_idx]).unsqueeze(0)

        return len_keep,mask_ids

    def get_mask(self,x,ps,i,attn,mrh=None):
        # 由外部获得attention,这是mask置信度最低的部分
        # if attn is not None and self.mask_ratio > 0.:
        #     if isinstance(attn,(list,tuple)):
        #         if self.attn_layer == -1:
        #             _attn = attn[1]
        #         else:
        #             _attn = attn[self.attn_layer]
        #     else:
        #         _attn = attn
        #     len_keep,mask_ids = self.select_mask_fn(ps,_attn,False,self.mask_ratio,select_inv=self.select_inv,random_ratio=self.mask_ratio_sr)
        # else:
        #     len_keep,mask_ids = 0,None

        if attn is not None and isinstance(attn,(list,tuple)):
            if self.attn_layer == -1:
                _attn = attn[1]
            else:
                _attn = attn[self.attn_layer]
        else:
            _attn = attn

        # random mask
        if _attn is not None and self.mask_ratio > 0.:
            len_keep,mask_ids = self.select_mask_fn(ps,_attn,False,self.mask_ratio,select_inv=self.select_inv,random_ratio=0.001)
        else:
            len_keep,mask_ids = ps,None

        # low mask
        if _attn is not None and self.mask_ratio_l > 0.:
            if mask_ids is None:
                len_keep,mask_ids = self.select_mask_fn(ps,_attn,False,self.mask_ratio_l,select_inv=self.select_inv)
            else:
                cls_attn_topk_idx_other = mask_ids[:,:len_keep].squeeze() if self.select_inv else mask_ids[:,len_keep:].squeeze()
                len_keep,mask_ids = self.select_mask_fn(ps,_attn,False,self.mask_ratio_l,select_inv=self.select_inv,mask_ids_other=mask_ids,len_keep_other=ps,cls_attn_topk_idx_other = cls_attn_topk_idx_other)
        # print(len_keep)

        # mask置信度最高的部分
        mask_ratio_h = self.mask_ratio_h
        if self.mrh_sche is not None:
            mask_ratio_h = self.mrh_sche[i]
        if mrh is not None:
            mask_ratio_h = mrh
        if mask_ratio_h > 0.:
            # mask high conf patch
            if self.mrh_type == 'stu':
                with torch.no_grad(): 
                    _,_,attn = self.online_encoder(x,return_attn=True,len_keep=len_keep,mask_ids=mask_ids,mask_enable=True)
                # 由学生来选最高
                if isinstance(attn,(list,tuple)):
                    if self.attn_layer == -1:
                        _attn = attn[0]
                    else:
                        _attn = attn[self.attn_layer]
                else:
                    _attn = attn
                len_keep,mask_ids = self.select_mask_fn(ps,_attn,largest=True,mask_ratio=mask_ratio_h,mask_ids_other=mask_ids,len_keep_other=len_keep,random_ratio=self.mask_ratio_hr)
            elif self.mrh_type == 'tea':
                # 两次都由teacher来选
                if isinstance(attn,(list,tuple)):
                    if self.attn_layer == -1:
                        _attn = attn[0]
                    else:
                        _attn = attn[self.attn_layer]
                else:
                    _attn = attn
                if mask_ids is None:
                    len_keep,mask_ids = self.select_mask_fn(ps,_attn,largest=True,mask_ratio=mask_ratio_h,len_keep_other=ps,random_ratio=self.mask_ratio_hr,select_inv=self.select_inv)
                else:
                    cls_attn_topk_idx_other = mask_ids[:,:len_keep].squeeze() if self.select_inv else mask_ids[:,len_keep:].squeeze()

                    len_keep,mask_ids = self.select_mask_fn(ps,_attn,largest=True,mask_ratio=mask_ratio_h,mask_ids_other=mask_ids,len_keep_other=ps,cls_attn_topk_idx_other = cls_attn_topk_idx_other,random_ratio=self.mask_ratio_hr,select_inv=self.select_inv)

        return len_keep,mask_ids

    @torch.no_grad()
    def forward_teacher(self,x,return_attn=False,no_fc=False):
        if not no_fc:
            x = self.patch_to_emb(x)
            x = self.dp(x)
        if return_attn:
            x,_,attn = self.online_encoder(x,return_attn=True)
        else:
            x,_ = self.online_encoder(x)
            attn = None

        x_cl = self.predictor_cl(x)

        return x,x_cl,attn

    @torch.no_grad()
    def forward_test(self,x,attn=None):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        ps = x.size(1)

        if self.test_type == 'ori':
            x,_ = self.online_encoder(x)
            x = self.predictor(x)
            return x

        elif self.test_type == 'mask':
            x_ori = x.clone()
            if attn is None:
                _,_,attn = self.online_encoder(x,return_attn=True)

            # 这里能不能mask 置信度高的？
            # len_keep,mask_ids = self.select_mask_fn(ps,attn,False,self.mask_ratio)
            len_keep,mask_ids = self.get_mask(x,ps,0,attn,0)
            x_masked,_ = self.online_encoder(x_ori,len_keep=len_keep,mask_ids=mask_ids,mask_enable=True)

            x_masked = self.predictor(x_masked)
            return x_masked
        
        elif self.test_type == 'mean':
            x_ori = x.clone()
            if attn is None:
                x,_,attn = self.online_encoder(x,return_attn=True)
            else:
                x,_ = self.online_encoder(x)

            # 这里能不能mask 置信度高的？
            #len_keep,mask_ids = self.select_mask_fn(ps,attn,False,self.mask_ratio)
            len_keep,mask_ids = self.get_mask(x,ps,0,attn,0)
            x_masked,_ = self.online_encoder(x_ori,len_keep=len_keep,mask_ids=mask_ids,mask_enable=True)
            x = self.predictor(x)
            x_masked = self.predictor(x_masked)

            return [x,x_masked]

    def test(self,x,return_attn=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        ps = x.size(1)

        if return_attn:
            x,_,attn = self.online_encoder(x,return_attn=True)
        else:
            x,_ = self.online_encoder(x)

        x = self.predictor(x)

        if self.training:
            if return_attn:
                return x, 0, 0,ps,ps,attn
            else:
                return x, 0, 0,ps,ps
        else:
            if return_attn:
                return x,attn
            else:
                return x

    def forward_loss(self, student_cls_feat, teacher_cls_feat,student_logit):

        if teacher_cls_feat is not None:
            # if self.cl_type == 'cls':
            #     cls_loss = self.cl_loss(student_logit,teacher_cls_feat.detach())
            # else:
            cls_loss = self.cl_loss(student_cls_feat,teacher_cls_feat.detach())
        else:
            cls_loss = 0.
        
        return cls_loss

    def forward(self, x,attn=None,teacher_cls_feat=None,i=None,no_fc=False):
        # 对特征进行编码,转换维度1024-512
        # if self.mfm_targ_ori:
        #     targ = x.clone()
        #     x = self.patch_to_emb(x) # n*512
        #     x = self.dp(x)
        # else:
        # 实验看起来在dropout之前的特征做target效果没有之后的好
        if not no_fc:
            x = self.patch_to_emb(x)
            x = self.dp(x)
            #targ = x.clone()
        targ = None
        ps = x.size(1)

        x1 = x.clone()

        # if self.patch_shuffle:
        #     x1 = patch_shuffle(x1,self.shuffle_group)

        # forward target network
        if self.cl_enable:
            with torch.no_grad():
                x2 = x.clone()
                if self.select_mask and attn is None:
                    teacher_cls_feat, _, attn = self.target_encoder(x2,return_attn=True)
                else:
                    teacher_cls_feat, _= self.target_encoder(x2)
                teacher_cls_feat = self.target_predictor(teacher_cls_feat)

        # get mask
        if self.select_mask:
            len_keep,mask_ids = self.get_mask(x1,ps,i,attn)
        else:
            len_keep,mask_ids = 0,None

        # forward online network
        student_cls_feat, mfm_loss = self.online_encoder(x1,targ,len_keep=len_keep,mask_ids=mask_ids,mask_enable=True,shuffle=self.patch_shuffle)

        # prediction
        student_logit = self.predictor(student_cls_feat)
        student_cls_feat = self.predictor_cl(student_cls_feat)

        cls_loss = self.forward_loss(student_cls_feat=student_cls_feat,teacher_cls_feat=teacher_cls_feat,student_logit=student_logit)

        return student_logit, mfm_loss, cls_loss,ps,len_keep