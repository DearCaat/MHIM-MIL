import torch
from torch import nn

from modules.mhim_modules.baseline import SAttention, DAttention, DSMIL
from modules.mhim_modules.masking import select_mask_fn, mask_fn
from modules.mhim_modules.scoring import get_pseudo_score, get_pseudo_score_trans
from modules.mhim_modules.merge import Merge
from modules.mhim_modules.losses import SoftTargetCrossEntropy
from modules.mhim_modules.utils import initialize_weights


class MHIM(nn.Module):
    """
    Multi-Head Instance Merging model.
    
    Implements a MIL framework with:
    - Dynamic patch masking based on attention scores
    - Patch merging mechanism
    - Self-supervised learning with teacher-student framework
    """
    
    def __init__(self, input_dim=1024, mlp_dim=512, mask_ratio=0, n_classes=2, temp_t=1., 
                 dropout=0.25, act='relu',
                 mask_ratio_h=0., mrh_sche=None, mask_ratio_hr=0., 
                 mask_ratio_l=0., da_act='gelu', baseline='selfattn', head=8,
                 attn2score=True, merge_enable=True, merge_k=1, merge_mm=0.9998, 
                 merge_ratio=0., merge_test=False):
        """
        Args:
            input_dim: Input feature dimension
            mlp_dim: MLP hidden dimension
            mask_ratio: Random masking ratio (for MHIM-v1)
            n_classes: Number of classes
            temp_t: Teacher temperature for distillation
            dropout: Dropout rate
            act: Activation function ('relu', 'gelu')
            mask_ratio_h: High-attention masking ratio
            mrh_sche: Masking ratio schedule
            mask_ratio_hr: Random ratio for high-attention masking
            mask_ratio_l: Low-attention masking ratio (for MHIM-v1)
            da_act: Activation for DAttention
            baseline: Baseline encoder ('selfattn', 'attn', 'dsmil')
            head: Number of attention heads
            attn_layer: Attention layer index for scoring (for ablation study)
            attn2score: Whether to convert attention to score 
            merge_enable: Whether to enable merging
            merge_k: Number of merge tokens
            merge_mm: Merge momentum
            merge_ratio: Merge ratio
            merge_test: Whether to use merge at test time
            merge_mask_type: Type of masking for merge ('random', 'low') (for ablation study)
        """
        super(MHIM, self).__init__()
 
        self.mask_ratio = mask_ratio
        self.mask_ratio_h = mask_ratio_h
        self.mask_ratio_hr = mask_ratio_hr
        self.mask_ratio_l = mask_ratio_l
        self.select_inv = False
        self.msa_fusion = "vote"
        self.mrh_sche = mrh_sche
        self.attn_layer = 0
        self.baseline = baseline
        self.merge_test = merge_test
        self.attn2score = attn2score
        self.head = head

        # Feature extraction
        self.feature = [nn.Linear(input_dim, mlp_dim)]

        if act.lower() == 'relu':
            self.feature += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.feature += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # Merge module
        self.merge = Merge(
            mlp_dim, k=merge_k, g_q_mm=merge_mm, merge_ratio=merge_ratio, 
            mask_type='random'
        ) if merge_enable else nn.Identity()

        self.feature = nn.Sequential(*self.feature)

        # Baseline encoder
        if baseline == 'selfattn':
            self.online_encoder = SAttention(mlp_dim=mlp_dim, head=head)
        elif baseline == 'attn':
            self.online_encoder = DAttention(mlp_dim, da_act)
        elif baseline == 'dsmil':
            self.online_encoder = DSMIL(
                n_classes=n_classes, mlp_dim=mlp_dim, mask_ratio=mask_ratio, 
                cls_attn=self.attn2score
            )

        self.predictor = nn.Linear(mlp_dim, n_classes)

        self.temp_t = temp_t
        self.temp_s = 1.

        self.cl_loss = SoftTargetCrossEntropy(self.temp_t, self.temp_s)

        self.predictor_cl = nn.Identity()
        self.target_predictor = nn.Identity()

        self.apply(initialize_weights)

    def get_mask(self, ps, i, attn, mrh=None):
        """
        Get masking indices based on attention scores.
        
        Args:
            ps: Number of patches
            i: Current iteration
            attn: Attention scores
            mrh: Manual mask ratio high (optional)
        
        Returns:
            len_keep: Number of patches to keep
            mask_ids: Masking indices
        """
        # Random mask, only for MHIM-ICCV
        if attn is not None and self.mask_ratio > 0.:
            len_keep, mask_ids = select_mask_fn(
                ps, attn, False, self.mask_ratio, select_inv=self.select_inv, 
                random_ratio=0.001, msa_fusion=self.msa_fusion
            )
        else:
            len_keep, mask_ids = ps, None

        # Low attention mask, only for MHIM-ICCV
        if attn is not None and self.mask_ratio_l > 0.:
            if mask_ids is None:
                len_keep, mask_ids = select_mask_fn(
                    ps, attn, False, self.mask_ratio_l, select_inv=self.select_inv, 
                    msa_fusion=self.msa_fusion
                )
            else:
                cls_attn_topk_idx_other = (
                    mask_ids[:, :len_keep].squeeze() if self.select_inv 
                    else mask_ids[:, len_keep:].squeeze()
                )
                len_keep, mask_ids = select_mask_fn(
                    ps, attn, False, self.mask_ratio_l, select_inv=self.select_inv, 
                    mask_ids_other=mask_ids, len_keep_other=ps, 
                    cls_attn_topk_idx_other=cls_attn_topk_idx_other, 
                    msa_fusion=self.msa_fusion
                )
        
        # High attention mask
        mask_ratio_h = self.mask_ratio_h
        if self.mrh_sche is not None:
            mask_ratio_h = self.mrh_sche[i]
        if mrh is not None:
            mask_ratio_h = mrh
            
        if mask_ratio_h > 0.:
            # Mask high confidence patches
            if mask_ids is None:
                len_keep, mask_ids = select_mask_fn(
                    ps, attn, largest=True, mask_ratio=mask_ratio_h, 
                    len_keep_other=ps, random_ratio=self.mask_ratio_hr, 
                    select_inv=self.select_inv, msa_fusion=self.msa_fusion
                )
            else:
                cls_attn_topk_idx_other = (
                    mask_ids[:, :len_keep].squeeze() if self.select_inv 
                    else mask_ids[:, len_keep:].squeeze()
                )
                len_keep, mask_ids = select_mask_fn(
                    ps, attn, largest=True, mask_ratio=mask_ratio_h, 
                    mask_ids_other=mask_ids, len_keep_other=ps, 
                    cls_attn_topk_idx_other=cls_attn_topk_idx_other, 
                    random_ratio=self.mask_ratio_hr, select_inv=self.select_inv, 
                    msa_fusion=self.msa_fusion
                )

        return len_keep, mask_ids

    @torch.no_grad()
    def forward_teacher(self, x):
        """
        Forward pass for teacher network (no gradient).
        
        Args:
            x: Input features
        
        Returns:
            x: Output features
            attn: Attention scores
        """
        x = self.feature(x)
        x = self.dp(x)
        
        if self.merge_test:
            p = x.size(1)
            self.training = False
            x = self.merge(x)
            self.training = True

        if self.baseline == 'dsmil':
            _, x, attn = self.online_encoder(x, return_attn=True)
            if self.merge_test:
                attn = attn[:, :p]
        else:
            x, attn, act = self.online_encoder(x, return_attn=True, return_act=True)
            if self.merge_test:
                # TransMIL: n_layers, 1, n_heads, L
                if type(attn) in (list, tuple):
                    attn = [attn[i][:, :, :p] for i in range(len(attn))]
                else:
                    attn = attn[:, :p]

            if self.attn2score:
                if self.baseline == 'selfattn':
                    attn = get_pseudo_score_trans(
                        self.predictor, act, attn[0], 
                        self.online_encoder.layer1.attn.to_out
                    )
                else:
                    attn = get_pseudo_score(self.predictor, act, attn)
            else:
                if attn is not None and isinstance(attn, (list, tuple)):
                    attn = attn[self.attn_layer]
                    
        return x, attn
    
    @torch.no_grad()
    def forward_test(self, x, return_attn=False, no_norm=False, return_act=False, **kwargs):
        """
        Forward pass for inference.
        
        Args:
            x: Input features
            return_attn: Whether to return attention scores
            no_norm: Whether to skip normalization in attention
            return_act: Whether to return activations
        
        Returns:
            x: Predictions
            a: Attention scores (if return_attn=True)
        """
        x = self.feature(x)
        x = self.dp(x)
        
        if self.merge_test:
            x = self.merge(x)

        if return_attn:
            if return_act:
                x, a, act = self.online_encoder(
                    x, return_attn=True, return_act=True, no_norm=no_norm, **kwargs
                )
                a = [a, act]
            else:
                if self.baseline == 'dsmil':
                    x, _, a = self.online_encoder(x, return_attn=True, no_norm=no_norm, **kwargs)
                else:
                    x, a = self.online_encoder(x, return_attn=True, no_norm=no_norm, **kwargs)
        else:
            x = self.online_encoder(x)

        if self.baseline == 'dsmil':
            pass
        else:    
            x = self.predictor(x)

        if return_attn:
            return x, a
        else:
            return x

    def pure(self, x):
        """
        Pure forward pass without masking or merging.
        
        Args:
            x: Input features
        
        Returns:
            Training: (x, 0, ps, ps)
            Inference: x
        """
        x = self.feature(x)
        x = self.dp(x)
        ps = x.size(1)

        if self.baseline == 'dsmil':
            x, _ = self.online_encoder(x)
        else:
            x = self.online_encoder(x)
            x = self.predictor(x)

        if self.training:
            return x, 0, ps, ps
        else:
            return x

    def forward_loss(self, student_cls_feat, teacher_cls_feat):
        """
        Compute contrastive loss between student and teacher.
        
        Args:
            student_cls_feat: Student features
            teacher_cls_feat: Teacher features
        
        Returns:
            cls_loss: Classification loss
        """
        if teacher_cls_feat is not None:
            cls_loss = self.cl_loss(student_cls_feat, teacher_cls_feat.detach())
        else:
            cls_loss = 0.
        
        return cls_loss

    def forward(self, x, attn=None, teacher_cls_feat=None, i=None, pos=None):
        """
        Forward pass with masking and merging.
        
        Args:
            x: Input features
            attn: Attention scores from teacher
            teacher_cls_feat: Teacher features for distillation
            i: Current iteration
            pos: Position information (unused)
        
        Returns:
            student_logit: Student predictions
            cls_loss: Contrastive loss
            ps: Original number of patches
            len_keep: Number of patches after masking
        """
        x = self.feature(x)
        x = self.dp(x)

        ps = x.size(1)

        # Get mask
        len_keep, mask_ids = self.get_mask(ps, i, attn)
        x = mask_fn(x, mask_ids, len_keep)
        ids_keep = mask_ids[:, :len_keep]
        if len(attn.size()) > 2:
            attn = torch.gather(
                attn, dim=-1, index=ids_keep.unsqueeze(1).repeat(1, self.head, 1)
            )
        else:
            attn = torch.gather(attn, dim=-1, index=ids_keep)

        x = self.merge(x, attn)

        len_keep = x.size(1)

        if self.baseline == 'dsmil':
            # Forward online network
            student_logit, student_cls_feat = self.online_encoder(x)

            # Contrastive loss
            cls_loss = self.forward_loss(
                student_cls_feat=student_cls_feat, teacher_cls_feat=teacher_cls_feat
            )

            return student_logit, cls_loss, ps, len_keep
        
        else:
            # Forward online network
            student_cls_feat = self.online_encoder(x)

            # Prediction
            student_logit = self.predictor(student_cls_feat)

            # Contrastive loss
            cls_loss = self.forward_loss(
                student_cls_feat=student_cls_feat, teacher_cls_feat=teacher_cls_feat
            )

            return student_logit, cls_loss, ps, len_keep
