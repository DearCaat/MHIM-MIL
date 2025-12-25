import torch
from copy import deepcopy

from .abmil import DAttention,AttentionGated
from .clam import CLAM_MB,CLAM_SB
from .mhim import MHIM
from .dsmil import MILNet
from .transmil import TransMIL
from .mean_max import MeanMIL,MaxMIL
from .dtfd import DTFD
from .rrt import RRTMIL
from .abmil_ibmil import Dattention_ori
from .mambamil_2d import MambaMIL_2D
from .utils import get_mil_model_params

import os 
from utils import cosine_scheduler

def load_mil_ckp(args,mil,mil_init_path):
    mil_ckp = torch.load(mil_init_path,weights_only=True)
    mil_ckp= mil_ckp['model']
    new_state_dict = {}
    for key in mil_ckp:
        # 将 'classifier.0.weight' 改为 'classifier.weight'
        if 'classifier.0.' in key:
            new_key = key.replace('classifier.0.', 'classifier.')
        elif '_fc1.' in key:
            new_key = key.replace('_fc1.', 'feature.')
        elif 'patch_to_embed.' in key:
            new_key = key.replace('patch_to_embed.', 'feature.')
        elif 'feature0.' in key:
            new_key = key.replace('feature0.', 'feature.')
        elif '_fc2.' in key:
            new_key = key.replace('_fc2.', 'classifier.')
        else:
            new_key = key
        new_state_dict[new_key] = mil_ckp[key]
    mil_ckp = new_state_dict

    if args.mil_init_type == 'main_fc':
        new_state_dict = {}
        if hasattr(mil,'feature'):
            for _k,v in mil_ckp.items():
                _k = _k.replace('feature.','') if 'feature' in _k else _k
                new_state_dict[_k]=v
            info = mil.feature.load_state_dict(new_state_dict,strict=False)
    else:
        info = mil.load_state_dict(mil_ckp,strict=False)

    if args.rank == 0:
        print(f"MIL Loading: {mil_init_path}")
        print(f"Results: {info}")
    
    return mil

def build_model(args,device,train_loader=None):
    return build_mil(args,args.model,device,train_loader)

def build_mil(args,model_name,device,train_loader):
    others = {}

    if args.teacher_init is not None:
        if not args.teacher_init.endswith('.pt'):
            _str = 'fold_{fold}_model_best.pt'.format(fold=args.fold_curr)
            _teacher_init = os.path.join(args.teacher_init,_str)
        else:
            _teacher_init = args.teacher_init

    genera_model_params,genera_trans_params = get_mil_model_params(args)

    if model_name == 'mhim':
        if args.mrh_sche:
            mrh_sche = cosine_scheduler(args.mask_ratio_h,0.,epochs=args.num_epoch,niter_per_ep=len(train_loader))
        else:
            mrh_sche = None

        model_params = {
            'baseline': args.baseline,
            'dropout': args.dropout,
            'input_dim':args.input_dim,
            'mask_ratio' : args.mask_ratio,
            'n_classes': args.n_classes,
            'temp_t': args.temp_t,
            'act': args.act,
            'head': args.n_heads,
            'mask_ratio_h': args.mask_ratio_h,
            'mask_ratio_hr': args.mask_ratio_hr,
            'mask_ratio_l': args.mask_ratio_l,
            'mrh_sche': mrh_sche,
            'da_act': args.da_act,
            'attn_layer': args.attn_layer,
            'attn2score': args.attn2score,
            'merge_enable': args.merge_enable,
            'merge_k': args.merge_k,
            'merge_mm': args.merge_mm,
            'merge_ratio': args.merge_ratio,
            'merge_test':args.merge_test,
        }
        
        model = MHIM(**model_params).to(device)
    elif model_name == 'mhim_pure':
        model = MHIM(input_dim=args.input_dim,select_mask=False,n_classes=args.n_classes,act=args.act,head=args.n_heads,da_act=args.da_act,baseline=args.baseline,merge_enable=False).to(device)
    elif model_name == 'rrtmil':
        model = RRTMIL(input_dim=args.input_dim,n_classes=args.n_classes,epeg_k=args.epeg_k,crmsa_k=args.crmsa_k,region_num=args.region_num,n_heads=args.rrt_n_heads,n_layers=args.rrt_n_layers,mil_norm=args.mil_norm,embed_norm_pos=args.embed_norm_pos).to(device)
    elif model_name == 'abmil':
        model = DAttention(**genera_model_params).to(device)
    elif model_name == 'gabmil':
        model = AttentionGated(**genera_model_params).to(device)
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif model_name == 'clam_sb':
        model = CLAM_SB(**genera_model_params).to(device)
    elif model_name == 'clam_mb':
        model = CLAM_MB(**genera_model_params).to(device)
    elif model_name == 'transmil':
        model = TransMIL(**genera_trans_params).to(device)
    elif model_name == 'dsmil':
        model = MILNet(**genera_model_params).to(device)
        #state_dict_weights = torch.load('./modules/init_cpk/dsmil_init.pth')
        #info = model.load_state_dict(state_dict_weights, strict=False)
        # print(info)
    elif model_name == 'dtfd':
        model = DTFD(device=device, lr=args.lr, weight_decay=args.weight_decay, steps=args.num_epoch, input_dim=args.input_dim, n_classes=args.n_classes).to(device)
    elif model_name == 'meanmil':
        model = MeanMIL(**genera_model_params).to(device)
    elif model_name == 'maxmil':
        model = MaxMIL(**genera_model_params).to(device)
    elif model_name == 'ibmil':
        if not args.confounder_path.endswith('.npy'):
            _confounder_path = os.path.join(args.confounder_path,str(args.fold_curr),'train_bag_cls_agnostic_feats_proto_'+str(args.confounder_k)+'.npy')
        else:
            _confounder_path =args.confounder_path
        model = Dattention_ori(confounder_path=_confounder_path,**genera_model_params).to(device)

    elif model_name == '2dmamba':
        model_params = {
            'in_dim': args.input_dim,
            'n_classes': args.n_classes,
            'mambamil_dim': args.mambamil_dim,
            'mambamil_layer': args.mambamil_layer,
            'mambamil_state_dim': args.mambamil_state_dim,
            'pscan': args.pscan,
            'cuda_pscan': args.cuda_pscan,
            'mamba_2d_max_w': args.mamba_2d_max_w,
            'mamba_2d_max_h': args.mamba_2d_max_h,
            'mamba_2d_pad_token': args.mamba_2d_pad_token,
            'mamba_2d_patch_size': args.mamba_2d_patch_size,
            'pos_emb_type': args.mamba_2d_pos_emb_type,
            'drop_out': args.dropout,
            'pos_emb_dropout': args.pos_emb_dropout,
        }
        model = MambaMIL_2D(**model_params).to(device)
    else:
        raise NotImplementedError
    
    #### Student Init  ####
    if args.init_stu_type != 'none':
        print('######### Model Initializing.....')
        pre_dict = torch.load(_teacher_init,weights_only=True)
        if 'model' in pre_dict:
            pre_dict = pre_dict['model']
        new_state_dict ={}
        if args.init_stu_type == 'fc':
        # only feature
            for _k,v in pre_dict.items():
                _k = _k.replace('feature.','') if 'feature' in _k else _k
                new_state_dict[_k]=v
            info = model.feature.load_state_dict(new_state_dict,strict=False)
        else:
        # init all
            info = model.load_state_dict(pre_dict,strict=False)
        print(info)

    #### MHIM Init  ####
    # teacher model
    if model_name == 'mhim':
        if args.mm_sche:
            mm_sche = cosine_scheduler(args.mm,1.,epochs=args.num_epoch,niter_per_ep=len(train_loader),start_warmup_value=1.)
        else:
            mm_sche = None
        others['mm_sche'] = mm_sche
        model_tea = deepcopy(model)
        if not args.no_tea_init and args.tea_type != 'same':
            print('######### Teacher Initializing.....')
            try:
                pre_dict = torch.load(_teacher_init,weights_only=True)
                if 'model' in pre_dict:
                    pre_dict = pre_dict['model']
                
                model_keys = set(model_tea.state_dict().keys())
                pretrain_keys = set(pre_dict.keys())
                
                model_has_module = any(k.startswith('module.') for k in model_keys)
                pretrain_has_module = any(k.startswith('module.') for k in pretrain_keys)
                
                if model_has_module and not pretrain_has_module:
                    new_state_dict = {}
                    for k, v in pre_dict.items():
                        new_state_dict['module.' + k] = v
                    pre_dict = new_state_dict
                elif not model_has_module and pretrain_has_module:
                    new_state_dict = {}
                    for k, v in pre_dict.items():
                        new_state_dict[k.replace('module.', '')] = v
                    pre_dict = new_state_dict
                
                info = model_tea.load_state_dict(pre_dict,strict=False)
                print(info)
            except Exception as e:
                print(f'########## Init Error: {e}')
        if args.tea_type == 'same':
            model_tea = model
        model_tea.merge_test = False
        others['model_ema'] = model_tea
    else:
        model_tea = None
    
    return model, others

