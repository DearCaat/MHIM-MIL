import time
import torch
import wandb
import numpy as np
from copy import deepcopy
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader,Sampler, WeightedRandomSampler, RandomSampler
import sys, argparse, os, copy, itertools
import torchvision.transforms.functional as VF
from optimizer.lookahead import Lookahead
from optimizer.radam import RAdam
from modules import attmil,clam,contrastive1_zxx,dsmil,transmil,mean_max,swin
from tqdm import tqdm
# import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from torch.cuda.amp import GradScaler
from contextlib import suppress
import time
from timm.utils import AverageMeter,dispatch_clip_grad
try:
    from timm.utils import trainer
except:
    pass
from timm.models import  model_parameters
from collections import OrderedDict

from utils import *

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # set seed
    def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    

    seed_torch(args.seed)

    loss_scaler = GradScaler() if args.amp else None
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    # --->划分数据集
    if args.datasets.lower() == 'camelyon16':
        label_path=os.path.join(args.dataset_root,'label.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index]
        # 官方划分
        if args.c16_offi:
            train_p,train_l,test_p,test_l,val_p,val_l = [],[],[],[],[],[]
            for i in range(len(p)):
                if 'test' in p[i]:
                    test_p.extend([p[i]])
                    test_l.extend([l[i]])
                else:
                    train_p.extend([p[i]])
                    train_l.extend([l[i]])

            train_p,train_l,test_p,test_l,val_p,val_l = np.array(train_p),np.array(train_l),np.array(test_p),np.array(test_l),np.array(val_p),np.array(val_l)

            if args.val_ratio > 0:
                val_index,train_index = data_split(list(range(len(train_p))),args.val_ratio,args.val_label_balance)
                train_p,train_l,val_p,val_l = train_p[train_index],train_l[train_index],train_p[val_index],train_l[val_index]

            train_p,train_l,test_p,test_l,val_p,val_l = [train_p for i in range(args.cv_fold) ],[train_l for i in range(args.cv_fold)],[test_p for i in range(args.cv_fold)],[test_l for i in range(args.cv_fold)],[val_p for i in range(args.cv_fold)],[val_l for i in range(args.cv_fold)]
            # print(train_p)

    elif args.datasets.lower() == 'tcga':
        label_path=os.path.join(args.dataset_root,'label.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index]

    # 当cv_fold == 1时，不使用交叉验证
    if args.cv_fold > 1 and not args.c16_offi:
        train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(args.cv_fold, p, l,args.val_ratio,args.val_label_balance)

    acs, pre, rec,fs,auc,te_auc=[],[],[],[],[],[]
    ckc_metric = [acs, pre, rec,fs,auc,te_auc]

    print('Dataset: ' + args.datasets)

    # resume
    if args.auto_resume:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        args.fold_start = ckp['k']
        acs, pre, rec,fs,auc,te_auc = ckp['ckc_metric']
        ckc_metric = [acs, pre, rec,fs,auc,te_auc]

    for k in range(args.fold_start, args.cv_fold):
        print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
        ckc_metric = one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l)

    if args.always_test:
        if args.wandb:
            wandb.log({
                "cross_val/te_auc_mean":np.mean(np.array(te_auc)),
                "cross_val/te_auc_std":np.std(np.array(te_auc)),

            })

    if args.wandb:
        wandb.log({
            "cross_val/acc_mean":np.mean(np.array(acs)),
            "cross_val/auc_mean":np.mean(np.array(auc)),
            "cross_val/f1_mean":np.mean(np.array(fs)),
            "cross_val/pre_mean":np.mean(np.array(pre)),
            "cross_val/recall_mean":np.mean(np.array(rec)),
            "cross_val/acc_std":np.std(np.array(acs)),
            "cross_val/auc_std":np.std(np.array(auc)),
            "cross_val/f1_std":np.std(np.array(fs)),
            "cross_val/pre_std":np.std(np.array(pre)),
            "cross_val/recall_std":np.std(np.array(rec)),
        })

    print('Cross validation accuracy mean: %.3f, std %.3f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
    print('Cross validation auc mean: %.3f, std %.3f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
    print('Cross validation precision mean: %.3f, std %.3f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
    print('Cross validation recall mean: %.3f, std %.3f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
    print('Cross validation fscore mean: %.3f, std %.3f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))

def one_fold(args,k,ckc_metric,train_p,train_l,test_p,test_l,val_p,val_l):
    # --->初始化
    seed_torch(args.seed)
    loss_scaler = GradScaler() if args.amp else None
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    acs,pre,rec,fs,auc,te_auc = ckc_metric

    # --->加载数据
    if args.datasets.lower() == 'camelyon16':

        train_set = C16Dataset(train_p[k],train_l[k],root=args.dataset_root,persistence=args.persistence)
        test_set = C16Dataset(test_p[k],test_l[k],root=args.dataset_root,persistence=args.persistence)
        if args.val_ratio != 0.:
            val_set = C16Dataset(val_p[k],val_l[k],root=args.dataset_root,persistence=args.persistence)
        else:
            val_set = test_set
        
    elif args.datasets.lower() == 'tcga':
        
        train_set = TCGADataset(train_p[k],train_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence)
        test_set = TCGADataset(test_p[k],test_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence)
        if args.val_ratio != 0.:
            val_set = TCGADataset(val_p[k],val_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence)
        else:
            val_set = test_set

    
    if args.weighted:
    # weighted 采样会使得每一轮得训练的样本都有所区别，这里的权重是样本类别的数目。
    # ref from clam@harvard
        if args.fix_loader_random:
            # 样本太少，shuffle会影响模型的收敛，主要是不同模型的shuffle不同
            # 该数由int(torch.empty((), dtype=torch.int64).random_().item())生成
            big_seed_list = 7784414403328510413
            generator = torch.Generator()
            generator.manual_seed(big_seed_list)
            weights = make_weights_for_balanced_classes_split(train_set)
            train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=WeightedRandomSampler(weights,len(weights),generator=generator), num_workers=args.num_workers)
        else:
            weights = make_weights_for_balanced_classes_split(train_set)
            train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=WeightedRandomSampler(weights,len(weights)), num_workers=args.num_workers)
    else:
        if args.fix_loader_random:
            # 样本太少，shuffle会影响模型的收敛，主要是不同模型的shuffle不同
            # 该数由int(torch.empty((), dtype=torch.int64).random_().item())生成
            big_seed_list = 7784414403328510413
            generator = torch.Generator()
            generator.manual_seed(big_seed_list)  
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,generator=generator)
        else:
            train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=RandomSampler(train_set), num_workers=args.num_workers)
            #train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    mm_sche = None

    if not args.teacher_init.endswith('.pt'):
        _str = 'fold_{fold}_model_best_auc.pt'.format(fold=k)
        _teacher_init = os.path.join(args.teacher_init,_str)
    else:
        _teacher_init =args.teacher_init
    # _str = 'fold_{fold}_model_best_auc.pt'.format(fold=k)
    # _teacher_init = os.path.join(args.teacher_init,_str)

    # bulid networks
    if args.model == 'mhim':
        
        if args.old:
            model = contrastive1_old.CLR(mask_ratio=args.mask_ratio,teacher_init=_teacher_init,n_classes=args.n_classes,mlp_decoder=args.mlp_decoder,mfm_loss=args.mfm_loss,temp_t=args.temp_t,temp_s=args.temp_s,cl_loss=args.cl_loss,cl_enable=args.cl_alpha!=0.,mfm_enable=args.mfm_alpha!=0.,zero_mask=args.zero_mask,cl_type=args.cl_type,cl_out_dim=args.cl_out_dim,dropout=args.dropout,no_tea_mask=args.no_tea_mask,n_robust=args.n_robust).to(device)
        else:
            # 
            if args.mrh_sche:
                mrh_sche = cosine_scheduler(args.mask_ratio_h,0.,epochs=args.num_epoch,niter_per_ep=len(train_loader))
            else:
                mrh_sche = None

            model_params = {
                'dropout': args.dropout,
                'mask_ratio' : args.mask_ratio,
                'n_classes': args.n_classes,
                'zero_mask': args.zero_mask,
                'mae_init': args.mae_init,
                'pos_pos': args.pos_pos,
                'mfm_loss': args.mfm_loss,
                'cl_loss': args.cl_loss,
                'cl_enable': args.cl_enable,
                'cl_out_dim': args.cl_out_dim,
                'cl_type': args.cl_type,
                'cl_pred_head': args.cl_pred_head,
                'cl_targ_no_pred': args.cl_targ_no_pred,
                'mfm_enable': args.mfm_alpha!=0. if not args.mfm_enable else args.mfm_enable,
                'teacher_init' : _teacher_init,
                'temp_t': args.temp_t,
                'temp_s': args.temp_s,
                'no_tea_mask': args.no_tea_mask,
                'drop_in_encoder':args.drop_in_encoder,
                'mfm_decoder_embed_dim': args.mfm_decoder_embed_dim,
                'mfm_decoder_out_dim': args.mfm_decoder_out_dim,
                'mfm_norm_targ_loss':  args.mfm_norm_targ_loss,
                'mfm_norm_pred_loss':  args.mfm_norm_pred_loss,
                'mlp_norm_last_layer': args.mlp_norm_last_layer,
                'mlp_nlayers': args.mlp_nlayers,
                'mlp_hidden_dim': args.mlp_hidden_dim,
                'mlp_bottleneck_dim': args.mlp_bottleneck_dim,
                'n_robust': args.n_robust,
                'mfm_decoder': args.mfm_decoder,
                'mfm_targ_ori': args.mfm_targ_ori,
                'mfm_temp_t': args.mfm_temp_t,
                'mfm_temp_s': args.mfm_temp_s,
                'pos': args.pos,
                'mfm_no_targ_grad':args.mfm_no_targ_grad,
                'act': args.act,
                'mfm_feats': args.mfm_feats,
                'multi_add_pos': args.multi_add_pos,
                'multi_type': args.multi_type,
                'attn': args.attn,
                'pool': args.pool,
                'region_num': args.region_num,
                'head': args.n_heads,
                'select_inv': args.select_inv,
                'select_mask': args.select_mask,
                'patch_shuffle': args.inner_patch_shuffle,
                'shuffle_group': args.shuffle_group,
                'msa_fusion': args.msa_fusion,
                'mask_ratio_h': args.mask_ratio_h,
                'mask_ratio_hr': args.mask_ratio_hr,
                'mask_ratio_l': args.mask_ratio_l,
                'mrh_sche': mrh_sche,
                'mrh_type': args.mrh_type,
                'attn_layer': args.attn_layer,
            }
            model = contrastive1_zxx.CLR(**model_params).to(device)
            
            if args.mm_sche:
                #mm_sche = cosine_scheduler(args.mm,1.0,epochs=args.num_epoch-args.mm_update_epoch,niter_per_ep=len(train_loader))
                mm_sche = cosine_scheduler(args.mm,args.mm_final,epochs=args.num_epoch-args.mm_update_epoch,niter_per_ep=len(train_loader),warmup_epochs=args.mm_warmup_epoch,start_warmup_value=1.)
            
    elif args.model == 'pure':
        if not args.zero_mask:
            args.mask_ratio=0. 

        if args.old:
            model = contrastive1_old.CLR(mask_ratio=args.mask_ratio,teacher_init=args.teacher_init,n_classes=args.n_classes,zero_mask=args.zero_mask,test=args.n_robust).to(device)
        else:
            model = contrastive1_zxx.CLR(mask_ratio=args.mask_ratio,n_classes=args.n_classes,zero_mask=args.zero_mask,cl_enable=False,mfm_enable=False,mae_init=args.mae_init,drop_in_encoder=args.drop_in_encoder,pos_pos=args.pos_pos,n_robust=args.n_robust,pos=args.pos,act=args.act,attn=args.attn,pool=args.pool,region_num=args.region_num,head=args.n_heads).to(device)
    elif args.model == 'attmil':
        model = attmil.DAttention(n_classes=args.n_classes,dropout=args.dropout,act=args.act,test=args.n_robust).to(device)
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif args.model == 'clam_sb':
        model = clam.CLAM_SB(n_classes=args.n_classes,dropout=args.dropout,test=args.n_robust,act=args.act).to(device)
    elif args.model == 'clam_mb':
        model = clam.CLAM_MB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'transmil':
        model = transmil.TransMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'dsmil':
        model = dsmil.MILNet(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5
        state_dict_weights = torch.load('./modules/init_cpk/dsmil_init.pth')
        info = model.load_state_dict(state_dict_weights, strict=False)
        print(info)
    elif args.model == 'rnnmil':
        model = rnnmil.rnn_single(n_classes=args.n_classes).to(device)
    elif args.model == 'meanmil':
        model = mean_max.MeanMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'maxmil':
        model = mean_max.MaxMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'swin':
        model = swin.Swin(n_classes=args.n_classes,dropout=args.dropout,test=args.n_robust,act=args.act,region_num=args.region_num,multi_scale=not args.swin_no_multi_scale, drop_path=args.drop_path,n_layers=args.swin_n_layers,n_heads=args.swin_n_heads,attn=args.attn).to(device)

    if args.init_stu_type != 'none':
        print('######### Model Initing.....')
        # pre_dict = torch.load(_teacher_init)
        pre_dict = torch.load(_teacher_init)
        new_state_dict ={}
        if args.init_stu_type == 'fc':
        # only patch_to_emb
            for _k,v in pre_dict.items():
                _k = _k.replace('patch_to_emb.','') if 'patch_to_emb' in _k else _k
                new_state_dict[_k]=v
            info = model.patch_to_emb.load_state_dict(new_state_dict,strict=False)
        else:
        # init all
            info = model.load_state_dict(pre_dict,strict=False)
        print(info)

    # teacher model
    if args.model_ema:
        model_tea = deepcopy(model)
        if not args.no_tea_init and args.tea_type != 'same':
            print('######### Teacher Initing.....')
            try:
                pre_dict = torch.load(_teacher_init)
                info = model_tea.load_state_dict(pre_dict,strict=False)
                print(info)
            except:
                # print(info)
                print("###### no teacher init......")
        if args.tea_type == 'inner':
            model_tea.patch_to_emb = model.patch_to_emb
            model_tea.predictor = model.predictor
        elif args.tea_type == 'same':
            model_tea = model
    else:
        model_tea = None

    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()

    # optimizer
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_sche == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch*len(train_loader), 0)
    elif args.lr_sche == 'step':
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=30 if args.datasets=='camelyon16' else 20, stop_epoch=args.max_epoch if args.datasets=='camelyon16' else 70,save_best_model_stage=int(args.save_best_model_stage * args.num_epoch))
    else:
        early_stopping = None

    optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch = 0, 0, 0, 0,0,0
    opt_te_auc,opt_tea_auc = 0., 0.
    opt_auc_mask,opt_auc_mean = 0.,0.
    epoch_start = 0

    if args.fix_train_random:
        seed_torch(args.seed)

    # resume
    if args.auto_resume:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        epoch_start = ckp['epoch']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['lr_sche'])
        early_stopping.load_state_dict(ckp['early_stop'])
        optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch = ckp['val_best_metric']
        opt_te_auc = ckp['te_best_metric'][0]
        np.random.set_state(ckp['random']['np'])
        torch.random.set_rng_state(ckp['random']['torch'])
        random.setstate(ckp['random']['py'])
        if args.fix_loader_random:
            train_loader.sampler.generator.set_state(ckp['random']['loader'])
        args.auto_resume = False

    # wandb.watch(model, log_freq=100)
    train_time_meter = AverageMeter()
    for epoch in range(epoch_start, args.num_epoch):
        if args.vit:
            trainer.step()
        train_loss,start,end = train_loop(args,model,model_tea,train_loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch)
        stop,accuracy, auc_value, precision, recall, fscore, test_loss = val_loop(args,model,val_loader,device,criterion,early_stopping,epoch,model_tea)
        train_time_meter.update(end-start)
        # test
        ######################
        if args.model == 'mhim' and args.val_type != 'ori':
            model.test_type = args.val_type
            _,accuracy_mean, auc_value_mean, precision_mean, recall_mean, fscore_mean, test_loss_mean = val_loop(args,model,val_loader,device,criterion,None,epoch,model_tea)
            if args.wandb:
                wandb.log({
                    "val_auc_mean":auc_value_mean,
                })
            if auc_value_mean > opt_auc_mean:
                opt_auc_mean = auc_value_mean
                if args.wandb:
                    wandb.log({
                        "best_mean_auc":opt_auc_mean,
                    })
            model.test_type = 'ori'
        ######################

        if model_tea is not None:
            _,accuracy_tea, auc_value_tea, precision_tea, recall_tea, fscore_tea, test_loss_tea = val_loop(args,model_tea,val_loader,device,criterion,None,epoch,model_tea)
            if args.wandb:
                rowd = OrderedDict([
                    ("val_acc_tea",accuracy_tea),
                    ("val_precision_tea",precision_tea),
                    ("val_recall_tea",recall_tea),
                    ("val_fscore_tea",fscore_tea),
                    ("val_auc_tea",auc_value_tea),
                    ("val_loss_tea",test_loss_tea),
                ])

                rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                wandb.log(rowd)

            if auc_value_tea > opt_tea_auc:
                opt_tea_auc = auc_value_tea
                if args.wandb:
                    rowd = OrderedDict([
                        ("best_tea_auc",opt_tea_auc)
                    ])
                    rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd)

        if args.always_test:

            _te_accuracy, _te_auc_value, _te_precision, _te_recall, _te_fscore,_te_test_loss_log = test(args,model,test_loader,device,criterion,model_tea)
            
            if args.wandb:
                rowd = OrderedDict([
                    ("te_acc",_te_accuracy),
                    ("te_precision",_te_precision),
                    ("te_recall",_te_recall),
                    ("te_fscore",_te_fscore),
                    ("te_auc",_te_auc_value),
                    ("te_loss",_te_test_loss_log),
                ])

                rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                wandb.log(rowd)

            if _te_auc_value > opt_te_auc:
                opt_te_auc = _te_auc_value
                if args.wandb:
                    rowd = OrderedDict([
                        ("best_te_auc",opt_te_auc)
                    ])
                    rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd)


        print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f , time: %.3f(%.3f)' % 
        (epoch+1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, train_time_meter.val,train_time_meter.avg))

        if args.wandb:
            rowd = OrderedDict([
                ("val_acc",accuracy),
                ("val_precision",precision),
                ("val_recall",recall),
                ("val_fscore",fscore),
                ("val_auc",auc_value),
                ("val_loss",test_loss),
                ("epoch",epoch),
            ])

            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            wandb.log(rowd)

        if auc_value > opt_auc and epoch >= args.save_best_model_stage*args.num_epoch:
            optimal_ac = accuracy
            opt_pre = precision
            opt_re = recall
            opt_fs = fscore
            opt_auc = auc_value
            opt_epoch = epoch

            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            if not args.no_log:
                best_pt = {
                    'model': model.state_dict(),
                    'teacher': model_tea.state_dict() if model_tea is not None else None,
                }
                torch.save(best_pt, os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
        if args.wandb:
            rowd = OrderedDict([
                ("val_best_acc",optimal_ac),
                ("val_best_precesion",opt_pre),
                ("val_best_recall",opt_re),
                ("val_best_fscore",opt_fs),
                ("val_best_auc",opt_auc),
                ("val_best_epoch",opt_epoch),
            ])

            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            wandb.log(rowd)
        
        # save checkpoint
        random_state = {
            'np': np.random.get_state(),
            'torch': torch.random.get_rng_state(),
            'py': random.getstate(),
            'loader': train_loader.sampler.generator.get_state() if args.fix_loader_random else '',
        }
        ckp = {
            'model': model.state_dict(),
            'lr_sche': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1,
            'k': k,
            'early_stop': early_stopping.state_dict(),
            'random': random_state,
            'ckc_metric': [acs,pre,rec,fs,auc,te_auc],
            'val_best_metric': [optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch],
            'te_best_metric': [opt_te_auc],
            'wandb_id': wandb.run.id if args.wandb else '',
        }
        if not args.no_log:
            torch.save(ckp, os.path.join(args.model_path, 'ckp.pt'))

        if stop:
            break
    
    # test
    best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
    info = model.load_state_dict(best_std['model'])
    print(info)
    if model_tea is not None and best_std['teacher'] is not None:
        info = model_tea.load_state_dict(best_std['teacher'])
        print(info)
    
    accuracy, auc_value, precision, recall, fscore,test_loss_log = test(args,model,test_loader,device,criterion,model_tea)
    
    if args.wandb:
        wandb.log({
            "test_acc":accuracy,
            "test_precesion":precision,
            "test_recall":recall,
            "test_fscore":fscore,
            "test_auc":auc_value,
            "test_loss":test_loss_log,
        })

    print('\n Optimal accuracy: %.3f ,Optimal auc: %.3f,Optimal precision: %.3f,Optimal recall: %.3f,Optimal fscore: %.3f' % (optimal_ac,opt_auc,opt_pre,opt_re,opt_fs))
    acs.append(accuracy)
    pre.append(precision)
    rec.append(recall)
    fs.append(fscore)
    auc.append(auc_value)

    if args.always_test:
        te_auc.append(opt_te_auc)
        
    # test
    ######################
    if args.model == 'mhim' and args.test_type != 'ori' and args.datasets == 'tcga':
        model.test_type = args.test_type
        accuracy_mean, auc_value_mean, precision_mean, recall_mean, fscore_mean, test_loss_mean = test(args,model,test_loader,device,criterion,model_tea)
        if args.wandb:
            wandb.log({
                "test_auc_mean":auc_value_mean,
            })
        model.test_type = 'ori'
    ######################

    if args.c16_offi:
        args.seed += 1

    return [acs,pre,rec,fs,auc,te_auc]

def train_loop(args,model,model_tea,loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch):
    start = time.time()
    loss_cls_meter = AverageMeter()
    loss_cl_meter = AverageMeter()
    loss_mfm_meter = AverageMeter()
    patch_num_meter = AverageMeter()
    keep_num_meter = AverageMeter()
    mm_meter = AverageMeter()
    train_loss_log = 0.
    model.train()
    if model_tea is not None:
        model_tea.train()
    # torch.autograd.set_detect_anomaly(True)
    for i, data in enumerate(loader):
        if args.vit:
            trainer.step()
        #torch.cuda.empty_cache()
        optimizer.zero_grad()
        bag=data[0].to(device)  # b*n*1024
        label=data[1].to(device)
        
        with amp_autocast():
            # if args.model != 'mhim':
            if args.patch_shuffle:
                bag = patch_shuffle(bag,args.shuffle_group)
            elif args.group_shuffle:
                bag = group_shuffle(bag,args.shuffle_group)

            if args.model == 'mhim':
                if args.tea_type == 'inner':
                    bag = model.patch_to_emb(bag)
                    bag = model.dp(bag)

                if model_tea is not None:
                    _,cls_tea,attn = model_tea.forward_teacher(bag,return_attn=True,no_fc=args.tea_type == 'inner')
                else:
                    attn,cls_tea = None,None
                if args.cl_alpha == 0.:
                    cls_tea = None
                train_logits, stu_loss, cls_loss,patch_num,keep_num = model(bag,attn,cls_tea,i=epoch*len(loader)+i,no_fc=args.tea_type == 'inner')

            elif args.model == 'pure':
                train_logits, stu_loss, cls_loss,patch_num,keep_num = model.test(bag)
            elif args.model in ('clam_sb','clam_mb','dsmil'):
                train_logits,cls_loss,patch_num = model(bag,label,criterion)
                stu_loss,keep_num = 0.,patch_num
            else:
                train_logits = model(bag)
                stu_loss,cls_loss,patch_num,keep_num = 0.,0.,0.,0.
        # print(x.shape,decoder_feat.shape)
            if args.loss == 'ce':
                logit_loss = criterion(train_logits.view(1,-1),label)
            elif args.loss == 'bce':
                logit_loss = criterion(train_logits.view(1,-1),label.view(1,-1).float())

        if args.cl_auto_alpha:
            args.cl_alpha = ((cls_loss / logit_loss) ** (-1)).detach()
        train_loss = args.cls_alpha * logit_loss +  cls_loss*args.cl_alpha + args.mfm_alpha*stu_loss
        # args.cl_alpha args.mfm_alpha
        #train_loss = train_loss / args.accumulation_steps

        # if args.amp:
        #     pass
        # else:
        #     train_loss.backward()
        
        # if args.clip_grad > 0.:
        #     dispatch_clip_grad(
        #         model_parameters(model),
        #         value=args.clip_grad, mode='norm')

        #if (i+1) % args.accumulation_steps == 0:
        if True:
            if args.amp:
                loss_scaler.scale(train_loss).backward()
                loss_scaler.step(optimizer)
                loss_scaler.update()
                # pass
            else:
                train_loss.backward()
                optimizer.step()
            if args.lr_supi and scheduler is not None:
                scheduler.step()
            if args.model == 'mhim':
                # if args.mm_head == 0.:
                #     mm_head = 0.
                # else:
                #     mm_head = mm_sche[epoch*len(loader)+i]
                if mm_sche is not None:
                    mm = mm_sche[epoch*len(loader)+i]
                else:
                    mm = args.mm
                # update target network
                model.update_target_network(mm,0.)
                if model_tea is not None and epoch >= args.mm_update_epoch:
                    if args.tea_type == 'inner':
                        ema_update(model.online_encoder,model_tea.online_encoder,mm)
                    elif args.tea_type == 'same':
                        pass
                    else:
                        ema_update(model,model_tea,mm)
            else:
                mm = 0.

        loss_cls_meter.update(logit_loss,1)
        loss_cl_meter.update(cls_loss,1)
        loss_mfm_meter.update(stu_loss,1)
        patch_num_meter.update(patch_num,1)
        keep_num_meter.update(keep_num,1)
        mm_meter.update(mm,1)

        if i % args.log_iter == 0 or i == len(loader)-1:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            rowd = OrderedDict([
                ('cls_loss',loss_cls_meter.avg),
                ('lr',lr),
                ('cl_loss',loss_cl_meter.avg),
                ('mfm_loss',loss_mfm_meter.avg),
                ('patch_num',patch_num_meter.avg),
                ('keep_num',keep_num_meter.avg),
                ('mm',mm_meter.avg),
            ])
            print('[{}/{}] logit_loss:{}, cls_loss:{}, st_loss:{}, patch_num:{}, keep_num:{} '.format(i,len(loader)-1,loss_cls_meter.avg,loss_cl_meter.avg,loss_mfm_meter.avg,patch_num_meter.avg, keep_num_meter.avg))
            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            if args.wandb:
                wandb.log(rowd)

        train_loss_log = train_loss_log + train_loss.item()

    end = time.time()
    train_loss_log = train_loss_log/len(loader)
    if not args.lr_supi and scheduler is not None:
        scheduler.step()
    
    return train_loss_log,start,end

def val_loop(args,model,loader,device,criterion,early_stopping,epoch,model_tea=None):
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    test_loss_log = 0.
    bag_logit, bag_labels=[], []
    # pred= []
    with torch.no_grad():
        for i, data in enumerate(loader):
            bag_labels.append(data[1].item())
            # if args.datasets == 'tcga':
            #     bag = torch.load(data[0][0], map_location = lambda storage, loc: storage.cuda(0))
            #     bag.unsqueeze_(0)
            # else:
            bag=data[0].to(device)  # b*n*1024
            label=data[1].to(device)

            if args.model in ('mhim','pure'):
                if model_tea is not None:
                    _,cls_tea,attn = model_tea.forward_teacher(bag,return_attn=True,no_fc=args.tea_type == 'inner')
                else:
                    attn,cls_tea = None,None
                test_logits = model.forward_test(bag,attn)
            elif args.model == 'dsmil':
                test_logits = model(bag)
            else:
                test_logits = model(bag)

            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))):
                    test_loss = criterion(test_logits[0].view(1,-1),label)
                    bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(1,-1),label)
                    bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits.view(1,-1),label)
                    bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits[0].view(1,-1),label.view(1,-1).float())
                bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            test_loss_log = test_loss_log + test_loss.item()
    
    # 保存日志文件
    accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit)
    test_loss_log = test_loss_log/len(loader)
    
    # earningly stop
    if early_stopping is not None:
        early_stopping(epoch,-auc_value,model)
        stop = early_stopping.early_stop
    else:
        stop = False
    return stop,accuracy, auc_value, precision, recall, fscore,test_loss_log

def test(args,model,loader,device,criterion,model_tea=None):
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    test_loss_log = 0.
    bag_logit, bag_labels=[], []
    # pred= []
    with torch.no_grad():
        for i, data in enumerate(loader):
            bag_labels.append(data[1].item())
            # if args.datasets == 'tcga':
            #     bag = torch.load(data[0][0], map_location = lambda storage, loc: storage.cuda(0))
            #     bag.unsqueeze_(0)
            # else:
            bag=data[0].to(device)  # b*n*1024
            label=data[1].to(device)
            if args.model in ('mhim','pure'):
                #test_logits = model.test(bag)
                if model_tea is not None:
                    _,cls_tea,attn = model_tea.forward_teacher(bag,return_attn=True,no_fc=args.tea_type == 'inner')
                else:
                    attn,cls_tea = None,None
                test_logits = model.forward_test(bag,attn)
            elif args.model == 'dsmil':
                test_logits = model(bag)
            else:
                test_logits = model(bag)

            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))):
                    test_loss = criterion(test_logits[0].view(1,-1),label)
                    bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(1,-1),label)
                    bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits[0].view(1,-1),label)
                    bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits.view(1,-1),label.view(1,-1).float())
                bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            test_loss_log = test_loss_log + test_loss.item()
    
    # 保存日志文件
    accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit)
    test_loss_log = test_loss_log/len(loader)

    return accuracy, auc_value, precision, recall, fscore,test_loss_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL Training Script')

    # dataset 
    parser.add_argument('--datasets', default='camelyon16', type=str, help='[camelyon16,tcga]')
    parser.add_argument('--dataset_root', default='/data/xxx/TransMIL', type=str, help='dataset root path')
    parser.add_argument('--tcga_max_patch', default=-1, type=int, help='Number of total training epochs [40]')
    parser.add_argument('--tcga_mini', action='store_true', help='Number of total training epochs [40]')
    parser.add_argument('--fix_loader_random', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--fix_train_random', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--no_fix_c16', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--val_ratio', default=0., type=float, help='Automatic Mixed Precision Training')
    parser.add_argument('--fold_start', default=0, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--val_label_balance', action='store_false', help='Automatic Mixed Precision Training')
    parser.add_argument('--weighted', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--persistence', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--c16_offi', action='store_true', help='Automatic Mixed Precision Training')

    # train
    parser.add_argument('--auto_resume', action='store_true', help='Camelyon16')
    parser.add_argument('--num_epoch', default=200, type=int, help='Number of total training epochs [40]')
    parser.add_argument('--max_epoch', default=130, type=int, help='Number of total training epochs [40]')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of total training epochs [40]')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--loss', default='ce', type=str, help='output log filepath')
    parser.add_argument('--opt', default='adam', type=str, help='output log filepath')
    parser.add_argument('--save_best_model_stage', default=0.1, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--model', default='mhim', type=str, help='Camelyon16')
    parser.add_argument('--seed', default=2021, type=int, help='random number [7]')
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lr_sche', default='cosine', type=str, help='Camelyon16')
    parser.add_argument('--lr_supi', action='store_true', help='lr_scheduler_update_per_iter')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Number of cross validation fold [2]')
    parser.add_argument('--clip_grad', default=.0, type=float, help='Number of cross validation fold [5]')
    parser.add_argument('--model_ema', action='store_true', help='lr_scheduler_update_per_iter')
    parser.add_argument('--ema_decay', default=0.9999, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--always_test', action='store_true', help='lr_scheduler_update_per_iter')
    parser.add_argument('--test_type', default='ori', type=str, help='lr_scheduler_update_per_iter')
    parser.add_argument('--val_type', default='ori', type=str, help='lr_scheduler_update_per_iter')

    # model
    parser.add_argument('--ds_average', action='store_true', help='lr_scheduler_update_per_iter')
    parser.add_argument('--n_heads', default=8, type=int, help='cl loss alpha')
    parser.add_argument('--act', default='relu', type=str, help='[gelu,relu]')
    parser.add_argument('--old', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--n_robust', default=0, type=int, help='lr_scheduler_update_per_iter')
    parser.add_argument('--dropout', default=0.25, type=float, help='lr_scheduler_update_per_iter')
    parser.add_argument('--cls_alpha', default=1.0, type=float, help='cl loss alpha')
    parser.add_argument('--mae_init', action='store_true', help='lr_scheduler_update_per_iter')
    parser.add_argument('--pos_pos', default=0, type=int, help='position of pos embed [-1,0]')
    parser.add_argument('--pos', default='ppeg', type=str, help='Camelyon16')
    parser.add_argument('--peg_k', default=7, type=int, help='Camelyon16')
    parser.add_argument('--attn', default='ntrans', type=str, help='Camelyon16')
    parser.add_argument('--pool', default='cls_token', type=str, help='Camelyon16')
    parser.add_argument('--region_num', default=8, type=int, help='position of pos embed [-1,0]')
    # shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='lr_scheduler_update_per_iter')
    parser.add_argument('--inner_patch_shuffle', action='store_true', help='lr_scheduler_update_per_iter')
    parser.add_argument('--group_shuffle', action='store_true', help='lr_scheduler_update_per_iter')
    parser.add_argument('--shuffle_group', default=0, type=int, help='position of pos embed [-1,0]')

    # mask
    parser.add_argument('--mask_ratio', default=0., type=float, help='mask ratio')
    parser.add_argument('--mask_ratio_l', default=0., type=float, help='mask ratio')
    parser.add_argument('--mask_ratio_h', default=0., type=float, help='mask ratio')
    parser.add_argument('--mask_ratio_hr', default=1., type=float, help='mask ratio')
    
    parser.add_argument('--zero_mask', action='store_true', help='lr_scheduler_update_per_iter')
    parser.add_argument('--drop_in_encoder', action='store_false', help='lr_scheduler_update_per_iter')
    # select mask
    parser.add_argument('--select_mask', action='store_true',  help='cl loss alpha')
    parser.add_argument('--select_inv', action='store_true', help='lr_scheduler_update_per_iter')
    parser.add_argument('--msa_fusion', default='vote', type=str, help='[mean,vote]')
    parser.add_argument('--mrh_sche', action='store_true', help='cl loss alpha')
    parser.add_argument('--mrh_type', default='tea', type=str, help='[mean,vote]')
    parser.add_argument('--attn_layer', default=1, type=int, help='cl loss alpha')

    # swin
    parser.add_argument('--drop_path', default=0.1, type=float, help='lr_scheduler_update_per_iter')
    parser.add_argument('--swin_n_layers', default=2, type=int, help='cl loss alpha')
    parser.add_argument('--swin_n_heads', default=8, type=int, help='cl loss alpha')
    parser.add_argument('--swin_no_multi_scale', action='store_true', help='mask ratio')
    
    # cl 
    parser.add_argument('--cl_alpha', default=0., type=float, help='cl loss alpha')
    parser.add_argument('--cl_enable', action='store_true', help='cl loss alpha')
    parser.add_argument('--cl_auto_alpha', action='store_true', help='lr_scheduler_update_per_iter')
    parser.add_argument('--no_tea_mask', action='store_false', help='lr_scheduler_update_per_iter')
    parser.add_argument('--cl_targ_no_pred', action='store_true', help='lr_scheduler_update_per_iter')
    parser.add_argument('--temp_t', default=0.1, type=float, help='mfm loss alpha')
    parser.add_argument('--temp_s', default=1., type=float, help='mfm loss alpha')
    parser.add_argument('--cl_loss', default='ce', type=str, help='mfm loss alpha')
    parser.add_argument('--cl_type', default='feat', type=str, help='mfm loss alpha')
    parser.add_argument('--cl_out_dim', default=512, type=int, help='cl loss alpha')
    parser.add_argument('--cl_pred_head', default='fc', type=str, help='mfm loss alpha')
    parser.add_argument('--teacher_init', default='modules/fold_2_model_best_auc.pt', type=str, help='output log filepath')
    parser.add_argument('--no_tea_init', action='store_true', help='lr_scheduler_update_per_iter')
    parser.add_argument('--init_stu_type', default='none', type=str, help='[none,fc,all]')
    parser.add_argument('--tea_type', default='none', type=str, help='[none,inner,outer]')
    parser.add_argument('--mm', default=0.9999, type=float, help='ema decay [0.9997]')
    parser.add_argument('--mm_final', default=1., type=float, help='ema decay [0.9997]')
    parser.add_argument('--mm_update_epoch', default=0, type=int, help='ema decay [0.9997]')
    parser.add_argument('--mm_warmup_epoch', default=0, type=int, help='ema decay [0.9997]')
    parser.add_argument('--mm_sche', action='store_true', help='cl loss alpha')
    parser.add_argument('--mm_head', default=0., type=float, help='ema decay [0.9997]')

    # mfm
    parser.add_argument('--mfm_enable', action='store_true',  help='cl loss alpha')
    parser.add_argument('--mfm_decoder', default='fc', type=str, help='lr_scheduler_update_per_iter')
    parser.add_argument('--mfm_alpha', default=0., type=float, help='mfm loss alpha')
    parser.add_argument('--mfm_loss', default='l2', type=str, help='mfm loss alpha')
    parser.add_argument('--mfm_temp_t', default=0.1, type=float, help='mfm loss alpha')
    parser.add_argument('--mfm_temp_s', default=1., type=float, help='mfm loss alpha')
    parser.add_argument('--mfm_targ_ori', action='store_true',  help='cl loss alpha')
    parser.add_argument('--mfm_no_targ_grad', action='store_true',  help='cl loss alpha')
    parser.add_argument('--mfm_feats', default='last_layer', type=str, help='mfm loss alpha')
    parser.add_argument('--multi_add_pos', action='store_true',  help='cl loss alpha')
    parser.add_argument('--multi_type', default='cat_all', type=str, help='mfm loss alpha')
    # mfm trans decoder 
    parser.add_argument('--mfm_decoder_embed_dim', default=512, type=int, help='cl loss alpha')
    parser.add_argument('--mfm_decoder_out_dim', default=512, type=int, help='cl loss alpha')
    # mfm loss norm
    parser.add_argument('--mfm_norm_targ_loss', default='none', type=str, help='[none,l1,l2]')
    parser.add_argument('--mfm_norm_pred_loss', default='none', type=str, help='[none,l1,l2]')
    # mfm mlp decoder
    parser.add_argument('--mlp_norm_last_layer', action='store_true',  help='cl loss alpha')
    parser.add_argument('--mlp_nlayers', default=1, type=int, help='cl loss alpha')
    parser.add_argument('--mlp_hidden_dim', default=512, type=int, help='cl loss alpha')
    parser.add_argument('--mlp_bottleneck_dim', default=256, type=int, help='cl loss alpha')

    # misc
    parser.add_argument('--title', default='default', type=str, help='output log filepath')
    parser.add_argument('--project', default='mil_new_c16', type=str, help='output log filepath')
    parser.add_argument('--log_iter', default=100, type=int, help='Number of total training epochs [40]')
    parser.add_argument('--early_stopping', action='store_false', help='Automatic Mixed Precision Training')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--wandb', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--force_cpu_ema', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--model_path', default='/data/xxx/checkpoint/c16/MIM+con/tang_new', type=str, help='MIL model')
    parser.add_argument('--swin', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--pict', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--vit', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--no_log', action='store_true', help='Automatic Mixed Precision Training')

    args = parser.parse_args()
    
    # sys.stdout = open('./log/MIM+con/new_tang/'+args.title+'.log', mode='a', encoding='utf-8')

    if not os.path.exists(os.path.join(args.model_path,args.project)):
        os.mkdir(os.path.join(args.model_path,args.project))
    args.model_path = os.path.join(args.model_path,args.project,args.title)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    
    if args.swin:
        trainer.step()

    if args.model == 'pure':
        args.teacher_init = ''
        args.cl_alpha=0.
        args.mfm_alpha = 0.
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif args.model == 'clam_sb':
        # args.cls_alpha=1.
        # args.cl_alpha = 0.
        args.cls_alpha= .7
        args.cl_alpha = .3
    elif args.model == 'clam_mb':
        args.cls_alpha= .7
        args.cl_alpha = .3
    elif args.model == 'dsmil':
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5

    if args.datasets == 'camelyon16' and not args.no_fix_c16:
        args.fix_loader_random = True
        args.fix_train_random = True

    # if args.datasets == 'camelyon16' and  args.model == 'mhim' and not args.no_fix_c16:
    #     args.fix_train_random = True
    
    if args.datasets == 'tcga':
        # 20 epoch后再计算最佳模型
        args.save_best_model_stage = args.save_best_model_stage
    else:
        args.save_best_model_stage = 0.

    if args.wandb:
        # dir 修改cache文件位置
        # wandb.util.generate_id() 2tk9m1xi 

        if args.auto_resume:
            ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
            wandb.init(project=args.project, entity='dearcat',name=args.title,config=args,dir=os.path.join(args.model_path),id=ckp['wandb_id'],resume='must')
        else:
            wandb.init(project=args.project, entity='dearcat',name=args.title,config=args,dir=os.path.join(args.model_path))
        wandb.save('./modules/contrastive1_zxx.py',policy='now')
        wandb.save('./modules/translayer.py',policy='now')
    print(args)

    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime)
    main(args=args)

    if args.pict:
        trainer.finish(main,wandb,args)