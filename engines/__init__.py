from .base_engine import BaseTrainer
from .common_mil import CommonMIL

from datasets.dataloader import build_feat_dataloader

def build_engine(args,device,dataset):

    _commom_mil = ('mhim','mhim_pure','rrtmil','abmil','gattmil','clam_sb','clam_mb','transmil','dsmil','dtfd','meanmil','maxmil','vitmil','abmilx','wikg','gigap','chief','2dmamba','titan','patch_gcn','patch_gcn_abx','hipt','head')
    
    if args.model in _commom_mil:
        engine = CommonMIL(args)
    elif args.model == 'ipsmil':
        engine = IPSMIL(args)
    elif args.model.startswith('e2e'):
        feat_train_dataloader = feat_val_dataloader = feat_test_dataloader = None
        if args.feat_test:
            if args.val_ratio == 0.:
                _,_,feat_test_dataloader = build_feat_dataloader(args,dataset,'test',root=args.feat_root,prefetch=False)
                _,_,feat_val_dataloader = build_feat_dataloader(args,dataset,'test',root=args.feat_root,prefetch=False)
            else:
                _,feat_val_dataloader,feat_test_dataloader = build_feat_dataloader(args,dataset,'no_train',root=args.feat_root,prefetch=False)

        if 'feat' in args.sel_type:
            # (args.adaLN_attn_weight and 'feat' in args.adaLN_attn_type):
            feat_train_dataloader,_,_ = build_feat_dataloader(args,dataset,'train',root=args.feat_root,prefetch=False)
        engine = E2E(args,device,feat_train_loader=feat_train_dataloader,feat_val_loader=feat_val_dataloader,feat_test_loader=feat_test_dataloader)
    else:
        raise NotImplementedError
    trainer = BaseTrainer(engine=engine,args=args)

    if args.datasets.lower().startswith('surv'):
        return trainer.surv_train,trainer.surv_validate
    else:
        return trainer.train,trainer.validate
