import torch
from torch.utils.data import DataLoader,RandomSampler

from .dataset_feat import FeatClsDataset,FeatSurvDataset
from .data_utils import *

def build_dataloader(args,dataset):
    return build_feat_dataloader(args,dataset,prefetch=args.prefetch)

def update_worker_dict(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.update_imgs()

def _get_feat_dataloader(args,dataset,root,train=True,prefetch=True):
    if args.datasets.lower().startswith('surv'):
        if train:
            _dataset = FeatSurvDataset(dataset,root=root,persistence=args.persistence,is_train=True,args=args)
        else:
            _dataset = FeatSurvDataset(dataset,root=root,persistence=args.persistence,is_train=False,args=args)
    else:
        p,l = dataset
        if train:
            _dataset = FeatClsDataset(p,l,root,persistence=args.persistence,is_train=True,_type=args.datasets,args=args)
        else:
            _dataset = FeatClsDataset(p,l,root,persistence=args.persistence,_type=args.datasets,args=args)

    loader_kwargs = {'pin_memory':args.pin_memory}
    loader_test_kwargs = None

    if 'gcn' in args.model.lower():
        loader_kwargs['collate_fn'] = collate_graph
        loader_test_kwargs = {'collate_fn': collate_graph}

    if train:
        _dataloader = DataLoader(_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last,num_workers=args.num_workers,**loader_kwargs)
    else:
        _num_workers_test = args.num_workers_test or args.num_workers
        if loader_test_kwargs is not None:
            _dataloader = DataLoader(_dataset, batch_size=1, shuffle=False, num_workers=_num_workers_test,pin_memory=args.pin_memory, **loader_test_kwargs)
        else:
            _dataloader = DataLoader(_dataset, batch_size=1, shuffle=False, num_workers=_num_workers_test,pin_memory=args.pin_memory)

    if prefetch:
        _dataloader = PrefetchLoader(_dataloader,device=args.device,need_norm=False)
        
    return _dataloader

def build_feat_dataloader(args,dataset,mode='all',root=None,prefetch=True):
    root = root or args.dataset_root

    if not args.datasets.lower().startswith('surv'):
        train_p, train_l, test_p, test_l,val_p,val_l = parse_dataframe(args,dataset)
        dataset = {'train':[train_p, train_l],"val":[val_p,val_l],"test":[test_p, test_l]}

    if mode == 'all':
        train_loader = _get_feat_dataloader(args,dataset=dataset['train'],root=root,train=True,prefetch=prefetch)
        if args.val_ratio == 0.:
            val_loader = _get_feat_dataloader(args,dataset=dataset['test'],root=root,train=False,prefetch=prefetch)
        else:
            val_loader = _get_feat_dataloader(args,dataset=dataset['val'],root=root,train=False,prefetch=prefetch)
        test_loader = _get_feat_dataloader(args,dataset=dataset['test'],root=root,train=False,prefetch=prefetch)
    elif mode == 'test':
        test_loader = _get_feat_dataloader(args,dataset=dataset['test'],root=root,train=False,prefetch=prefetch)
        train_loader = val_loader = None
    elif mode == 'no_train':
        if args.val_ratio == 0.:
            val_loader = _get_feat_dataloader(args,dataset=dataset['test'],root=root,train=False,prefetch=prefetch)
        else:
            val_loader = _get_feat_dataloader(args,dataset=dataset['val'],root=root,train=False,prefetch=prefetch)
        test_loader = _get_feat_dataloader(args,dataset=dataset['test'],root=root,train=False,prefetch=prefetch)
        train_loader = None
    elif mode == 'train':
        train_loader = _get_feat_dataloader(args,dataset=dataset['train'],root=root,train=True,prefetch=prefetch)
        val_loader = test_loader = None

    return train_loader,val_loader,test_loader