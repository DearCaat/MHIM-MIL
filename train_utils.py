import torch
import torch.nn as nn
from timm.scheduler import create_scheduler_v2

from utils import *

############# Survival Prediction ###################
def nll_loss(hazards,S,Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1)  # censorship status, 0 or 1
    # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

class NLLSurvLoss(object):
    def __init__(self, alpha=0.):
        self.alpha = alpha

    def __call__(self, Y, c, logits=None,hazards=None, S=None,alpha=None):
        if alpha is None:
            alpha = self.alpha
        if hazards is None:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
        return nll_loss(hazards, S, Y, c, alpha=alpha)

def build_train(args,model,train_loader):
    # build criterion
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "nll_surv":
        criterion = NLLSurvLoss(alpha=0.0)
    else:
        raise NotImplementedError

    # build optimizer
    if args.distributed:
        _model = model.module
    else:
        _model = model

    lr = args.lr

    params = [
        {'params': filter(lambda p: p.requires_grad, _model.parameters()), 'lr': lr,'weight_decay': args.weight_decay},]
            
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(params)
    elif args.opt == 'adam':
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(params)
    else:
        raise NotImplementedError

    # build scheduler
    if args.lr_sche == 'cosine':
        scheduler,_ = create_scheduler_v2(optimizer,sched='cosine',num_epochs=args.num_epoch,warmup_lr=args.warmup_lr,warmup_epochs=args.warmup_epochs,min_lr=1e-7)

    elif args.lr_sche == 'step':
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None
    else:
        raise NotImplementedError

    # build early stopping
    if args.early_stopping:
        patience,stop_epoch = args.patient,args.max_epoch
        early_stopping = EarlyStopping(patience=patience, stop_epoch=stop_epoch)
    else:
        early_stopping = None
    
    return criterion,optimizer,scheduler,early_stopping