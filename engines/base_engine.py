import time
import wandb
from timm.models import model_parameters
from timm.utils import AverageMeter, dispatch_clip_grad
from collections import OrderedDict

from utils import *
from .metrics import get_metric_val

class BaseTrainer():
    def __init__(self, engine, args, **kwargs):
        self.engine = engine

    def train(self, args, model, model_ema, loader, optimizer, device, amp_autocast, criterion, loss_scaler, scheduler, epoch, model_others):
        start = time.time()
        loss_cls_meter = AverageMeter()
        loss_cl_meter = AverageMeter()
        patch_num_meter = AverageMeter()
        keep_num_meter = AverageMeter()
        pad_ratio_meter = AverageMeter()
        mm_meter = AverageMeter()
        update_time_m = AverageMeter()
        data_time_m = AverageMeter()

        has_no_sync = hasattr(model, "no_sync")
        model.train()

        train_loss_log = 0.
        accum_steps = args.accumulation_steps
        last_accum_steps = len(loader) % accum_steps
        updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
        num_updates = epoch * updates_per_epoch
        last_batch_idx = len(loader) - 1
        last_batch_idx_to_accum = len(loader) - last_accum_steps

        model_unddp = model.module if args.distributed else model
        if model_ema is not None:
            model_ema.train()

        data_start_time = update_start_time = time.time()

        self.engine.init_func_train(args, model=model_unddp, others=model_others, epoch=epoch, optimizer=optimizer)
        optimizer.zero_grad()
        update_sample_count = 0

        for batch_idx, batch in enumerate(loader):
            last_batch = batch_idx == last_batch_idx
            need_update = last_batch or (batch_idx + 1) % accum_steps == 0
            update_idx = batch_idx // accum_steps
            if batch_idx >= last_batch_idx_to_accum:
                accum_steps = last_accum_steps

            if not args.prefetch:
                batch = {
                    k: ([v.to(device=device, non_blocking=True) for v in v] if isinstance(v, list) and k == 'input' else
                        v.to(device=device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
            bag = batch['input']
            label = batch['target']
            batch_size = label.size(0)
            pos = batch.get('pos', None)  
            idx = batch.get('idx', None)
            feat = batch.get('feat', None)

            self.engine.after_get_data_func(args, device=device, bag=bag, optimizer=optimizer, loader=loader,
                                            n_iter=epoch * len(loader) + batch_idx, epoch=epoch, model=model_unddp,
                                            others=model_others)

            # multiply by accum steps to get equivalent for full update
            data_time_m.update(accum_steps * (time.time() - data_start_time))

            if args.patch_shuffle:
                bag = patch_shuffle(bag)

            def _forward():
                with amp_autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                    logits, labels, aux_loss, patch_num, keep_num, pad_ratio, kn_std = self.engine.forward_func(args,
                    model=model,
                    model_ema=model_ema,
                    bag=bag,
                    label=label,
                    criterion=criterion,
                    batch_size=batch_size,
                    i=batch_idx,
                    epoch=epoch,
                    n_iter=epoch * len(loader) + batch_idx,
                    loader=loader,
                    device=device,
                    others=model_others,
                    pos=pos,
                    idx=idx,
                    feat=feat)

                    if logits is None:
                        return None, None, None, patch_num, keep_num, pad_ratio
                    # continue
                    bs = logits.size(0)
                    logit_loss = criterion(logits.view(bs, -1), labels.view(bs))
                    loss = args.main_alpha * logit_loss + aux_loss * args.aux_alpha

                    loss /= accum_steps
                    return loss, aux_loss, logit_loss, patch_num, keep_num, pad_ratio

            def _backward(_loss):
                if loss_scaler:
                    
                    loss_scaler.scale(_loss).backward()
                    if need_update:
                        loss_scaler.step(optimizer)
                        loss_scaler.update()
                else:
                    _loss.backward()
                    if need_update:
                        if args.clip_grad is not None:
                            dispatch_clip_grad(
                                model_parameters(model),
                                value=args.clip_grad,
                            )
                        optimizer.step()

            if args.debug:
                torch.autograd.set_detect_anomaly(True)
            if has_no_sync and not need_update:
                with model.no_sync():
                    loss, aux_loss, logit_loss, patch_num, keep_num, pad_ratio = _forward()
                    if loss is None:
                        continue
                    _backward(loss)
            else:
                loss, aux_loss, logit_loss, patch_num, keep_num, pad_ratio = _forward()
                if loss is None:
                    continue
                _backward(loss)

            self.engine.after_backward_func(args, model=model, others=model_others, num_updates=num_updates)

            if not args.distributed:
                loss_cls_meter.update(logit_loss * accum_steps * args.main_alpha, batch_size)
                loss_cl_meter.update(aux_loss * accum_steps * args.aux_alpha, batch_size)
                patch_num_meter.update(patch_num, 1)
                keep_num_meter.update(keep_num, 1)
                pad_ratio_meter.update(pad_ratio, 1)

            update_sample_count += batch_size
            if not need_update:
                data_start_time = time.time()
                continue

            num_updates += 1
            optimizer.zero_grad()
            if args.lr_supi and scheduler is not None:
                scheduler.step()

            if model_ema is not None:
                if args.model == 'mhim':
                    if args.tea_type == 'same':
                        pass
                    else:
                        if model_others['mm_sche'] is not None:
                            mm = model_others['mm_sche'][epoch*len(loader)+batch_idx]
                        else:
                            mm = args.mm
                        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
                        
                        for param_q, param_k in zip(model.parameters(), model_ema.parameters()):
                            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm) 
                        
            else:
                mm = 0.

            mm_meter.update(mm, 1)


            time_now = time.time()
            update_time_m.update(time.time() - update_start_time)
            update_start_time = time_now

            if update_idx % args.log_iter == 0 or batch_idx == len(loader) - 1:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                rowd = OrderedDict([
                    ('main_loss', loss_cls_meter.avg),
                    ('lr', lr),
                    ('aux_loss', loss_cl_meter.avg),
                    ('patch_num', patch_num_meter.avg),
                    ('keep_num', keep_num_meter.avg),
                    ('pad_ratio', pad_ratio_meter.avg),
                    ('mm', mm_meter.avg),
                ])

                if args.rank == 0:

                    if loss_cl_meter.val != 0:
                        loss_cl_meter_avg_format = f'AuxLoss: {loss_cl_meter.val:#.4g} ({loss_cl_meter.avg:#.16g})  '
                    else:
                        loss_cl_meter_avg_format = ''

                    if keep_num_meter.avg == patch_num_meter.avg:
                        keep_num_meter_avg_format = ""
                    else:
                        keep_num_meter_avg_format = f'KN: {keep_num_meter.val:#.1f} ({keep_num_meter.avg:#.1f})  '
                    print(
                        f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                        f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                        f'Loss: {loss_cls_meter.val:#.4g} ({loss_cls_meter.avg:#.16g})  '
                        f'{loss_cl_meter_avg_format}'
                        f'PN: {patch_num_meter.val:#.1f} ({patch_num_meter.avg:#.1f})  '
                        f'{keep_num_meter_avg_format}'
                        f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                        f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                        f'LR: {lr:.3e}  '
                        f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                    )
                    rowd = OrderedDict([(str(args.fold_curr) + '-fold/' + _k, _v) for _k, _v in rowd.items()])
                    if args.wandb and args.rank == 0:
                        wandb.log(rowd)

            update_sample_count = 0
            data_start_time = time.time()
            train_loss_log = train_loss_log + loss.item()

        self.engine.final_train_func(args, model=model)

        end = time.time()
        train_loss_log = train_loss_log / len(loader)

        if not args.lr_supi and scheduler is not None:
            scheduler.step(epoch + 1)

        return train_loss_log, start, end

    def validate(self, args, model, loader, device, criterion, amp_autocast, early_stopping=None, epoch=None, status="val", others=None):
        model.eval()
        loss_cls_meter = AverageMeter()
        loader_time_meter = AverageMeter()
        process_time_meter = AverageMeter()
        bag_logit, bag_labels = None, None
        bag_logit_sub = None

        model_unddp = model.module if args.distributed else model

        self.engine.init_func_val(args, status=status, amp_autocast=amp_autocast, model=model_unddp, loader=loader,
                                  others=others, epoch=epoch)
        start_all_time = time.time()  
        accu_time = 0
        acc_time_loader = 0
        start_all_time = time.time()  

        with torch.no_grad():
            for i, batch in enumerate(loader):
                start_time = time.time()  
                if not args.prefetch:
                    batch = {
                        k: ([v.to(device=device, non_blocking=True) for v in v] if isinstance(v, list) and k == 'input' else
                            v.to(device=device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()
                    }
                bag = batch['input']
                label = batch['target']
                pos = batch.get('pos', None) 
                idx = batch.get('idx', None)

                batch_size = label.size(0)
                bag_labels = torch.cat((bag_labels, label)) if bag_labels is not None else torch.clone(label)

                self.engine.after_get_data_func(args, device=device, bag=bag, optimizer=None, loader=loader,
                                                n_iter=None, epoch=None, model=model_unddp)

                with amp_autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                    logits, labels = self.engine.validate_func(args, model=model, bag=bag, label=label,
                    criterion=criterion, batch_size=batch_size, i=i,
                    loader=loader, device=device, others=others, pos=pos,
                    idx=idx)

                    if logits is None:
                        continue

                    if type(logits) in (list, tuple):
                        logits, logits_sub = logits
                    else:
                        logits_sub = None

                    bs = logits.size(0)

                    bag_logit = torch.cat((bag_logit, logits)) if bag_logit is not None else logits.clone()
                    if logits_sub is not None:
                        bag_logit_sub = torch.cat(
                            (bag_logit_sub, logits_sub)) if bag_logit_sub is not None else logits_sub.clone()
                    test_loss = criterion(logits.view(bs, -1), labels.view(bs))

                loss_cls_meter.update(test_loss, 1)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.time()  
                iteration_time = end_time - start_time
                accu_time += iteration_time
                all_time_gap = end_time - start_all_time
                loader_time = all_time_gap - acc_time_loader
                acc_time_loader += loader_time

                loader_time_meter.update(loader_time, 1)
                process_time_meter.update(iteration_time, 1)

                if args.rank == 0 and (i + 1) == len(loader):
                    print(f"Iter{i + 1}: Loader took {loader_time_meter.avg:.2f} seconds "
                          f"Process took {process_time_meter.avg:.2f} seconds ")

        suffix_main = suffix_sub = None

        output = get_metric_val(args, bag_logit=bag_logit, bag_labels=bag_labels, model=model, status=status,early_stopping=early_stopping, epoch=epoch,
                                     loss_cls_meter=loss_cls_meter, suffix=suffix_main)
        output = list(output)

        if bag_logit_sub is not None:
            output_sub = get_metric_val(args, bag_logit=bag_logit_sub, bag_labels=bag_labels, model=model,
                                             status=status, early_stopping=None, epoch=epoch,
                                             loss_cls_meter=loss_cls_meter, suffix=suffix_sub)

            output_sub = list(output_sub)

            output[-1].update(output_sub[-1])

            output[0] = [output[0], output_sub[0]]

        return output

    ############# Survival Prediction ###################
    def surv_train(self, args, model, model_ema, loader, optimizer, device, amp_autocast, criterion, loss_scaler,
                   scheduler, epoch, model_others):
        start = time.time()
        
        loss_cls_meter = AverageMeter()
        loss_cl_meter = AverageMeter()
        patch_num_meter = AverageMeter()
        keep_num_meter = AverageMeter()
        pad_ratio_meter = AverageMeter()
        mm_meter = AverageMeter()
        update_time_m = AverageMeter()
        data_time_m = AverageMeter()

        model.train()
        if model_ema is not None:
            model_ema.train()

        model_unddp = model.module if args.distributed else model

        accum_steps = args.accumulation_steps
        updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
        num_updates = epoch * updates_per_epoch
        last_batch_idx = len(loader) - 1

        self.engine.init_func_train(args, model=model_unddp, others=model_others, epoch=epoch, optimizer=optimizer)
        optimizer.zero_grad()

        train_loss_log = 0.
        update_sample_count = 0
        data_start_time = update_start_time = time.time()

        for batch_idx, batch in enumerate(loader):

            last_batch = batch_idx == last_batch_idx
            need_update = last_batch or (batch_idx + 1) % accum_steps == 0
            update_idx = batch_idx // accum_steps

            if not args.prefetch:
                batch = {
                    k: ([v.to(device=device, non_blocking=True) for v in v] if isinstance(v, list) and k == 'input' else
                        v.to(device=device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }

            bag = batch['input']

            label = batch['target'].to(dtype=torch.long, device=batch['target'].device)
            data_Censorship = batch['censorship'].to(dtype=torch.float, device=batch['censorship'].device)

            pos = batch.get('pos', None)
            idx = batch.get('idx', None)
           
            batch_size = label.size(0)

            self.engine.after_get_data_func(args, device=device, bag=bag, optimizer=optimizer, loader=loader,
                                            n_iter=epoch * len(loader) + batch_idx, epoch=epoch, model=model_unddp,
                                            others=model_others)

            data_time_m.update(accum_steps * (time.time() - data_start_time))

            if args.patch_shuffle:
                bag = patch_shuffle(bag)

            def _forward():
                with amp_autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                    logits, labels_surv, aux_loss, patch_num, keep_num, pad_ratio, kn_std = self.engine.forward_func(
                        args,
                        model=model,
                        model_ema=model_ema,
                        bag=bag,
                        label=[label, data_Censorship],
                        criterion=criterion,
                        batch_size=batch_size,
                        i=batch_idx,
                        epoch=epoch,
                        n_iter=epoch * len(loader) + batch_idx,
                        loader=loader,
                        device=device,
                        others=model_others,
                        pos=pos,
                        idx=idx
                    )
                    if logits is None:
                        return None, None, None, patch_num, keep_num, pad_ratio


                    with amp_autocast(device_type='cuda', enabled=False):
                        logit_loss = criterion(logits=logits, Y=labels_surv[0], c=labels_surv[1])

                        loss = args.main_alpha * logit_loss + aux_loss * args.aux_alpha
                        loss /= accum_steps

                    return loss, aux_loss, logit_loss, patch_num, keep_num, pad_ratio

            def _backward(_loss):
                if loss_scaler:
                    loss_scaler.scale(_loss).backward()

                    if need_update:
                        loss_scaler.step(optimizer)
                        loss_scaler.update()
                else:
                    _loss.backward()
                    if need_update:
                        if args.clip_grad is not None:
                            dispatch_clip_grad(
                                model_parameters(model),
                                value=args.clip_grad,
                            )
                        optimizer.step()

            has_no_sync = hasattr(model, "no_sync")
            if has_no_sync and not need_update:
                with model.no_sync():
                    loss, aux_loss, logit_loss, patch_num, keep_num, pad_ratio = _forward()
                    if loss is None:
                        data_start_time = time.time()
                        continue
                    _backward(loss)
            else:
                loss, aux_loss, logit_loss, patch_num, keep_num, pad_ratio = _forward()
                if loss is None:
                    data_start_time = time.time()
                    continue
                _backward(loss)

            self.engine.after_backward_func(args, model=model, others=model_others, num_updates=num_updates)

            if not args.distributed:
                loss_cls_meter.update(loss.item() * accum_steps, batch_size)
                loss_cl_meter.update(aux_loss, batch_size)
                patch_num_meter.update(patch_num, 1)
                keep_num_meter.update(keep_num, 1)
                pad_ratio_meter.update(pad_ratio, 1)

            update_sample_count += batch_size

            if not need_update:
                data_start_time = time.time()
                continue

            num_updates += 1
            optimizer.zero_grad()
            if args.lr_supi and scheduler is not None:
                scheduler.step()

            if model_ema is not None:
                if args.model == 'mhim':
                    if args.tea_type == 'same':
                        pass
                    else:
                        if model_others['mm_sche'] is not None:
                            mm = model_others['mm_sche'][epoch*len(loader)+batch_idx]
                        else:
                            mm = args.mm
                        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
                        for param_q, param_k in zip(model.parameters(), model_ema.parameters()):
                            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm) 
                else:
                    model_ema.update(model, epoch * len(loader) + batch_idx)
                    mm = model_ema.get_decay(epoch * len(loader) + batch_idx)
            else:
                mm = 0.
            mm_meter.update(mm, 1)

            time_now = time.time()
            update_time_m.update(time_now - update_start_time)
            update_start_time = time_now

            if update_idx % args.log_iter == 0 or batch_idx == last_batch_idx:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if args.rank == 0:
                    if loss_cl_meter.val != 0:
                        loss_cl_meter_avg_format = f'AuxLoss: {loss_cl_meter.val:#.4g} ({loss_cl_meter.avg:#.16g})  '
                    else:
                        loss_cl_meter_avg_format = ''

                    if keep_num_meter.avg == patch_num_meter.avg:
                        keep_num_meter_avg_format = ""
                    else:
                        keep_num_meter_avg_format = f'KN: {keep_num_meter.val:#.1f} ({keep_num_meter.avg:#.1f})  '


                    print(
                        f'Train(Surv): {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                        f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                        f'Loss: {loss_cls_meter.val:#.4g} ({loss_cls_meter.avg:#.16g})  '
                        f'{loss_cl_meter_avg_format}'
                        f'PN: {patch_num_meter.val:#.1f} ({patch_num_meter.avg:#.1f})  '
                        f'{keep_num_meter_avg_format}'
                        f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                        f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                        f'LR: {lr:.3e}  '
                        f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                    )

                    rowd = OrderedDict([
                        ('main_loss', loss_cls_meter.avg),
                        ('lr', lr),
                        ('aux_loss', loss_cl_meter.avg),
                        ('patch_num', patch_num_meter.avg),
                        ('keep_num', keep_num_meter.avg),
                        ('pad_ratio', pad_ratio_meter.avg),
                        ('mm', mm_meter.avg),
                    ])
                    rowd = OrderedDict([(str(args.fold_curr) + '-fold/' + _k, _v) for _k, _v in rowd.items()])
                    if args.wandb and args.rank == 0:
                        wandb.log(rowd)

            train_loss_log += loss.item()
            update_sample_count = 0
            data_start_time = time.time()

        self.engine.final_train_func(args, model=model)

        end = time.time()
        train_loss_log = train_loss_log / len(loader)

        if not args.lr_supi and scheduler is not None:
            scheduler.step(epoch + 1)

        return train_loss_log, start, end

    def surv_validate(self, args, model, loader, device, criterion, amp_autocast, early_stopping=None, epoch=None,
                      status="val", others=None):
        model.eval()
        loss_cls_meter = AverageMeter()
        loader_time_meter = AverageMeter()
        process_time_meter = AverageMeter()

        all_risk_scores = None
        all_risk_scores_sub = None
        all_censorships = None
        all_event_times = None

        model_unddp = model.module if args.distributed else model

        self.engine.init_func_val(args, status=status, amp_autocast=amp_autocast, model=model_unddp, loader=loader,
                                  others=others, epoch=epoch)

        start_all_time = time.time()
        acc_time_loader = 0
        accu_time = 0

        with torch.no_grad():
            for i, batch in enumerate(loader):
                start_time = time.time()

                if not args.prefetch:
                    batch = {
                        k: ([v.to(device=device, non_blocking=True) for v in v] if isinstance(v, list) and k == 'input' else
                            v.to(device=device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()
                    }

                bag = batch['input']
                label = batch['target'].to(dtype=torch.long, device=batch['target'].device)
                data_Censorship = batch['censorship'].to(dtype=torch.float, device=batch['censorship'].device)
                data_Event = batch['event']
                pos = batch.get('pos', None)
                idx = batch.get('idx', None)
                batch_size = label.size(0)

                self.engine.after_get_data_func(args, device=device, bag=bag, optimizer=None, loader=loader,
                                                n_iter=None, epoch=None, model=model_unddp, others=others)

                with amp_autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                    
                    logits, labels = self.engine.validate_func(
					args,
					model=model,
					bag=bag,
					label=label,
					criterion=criterion,
					batch_size=batch_size,
					i=i,
					loader=loader,
					device=device,
					others=others,
					pos=pos,
                    idx=idx
				)

                    if logits is None:
                        end_time = time.time()
                        iteration_time = end_time - start_time
                        accu_time += iteration_time
                        all_time_gap = end_time - start_all_time
                        loader_time = all_time_gap - acc_time_loader
                        acc_time_loader += loader_time
                        loader_time_meter.update(loader_time, 1)
                        process_time_meter.update(iteration_time, 1)
                        if args.rank == 0 and (i + 1) == len(loader):
                            print(f"Iter{i + 1}: Loader took {loader_time_meter.avg:.2f} seconds "
                                  f"Process took {process_time_meter.avg:.2f} seconds ")
                        continue
                    
                    if type(logits) in (list, tuple):
                        logits, logits_sub = logits
                    else:
                        logits_sub = None

                    hazards = torch.sigmoid(logits)
                    S = torch.cumprod(1 - hazards, dim=1)

                    test_loss = criterion(hazards=hazards, S=S, Y=labels, c=data_Censorship)
                    loss_cls_meter.update(test_loss,batch_size)

                    # risk score
                    risk = -torch.sum(S, dim=1)
                    all_risk_scores = torch.cat(
                        (all_risk_scores, risk)) if all_risk_scores is not None else risk.clone()
                    all_censorships = torch.cat(
                        (all_censorships, data_Censorship)) if all_censorships is not None else data_Censorship.clone()
                    all_event_times = torch.cat(
                        (all_event_times, data_Event)) if all_event_times is not None else data_Event.clone()

                    if logits_sub is not None:
                        hazards_sub = torch.sigmoid(logits_sub)
                        S_sub = torch.cumprod(1 - hazards_sub, dim=1)
                        risk_sub = -torch.sum(S_sub, dim=1)
                        all_risk_scores_sub = torch.cat(
                        (all_risk_scores_sub, risk_sub)) if all_risk_scores_sub is not None else risk_sub.clone()

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.time()
                iteration_time = end_time - start_time
                accu_time += iteration_time
                all_time_gap = end_time - start_all_time
                loader_time = all_time_gap - acc_time_loader
                acc_time_loader += loader_time

                loader_time_meter.update(loader_time, 1)
                process_time_meter.update(iteration_time, 1)

                if args.rank == 0 and (i + 1) == len(loader):
                    print(f"Iter{i + 1}: Loader took {loader_time_meter.avg:.2f} seconds "
                          f"Process took {process_time_meter.avg:.2f} seconds ")
                    
        suffix_main = suffix_sub = None

        output = get_metric_val(args, bag_logit=all_risk_scores, 
        bag_labels={'censorships':all_censorships,'event_times':all_event_times}, 
        model=model, status=status,
        early_stopping=early_stopping, epoch=epoch,
        loss_cls_meter=loss_cls_meter, suffix=suffix_main,surv=True)

        output = list(output)

        if all_risk_scores_sub is not None:
            output_sub = get_metric_val(args, bag_logit=all_risk_scores_sub, 
            bag_labels={'censorships':all_censorships,'event_times':all_event_times}, 
            model=model,status=status, early_stopping=None, epoch=epoch,
            loss_cls_meter=loss_cls_meter, suffix=suffix_sub,surv=True)

            output_sub = list(output_sub)

            output[-1].update(output_sub[-1])

            output[0] = [output[0], output_sub[0]]

        return output