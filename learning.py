import torch
import utils
import logging
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast


def ddp_train(args, train_queue, model, criterion, optimizer, epoch, writer, device, scaler, scheduler):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    train_queue.sampler.set_epoch(epoch)

    for step, (images, labels) in enumerate(train_queue):

        images, labels = images.to(device), labels.to(device)

        if args.amp:
            with autocast():
                logits, aux_logits = model(images)
                loss = criterion(logits, labels)
                if args.auxiliary:
                    aux_loss = criterion(aux_logits, labels)
                    loss += args.auxiliary_weight * aux_loss
        else:
            logits, aux_logits = model(images)
            loss = criterion(logits, labels)
            if args.auxiliary:
                aux_loss = criterion(aux_logits, labels)
                loss += args.auxiliary_weight * aux_loss

        n = logits.size(0)
        prec1, prec5 = utils.accuracy(logits, labels, topk=(1, 5))
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        objs.update(loss.item(), n)

        iter_num = step + epoch * len(train_queue)
        if dist.get_rank() in [-1, 0]:
            writer.add_scalar('train/loss', loss.item(), iter_num)
            writer.add_scalar('train/acc', prec1.item(), iter_num)

        if args.amp:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = (scaler.get_scale() < scale)  # skip the schedule if scale is reduced
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if not skip_lr_sched:
            scheduler.step()

        if step % args.report_freq == 0 or step == len(train_queue) - 1:
            logging.info('train %05d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def ddp_infer(valid_queue, model, criterion, device):
    model.eval()

    with torch.no_grad():
        preds, gt = [], []
        for step, (images, labels) in enumerate(valid_queue):

            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            preds.append(logits)
            gt.append(labels)

        preds = utils.distributed_concat(torch.cat(preds, dim=0), len(valid_queue.dataset))
        gt = utils.distributed_concat(torch.cat(gt, dim=0), len(valid_queue.dataset))
        loss = criterion(preds, gt)
        prec1, prec5 = utils.accuracy(preds, gt, topk=(1, 5))

    return prec1.item(), prec5.item(), loss.item()


def train(args, train_queue, model, criterion, optimizer, epoch, writer, scheduler):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (images, labels) in enumerate(train_queue):

        images, labels = images.cuda(), labels.cuda()

        logits, aux_logits = model(images)
        loss = criterion(logits, labels)
        if args.auxiliary:
            aux_loss = criterion(aux_logits, labels)
            loss += args.auxiliary_weight * aux_loss

        n = logits.size(0)
        prec1, prec5 = utils.accuracy(logits, labels, topk=(1, 5))
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        objs.update(loss.item(), n)

        iter_num = step + epoch * len(train_queue)
        writer.add_scalar('train/loss', loss.item(), iter_num)
        writer.add_scalar('train/acc', prec1.item(), iter_num)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.report_freq == 0 or step == len(train_queue) - 1:
            logging.info('train %05d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(args, valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.eval()

    with torch.no_grad():
        for step, (images, labels) in enumerate(valid_queue):

            images, labels = images.cuda(), labels.cuda()
            logits, _ = model(images)
            loss = criterion(logits, labels)

            prec1, prec5 = utils.accuracy(logits, labels, topk=(1, 5))
            n = logits.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0 or step == len(valid_queue) - 1:
                logging.info('valid %05d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg
