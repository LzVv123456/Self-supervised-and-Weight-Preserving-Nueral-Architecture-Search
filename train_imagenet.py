import os
import time
import utils
import torch
import logging
import argparse
import torch.nn as nn
import torch.distributed as dist
from datetime import timedelta
from model_train import Network, NetworkImageNet
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from data_provider import DataProvider


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--note', type=str, default='exp')
    parser.add_argument('--exp_group', type=str, default='exp', help='name of the experment')
    parser.add_argument('--path', type=str, default=None, help='the corresponding exps dir')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--train_epochs', type=int, default=245)
    parser.add_argument('--data_mode', type=str, default='supervised')

    # data
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'imagenet-tiny'])
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')

    # network
    parser.add_argument('--arch_name', type=str, default='highest_acc')
    parser.add_argument('--init_channels', type=int, default=48)
    parser.add_argument('--auxiliary', action='store_true', default=True)
    parser.add_argument('--auxiliary_weight', type=float, default=0.4)
    parser.add_argument('--affine', action='store_true', default=True)

    # optm
    parser.add_argument('--optm', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.4)
    parser.add_argument('--lr_min', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=3e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--grad_clip', type=float, default=5.0)

    # miscellaneous
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--report_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=5)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--amp', action='store_true')

    return parser.parse_args()


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main(args, arch):
    
    args, writer = utils.init_exp('train', args)
    assert args.batch_size % 2 == 0

    with open(os.path.join(args.save_path, 'net.config'), 'a') as f:
        f.write('{}:'.format('model') + str(arch) + '\n')

    if args.ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda")

    # model
    if args.dataset == 'imagenet':
        model = NetworkImageNet(args.init_channels, args.num_classes, arch, args.auxiliary, args.affine).to(device)
    else:
        model = Network(args.init_channels, args.num_classes, arch, args.auxiliary, args.affine).to(device)

    model.drop_path_prob = 0.0

    if args.ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # loss
    valid_criterion = nn.CrossEntropyLoss()
    valid_criterion = valid_criterion.cuda()
    train_criterion = CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)
    train_criterion = train_criterion.cuda()

    # optimizers
    if args.optm == 'adam':
        if args.ddp:
            optimizer = torch.optim.Adam(model.module.parameters(), args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optm == 'sgd':
        if args.ddp:
            optimizer = torch.optim.SGD(model.module.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise NotImplementedError

    # data_provider
    data_provider = DataProvider(args)
    if args.ddp:
        train_queue, test_queue = data_provider.build_loaders_train_ddp(args.batch_size, args.num_workers)
    else:
        train_queue, test_queue = data_provider.build_loaders_train_imagenet(args.batch_size, args.num_workers)

    # schedulers
    linear_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda iter:(iter / (args.warmup_epochs * len(train_queue))))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.train_epochs * len(train_queue)), eta_min=args.lr_min)
    valid_stamps = set(range(0, args.warmup_epochs + args.train_epochs, args.valid_freq)) | \
    set(range(int((args.warmup_epochs + args.train_epochs)*0.95), args.warmup_epochs + args.train_epochs))

    scaler = GradScaler()
    mixup_fn = utils.Mixup(num_classes=args.num_classes)

    best_top1 = 0
    best_top5 = 0

    for epoch in range(args.warmup_epochs + args.train_epochs):

        start = time.time()
        logging.info('epoch %d', epoch)

        if epoch < args.warmup_epochs:
            scheduler = linear_scheduler
        else:
            scheduler = cosine_scheduler

        if args.ddp:
            train_obj = ddp_train(args, train_queue, model, train_criterion, optimizer, epoch, writer, device, scaler, scheduler, mixup_fn)
        else:
            train_obj = train(args, train_queue, model, train_criterion, optimizer, epoch, writer, scheduler, mixup_fn)

        logging.info('train_loss %e', train_obj)

        if epoch in valid_stamps:
            if args.ddp:
                valid_top1, valid_top5, valid_obj = ddp_infer(test_queue, model, valid_criterion, device)
            else:
                valid_top1, valid_top5, valid_obj = infer(args, test_queue, model, valid_criterion)

            logging.info('valid_top1 %f, best_top1 %f, valid_top5 %f, best_top5 %f, valid_loss %e', valid_top1, best_top1, valid_top5, best_top5, valid_obj)

            if writer is not None:
                writer.add_scalar('train/valid_acc', valid_top1, epoch)
                writer.add_scalar('train/valid_loss', valid_obj, epoch)

            if valid_top1 > best_top1:
                best_top1 = valid_top1
                best_top5 = valid_top5
                torch.save(model.state_dict(), os.path.join(args.save_path, 'model.pth.tar'))

        time_per_epoch = time.time() - start
        seconds_left = int((args.warmup_epochs + args.train_epochs - epoch - 1) * time_per_epoch)
        logging.info('Time per epoch: %s, Est. complete in: %s' % (str(timedelta(seconds=time_per_epoch)), str(timedelta(seconds=seconds_left))))
        logging.info('--' * 60)
    
    logging.info('best_top1 %f, best_top5 %f', best_top1, best_top5)
    # stop current logging
    logging.shutdown()


def ddp_train(args, train_queue, model, criterion, optimizer, epoch, writer, device, scaler, scheduler, mixup_fn):
    objs = utils.AvgrageMeter()
    model.train()

    train_queue.sampler.set_epoch(epoch)

    for step, (images, labels) in enumerate(train_queue):

        images, labels = images.to(device), labels.to(device)

        # images, labels = mixup_fn(images, labels)

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

        objs.update(loss.item(), logits.size(0))

        iter_num = step + epoch * len(train_queue)
        if dist.get_rank() in [-1, 0]:
            writer.add_scalar('train/loss', loss.item(), iter_num)

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
            logging.info('train %05d %e', step, objs.avg)

    return objs.avg


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


def train(args, train_queue, model, criterion, optimizer, epoch, writer, scheduler, mixup_fn):
    objs = utils.AvgrageMeter()
    model.train()

    for step, (images, labels) in enumerate(train_queue):

        images, labels = images.cuda(), labels.cuda()

        # images, labels = mixup_fn(images, labels)

        logits, aux_logits = model(images)
        loss = criterion(logits, labels)
        if args.auxiliary:
            aux_loss = criterion(aux_logits, labels)
            loss += args.auxiliary_weight * aux_loss

        objs.update(loss.item(), logits.size(0))

        iter_num = step + epoch * len(train_queue)
        writer.add_scalar('train/loss', loss.item(), iter_num)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.report_freq == 0 or step == len(train_queue) - 1:
            logging.info('train %05d %e', step, objs.avg)

    return objs.avg


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


if __name__ == '__main__':
    args = parse_args()
    arch_dict = utils.get_arch_dict(args.path, args.arch_name)
    note = args.note

    for weight_fname, arch in arch_dict.items():
        args.note = '_'.join([note, weight_fname])
        print('training: ' + weight_fname)
        main(args, arch)
