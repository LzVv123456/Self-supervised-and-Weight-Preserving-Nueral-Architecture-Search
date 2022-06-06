import os
import time
import utils
import torch
import logging
import argparse
import torch.nn as nn
import torch.distributed as dist
from datetime import timedelta
from model_train import Network
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from learning import ddp_train, ddp_infer, train, infer
from data_provider import DataProvider


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--note', type=str, default='exp')
    parser.add_argument('--exp_group', type=str, default='exp', help='name of the experment')
    parser.add_argument('--path', type=str, default=None, help='the corresponding exps dir')
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--train_epochs', type=int, default=580)
    parser.add_argument('--data_mode', type=str, default='supervised')

    # data
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--label_smooth', type=float, default=0.0, help='label smoothing')

    # network
    parser.add_argument('--arch_name', type=str, default='highest_acc')
    parser.add_argument('--init_channels', type=int, default=36)
    parser.add_argument('--drop_path_prob', type=float, default=0.3)
    parser.add_argument('--auxiliary', action='store_true', default=True)
    parser.add_argument('--auxiliary_weight', type=float, default=0.4)
    parser.add_argument('--affine', action='store_true', default=True)
    parser.add_argument('--fix', action='store_true', default=False)

    # optim
    parser.add_argument('--optm', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--grad_clip', type=float, default=5.0)

    # others
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--report_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=5)
    parser.add_argument('--local_rank', type=int, default=-1)

    # boost
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--amp', action='store_true')

    return parser.parse_args()


def main(args, arch, searched_dict):

    args, writer = utils.init_exp('finetune', args)

    with open(os.path.join(args.save_path, 'net.config'), 'a') as f:
        f.write('{}:'.format('model') + str(arch) + '\n')

    if args.ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda")

    # model
    model = Network(args.init_channels, args.num_classes, arch, args.auxiliary, args.affine).to(device)
    model.load_weight(searched_dict)

    if args.fix:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    if args.ddp:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # loss
    if args.label_smooth != 0:
        train_criterion = utils.CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)
    else:
        train_criterion = nn.CrossEntropyLoss()
    valid_criterion = nn.CrossEntropyLoss()

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
        train_queue, test_queue = data_provider.build_loaders_train(args.train_ratio, args.batch_size, args.num_workers)

    # schedulers
    linear_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda iter:(iter / (args.warmup_epochs * len(train_queue))))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.train_epochs * len(train_queue)), eta_min=args.lr_min)
    valid_stamps = set(range(0, args.warmup_epochs + args.train_epochs, args.valid_freq)) | set(range(500, args.warmup_epochs + args.train_epochs))

    scaler = GradScaler()

    best_top1 = 0
    best_top5 = 0

    for epoch in range(args.warmup_epochs + args.train_epochs):

        start = time.time()
        logging.info('epoch %d', epoch)

        if args.fix and epoch == args.warmup_epochs:
            if args.ddp:
                for param in model.module.parameters():
                    param.requires_grad = True
            else:
                for param in model.parameters():
                    param.requires_grad = True

        if args.ddp:
            model.module.drop_path_prob = args.drop_path_prob * epoch / (args.warmup_epochs + args.train_epochs)
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / (args.warmup_epochs + args.train_epochs)

        if epoch < args.warmup_epochs:
            scheduler = linear_scheduler
        else:
            scheduler = cosine_scheduler

        if args.ddp:
            train_acc, train_obj = ddp_train(args, train_queue, model, train_criterion, optimizer, epoch, writer, device, scaler, scheduler)
        else:
            train_acc, train_obj = train(args, train_queue, model, train_criterion, optimizer, epoch, writer, scheduler)
        
        logging.info('train_acc %f, train_loss %e', train_acc, train_obj)

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
    
    logging.info('best_top1 %f, best_top5', best_top1, best_top5)


if __name__ == '__main__':
    args = parse_args()
    arch_dict = utils.get_arch_dict(args.path, args.arch_name)
    note = args.note

    for weight_fname, arch in arch_dict.items():
        searched_dict = torch.load(os.path.join(args.path, weight_fname + '.pth.tar'))
        args.note = '_'.join([note, weight_fname])
        print('finetuning: ' + weight_fname)
        main(args, arch, searched_dict)
