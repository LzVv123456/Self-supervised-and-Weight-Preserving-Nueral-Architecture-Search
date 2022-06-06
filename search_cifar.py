import torch
import utils
import time
import logging
import argparse
from datetime import timedelta
import numpy as np
import torch.nn as nn
from model_search import Network
from data_provider import DataProvider
from byol_pytorch import BYOL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--note', type=str, default='exp', help='name of the experment')
    parser.add_argument('--exp_group', type=str, default='exp', help='name of the experment')
    parser.add_argument('--data_mode', type=str, default='SimCLR', choices=['supervised', 'SimCLR', 'byol'])

    # epochs for stages
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--epoch_ratios', type=str, default='0.2,0.4,0.4,0.0', help='epoch ratio for each phase')
    parser.add_argument('--min_warmup', type=int, default=40, help='minimum warmup epoch')

    # data
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--num_workers', type=int, default=8)

    # network
    parser.add_argument('--layers', type=int, default=20)
    parser.add_argument('--init_channels', type=int, default=36)
    parser.add_argument('--net_mode', type=str, default='two', choices=['full_v2', 'two'])
    parser.add_argument('--prune_direct', type=str, default='forward', choices=['forward', 'backward'])
    parser.add_argument('--affine', action='store_true', default=False, help='add affine for batchnorm in the network')
    parser.add_argument('--weight_skip_drop_prob', type=float, default=0.2, help='prob to drop unconv ops when weight sampling')
    parser.add_argument('--weight_pool_drop_prob', type=float, default=0, help='prob to drop unconv ops when weight sampling')
    parser.add_argument('--arch_skip_drop_prob', type=float, default=0, help='prob to drop unconv ops when arch sampling')
    parser.add_argument('--arch_pool_drop_prob', type=float, default=0, help='prob to drop unconv ops when arch sampling')
    parser.add_argument('--skip_r_num', type=int, default=2, help='maximum num of skip connection in a cell')
    parser.add_argument('--pool_r_num', type=int, default=2, help='maximum num of pooling in a cell')
    parser.add_argument('--compatible_mode', action='store_true', default=False, help='stop dropout during fpp')

    # optm
    parser.add_argument('--weight_optm', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--weight_decay', type=float, default=4e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--arch_lr', type=float, default=1e-3)
    parser.add_argument('--arch_weight_decay', type=float, default=0)
    parser.add_argument('--arch_adam_beta1', type=float, default=0)
    parser.add_argument('--arch_adam_beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--update_arch_every', type=int, default=5)
    parser.add_argument('--arch_update_steps', type=int, default=1)

    # miscellaneous
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--report_freq', type=int, default=40)
    parser.add_argument('--valid_freq', type=int, default=5)
    parser.add_argument('--local_rank', type=int, default=-1)

    return parser.parse_args()


def main(args):
    args, writer = utils.init_exp('search', args)

    model = Network(args.init_channels, args.num_classes, args.layers, args.net_mode, \
    args.data_mode, args.prune_direct, args.affine, args.skip_r_num, args.pool_r_num).cuda()
    if args.data_mode == 'byol':
        model = BYOL(model, 32, hidden_layer='globalpooling')
    model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()

    # optimizers
    if args.weight_optm == 'adam':
        weight_optimizer = torch.optim.Adam(model.module.weight_params(), args.lr, weight_decay=args.weight_decay)
    elif args.weight_optm == 'sgd':
        weight_optimizer = torch.optim.SGD(model.module.weight_params(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise NotImplementedError
    
    arch_optimizer = torch.optim.Adam(model.module.alpha_params(), lr=args.arch_lr, \
    weight_decay=args.arch_weight_decay, betas=(args.arch_adam_beta1, args.arch_adam_beta2))

    # data_provider
    data_provider = DataProvider(args)
    train_queue, valid_queue, test_queue = data_provider.build_loaders_search(args.train_ratio, args.batch_size, args.num_workers, args.debug)

    # schedulers
    weight_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, T_max=args.epochs)
    arch_update_scheduler = utils.get_update_schedule_grad(len(train_queue), args.update_arch_every, args.arch_update_steps)

    finish_prune = args.warmup_epochs + args.search_epochs + args.prune_epochs
    save_stamps = set({finish_prune, finish_prune + args.stabilize_epochs // 3, finish_prune + 2 * args.stabilize_epochs // 3, finish_prune + args.stabilize_epochs})

    if args.progressive:
        valid_stamps = set(range(finish_prune, args.epochs, args.valid_freq)) | save_stamps
    else:
        valid_stamps = set(range(args.warmup_epochs, args.epochs, args.valid_freq)) | save_stamps

    highest_acc = 0
    lowest_loss = np.inf

    for epoch in range(args.epochs):
        start = time.time()
        logging.info('epoch %d', epoch)

        train_acc, train_obj = train(args, model, train_queue, valid_queue, criterion, weight_optimizer, arch_optimizer, epoch, arch_update_scheduler, writer)
        logging.info('train_acc %f, train_loss %e', train_acc, train_obj)

        weight_lr_scheduler.step()

        if epoch in valid_stamps:
            valid_acc, valid_obj = infer(args, model, test_queue, criterion)
            logging.info('train_valid_acc %f, train_valid_loss %e', valid_acc, valid_obj)
            writer.add_scalar('train/valid_acc', valid_acc, epoch)
            writer.add_scalar('train/valid_loss', valid_obj, epoch)

            if valid_acc > highest_acc:
                model.module.save_arch_weights(args.save_path, 'highest_acc')
                highest_acc = valid_acc

            if valid_obj < lowest_loss:
                model.module.save_arch_weights(args.save_path, 'lowest_loss')
                lowest_loss = valid_obj

        if epoch in save_stamps or epoch==args.epochs-1:
            model.module.save_arch_weights(args.save_path, epoch)
                
        if epoch > args.warmup_epochs:
            model.module.get_cells_arch(epoch, add_restrict=False)
            model.module.get_cells_arch(epoch, add_restrict=True)

        time_per_epoch = time.time() - start
        seconds_left = int((args.epochs - epoch - 1) * time_per_epoch)
        print('Time per epoch: %s, Est. complete in: %s' % (str(timedelta(seconds=time_per_epoch)), str(timedelta(seconds=seconds_left))))
        print('--' * 60)


def train(args, model, train_queue, valid_queue, criterion, weight_optimizer, arch_optimizer, epoch, arch_update_scheduler, writer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    valid_iter = None

    if args.progressive:
        model.module.progressive_prune(args, epoch)

    if args.compatible_mode and epoch >= args.warmup_epochs + args.search_epochs - 1:
        args.weight_skip_drop_prob = 0.0
        args.weight_pool_drop_prob = 0.0
    
    for step, (images, labels) in enumerate(train_queue):
        iter_num = epoch * len(train_queue) + step
        model.module.reset_binary_gates(args.weight_skip_drop_prob, args.weight_pool_drop_prob, warmup=(epoch <= args.warmup_epochs))  # sample random path
        model.module.unused_ops_off()  # turn off unused ops

        if args.data_mode == 'supervised':
            images, labels = images.cuda(), labels.cuda()
            logits = model(images)
            loss = criterion(logits, labels)
        elif args.data_mode == 'SimCLR':
            images = torch.cat(images, dim=0)
            images = images.cuda()
            features = model(images)
            logits, labels = utils.info_nce_loss(features)
            loss = criterion(logits, labels)
        elif args.data_mode == 'byol':
            images = images.cuda()
            loss = model(images)
        else:
            raise NotImplementedError

        if args.data_mode == 'byol':
            objs.update(loss.item(), images.size(0))
        else:
            n = logits.size(0)
            prec1, prec5 = utils.accuracy(logits, labels, topk=(1, 5))
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            objs.update(loss.item(), n)

        iter_num = step + epoch * len(train_queue)
        writer.add_scalar('train/loss', loss.item(), iter_num)

        if args.data_mode != 'byol':
            writer.add_scalar('train/acc', prec1.item(), iter_num)

        weight_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.module.weight_params(), args.grad_clip)
        weight_optimizer.step()

        model.module.unused_ops_back()  # get unused operations back
            
        if epoch > args.warmup_epochs:
            for j in range(arch_update_scheduler.get(step, 0)):
                model.train()

                if valid_iter is None:
                    valid_iter = iter(valid_queue)
                try:
                    images_valid, labels_valid = next(valid_iter)
                except StopIteration:
                    valid_iter = iter(valid_queue)
                    images_valid, labels_valid = next(valid_iter)

                model.module.reset_binary_gates(args.arch_skip_drop_prob, args.arch_pool_drop_prob, warmup=(epoch <= args.warmup_epochs))  # sample random path
                model.module.unused_ops_off()

                if args.data_mode == 'supervised':
                    images_valid, labels_valid = images_valid.cuda(), labels_valid.cuda()
                    logits = model(images_valid)
                    loss = criterion(logits, labels_valid)
                elif args.data_mode == 'SimCLR':
                    images_valid = torch.cat(images_valid, dim=0)
                    images_valid = images_valid.cuda()
                    features = model(images_valid)
                    logits, labels = utils.info_nce_loss(features)
                    loss = criterion(logits, labels)
                elif args.data_mode == 'byol':
                    images = images.cuda()
                    loss = model(images)
                else:
                    raise NotImplementedError

                arch_optimizer.zero_grad()  # reset the grads of weight and arch params
                loss.backward()
            
                model.module.set_arch_param_grad()  # compute the grads of arch params

                arch_optimizer.step()

                model.module.rescale_updated_arch_param()  # rescale the arch params
                model.module.unused_ops_back()  # get unused operations back

        if step % args.report_freq == 0 or step == len(train_queue) - 1:
            if args.data_mode != 'byol':
                logging.info('train %05d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            else:
                logging.info('train %05d %e', step, objs.avg)

    return top1.avg, objs.avg


def infer(args, model, valid_queue, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.module.set_chosen_op_active()  # select one path according to current arch params
    model.module.unused_ops_off()  # mask unselected paths
    model.eval()

    with torch.no_grad():
        for step, (images, labels) in enumerate(valid_queue):

            if args.data_mode == 'supervised':
                images, labels = images.cuda(), labels.cuda()
                logits = model(images)
                loss = criterion(logits, labels)
            elif args.data_mode == 'SimCLR':
                images = torch.cat(images, dim=0)
                images = images.cuda()
                features = model(images)
                logits, labels = utils.info_nce_loss(features)
                loss = criterion(logits, labels)
            elif args.data_mode == 'byol':
                images = images.cuda()
                loss = model(images)
            else:
                raise NotImplementedError

            if args.data_mode == 'byol':
                objs.update(loss.item(), images.size(0))
            else:
                n = logits.size(0)
                prec1, prec5 = utils.accuracy(logits, labels, topk=(1, 5))
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)
                objs.update(loss.item(), n)

            if step % args.report_freq == 0 or step == len(valid_queue) - 1:
                if args.data_mode != 'byol':
                    logging.info('valid %05d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                else:
                    logging.info('valid %05d %e', step, objs.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    args = parse_args()
    main(args)
