import os
import time
import utils
import torch
import logging
import argparse
import torch.nn as nn
from datetime import timedelta
from model_pretrain import Network
from data_provider import DataProvider


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--note', type=str, default='exp')
    parser.add_argument('--exp_group', type=str, default='exp', help='name of the experment')
    parser.add_argument('--path', type=str, default=None, help='the corresponding exps dir')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_mode', type=str, default='SimCLR')

    # data
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')

    # network
    parser.add_argument('--arch_name', type=str, default='highest_acc')
    parser.add_argument('--init_channels', type=int, default=36)
    parser.add_argument('--auxiliary', action='store_true', default=False)
    parser.add_argument('--affine', action='store_true', default=False)
    parser.add_argument('--preserve_weight', action='store_true', default=False)

    # optim
    parser.add_argument('--optm', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--lr_min', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--grad_clip', type=float, default=5.0)

    # others
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--report_freq', type=int, default=40)
    parser.add_argument('--valid_freq', type=int, default=5)
    parser.add_argument('--local_rank', type=int, default=-1)
    
    return parser.parse_args()


def main(args, arch, searched_dict):
    
    args, writer = utils.init_exp('pretrain', args)

    with open(os.path.join(args.save_path, 'net.config'), 'a') as f:
        f.write('{}:'.format('pretrained') + str(arch) + '\n')

    # model
    model = Network(args.init_channels, args.num_classes, arch, args.auxiliary, args.affine).cuda()
    model.drop_path_prob = 0.0
    if args.preserve_weight:
        model.load_weight(searched_dict)

    # loss
    criterion = nn.CrossEntropyLoss()

    # optimizers
    if args.optm == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optm == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise NotImplementedError

    # data_provider
    data_provider = DataProvider(args)
    train_queue, test_queue = data_provider.build_loaders_train(args.train_ratio, args.batch_size, args.num_workers)

    # schedulers
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs * len(train_queue)), eta_min=args.lr_min)
    valid_stamps = set(range(0, args.epochs, args.valid_freq)) | set(range(500, args.epochs))
    save_stamps = [i for i in range(0, args.epochs, 50)] + [args.epochs-1]

    best_acc = 0

    for epoch in range(args.epochs):

        if epoch in save_stamps:
            with open(os.path.join(args.save_path, 'net.config'), 'a') as f:
                f.write('Epoch_{}:'.format(epoch) + str(arch) + '\n')
            torch.save(model.state_dict(), os.path.join(args.save_path, 'Epoch_{}.pth.tar'.format(epoch)))
        
        start = time.time()
        logging.info('epoch %d', epoch)

        train_acc, train_obj = train(args, train_queue, model, criterion, optimizer, epoch, writer, scheduler)
        logging.info('train_acc %f, train_loss %e', train_acc, train_obj)

        if epoch in valid_stamps:
            valid_acc, valid_obj = infer(args, test_queue, model, criterion)
            logging.info('valid_acc %f, best_acc %f, valid_loss %e', valid_acc, best_acc, valid_obj)

            writer.add_scalar('train/valid_acc', valid_acc, epoch)
            writer.add_scalar('train/valid_loss', valid_obj, epoch)

            if valid_acc > best_acc:
                best_acc = valid_acc
                with open(os.path.join(args.save_path, 'net.config'), 'a') as f:
                    f.write('Epoch_{}:'.format(epoch) + str(arch) + '\n')
                torch.save(model.state_dict(), os.path.join(args.save_path, 'pretrained.pth.tar'))
            
        time_per_epoch = time.time() - start
        seconds_left = int((args.epochs - epoch - 1) * time_per_epoch)
        logging.info('Time per epoch: %s, Est. complete in: %s' % (str(timedelta(seconds=time_per_epoch)), str(timedelta(seconds=seconds_left))))
        logging.info('--' * 60)
    
    logging.info('best_acc %f', best_acc)
    torch.save(model.state_dict(), os.path.join(args.save_path, 'last_pretrain.pth.tar'))
    # stop current logging
    logging.shutdown()


def train(args, train_queue, model, criterion, optimizer, epoch, writer, scheduler):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (images, labels) in enumerate(train_queue):

        images = torch.cat(images, dim=0)
        images = images.cuda()
        features, _ = model(images)
        logits, labels = utils.info_nce_loss(features)
        loss = criterion(logits, labels)

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

            images = torch.cat(images, dim=0)
            images = images.cuda()
            features, _ = model(images)
            logits, labels = utils.info_nce_loss(features)
            loss = criterion(logits, labels)

            prec1, prec5 = utils.accuracy(logits, labels, topk=(1, 5))
            n = logits.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0 or step == len(valid_queue) - 1:
                logging.info('valid %05d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    args = parse_args()
    arch_dict = utils.get_arch_dict(args.path, args.arch_name)
    note = args.note

    for weight_fname, arch in arch_dict.items():
        searched_dict = torch.load(os.path.join(args.path, weight_fname + '.pth.tar'))
        args.note = '_'.join([note, weight_fname])
        print('SimCLR pretraining: ' + weight_fname)
        main(args, arch, searched_dict)
