import os
import sys
import torch
import random
import logging
import numpy as np
from tensorboardX import SummaryWriter
from utils.pytorch_utils import *
from genotypes import DARTS, PARTS
from utils.mixup import Mixup


def init_exp(stage, args):
    assert torch.cuda.is_available(), print('No gpu')

    if args.local_rank != -1:
        args.seed = args.seed + args.local_rank

    args.save_path = './exps/{}/{}-{}'.format(args.exp_group, stage, args.note)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    print('Experiment dir : {}'.format(args.save_path))

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(os.path.join(args.save_path))
    else:
        writer = None

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if stage == 'search':
        if args.prune_direct is not None:
            args.progressive = True
        else:
            args.progressive = False
      
        args.warmup_epochs, args.search_epochs, args.prune_epochs, args.stabilize_epochs = [int(float(ratio) * args.epochs) for ratio in args.epoch_ratios.split(',')]
        if args.warmup_epochs < args.min_warmup:
            gap = args.min_warmup - args.warmup_epochs
            args.warmup_epochs += gap
            args.search_epochs -= gap

        if not args.progressive:
            args.search_epochs = args.search_epochs + args.prune_epochs + args.stabilize_epochs
            args.prune_epochs = 0
            args.stabilize_epochs = 0
    
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'imagenet-tiny':
        args.num_classes = 200
    elif args.dataset == 'imagenet':
        args.num_classes = 1000
    else:
        raise NotImplementedError
    
    logging.info(args)

    return args, writer


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def get_update_schedule_grad(nBatch, update_arch_every, arch_update_steps):
    schedule = {}
    for i in range(nBatch):
        if (i+1) % update_arch_every == 0:
            schedule[i] = arch_update_steps
    return schedule


def get_split_list(in_dim, child_num):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    return in_dim_list


def get_arch_dict(path, arch_name):

    if path is not None:
        arch_name = arch_name.split(',')
        arch_meta_dict = {}

        with(open(os.path.join(path, 'net.config'), 'r')) as f:
            lines = f.readlines()
            for line in lines:
                name, structure = line.split(':')[0], line.split(':')[-1]
                arch_meta_dict[name] = eval(structure.strip())
        # get target arch
        assert type(arch_name) == list, print('wrong arch_name type')
        arch_dict = {}
        for name in arch_name:
            if name in arch_meta_dict.keys():
                arch_dict[name] = arch_meta_dict[name]
    else:
        arch_dict = {}
        if arch_name == 'darts':
            arch_dict[arch_name] = DARTS
        elif arch_name == 'pdarts':
            arch_dict[arch_name] = PARTS

    assert len(arch_dict.keys()) > 0, print('no matched arch_name')

    return arch_dict
