from random import shuffle
import torch
import os
import math
import numpy as np
from utils import get_split_list
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, DistributedSampler
from utils.data_aug import get_transforms


class DataProvider:
    def __init__(self, args):
        
        train_transform, test_transform = get_transforms(args)

        if args.dataset == 'cifar10':
            self.n_classes = 10
            self.train_dataset = datasets.CIFAR10(args.data_path, train=True, download=True, transform=train_transform)
            self.test_dataset = datasets.CIFAR10(args.data_path, train=False, download=True, transform=test_transform)
        elif args.dataset == 'cifar100':
            self.n_classes = 100
            self.train_dataset = datasets.CIFAR100(args.data_path, train=True, download=True, transform=train_transform)
            self.test_dataset = datasets.CIFAR100(args.data_path, train=False, download=True, transform=test_transform)
        elif args.dataset == 'imagenet-tiny':
            self.n_classes = 200
            self.train_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=train_transform)
            self.test_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=test_transform)
        elif args.dataset == 'imagenet':
            self.n_classes = 1000
            self.train_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=train_transform)
            self.test_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=test_transform)
        else:
            raise NotImplementedError
        
    def build_loaders_train_ddp(self, batch_size, num_workers):
        train_sampler = DistributedSampler(self.train_dataset)
        test_sampler = SequentialDistributedSampler(self.test_dataset, batch_size=batch_size)
        train_loader = DataLoader(self.train_dataset, batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(self.test_dataset, batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)
        return train_loader, test_loader

    def build_loaders_train_imagenet(self, batch_size, num_workers):
        train_loader = DataLoader(self.train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(self.test_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, test_loader

    def build_loaders_train(self, train_ratio, batch_size, num_workers):
        valid_size = int(len(self.train_dataset) - train_ratio * len(self.train_dataset))
        (train_indexes, _), test_indexes = self.random_sample_valid_set(self.train_dataset.targets, valid_size, self.n_classes), list(range(len(self.test_dataset)))
        train_sampler, test_sampler = SubsetRandomSampler(train_indexes), SubsetRandomSampler(test_indexes)
        train_loader = DataLoader(self.train_dataset, batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(self.test_dataset, batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)
        return train_loader, test_loader

    def build_loaders_search(self, train_ratio, batch_size, num_workers, debug):
        valid_size = int(len(self.train_dataset) - train_ratio * len(self.train_dataset))
        (train_indexes, valid_indexes), test_indexes = self.random_sample_valid_set(self.train_dataset.targets, valid_size, self.n_classes), list(range(len(self.test_dataset)))

        if debug:
            train_indexes, valid_indexes, test_indexes = train_indexes[:200], valid_indexes[:200], test_indexes[:200]

        train_sampler, valid_sampler, test_sampler = SubsetRandomSampler(train_indexes), SubsetRandomSampler(valid_indexes), SubsetRandomSampler(test_indexes)
        train_loader = DataLoader(self.train_dataset, batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        valid_loader = DataLoader(self.train_dataset, batch_size, sampler=valid_sampler, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(self.test_dataset, batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)
        return train_loader, valid_loader, test_loader

    @staticmethod
    def random_sample_valid_set(train_labels, valid_size, n_classes):
        train_size = len(train_labels)
        assert train_size > valid_size

        g = torch.Generator()
        g.manual_seed(0)  # set random seed before sampling validation set
        rand_indexes = torch.randperm(train_size, generator=g).tolist()

        train_indexes, valid_indexes = [], []
        per_class_remain = get_split_list(valid_size, n_classes)

        for idx in rand_indexes:
            label = train_labels[idx]
            if isinstance(label, float):
                label = int(label)
            elif isinstance(label, np.ndarray):
                label = np.argmax(label)
            else:
                assert isinstance(label, int)
            if per_class_remain[label] > 0:
                valid_indexes.append(idx)
                per_class_remain[label] -= 1
            else:
                train_indexes.append(idx)
        return train_indexes, valid_indexes


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
