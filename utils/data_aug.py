import torch
import numpy as np
from torchvision.transforms import transforms


class ContrastiveLearningViewGenerator(object):
    """ Take two random crops of one image as query and key. """

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
    
    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def get_transforms(args):

    if args.dataset == 'cifar10':
        image_size, mean, std = 32, [0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]
    elif args.dataset == 'cifar100':
        image_size, mean, std = 32, [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    elif args.dataset == 'imagenet-tiny':
        image_size, mean, std = 64, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif args.dataset == 'imagenet':
        image_size, mean, std = 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise NotImplementedError
    
    if args.data_mode == 'SimCLR':
        data_transforms = transforms.Compose([transforms.RandomResizedCrop((image_size, image_size)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p = 0.3),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ToTensor(),
                                              transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 2.0))], p = 0.2),
                                              transforms.Normalize(mean, std)])
        
        train_transform, test_transform = ContrastiveLearningViewGenerator(data_transforms), ContrastiveLearningViewGenerator(data_transforms)
    elif args.data_mode == 'supervised' and args.dataset in ['imagenet', 'imagenet-tiny']:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if args.dataset == 'imagenet-tiny':
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        if args.dataset == 'imagenet':
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if args.cutout:
            train_transform.transforms.append(Cutout(args.cutout_length))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return train_transform, test_transform
