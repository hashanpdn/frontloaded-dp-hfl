"""
Dataset utilities for MNIST and CIFAR-10.

This module downloads datasets (if needed), partitions them across clients
under multiple schemes (IID equal-size, IID non-equal-size, non-IID label
shards, and one-class), and returns per-client DataLoaders for training and
testing. It also provides helpers for dataset statistics and per-client class
distributions.
"""

import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from options import args_parser


class DatasetSplit(Dataset):
    """Expose a subset of a dataset using a list of indices."""

    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, target = self.dataset[self.idxs[item]]
        return image, target


def gen_ran_sum(_sum, num_users):
    """Randomly split integer `_sum` into `num_users` positive parts (Dirichlet–Multinomial)."""
    base = 100 * np.ones(num_users, dtype=np.int32)
    _sum = _sum - 100 * num_users
    p = np.random.dirichlet(np.ones(num_users), size=1)[0]
    size_users = np.random.multinomial(_sum, p, size=1)[0] + base
    return size_users


def get_mean_and_std(dataset):
    """Compute per-channel mean and std for a vision dataset."""
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("=>compute mean and std")
    for inputs, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def iid_esize_split(dataset, args, kwargs, is_shuffle=True):
    """
    IID partition with equal samples per client.

    Returns
    -------
    list[DataLoader]
    """
    sum_samples = len(dataset)
    num_samples_per_client = int(sum_samples / args.num_clients)
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(args.num_clients):
        dict_users[i] = np.random.choice(all_idxs, num_samples_per_client, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(
            DatasetSplit(dataset, dict_users[i]),
            batch_size=args.batch_size,
            shuffle=is_shuffle,
            **kwargs
        )
    return data_loaders


def iid_nesize_split(dataset, args, kwargs, is_shuffle=True):
    """
    IID partition with non-equal client sizes (imbalanced counts).
    Label distribution is IID; sample counts differ per client.
    """
    sum_samples = len(dataset)
    num_samples_per_client = gen_ran_sum(sum_samples, args.num_clients)
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for (i, num_samples_client) in enumerate(num_samples_per_client):
        dict_users[i] = np.random.choice(all_idxs, num_samples_client, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(
            DatasetSplit(dataset, dict_users[i]),
            batch_size=args.batch_size,
            shuffle=is_shuffle,
            **kwargs
        )
    return data_loaders


def niid_esize_split(dataset, args, kwargs, is_shuffle=True):
    """
    Non-IID label-shard split: each client receives two label shards.
    """
    data_loaders = [0] * args.num_clients
    num_shards = 2 * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)

    labels = dataset.targets if is_shuffle else dataset.targets
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].astype(int)

    for i in range(args.num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0
            ).astype(int)
        data_loaders[i] = DataLoader(
            DatasetSplit(dataset, dict_users[i]),
            batch_size=args.batch_size,
            shuffle=is_shuffle,
            **kwargs
        )
    return data_loaders


def niid_esize_split_train(dataset, args, kwargs, is_shuffle=True):
    """
    Non-IID (K classes per client) for training; returns loaders and shard pattern.

    Returns
    -------
    list[DataLoader], dict
    """
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].astype(int)

    split_pattern = {i: [] for i in range(args.num_clients)}
    for i in range(args.num_clients):
        rand_set = np.random.choice(idx_shard, 2, replace=False)
        split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0
            ).astype(int)
        data_loaders[i] = DataLoader(
            DatasetSplit(dataset, dict_users[i]),
            batch_size=args.batch_size,
            shuffle=is_shuffle,
            **kwargs
        )
    return data_loaders, split_pattern


def niid_esize_split_test(dataset, args, kwargs, split_pattern, is_shuffle=False):
    """Apply the recorded training shard pattern to the test set."""
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].astype(int)

    for i in range(args.num_clients):
        rand_set = split_pattern[i][0]
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0
            ).astype(int)
        data_loaders[i] = DataLoader(
            DatasetSplit(dataset, dict_users[i]),
            batch_size=args.batch_size,
            shuffle=is_shuffle,
            **kwargs
        )
    return data_loaders, None


def niid_esize_split_train_large(dataset, args, kwargs, is_shuffle=True):
    """Non-IID (K classes) variant that also records class IDs for later test split."""
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].astype(int)

    split_pattern = {i: [] for i in range(args.num_clients)}
    for i in range(args.num_clients):
        rand_set = np.random.choice(idx_shard, 2, replace=False)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0
            ).astype(int)
            split_pattern[i].append(dataset.__getitem__(idxs[rand * num_imgs])[1])
        data_loaders[i] = DataLoader(
            DatasetSplit(dataset, dict_users[i]),
            batch_size=args.batch_size,
            shuffle=is_shuffle,
            **kwargs
        )
    return data_loaders, split_pattern


def niid_esize_split_test_large(dataset, args, kwargs, split_pattern, is_shuffle=False):
    """Create non-IID test splits using recorded class IDs (large-split variant)."""
    data_loaders = [0] * args.num_clients
    num_shards = 10  # CIFAR-10 / MNIST
    num_imgs = int(len(dataset) / num_shards)
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(len(dataset))
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].astype(int)

    for i in range(args.num_clients):
        rand_set = split_pattern[i]
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0
            ).astype(int)
        data_loaders[i] = DataLoader(
            DatasetSplit(dataset, dict_users[i]),
            batch_size=args.batch_size,
            shuffle=is_shuffle,
            **kwargs
        )
    return data_loaders, None


def niid_esize_split_oneclass(dataset, args, kwargs, is_shuffle=True):
    """Non-IID one-class split: each client receives a single class."""
    data_loaders = [0] * args.num_clients
    num_shards = args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels if is_shuffle else dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].astype(int)

    for i in range(args.num_clients):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0
            ).astype(int)
        data_loaders[i] = DataLoader(
            DatasetSplit(dataset, dict_users[i]),
            batch_size=args.batch_size,
            shuffle=is_shuffle,
            **kwargs
        )
    return data_loaders


def split_data(dataset, args, kwargs, is_shuffle=True):
    """
    Dispatch to a partitioning scheme based on `args.iid`.

    1: IID equal-size; 0: non-IID label shards; -1: IID labels, non-equal sizes; -2: one-class.
    """
    if args.iid == 1:
        data_loaders = iid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == 0:
        data_loaders = niid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == -1:
        data_loaders = iid_nesize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == -2:
        data_loaders = niid_esize_split_oneclass(dataset, args, kwargs, is_shuffle)
    else:
        raise ValueError('Data Distribution pattern `{}` not implemented '.format(args.iid))
    return data_loaders


def get_dataset(dataset_root, dataset, args):
    """Return per-client loaders and global validation loaders for the chosen dataset."""
    if dataset == 'mnist':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_mnist(dataset_root, args)
    elif dataset == 'cifar10':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_cifar10(dataset_root, args)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_mnist(dataset_root, args):
    """Download/prepare MNIST and build client splits + validation loaders."""
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=True,
                           download=True, transform=transform)
    test = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=False,
                          download=True, transform=transform)
    train_loaders = split_data(train, args, kwargs, is_shuffle=True)
    test_loaders = split_data(test, args, kwargs, is_shuffle=False)
    v_train_loader = DataLoader(train, batch_size=args.batch_size * args.num_clients,
                                shuffle=True, **kwargs)
    v_test_loader = DataLoader(test, batch_size=args.batch_size * args.num_clients,
                               shuffle=False, **kwargs)
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_cifar10(dataset_root, args):
    """Download/prepare CIFAR-10 and build client splits + validation loaders."""
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
    if args.model == 'cnn_complex_2':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.model == 'cnn_complex_3':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError("this nn for cifar10 not implemented")

    train = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=True,
                             download=True, transform=transform_train)
    test = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=False,
                            download=True, transform=transform_test)
    v_train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)
    v_test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)
    train_loaders = split_data(train, args, kwargs, is_shuffle=True)
    test_loaders = split_data(test, args, kwargs, is_shuffle=False)
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def show_distribution(dataloader, args):
    """
    Compute the label distribution for a client dataloader.

    Returns
    -------
    np.ndarray: proportions per class (sum to 1.0).
    """
    if args.dataset == 'mnist':
        try:
            labels = dataloader.dataset.dataset.train_labels.numpy()
        except:
            print("Using test_labels")
            labels = dataloader.dataset.dataset.test_labels.numpy()
    elif args.dataset == 'cifar10':
        try:
            labels = dataloader.dataset.dataset.train_labels
        except:
            print("Using test_labels")
            labels = dataloader.dataset.dataset.test_labels
    else:
        raise ValueError("`{}` dataset not included".format(args.dataset))

    num_samples = len(dataloader.dataset)
    idxs = [i for i in range(num_samples)]
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    distribution = [0] * len(unique_labels)
    for idx in idxs:
        _, label = dataloader.dataset[idx]
        distribution[label] += 1
    distribution = np.array(distribution) / num_samples
    return distribution


if __name__ == '__main__':
    # Smoke test for dataset construction and per-client distributions.
    args = args_parser()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loaders, test_loaders, _, _ = get_dataset(args.dataset_root, args.dataset, args)
    print(f"The dataset is {args.dataset} divided into {args.num_clients} clients/tasks in an iid = {args.iid} way")
    for i in range(args.num_clients):
        train_loader = train_loaders[i]
        print(len(train_loader.dataset))
        distribution = show_distribution(train_loader, args)
        print("dataloader {} distribution".format(i))
        print(distribution)
