import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.search_dataset import search_train_dataset, search_test_dataset
from torch.utils.data import Dataset
from collections.abc import Callable


def create_dataset(config, preprocess: Callable | None, evaluate=False):
    train_transform = preprocess
    test_transform = train_transform

    test_dataset = search_test_dataset(config, test_transform)
    if evaluate:
        return None, test_dataset

    train_dataset = search_train_dataset(config, train_transform)

    return train_dataset, test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(
    dataset: Dataset,
    sampler: bool | None,
    batch_size: int,
    num_worker: int,
    is_train: bool,
    collate_fn: Callable | None,
):
    if is_train:
        shuffle = sampler is None
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        pin_memory=True,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    return loader
