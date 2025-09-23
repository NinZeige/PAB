from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from dataset.search_dataset import search_train_dataset, search_test_dataset
from collections.abc import Callable


def create_dataset(config: dict, preprocess: Callable | None, evaluate=False):
    dataset = None
    if evaluate:
        dataset = search_test_dataset(config, preprocess)
    else:
        dataset = search_train_dataset(config, preprocess)
    return dataset


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
    batch_size: int,
    sampler: Optional[Sampler] = None,
    num_worker: int = 1,
    is_train: bool = False,
    collate_fn: Callable | None = None,
):
    assert isinstance(dataset, Dataset)  # 防止忘记处理 `create_dataset` 返回列表
    shuffle = (sampler is not None) and is_train
    drop_last = is_train

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    return loader
