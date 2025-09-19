import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.search_dataset import search_train_dataset, search_test_dataset


def create_dataset(config, preprocess, evaluate=False):

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    train_transform = preprocess

    # model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="./checkpoints/mobileclip2_s4.pt", **model_kwargs)
    # tokenizer = open_clip.get_tokenizer(model_name)

    # # Model needs to be in eval mode for inference because of batchnorm layers unlike ViTs
    # model.eval()

    # # For inference/model exporting purposes, please reparameterize first
    # model = reparameterize_model(model)

    # normal_image = preprocess(Image.open("test/pict.jpg").convert("RGB")).unsqueeze(0)
    # abnoml_image = preprocess(Image.open("test/bad_pict1.jpg").convert("RGB")).unsqueeze(0)

    test_transform = preprocess

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


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders: list[DataLoader] = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(
            f"### be careful: func create_loader returns a list length of {len(loaders)}"
        )

    return loaders
