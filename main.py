import yaml
from rich.table import Table, Column
from rich.console import Console
from rich.progress import track

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import Siglip2Model

from dataset import create_dataset, create_loader
from evaluate import evaluate_once
from models.siglip2 import build_model, make_eval_collate_fn, make_train_collate_fn


def rich_table_setup():
    # Pretty Print
    t = Table(
        *map(lambda x: Column(x, justify='center'), ['R1', 'R5', 'R10', 'mAP', 'mINP']),
        title='Evaluation Result',
    )
    c = Console()
    return t, c


def evaluate_main(cfg: dict[str, str | list[str]]):
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, processor, tokenizer = build_model(dev)
    model.eval()

    _, test_dataset = create_dataset(cfg, None, evaluate=True)
    loader = create_loader(
        test_dataset,
        batch_size=cfg['batch_size_test'],
        collate_fn=make_eval_collate_fn(processor, tokenizer, cfg['max_words']),
    )

    t, c = rich_table_setup()
    evaluate_once(model, loader, t, c)


def train_main(cfg: dict[str, int | str | list[str]]):
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model, processor, tokenizer = build_model(dev)
    model.train()

    train_dataset, test_dataset = create_dataset(cfg, None)
    train_loader = create_loader(
        train_dataset,
        batch_size=cfg['batch_size_train'],
        is_train=True,
        collate_fn=make_train_collate_fn(processor, tokenizer),
    )
    test_loader = create_loader(
        test_dataset,
        batch_size=cfg['batch_size_test'],
        collate_fn=make_eval_collate_fn(processor),
    )

    optim = AdamW(
        [p for p in list(model.parameters()) if p.requires_grad],
        lr=cfg['optimizer']['lr'],
        weight_decay=cfg['optimizer']['weight_decay'],
    )

    max_epoch = cfg['scheduler']['epochs']
    table, console = rich_table_setup()

    for epoch_no in range(max_epoch):
        train_once(model, train_loader, optim, epoch_no)

        # Eval the model, and [TODO] save best result
        evaluate_once(model, test_loader, table, console)
        raise NotImplemented('Save Result')

    raise NotImplemented


def to_device(batch, device):
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def train_once(
    model: Siglip2Model,
    train_loader: DataLoader,
    optim,
    epoch_no: int,
):
    dev = model.device
    optim.zero_grad()

    for img, txt, _ in track(train_loader, description=f'Ep {epoch_no}'):
        img = to_device(img, dev)
        txt = to_device(txt, dev)

        output = model(**img, **txt, return_loss=True)
        loss = output.loss
        loss.backward()
        optim.step()

    raise NotImplemented()


def main():
    with open('config/siglip2.yaml', 'r') as f:
        cfg = yaml.load(f.read(), yaml.Loader)

    evaluate_main(cfg)


if __name__ == '__main__':
    main()
