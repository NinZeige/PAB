import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml

import torch
from torch import GradScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import open_clip


from rich.table import Table, Column
from rich.console import Console
from rich.progress import track

from models import mobileclip
from dataset import create_dataset, create_loader
from evaluate import evaluate_once


@dataclass
class ClipLossArgs:
    local_loss: bool = False
    gather_with_grad: bool = False
    cache_labels: bool = True
    rank: int = 0
    world_size: int = 1
    horovod: bool = False
    distill: bool = False
    model: str = 'mobileclip2'
    siglip: bool = False


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

    assert isinstance(cfg['pretrained'], str)

    ckpt = None
    if 'checkpoint' in cfg:
        assert isinstance(cfg['checkpoint'], str)
        ckpt = Path(cfg['checkpoint'])
    pretrained = Path(cfg['pretrained'])

    model, _, preproc, tokenizer = mobileclip.load_pretrained(
        pretrained, device=dev, ckpt=ckpt
    )
    assert isinstance(model, open_clip.CustomTextCLIP)

    test_dataset = create_dataset(
        cfg,
        preproc,
        True,
    )
    assert isinstance(cfg['batch_size_test'], int)
    loader = create_loader(
        test_dataset,
        batch_size=cfg['batch_size_test'],
        sampler=None,
        num_worker=4,
        is_train=False,
        collate_fn=mobileclip.make_eval_collate_fn(tokenizer),
    )

    t, c = rich_table_setup()
    evaluate_once(model, loader, t, c)


def train_main(cfg: dict):
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    assert isinstance(cfg['batch_size_test'], int)
    assert isinstance(cfg['batch_size_train'], int)
    assert isinstance(cfg['max_words'], int)

    ckpt = None
    if 'checkpoint' in cfg:
        assert isinstance(cfg['checkpoint'], str)
        ckpt = Path(cfg['checkpoint'])

    pretrained = Path(cfg['pretrained'])
    model, train_proc, test_proc, tokenizer = mobileclip.load_pretrained(
        pretrained, device=dev, ckpt=ckpt
    )

    train_dataset = create_dataset(
        cfg,
        train_proc,
        False,
    )
    test_dataset = create_dataset(
        cfg,
        test_proc,
        True,
    )
    assert train_dataset is not None

    train_loader = create_loader(
        train_dataset,
        batch_size=cfg['batch_size_train'],
        sampler=None,
        num_worker=4,
        is_train=True,
        collate_fn=mobileclip.make_train_collate_fn(tokenizer),
    )
    test_loader = create_loader(
        test_dataset,
        batch_size=cfg['batch_size_test'],
        sampler=None,
        num_worker=4,
        is_train=False,
        collate_fn=mobileclip.make_eval_collate_fn(tokenizer),
    )

    max_epoch = cfg['scheduler']['epochs']

    optim = AdamW(
        [p for p in list(model.parameters()) if p.requires_grad],
        lr=float(cfg['optimizer']['lr']),  # '1e-9' 会被yaml解析为`str`类型
        weight_decay=cfg['optimizer']['weight_decay'],
        betas=(0.9, 0.98),
        eps=1e-8,
        fused=True,
    )
    device = next(model.parameters()).device

    # Prepare Optimizers
    scaler = torch.GradScaler(device=device.type, enabled=torch.cuda.is_available())
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        cfg['scheduler']['num_warmup_steps'],
        max_epoch * len(train_loader),
    )
    loss_func = open_clip.create_loss(ClipLossArgs())
    assert isinstance(loss_func, open_clip.ClipLoss)

    t, c = rich_table_setup()
    best_mAP = -1.0

    for epoch_no in range(max_epoch):
        loss = train_one_epoch(
            model,
            train_loader,
            epoch_no,
            optim,
            scaler,
            loss_func,
            scheduler,
            c,
        )

        res = evaluate_once(model, test_loader, t, c)
        cur_mAP = float(res['mAP'])
        c.print(f'[Epoch {epoch_no}] train_loss={loss:.4f}  mAP={cur_mAP:.4f}')

        if cur_mAP > best_mAP:
            best_mAP = cur_mAP
            mobileclip.save_ckpt(
                Path(cfg['save_dir']),
                model,
                optim,
                scheduler,
                scaler,
                epoch_no,
                best_mAP,
                cfg,
            )


def to_device(batch, device):
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def train_one_epoch(
    model: open_clip.CustomTextCLIP,
    train_loader: DataLoader,
    epoch_no: int,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
    loss_fn: open_clip.ClipLoss,
    scheduler,
    console: Console,
):
    dev = next(model.parameters()).device
    assert isinstance(dev, torch.device)
    assert isinstance(dev.type, str)

    running = 0
    model.train()

    cnt = 0
    INTV = 5

    for imgs, text, _ in track(
        train_loader, description=f'Ep {epoch_no}', console=console
    ):
        optim.zero_grad(set_to_none=True)

        imgs = imgs.to(dev)
        text = text.to(dev)

        with torch.autocast(
            device_type=dev.type,
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            image_feat: torch.Tensor
            text_feat: torch.Tensor
            logit_scale: torch.Tensor

            image_feat, text_feat, logit_scale = model(image=imgs, text=text)  # type: ignore
            loss = loss_fn(
                image_features=image_feat,
                text_features=text_feat,
                logit_scale=logit_scale,
                output_dict=True,
            )
            total_loss = sum(loss.values())

        scaler.scale(total_loss).backward()  # type: ignore
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optim)
        scaler.update()
        scheduler.step()
        running += total_loss
        cur_lr = optim.param_groups[0]['lr']

        cnt = (cnt + 1) % INTV
        if not cnt and console is not None:
            console.print(f'lr={cur_lr:.2e}  loss={total_loss:.4f}')

    return running / max(1, len(train_loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--pretrained')
    parser.add_argument('--save_dir')
    parser.add_argument('-e', '--evaluate', action='store_true')
    arg = parser.parse_args()

    with open('config/mobile-clip2.yaml', 'r') as f:
        cfg = yaml.load(f.read(), yaml.Loader)

    if arg.checkpoint:
        cfg['checkpoint'] = arg.checkpoint
    if arg.save_dir:
        cfg['save_dir'] = arg.save_dir
    if arg.pretrained:
        cfg['pretrained'] = arg.pretrained

    if arg.evaluate:
        evaluate_main(cfg)
    else:
        train_main(cfg)


if __name__ == '__main__':
    main()
