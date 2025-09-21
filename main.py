import yaml
import argparse
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from rich.table import Table, Column

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, Siglip2Model

from dataset import create_dataset, create_loader
from evaluate import evaluate_once
from models.siglip2 import (
    build_model,
    make_eval_collate_fn,
    make_train_collate_fn,
    save_ckpt,
)


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
    best_mAP = -1.0

    model, processor, tokenizer = build_model(dev)

    # Prepare dataset
    train_dataset, test_dataset = create_dataset(cfg, None)
    train_loader = create_loader(
        train_dataset,
        batch_size=cfg['batch_size_train'],
        is_train=True,
        num_worker=4,
        collate_fn=make_train_collate_fn(processor, tokenizer, cfg['max_words']),
    )
    test_loader = create_loader(
        test_dataset,
        batch_size=cfg['batch_size_test'],
        collate_fn=make_eval_collate_fn(processor, tokenizer, cfg['max_words']),
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

    # Prepare Optimizers
    scaler = torch.GradScaler(
        device=model.device.type, enabled=torch.cuda.is_available()
    )
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        cfg['scheduler']['num_warmup_steps'],
        max_epoch * len(train_loader),
    )

    # For `rich` pretty print
    table, console = rich_table_setup()

    for epoch_no in range(max_epoch):
        loss = train_once(model, train_loader, epoch_no, optim, scaler, scheduler)

        # Eval the model and then save best
        res = evaluate_once(model, test_loader, table, console)
        cur_mAP = float(res['mAP'])

        print(f'[Epoch {epoch_no}] train_loss={loss:.4f}  mAP={cur_mAP:.4f}')

        if cur_mAP > best_mAP:
            best_mAP = cur_mAP
            save_ckpt(
                Path('output/best.pt'),
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


def train_once(
    model: Siglip2Model,
    train_loader: DataLoader,
    epoch_no: int,
    optim: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    scheduler,
):
    dev: torch.device = model.device
    running = 0
    model.train()

    with Progress() as prog:
        train_task = prog.add_task(f'Ep {epoch_no}', total=len(train_loader))
        cnt = 0
        INTV = 50

        for img, txt, _ in train_loader:
            optim.zero_grad(set_to_none=True)

            img = to_device(img, dev)
            txt = to_device(txt, dev)
            with torch.autocast(
                device_type=dev.type,
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_available(),
            ):
                # 直接用内置 loss（SigLIP 风格的 binary logistic 对比损失）
                output = model(**img, **txt, return_loss=True)
                loss = output.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optim)
            scaler.update()
            scheduler.step()

            running += loss.item()
            cur_lr = optim.param_groups[0]['lr']

            cnt = (cnt + 1) % INTV
            if not cnt:
                prog.console.print(f'lr={cur_lr:.2e}  loss={loss.item():.4f}')
            prog.update(train_task, advance=1)

    return running / max(1, len(train_loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--evaluation', action='store_true')
    args = parser.parse_args()

    with open('config/siglip2.yaml', 'r') as f:
        cfg = yaml.load(f.read(), yaml.Loader)

    if args.evaluation:
        evaluate_main(cfg)
    else:
        train_main(cfg)


if __name__ == '__main__':
    main()
