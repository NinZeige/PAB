import yaml
import argparse
from pathlib import Path

from rich.console import Console
from rich.progress import track
from rich.table import Table, Column

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from dataset import create_dataset, create_loader
from evaluate import evaluate_once
from models.siglip2 import (
    make_eval_collate_fn,
    make_train_collate_fn,
    save_ckpt,
)
from models.siglip2cmp import SigLIP2CMP, SigLIP2CMPConfig


def build_model(yaml_obj):
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    conf = SigLIP2CMPConfig.from_yaml_obj(yaml_obj)
    return SigLIP2CMP.build_model(conf, dev)


def build_optim(obj, model: SigLIP2CMP):
    params = {
        'siglip_lr': obj['siglip_lr'],
        'mlp_lr': obj['mlp_lr'],
        'bert_lr': obj['bert_lr'],
        'itm_lr': obj['itm_lr'],
        'weight_decay': obj['weight_decay'],
    }
    params = {k: float(v) for k, v in params.items()}

    optim = AdamW(
        [
            {
                'params': model.siglip2.parameters(),
                'lr': params['siglip_lr'],
            },  # SigLIP应该给低学习率，避免早期训练参数过大破坏SigLIP的预训练参数
            {
                'params': list(model.text_proj.parameters())
                + list(model.image_proj.parameters()),
                'lr': params['mlp_lr'],
            },
            {'params': model.itm_head.parameters(), 'lr': params['itm_lr']},
            {'params': model.bert.parameters(), 'lr': params['bert_lr']},
        ],
        weight_decay=params['weight_decay'],
        betas=(0.9, 0.98),
        eps=1e-8,
        fused=True,
    )
    return optim


def rich_table_setup():
    # Pretty Print
    t = Table(
        *map(lambda x: Column(x, justify='center'), ['R1', 'R5', 'R10', 'mAP', 'mINP']),
        title='Evaluation Result',
    )
    c = Console()
    return t, c


def evaluate_main(cfg: dict[str, str | list[str]]):
    model, processor, tokenizer = build_model(cfg)
    model.eval()

    _, test_dataset = create_dataset(cfg, None, evaluate=True)
    loader = create_loader(
        test_dataset,
        batch_size=cfg['batch_size_test'],
        num_worker=4,
        collate_fn=make_eval_collate_fn(processor, tokenizer, cfg['max_words']),
    )

    t, c = rich_table_setup()
    evaluate_once(model, loader, t, c)


def train_main(cfg: dict[str, int | str | list[str]]):
    best_mAP = -1.0
    assert isinstance(cfg['save_dir'], str)
    assert isinstance(cfg['batch_size_train'], int)
    assert isinstance(cfg['batch_size_test'], int)

    model, processor, tokenizer = build_model(cfg)

    # Prepare dataset
    train_dataset, test_dataset = create_dataset(cfg, None)
    assert train_dataset is not None

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
        num_worker=4,
        collate_fn=make_eval_collate_fn(processor, tokenizer, cfg['max_words']),
    )

    max_epoch = cfg['scheduler']['epochs']
    optim = build_optim(cfg['optimizer'], model)

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
        loss = train_once(
            model, train_loader, epoch_no, optim, scaler, scheduler, console=console
        )

        # Eval the model and then save best
        res = evaluate_once(model, test_loader, table, console)
        cur_mAP = float(res['mAP'])

        print(f'[Epoch {epoch_no}] train_loss={loss:.4f}  mAP={cur_mAP:.4f}')

        if cur_mAP > best_mAP:
            best_mAP = cur_mAP
            save_ckpt(
                Path(cfg['save_dir']) / 'best.pt',
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
    model: SigLIP2CMP,
    train_loader: DataLoader,
    epoch_no: int,
    optim: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    scheduler,
    console: Console | None = None,
):
    dev: torch.device = model.device
    running = 0
    model.train()

    cnt = 0
    INTV = 50

    for img, txt, idx in track(
        train_loader, description=f'Ep {epoch_no}', console=console
    ):
        optim.zero_grad(set_to_none=True)

        img = to_device(img, dev)
        txt = to_device(txt, dev)
        idx = idx.to(dev)

        with torch.autocast(
            device_type=dev.type,
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            # 直接用内置 loss（SigLIP 风格的 binary logistic 对比损失）
            output = model.forward(idx=idx, **img, **txt, return_loss=True)
            loss = output.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optim)
        scaler.update()
        scheduler.step()

        running += loss.item()
        lrs = [optim.param_groups[i]['lr'] for i in range(4)]

        cnt = (cnt + 1) % INTV
        if not cnt and console is not None:
            console.print(
                f'siglip lr={lrs[0]:.2e} mlp lr={lrs[1]:.2e} \
bert lr={lrs[2]:.2e} itm lr={lrs[2]:.2e} loss={loss.item():.4f}'
            )

    return running / max(1, len(train_loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--evaluation', action='store_true')
    parser.add_argument('--save-dir', required=True)
    args = parser.parse_args()

    with open('config/siglip2.yaml', 'r') as f:
        cfg = yaml.load(f.read(), yaml.Loader)

    if args.save_dir:
        cfg['save_dir'] = args.save_dir

    if args.evaluation:
        evaluate_main(cfg)
    else:
        train_main(cfg)


if __name__ == '__main__':
    main()
