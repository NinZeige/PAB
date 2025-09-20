from pathlib import Path
from functools import partial

import torch
import yaml
from dataset import create_dataset, create_loader
from evaluate import evaluate_itc, mAP
from transformers import (
    Siglip2Model,
    Siglip2ImageProcessor,
    AutoTokenizer,
    GemmaTokenizer,
)

from rich.table import Table, Column
from rich.console import Console


def build_model():
    MODEL_NAME = 'google/siglip2-base-patch16-naflex'
    model = Siglip2Model.from_pretrained(MODEL_NAME)
    processor = Siglip2ImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer: GemmaTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(dev)

    return model, processor, tokenizer


def main():
    with open('config/mobile-clip2.yaml', 'r') as f:
        cfg = yaml.load(f.read(), yaml.Loader)

    model, processor, tokenizer = build_model()
    processor = partial(processor, return_tensors='pt')
    _, test_dataset = create_dataset(cfg, processor, evaluate=True)

    test_loader = create_loader(
        test_dataset,
        None,
        batch_size=cfg['batch_size_test'],
        num_worker=1,
        is_train=False,
        collate_fn=None,
    )

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sim_mat, *_ = evaluate_itc(model, test_loader, tokenizer, processor, dev, cfg)
    res = mAP(sim_mat, test_loader.dataset.g_pids, test_loader.dataset.q_pids)

    # Pretty Print
    t = Table(
        *map(lambda x: Column(x, justify='center'), ['R1', 'R5', 'R10', 'mAP', 'mINP']),
        title='Evaluation Result',
    )
    t.add_row(*map(lambda x: f'{x:.2f}%', res.values()))
    Console().print(t)


if __name__ == '__main__':
    main()
