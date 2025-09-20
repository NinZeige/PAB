from functools import partial

import torch
import yaml
from rich.table import Table, Column
from rich.console import Console


from dataset import create_dataset, create_loader
from evaluate import evaluate_itc, mAP
from models.siglip2 import build_model, collate_func


def evaluate_main(cfg: dict[str, str | list[str]]):
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, processor, tokenizer = build_model(dev)

    _, test_dataset = create_dataset(cfg, None, evaluate=True)
    test_loader = create_loader(
        test_dataset,
        batch_size=cfg['batch_size_test'],
        collate_fn=partial(collate_func, processor=processor),
    )

    # Evaluation
    sim_mat, *_ = evaluate_itc(model, test_loader, tokenizer, processor, dev, cfg)
    res = mAP(sim_mat, test_loader.dataset.g_pids, test_loader.dataset.q_pids)

    # Pretty Print
    t = Table(
        *map(lambda x: Column(x, justify='center'), ['R1', 'R5', 'R10', 'mAP', 'mINP']),
        title='Evaluation Result',
    )
    t.add_row(*map(lambda x: f'{x:.2f}%', res.values()))
    Console().print(t)


def main():
    with open('config/siglip2.yaml', 'r') as f:
        cfg = yaml.load(f.read(), yaml.Loader)

    evaluate_main(cfg)


if __name__ == '__main__':
    main()
