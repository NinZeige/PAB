from pathlib import Path
import unittest
from collections.abc import Callable

import torch

from models.mobileclip import (
    load_pretrained,
    make_eval_collate_fn,
)
from dataset import create_dataset, create_loader
from .utils import load_config

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 160


class TestMobileClip(unittest.TestCase):
    def test_load(self):
        ckpt = Path('/root/autodl-tmp/saved_model/mobileclip/mobileclip2_s4.pt')
        model, _, preproc, tok = load_pretrained(ckpt, DEVICE)
        assert isinstance(preproc, Callable)

        test_datasets = create_dataset(
            load_config(),
            preproc,
            True,
        )
        loader = create_loader(
            test_datasets,
            batch_size=BATCH_SIZE,
            sampler=None,
            num_worker=4,
            is_train=False,
            collate_fn=make_eval_collate_fn(tok),
        )

        img, txt, idx = next(iter(loader))
        breakpoint()
        self.assertEqual(img.shape, torch.Size([BATCH_SIZE, 768]))
        self.assertEqual(txt.shape, torch.Size([BATCH_SIZE, 768]))
        self.assertEqual(idx.shape, torch.Size([BATCH_SIZE]))
