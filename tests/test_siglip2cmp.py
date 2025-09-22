import unittest

import torch

from main import to_device
from models import siglip2, siglip2cmp
from dataset import create_dataset, create_loader


def load_config() -> dict[str, str | list[str]]:
    import yaml

    with open('config/siglip2.yaml') as f:
        return yaml.load(f, Loader=yaml.Loader)


class TestSigLIP2CMPForward(unittest.TestCase):
    @staticmethod
    def build_model(cfg, device: torch.device | str):
        return siglip2cmp.SigLIP2CMP.build_model(cfg, device)

    def test_forward(self):
        cfg = load_config()
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, proc, tok = TestSigLIP2CMPForward.build_model(cfg, dev)
        model.eval()

        _, test_set = create_dataset(cfg, None, True)
        loader = create_loader(
            test_set, 32, None, 2, False, siglip2.make_eval_collate_fn(proc, tok, 64)
        )

        img, txt, idx = next(iter(loader))
        img = to_device(img, dev)
        txt = to_device(txt, dev)
        idx = idx.to(dev)

        output = model.forward(idx=idx, **img, **txt, return_loss=True)
        self.assertTrue(output.loss is not None)
        self.assertEqual(type(float(output.loss)), float)


if __name__ == '__main__':
    # 可用 PYTEST 运行，也可直接 unittest 运行
    unittest.main()
