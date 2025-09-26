import unittest

import torch

from main import to_device
from models import siglip2
from models.siglip2itm import Siglip2ITMConfig, Siglip2ITM
from dataset import create_dataset, create_loader


def load_config() -> dict[str, str | list[str]]:
    import yaml

    with open('config/siglip2.yaml') as f:
        return yaml.load(f, Loader=yaml.Loader)


class TestSigLIP2CMPForward(unittest.TestCase):
    @staticmethod
    def build_model():
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        conf = Siglip2ITMConfig(None)
        return Siglip2ITM(conf, dev)

    def test_forward(self):
        model = TestSigLIP2CMPForward.build_model()
        proc = Siglip2ITM.get_processor()
        tok = Siglip2ITM.get_tokenizer()
        cfg = load_config()
        dev = model.siglip2.device

        _, ds = create_dataset(cfg, None, True)
        loader = create_loader(
            ds, 32, None, 4, False, siglip2.make_eval_collate_fn(proc, tok, 64)
        )

        img, txt, idx = next(iter(loader))
        img = to_device(img, dev)
        txt = to_device(txt, dev)
        _ = idx.to(dev)

        _ = model.forward(**img, **txt)  # type: ignore
        res = model.forward(**img, **txt, return_loss=True)  # type: ignore
        self.assertIsNotNone(res.loss)


if __name__ == '__main__':
    unittest.main()
