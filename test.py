import unittest
import torch
from pathlib import Path
import logging

MODEL_NAME = 'google/siglip2-base-patch16-naflex'
from functools import partial
from models import siglip2


class TestDataloader(unittest.TestCase):
    def test_infer(self):
        from PIL import Image
        import os

        model, proc, _ = siglip2.build_model()
        PAB_ROOT = Path(os.environ['PABROOT'])

        TEST_FILE = PAB_ROOT / 'test' / '0.jpg'

        with Image.open(TEST_FILE) as f:
            img_input = proc(
                images=[
                    f,
                ]
            )

        img_feat = model.get_image_features(**img_input)
        self.assertTrue(isinstance(img_feat, torch.Tensor))
        self.assertEqual(img_feat.shape, torch.Size((1, 768)))

    @staticmethod
    def load_config() -> dict[str, str | list[str]]:
        import yaml

        with open('config/siglip2.yaml') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        return cfg

    def test_load(self):
        from dataset import create_dataset, create_loader

        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model, processor, _ = siglip2.build_model(dev)

        cfg = TestDataloader.load_config()

        BATCH_SIZE = 50
        _, testset = create_dataset(cfg, None, True)
        loader = create_loader(
            testset,
            batch_size=BATCH_SIZE,
            sampler=None,
            collate_fn=siglip2.make_eval_collate_fn(processor),
        )

        img, idx = next(iter(loader))  # 获取数据

        # 基本形状确认
        self.assertEqual(len(idx), BATCH_SIZE)
        img = {k: v.to(dev) for k, v in img.items()}
        output = model.get_image_features(**img)
        # 判断输出特征形状
        self.assertEqual(output.shape, torch.Size((50, 768)))

    def test_train_set(self):
        cfg = TestDataloader.load_config()


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    unittest.main()
