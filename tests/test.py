import unittest
import torch
from pathlib import Path
import logging

from main import to_device

MODEL_NAME = 'google/siglip2-base-patch16-naflex'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from models import siglip2, siglip2cmp


def load_config() -> dict[str, str | list[str]]:
    import yaml

    with open('config/siglip2.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    return cfg


def load_single_batch(processor, tokenizer, device=torch.device | str):
    import os
    from PIL import Image

    PAB_ROOT = os.environ['PABROOT']
    PAB_ROOT = Path(PAB_ROOT)

    TEST_FILE = PAB_ROOT / 'test' / '0.jpg'

    with Image.open(TEST_FILE) as f:
        img_input = processor(
            images=[
                f,
            ]
        )

    txt_input = tokenizer(
        [
            'a man holding a cat',
            'a child leaving hometown',
            'the cat fall on the ground',
            'a cat chasing a mouse',
        ],
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors='pt',
    )
    txt_input = to_device(txt_input, device)
    img_input = to_device(img_input, device)

    return img_input, txt_input


class TestDataloader(unittest.TestCase):
    def test_infer(self):
        from PIL import Image
        import os

        model, proc, tokenizer = siglip2.build_model(DEVICE)
        PAB_ROOT = Path(os.environ['PABROOT'])

        TEST_FILE = PAB_ROOT / 'test' / '0.jpg'

        [img_input, txt_input] = [to_device(x, DEVICE) for x in [img_input, txt_input]]

        img_feat = model.get_image_features(**img_input)
        _ = model(**txt_input, **img_input, return_loss=True)
        self.assertTrue(isinstance(img_feat, torch.Tensor))
        self.assertEqual(img_feat.shape, torch.Size((1, 768)))

    def test_load(self):
        from dataset import create_dataset, create_loader

        model, processor, tok = siglip2.build_model(DEVICE)

        cfg = load_config()

        BATCH_SIZE = 50
        _, testset = create_dataset(cfg, None, True)
        loader = create_loader(
            testset,
            batch_size=BATCH_SIZE,
            sampler=None,
            collate_fn=siglip2.make_eval_collate_fn(processor, tok, cfg['max_words']),
        )

        img, txt, idx = next(iter(loader))  # 获取数据

        # 基本形状确认
        self.assertEqual(len(idx), BATCH_SIZE)
        img = to_device(img, DEVICE)
        txt = to_device(txt, DEVICE)
        img_output = model.get_image_features(**img)
        txt_output = model.get_text_features(**txt)
        # 判断输出特征形状
        self.assertEqual(img_output.shape, torch.Size((50, 768)))
        self.assertEqual(txt_output.shape, torch.Size((50, 768)))


class Siglip2CMPTest(unittest.TestCase):
    def test_forward(self):
        from dataset import create_dataset, create_loader

        model, proc, token = siglip2cmp.SigLIP2CMP.build_model(DEVICE)

        cfg = load_config()

        BATCH_SIZE = 50
        _, testset = create_dataset(cfg, None, True)
        loader = create_loader(
            testset,
            batch_size=BATCH_SIZE,
            sampler=None,
            collate_fn=siglip2.make_eval_collate_fn(proc, token, cfg['max_words']),
        )

        img, txt, idx = next(iter(loader))

        output = model.forward(idx=idx, **img, **txt, return_loss=True)
        self.assertTrue(isinstance(output, siglip2cmp.ITMOutput))


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    unittest.main()
