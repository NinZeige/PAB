import unittest
import torch
from pathlib import Path

MODEL_NAME = 'google/siglip2-base-patch16-naflex'
from functools import partial


class test_dataloader(unittest.TestCase):
    @staticmethod
    def load_processor():
        from transformers import Siglip2ImageProcessorFast

        processor = Siglip2ImageProcessorFast.from_pretrained(MODEL_NAME)
        return partial(processor, return_tensors='pt')

    @staticmethod
    def load_model():
        from transformers import Siglip2Model

        model = Siglip2Model.from_pretrained(MODEL_NAME)
        return model

    def test_infer(self):
        from PIL import Image
        import os

        model = self.load_model()
        proc = self.load_processor()
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

    def test_load(self):
        from dataset import create_dataset, create_loader
        import yaml

        with open('config/mobile-clip2.yaml') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        model = test_dataloader.load_model()
        processor = test_dataloader.load_processor()

        BATCH_SIZE = 50
        _, testset = create_dataset(cfg, processor, True)
        loader = create_loader(testset, None, BATCH_SIZE, 1, False, None)

        i = next(iter(loader))  # 获取数据

        # 基本形状确认
        self.assertEqual(len(i[2]), BATCH_SIZE)
        img_input = i[0]
        [p, m, s] = map(lambda t: t.squeeze(1), list(img_input.values()))

        output = model.get_image_features(
            pixel_values=p,
            pixel_attention_mask=m,
            spatial_shapes=s,
        )
        # 判断输出特征形状
        self.assertEqual(output.shape, torch.Size((50, 768)))


if __name__ == '__main__':
    unittest.main()
