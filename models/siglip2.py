import os
from pathlib import Path

import torch
from transformers import (
    Siglip2Model,
    Siglip2ImageProcessorFast,
    AutoTokenizer,
    GemmaTokenizer,
)


def build_model(device: torch.device | str, local_file: Path | None = None):
    MODEL_NAME = 'google/siglip2-base-patch16-naflex'
    model = Siglip2Model.from_pretrained(MODEL_NAME)
    if local_file is not None:
        model.load_state_dict(load(local_file)['model'])
    processor = Siglip2ImageProcessorFast.from_pretrained(MODEL_NAME)
    tokenizer: GemmaTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = model.to(device)
    return model, processor, tokenizer


def make_eval_collate_fn(
    processor: Siglip2ImageProcessorFast,
    tokenizer: GemmaTokenizer,
    text_max_len: int | None = None,
):
    """
    训练用的collate函数目前逻辑与评估相同，后续需要不同参数输入的时候再改动
    """
    return make_train_collate_fn(processor, tokenizer, text_max_len)


def make_train_collate_fn(
    processor: Siglip2ImageProcessorFast,
    tokenizer: GemmaTokenizer,
    text_max_len: int,
):
    """
    训练和评估使用的collate函数的输入batch形状不同，通过键值对形式对齐
    """

    def collate(batch):
        images = [b['image'] for b in batch]
        texts = [b['caption'] for b in batch]
        # NaFlex/FixRes 通吃：一次性处理整批图像
        img_inputs = processor(images=images, return_tensors='pt')
        txt_inputs = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=text_max_len,
            return_tensors='pt',
        )
        indices = torch.tensor([b['index'] for b in batch], dtype=torch.long)
        return img_inputs, txt_inputs, indices

    return collate


def save_ckpt(path: Path, model, optim, scheduler, scaler, epoch, best_map, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'best_mAP': best_map,
            'cfg': cfg,
        },
        path,
    )


def load(path: Path):
    obj = torch.load(path)
    return obj
