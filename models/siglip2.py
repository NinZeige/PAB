from functools import partial

import torch
from transformers import (
    Siglip2Model,
    Siglip2ImageProcessorFast,
    AutoTokenizer,
    GemmaTokenizer,
)


def build_model(device: str = 'cpu'):
    MODEL_NAME = 'google/siglip2-base-patch16-naflex'
    model = Siglip2Model.from_pretrained(MODEL_NAME)
    processor = Siglip2ImageProcessorFast.from_pretrained(MODEL_NAME)
    tokenizer: GemmaTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model.to(device)
    return model, processor, tokenizer


def collate_func(batch, processor: Siglip2ImageProcessorFast):
    images = [b[0] for b in batch]
    indices = [b[2] for b in batch]
    images_input = processor(images, return_tensors='pt')

    return images_input, torch.tensor(indices)
