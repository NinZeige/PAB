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


def make_eval_collate_fn(processor: Siglip2ImageProcessorFast):
    def collate(batch):
        images = [b['image'] for b in batch]
        indices = [b['index'] for b in batch]
        images_input = processor(images, return_tensors='pt')
        return images_input, torch.tensor(indices)

    return collate


def make_train_collate_fn(
    processor: Siglip2ImageProcessorFast,
    tokenizer: GemmaTokenizer,
    text_max_len: int = None,
):
    '''
    训练和评估使用的collate函数的输入batch形状不同，通过键值对形式对齐
    '''
    def collate(batch):
        images = [b['image'] for b in batch]
        texts = [b['caption'] for b in batch]
        # NaFlex/FixRes 通吃：一次性处理整批图像
        img_inputs = processor(images=images, return_tensors='pt')
        txt_inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=text_max_len,
            return_tensors='pt',
        )
        indices = torch.tensor([b['idx'] for b in batch], dtype=torch.long)
        return img_inputs, txt_inputs, indices

    return collate
