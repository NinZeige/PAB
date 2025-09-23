from collections.abc import Callable
from pathlib import Path

import torch
import open_clip

MODEL_NAME = 'MobileCLIP2-S4'


def load_pretrained(local_file: Path, device: torch.device | str):
    if not local_file.is_file():
        raise ValueError(f'Given local_file {local_file} does not exist')

    weight_file = str(local_file)

    model, train_proc, test_proc = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=weight_file,
        device=device,
    )

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    assert isinstance(model, open_clip.CustomTextCLIP)
    assert isinstance(train_proc, Callable)
    assert isinstance(test_proc, Callable)
    return model, train_proc, test_proc, tokenizer


def make_eval_collate_fn(
    tokenizer: Callable,
):
    """
    训练用的collate函数目前逻辑与评估相同，后续需要不同参数输入的时候再改动
    """
    return make_train_collate_fn(tokenizer)


def make_train_collate_fn(
    tokenizer: Callable,
):
    """
    训练和评估使用的collate函数的输入batch形状不同，通过键值对形式对齐
    """

    def collate(batch):
        images = [b['image'] for b in batch]
        texts = [b['caption'] for b in batch]
        assert isinstance(images[-1], torch.Tensor)

        txt_inputs = tokenizer(
            texts,
        )
        img_inputs = torch.stack(images)
        indices = torch.tensor([b['index'] for b in batch], dtype=torch.long)
        return img_inputs, txt_inputs, indices

    return collate


def save_ckpt(path: Path, model, optim, scheduler, scaler, epoch, best_map, cfg):
    import os

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
