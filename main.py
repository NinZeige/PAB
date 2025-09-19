import torch
import yaml
import open_clip
from PIL import Image
from mobileclip.modules.common.mobileone import reparameterize_model

from dataset import create_dataset, create_loader
from evaluate import evaluate_itc


def load_pretrained_mclip2():
    MODEL_NAME = "MobileCLIP2-S4"
    PRETRAINED = "./checkpoints/mobileclip2_s4.pt"

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = reparameterize_model(model.eval())
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(dev)

    return model, preprocess, tokenizer


def main():
    with open("config/mobile-clip2.yaml", "r") as f:
        cfg = yaml.load(f.read(), yaml.Loader)

    model, preprocess, tokenizer = load_pretrained_mclip2()
    _, test_dataset = create_dataset(cfg, preprocess, evaluate=True)

    test_loader = create_loader(
        [test_dataset],
        [None],
        batch_size=[cfg["batch_size_test"]],
        num_workers=[1],
        is_trains=[False],
        collate_fns=[None],
    )[0]

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sim_mat, *_ = evaluate_itc(model, test_loader, tokenizer, dev, cfg)
    print(f"{sim_mat.shape!r}")
    breakpoint()


if __name__ == "__main__":
    main()
