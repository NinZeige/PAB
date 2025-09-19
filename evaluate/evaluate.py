import torch
import torch.nn.functional as F
import open_clip
from torch.utils.data import DataLoader
from rich.progress import track


@torch.no_grad
def evaluate_itc(
    model: open_clip.CLIP,
    loader: DataLoader,
    tokenizer: open_clip.SimpleTokenizer | open_clip.tokenizer.SigLipTokenizer,
    device: str,
    cfg: dict,
):

    texts = loader.dataset.text
    num_text = len(texts)
    text_bs = cfg["batch_size_test_text"]

    text_embeds = []
    for i in track(range(0, num_text, text_bs)):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
        ).to(device)
        
        text_embed = model.encode_text(text_input)
        text_embeds.append(text_embed)
    text_embeds = torch.cat(text_embeds, dim=0)

    image_embeds = []
    for image, _, _ in track(loader):
        image = image.to(device)

        image_embed = model.encode_image(image)
        image_embeds.append(image_embed)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    sims_matrix_t2i = sims_matrix.t()

    return sims_matrix_t2i, image_embeds, text_embeds
