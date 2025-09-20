import torch
import torch.nn.functional as F
from transformers import Siglip2Model
from torch.utils.data import DataLoader
from rich.progress import track


@torch.no_grad()
def evaluate_itc(
    model: Siglip2Model,
    loader: DataLoader,
    tokenizer,
    processor,
    device: str,
    cfg: dict,
):
    texts = loader.dataset.text
    images = loader.dataset.image
    num_text = len(texts)
    text_bs = cfg['batch_size_test_text']

    text_embeds = []
    for i in track(range(0, num_text, text_bs), description='Encode Ft'):
        text = texts[i : min(num_text, i + text_bs)]

        text_input = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt',
        ).to(device)

        text_embed = model.get_text_features(**text_input)
        text_embeds.append(F.normalize(text_embed, dim=-1))
    text_embeds = torch.cat(text_embeds, dim=0)

    image_embeds = []
    for images, _, _ in track(loader, description='Encode Fi'):
        [p, m, s] = map(lambda t: t.squeeze(1).to(device), list(images.values()))

        image_embed = model.get_image_features(
            pixel_values=p, pixel_attention_mask=m, spatial_shapes=s
        )
        image_embeds.append(F.normalize(image_embed, dim=-1))
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    sims_matrix_t2i = sims_matrix.t()

    return sims_matrix_t2i, image_embeds, text_embeds


def mAP(scores_t2i: torch.Tensor, g_pids, q_pids) -> dict[str, float]:
    similarity = scores_t2i.clone()
    indices = torch.argsort(similarity, dim=1, descending=True)
    g_pids = torch.tensor(g_pids)
    q_pids = torch.tensor(q_pids)
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :10].cumsum(1)  # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [
        tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.0)
        for i, match_row in enumerate(matches)
    ]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    t2i_cmc, t2i_mAP, t2i_mINP, _ = all_cmc, mAP, mINP, indices
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()

    eval_result = {
        'R1': t2i_cmc[0],
        'R5': t2i_cmc[4],
        'R10': t2i_cmc[9],
        'mAP': t2i_mAP,
        'mINP': t2i_mINP,
    }

    return eval_result
