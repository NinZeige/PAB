import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rich.progress import track
from rich.table import Table
from rich.console import Console

from models.siglip2cmp import SigLIP2CMP


def evaluate_once(
    model: SigLIP2CMP, loader: DataLoader, table: Table, console: Console
):
    model.eval()

    sim_mat, image_embeds, text_embeds = evaluate_itc(model, loader, console=console)
    score = evaluation_itm(
        model,
        sim_mat,
        image_embeds,
        text_embeds,
        console=console,
    )
    res = mAP(score, loader.dataset.g_pids, loader.dataset.q_pids)

    # Pretty Print
    table.add_row(*map(lambda x: f'{x:.2f}%', res.values()))
    console.print(table)
    return res


@torch.no_grad()
def evaluate_itc(
    model: SigLIP2CMP,
    loader: DataLoader,
    console: Console | None = None,
):
    model.eval()
    image_features = []
    text_features = []
    image_embeds = []
    text_embeds = []

    for imgs, text, idx in track(loader, description='Eval ITC', console=console):
        imgs = {k: v.to(model.device) for k, v in imgs.items()}
        text = {k: v.to(model.device) for k, v in text.items()}
        idx = idx.to(model.device)

        output = model.forward(idx=idx, **imgs, **text)

        image_features.append(F.normalize(output.image_feat, dim=-1))
        text_features.append(F.normalize(output.text_feat, dim=-1))
        image_embeds.append(output.image_embeds)
        text_embeds.append(output.image_embeds)

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)
    image_embeds = torch.cat(image_embeds)
    text_embeds = torch.cat(text_embeds)

    sims_matrix = image_features @ text_features.t()
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


@torch.no_grad()
def evaluation_itm(
    model: SigLIP2CMP,
    sims_matrix,
    image_embeds,
    text_embeds,
    console: Console | None = None,
):
    model.eval()
    device = model.device
    k_test = 128

    score_matrix_t2i = torch.full(sims_matrix.size(), 1000.0).to(device)
    text_atts = torch.ones(text_embeds.shape[:2], device=device)

    for i, sims in enumerate(
        track(sims_matrix, description='Eval ITM', console=console)
    ):
        _, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_embeds[topk_idx]
        encoder_att = torch.ones(
            encoder_output.size()[:2], dtype=torch.long, device=device
        )

        output = model.get_cross_embeds(
            encoder_output,
            encoder_att,
            text_embeds=text_embeds[i].repeat(k_test, 1, 1),
            text_atts=text_atts[i].repeat(k_test, 1),
        )[:, 0, :]
        score = model.itm_head(output)[:, 1]
        score_matrix_t2i[i, topk_idx] = score

    min_values, _ = torch.min(score_matrix_t2i, dim=1)
    replacement_tensor = min_values.view(-1, 1).expand(-1, score_matrix_t2i.size(1))
    for i in range(sims_matrix.size(0)):
        score_matrix_t2i[i][score_matrix_t2i[i] == 1000.0] = replacement_tensor[i][
            score_matrix_t2i[i] == 1000.0
        ]
    score_matrix_t2i[score_matrix_t2i == 1000.0] = replacement_tensor[
        score_matrix_t2i == 1000.0
    ]
    score_matrix_t2i = (score_matrix_t2i - score_matrix_t2i.min()) / (
        score_matrix_t2i.max() - score_matrix_t2i.min()
    )

    score_sim_t2i = sims_matrix.clone()
    score_sim_t2i = (score_sim_t2i - score_sim_t2i.min()) / (
        score_sim_t2i.max() - score_sim_t2i.min()
    )
    score_matrix_t2i = score_matrix_t2i + 0.002 * score_sim_t2i  #

    return score_matrix_t2i.cpu()
