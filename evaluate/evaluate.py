import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip

from rich.progress import track
from rich.table import Table
from rich.console import Console

from dataset import search_test_dataset


def evaluate_once(
    model: open_clip.CustomTextCLIP, loader: DataLoader, table: Table, console: Console
):
    model.eval()

    assert isinstance(loader.dataset, search_test_dataset)
    sim_mat, image_embeds, text_embeds = evaluate_itc(model, loader, console=console)
    res = mAP(sim_mat, loader.dataset.g_pids, loader.dataset.q_pids)

    # Pretty Print
    table.add_row(*map(lambda x: f'{x:.2f}%', res.values()))
    console.print(table)
    return res


@torch.no_grad()
def evaluate_itc(
    model: open_clip.CustomTextCLIP,
    loader: DataLoader,
    console: Console | None = None,
):
    model.eval()
    device = next(model.parameters()).device
    image_features = []
    text_features = []
    image_embeds = []
    text_embeds = []

    for imgs, text, idx in track(loader, description='Eval ITC', console=console):
        imgs = imgs.to(device)
        text = text.to(device)
        idx = idx.to(device)

        image_feat, text_feat, _ = model.forward(image=imgs, text=text)  # type: ignore

        image_features.append(F.normalize(image_feat, dim=-1))  # type: ignore
        text_features.append(F.normalize(text_feat, dim=-1))  # type: ignore

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

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
