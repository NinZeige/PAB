import torch
import torch.nn.functional as F
from transformers import Siglip2Model
from torch.utils.data import DataLoader

from rich.progress import track
from rich.table import Table
from rich.console import Console

from models.siglip2cmp import SigLIP2CMP


def evaluate_once(
    model: Siglip2Model, loader: DataLoader, table: Table, console: Console
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


@torch.no_grad()
def evaluation_itm_single_gpu(
    model,
    sims_matrix: torch.Tensor,  # [N_text, N_image]
    image_embeds: torch.Tensor,  # [N_image, D]
    text_embeds: torch.Tensor,  # [N_text,  D]
    k: int = 128,
):
    raise NotImplementedError()
    model.eval()

    # 更稳的 device 获取方式（兼容无 model.device 的情况）
    device = getattr(model, 'device', next(model.parameters()).device)

    sims_matrix = sims_matrix.to(device)
    image_embeds = image_embeds.to(device)
    text_embeds = text_embeds.to(device)

    N_text, N_image = sims_matrix.shape
    k = min(int(k), N_image)  # 防止 k > N_image

    # 注意：你的 text_embeds / image_embeds 没有序列维度，这里构造 L=1 的 mask
    # text_atts: [N_text, 1]
    text_atts = torch.ones((N_text, 1), dtype=torch.long, device=device)

    # 用 NaN 占位，后面用每行的最小有效值填补
    score_mat = torch.full((N_text, N_image), float('nan'), device=device)

    for i in track(range(N_text), description='Eval ITM'):
        # 1) 取该文本与所有图片的双塔相似度，选 top-k 图片
        sims = sims_matrix[i]  # [N_image]
        _, topk_idx = torch.topk(sims, k, dim=0)

        # 2) 候选图片特征 [k, D] -> [k, 1, D]，并构造 image_att [k, 1]
        enc_out = image_embeds[topk_idx].unsqueeze(1).contiguous()  # [k, 1, D]
        enc_att = torch.ones((k, 1), dtype=torch.long, device=device)

        # 3) 当前文本特征 [D] -> [1, 1, D]，再 expand 成 [k, 1, D]
        txt_emb = (
            text_embeds[i].view(1, 1, -1).expand(k, 1, -1).contiguous()
        )  # [k, 1, D]
        txt_att = text_atts[i].view(1, 1).expand(k, 1).contiguous()  # [k, 1]

        # 4) 过 cross-encoder：取 [CLS]（第 0 个 token）向量
        cross = model.get_cross_embeds(
            image_embeds=enc_out,
            image_atts=enc_att,
            text_embeds=txt_emb,
            text_atts=txt_att,
        )[:, 0, :]  # [k, D_bert]

        # 5) ITM 头部：取正类 logits 作为匹配分
        logits = model.itm_head(cross)[:, 1]  # [k]
        score_mat[i, topk_idx] = logits

    # --- 用每行最小有效值填补 NaN ---
    is_all_nan = torch.isnan(score_mat).all(dim=1)  # [N_text]
    # 用 +inf 替换 NaN 后求最小值，再把“全 NaN 行”的最小值置 0
    row_min = torch.min(
        torch.where(
            torch.isnan(score_mat), torch.tensor(float('inf'), device=device), score_mat
        ),
        dim=1,
    ).values
    row_min = torch.where(is_all_nan, torch.zeros_like(row_min), row_min)

    score_mat = torch.where(torch.isnan(score_mat), row_min[:, None], score_mat)

    # --- 全局 min-max 归一化 ---
    smin, smax = score_mat.min(), score_mat.max()
    if (smax - smin) > 0:
        score_mat = (score_mat - smin) / (smax - smin)
    else:
        score_mat.zero_()

    # --- 双塔相似度同样归一化并轻量融合 ---
    sims_norm = sims_matrix
    smin2, smax2 = sims_norm.min(), sims_norm.max()
    if (smax2 - smin2) > 0:
        sims_norm = (sims_norm - smin2) / (smax2 - smin2)
    else:
        sims_norm = torch.zeros_like(sims_norm)

    score_mat = score_mat + 0.002 * sims_norm

    return score_mat.detach().cpu()
