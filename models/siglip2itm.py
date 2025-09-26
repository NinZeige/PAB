from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import os

import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    Siglip2Model,
    Siglip2ImageProcessorFast,
    AutoTokenizer,
)

from . import siglip2


@dataclass
class Siglip2ITMOutput:
    text_feat: torch.FloatTensor
    image_feat: torch.FloatTensor
    text_embeds: torch.Tensor
    image_embeds: torch.Tensor
    loss: Optional[torch.FloatTensor] = None


@dataclass
class Siglip2ITMConfig:
    siglip2_checkpoint: Optional[str]


class Siglip2ITM(nn.Module):
    """
    Use Siglip2Model as text and vision encoder
    Use a simple MHA + ITM to score the similarity
    """

    EMBEDS_DIM = 768
    SIG_MODEL_NAME = 'google/siglip2-base-patch16-naflex'

    def __init__(
        self, conf: Siglip2ITMConfig, device: torch.device | str | None = None
    ):
        super().__init__()

        siglip2_ckpt = (
            Path(conf.siglip2_checkpoint) if conf.siglip2_checkpoint else None
        )
        sig_model, *_ = siglip2.build_model(device, siglip2_ckpt)

        # Freeze SigLIP2 and train Text2Image Matching head only
        self.siglip2 = sig_model
        Siglip2ITM.freeze_siglip2(self.siglip2)

        self.itm = Text2ImageScorer(d=512)
        if device:
            self.itm = self.itm.to(device)

    def save_ckpt(self, path: Path, best_mAP: float):
        if path.is_dir():
            raise ValueError('Invalid path')
        if not path.parent.exists():
            os.makedirs(path.parent)

        torch.save(
            {
                'model': self.state_dict(),
                'best_mAP': best_mAP,
            },
            f=path,
        )

    @classmethod
    def from_ckpt(cls, path: Path):
        if not path.exists():
            raise ValueError('Invalid path')
        obj = torch.load(path)
        model = cls(Siglip2ITMConfig(None))
        model.load_state_dict(obj['model'])
        best_mAP: float = obj['best_mAP']
        return model, best_mAP

    @staticmethod
    def get_processor():
        return Siglip2ImageProcessorFast.from_pretrained(Siglip2ITM.SIG_MODEL_NAME)

    @classmethod
    def get_tokenizer(cls):
        tok = AutoTokenizer.from_pretrained(Siglip2ITM.SIG_MODEL_NAME)
        return tok

    @staticmethod
    def freeze_siglip2(model: Siglip2Model):
        for p in model.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: bool = False,
    ):
        with torch.no_grad():
            sigout = self.siglip2.forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )

        image_embeds: torch.FloatTensor = sigout.vision_model_output.hidden_states[-1]  # type: ignore
        image_feat: torch.FloatTensor = sigout.image_embeds  # type: ignore
        text_embeds: torch.FloatTensor = sigout.text_model_output.hidden_states[-1]  # type: ignore
        text_feat: torch.FloatTensor = sigout.text_embeds  # type: ignore

        loss: torch.FloatTensor | None = None
        if return_loss:
            loss = self.get_matching_loss(
                image_embeds, text_embeds, image_feat, text_feat
            )  # type: ignore

        return Siglip2ITMOutput(text_feat, image_feat, text_embeds, image_embeds, loss)

    def get_matching_loss(
        self,
        image_embeds: torch.FloatTensor,
        text_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
    ):
        """
        仅训练时使用，获取相似度中前k名，并传给Text2Image头匹配
        """
        batch_size = text_embeds.shape[0]
        select_range = 8
        sim_mat = text_features @ image_features.t()
        relative_idx = torch.arange(
            batch_size,
            device=sim_mat.device,
        )  # 在Batch内的临时编号，便于确定正确的lable

        # 获取topk的编号，并修正未在前`select_range`名的情况
        _, topk_indices = sim_mat.topk(k=select_range, dim=1)
        matched: torch.BoolTensor = topk_indices == relative_idx.unsqueeze(-1)  # type: ignore
        assert matched.shape == torch.Size([batch_size, select_range])
        no_correct_match = matched.sum(1) == 0
        topk_indices[no_correct_match, -1] = relative_idx[no_correct_match]
        image_embeds_candidate = image_embeds[topk_indices]

        # Shuffle
        perm = torch.randperm(select_range, device=sim_mat.device)
        topk_indices = topk_indices[:, perm]
        image_embeds_candidate = image_embeds_candidate[:, perm]

        # Update `matched`
        matched: torch.BoolTensor = topk_indices == relative_idx.unsqueeze(-1)  # type: ignore
        real_label = matched.nonzero()[:, 1]

        assert image_embeds_candidate.shape == torch.Size(
            [batch_size, 8, *image_embeds.shape[1:]]
        )

        itm_out = self.itm.forward(
            text_embeds,
            image_embeds_candidate,
        )

        return F.cross_entropy(itm_out, real_label)

    def get_matching_result(
        self,
        sim_mat: torch.Tensor,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ):
        """
        评估时使用
        Args:
        sim_mat: `softmax(text_feat @ image_feat.t())`的结果
        image_embeds: 所有测试集的image_embeds，从forward中获取。即 `[len(loader), 256, 768]`
        text_embeds: 所有测试集的text_embeds。形状 `[len(loader), 64, 768]`
        """
        K = 8

        _, topk_indices = sim_mat.topk(K, dim=1)  # [BS, 8]
        itmout = self.itm.forward(
            text_embeds,
            image_embeds[topk_indices],
        )  # [BS, 8]

        low_score = itmout.min().min().item()  # [BS, 1]
        return_score = torch.full(
            sim_mat.shape, low_score, device=sim_mat.device
        )  # [BS, BS]

        return_score.scatter_(1, topk_indices, itmout)
        return return_score


class Text2ImageScorer(nn.Module):
    def __init__(self, d: int, heads=8) -> None:
        super().__init__()
        self.text_proj = nn.Linear(Siglip2ITM.EMBEDS_DIM, d, bias=False)
        self.image_proj = nn.Linear(Siglip2ITM.EMBEDS_DIM, d, bias=False)
        self.u = nn.Parameter(torch.randn(d))  # Text pooling
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.Wb = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.Wb)

        self.mlp = nn.Sequential(nn.Linear(3 * d, 512), nn.ReLU(), nn.Linear(512, 1))
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.5))
        self.tau = nn.Parameter(torch.tensor(0.07))
        self.d = d

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        batch_size = text_embeds.shape[0]
        Q_, t_q = self.text_to_query(text_embeds)
        Q = Q_.reshape(8 * batch_size, 1, self.d)
        K = self.image_to_key(image_embeds)
        V = K

        C_, _ = self.mha(Q, K, V)
        assert C_.shape == torch.Size([batch_size * 8, 1, self.d])
        c = C_.squeeze(1).view(batch_size, 8, self.d)

        # 相似度通道
        t_hat = F.normalize(t_q, dim=-1)  # [BS,d]
        C_hat = F.normalize(c, dim=-1)  # [BS,8,d]
        sim_cos = (t_hat.unsqueeze(1) * C_hat).sum(-1)  # [BS,8]

        sim_bil = torch.einsum('bd,dd,bed->be', t_q, self.Wb, c)  # [BS,8]

        # MLP 通道
        feat = torch.cat(
            [c, t_q.unsqueeze(1).expand(-1, 8, -1), t_q.unsqueeze(1) * c], dim=-1
        )
        mlp_out = self.mlp(feat).squeeze(-1)  # [BS,8]
        assert mlp_out.shape == torch.Size([batch_size, 8])

        s: torch.FloatTensor = (
            self.w1 * sim_cos + self.w2 * sim_bil + mlp_out
        ) * self.tau.exp()

        assert s.shape == torch.Size([batch_size, 8])
        return s

    def text_to_query(self, text_embeds: torch.Tensor):
        batch_size = len(text_embeds)
        t: torch.Tensor = self.text_proj(text_embeds)  # [BS, 64, d]
        assert t.shape == torch.Size([batch_size, 64, self.d])

        alpha = torch.softmax(t @ self.u, dim=1)
        t_q = (alpha.unsqueeze(-1) * t).sum(1)
        assert t_q.shape == torch.Size([batch_size, self.d])

        return t_q.view(batch_size, 1, 1, -1).expand(-1, 8, 1, -1), t_q  # [BS, 8, 1, d]

    def image_to_key(self, image_embes: torch.Tensor):
        batch_size = len(image_embes)
        assert image_embes.shape == torch.Size([batch_size, 8, 256, 768])
        k: torch.Tensor = self.image_proj(image_embes)

        K_ = k.reshape(batch_size * 8, 256, self.d)
        return K_
