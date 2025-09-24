from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Siglip2Model

from . import siglip2
from .bert import BertConfig, BertForMaskedLM


@dataclass
class SigLIP2CMPOutput:
    loss: Optional[torch.Tensor] = None
    image_embeds: Optional[torch.Tensor] = None
    text_embeds: Optional[torch.Tensor] = None
    image_feat: Optional[torch.Tensor] = None
    text_feat: Optional[torch.Tensor] = None


@dataclass
class SigLIP2CMPConfig:
    siglip_pretrained: Optional[Path]
    bert_conf: BertConfig
    siglip_loss_coeff: float = 0.4
    itm_loss_coeff: float = 0.6
    label_smooth: float = 0.2

    @staticmethod
    def from_yaml_obj(obj: dict[str, str | int | object]):
        siglip_pretrained = obj['siglip_pretrained']
        label_smooth = obj['label_smooth']
        bert_conf_file = obj['text_config']

        assert isinstance(siglip_pretrained, str)
        assert isinstance(label_smooth, float)
        assert isinstance(bert_conf_file, str)

        bert_conf = BertConfig.from_json_file(Path(bert_conf_file))

        return SigLIP2CMPConfig(
            siglip_pretrained=Path(siglip_pretrained),
            bert_conf=bert_conf,
            label_smooth=label_smooth,
        )


class SigLIP2CMP(nn.Module):
    MODEL_NAME = 'google/siglip2-base-patch16-naflex'
    SIGLIP_DIM = 768

    def __init__(
        self,
        siglip2: Siglip2Model,
        conf: SigLIP2CMPConfig,
    ):
        """
        Args:
        siglip2: SigLIP2的模型本身，使用`models.siglip2`
        """
        super().__init__()
        self.siglip2 = siglip2
        self.device = self.siglip2.device
        self.bert = SigLIP2CMP.build_bert(conf.bert_conf)
        self.epsilon = conf.label_smooth
        self.siglip_loss_coeff = conf.siglip_loss_coeff
        self.itm_loss_coeff = conf.itm_loss_coeff
        self.conf = conf

        input_dim = SigLIP2CMP.SIGLIP_DIM
        output_dim = 2
        self.itm_head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim),
        )

        self.text_proj = SigLIP2CMP.feature_mapping(input_dim, input_dim, 0.4)
        self.image_proj = SigLIP2CMP.feature_mapping(input_dim, input_dim, 0.4)

    @staticmethod
    def verify_bert_conf(bert_conf: BertConfig):
        """
        对用于初始化的BertConfig进行验证
        """
        if bert_conf.encoder_width != SigLIP2CMP.SIGLIP_DIM:
            raise ValueError(
                f'Invalid BERT encoder width: {bert_conf.encoder_width}, expected: {SigLIP2CMP.SIGLIP_DIM}'
            )

    @staticmethod
    def build_model(
        config: SigLIP2CMPConfig,
        device: torch.device | str,
    ):
        sigmodel, proc, tokenizer = siglip2.build_model(
            device, local_file=config.siglip_pretrained
        )
        model = SigLIP2CMP(sigmodel, config).to(device)
        return model, proc, tokenizer

    @staticmethod
    def from_pretrained(siglip2_ckpt: Path):
        raise NotImplementedError()

    @staticmethod
    def build_bert(config: BertConfig):
        bert = BertForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=config.name_or_path,
            config=config,
            output_loading_info=False,
            local_files_only=True,
        )
        p = bert.cls.predictions.decoder.bias
        bert.cls.predictions.decoder.bias = nn.Parameter(torch.zeros(p.shape))

        return bert

    def forward(
        self,
        idx,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
    ):
        sigout = self.siglip2.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=False,
            output_hidden_states=True,
        )
        dev = self.siglip2.device

        image_embeds: torch.FloatTensor = sigout.vision_model_output.hidden_states[-1]  # type: ignore
        image_feat = sigout.image_embeds
        image_atts = torch.ones(image_embeds.shape[:2], device=dev)
        text_embeds: torch.FloatTensor = sigout.text_model_output.hidden_states[-1]  # type: ignore
        text_feat = sigout.text_embeds
        text_atts: torch.Tensor
        if attention_mask is not None:
            text_atts = attention_mask  # CMP的做法
        else:
            text_atts = torch.ones(text_embeds.shape[:2], device=dev)

        # Loss Calculation
        loss: Optional[torch.Tensor] = None
        if return_loss is not None and return_loss:
            loss = torch.tensor(0.0, device=dev)
            if sigout.loss:
                loss += sigout.loss * self.siglip_loss_coeff

            loss += (
                self.get_matching_loss(
                    image_embeds=image_embeds,
                    image_atts=image_atts,
                    image_feat=image_feat,
                    text_embeds=text_embeds,
                    text_atts=text_atts,
                    text_feat=text_feat,
                    idx=idx,
                )
                * self.itm_loss_coeff
            )

        output = SigLIP2CMPOutput(
            loss=loss,
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            image_feat=image_feat,
            text_feat=text_feat,
        )
        return output

    def get_image_features(self, **kwargs):
        return self.siglip2.get_image_features(**kwargs)

    def get_text_features(self, **kwargs):
        return self.siglip2.get_text_features(**kwargs)

    def get_cross_embeds(
        self,
        image_embeds,
        image_atts,
        text_embeds,
        text_atts,
    ):
        encoder = self.bert.bert
        return encoder(
            encoder_embeds=text_embeds,
            attention_mask=text_atts,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode='fusion',
        ).last_hidden_state

    def get_matching_loss(
        self,
        image_embeds,
        image_atts,
        image_feat,
        text_embeds,
        text_atts,
        text_feat,
        idx,
    ):
        """
        Matching Loss with in-batch hard negatives
        """
        bs = image_embeds.size(0)

        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        with torch.no_grad():
            logit_scale = self.siglip2.logit_scale.exp().clamp(1e-3, 100)
            sim_i2t = image_feat @ text_feat.t() * logit_scale
            sim_t2i = text_feat @ image_feat.t() * logit_scale
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5

            idx = idx.view(-1, 1)
            assert idx.size(0) == bs
            mask = torch.eq(idx, idx.t())
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0)

        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

        cross_pos = self.get_cross_embeds(
            image_embeds,
            image_atts,
            text_embeds=text_embeds,
            text_atts=text_atts,
        )[:, 0, :]
        cross_neg = self.get_cross_embeds(
            image_embeds_all,
            image_atts_all,
            text_embeds=text_embeds_all,
            text_atts=text_atts_all,
        )[:, 0, :]

        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image_embeds.device)
        itm_loss = F.cross_entropy(output, itm_labels)

        return itm_loss

    @staticmethod
    def feature_mapping(input_dim: int, output_dim: int, dropout_p: float = 0.0):
        from torch.nn import init

        mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(input_dim, output_dim),
        )
        assert isinstance(mlp[2].weight.data, torch.Tensor)
        assert isinstance(mlp[2].bias.data, torch.Tensor)

        init.normal_(mlp[2].weight.data, std=0.00001)
        init.constant_(mlp[2].bias.data, 0.0)
        return mlp


__all__ = [
    'SigLIP2CMP',
    'SigLIP2CMPOutput',
]
