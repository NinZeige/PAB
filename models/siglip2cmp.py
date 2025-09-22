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


class SigLIP2CMP(nn.Module):
    MODEL_NAME = 'google/siglip2-base-patch16-naflex'
    SIGLIP_LOSS_COEFF = 1.0
    ITMHEAD_LOSS_COEFF = 1.0

    def __init__(
        self,
        siglip2: Siglip2Model,
        cfg: dict[str, int | str | list[str]],
        embed_dim: int = 768,
        itm_hidden: int = 256,
    ):
        """
        `embed_dim`: SigLIP2 输出的特征向量维度
        """
        super().__init__()
        self.siglip2 = siglip2
        self.bert = build_bert(config=cfg, vision_width=embed_dim)
        self.temp = nn.Parameter(torch.ones([]) * cfg['temp'])
        self.epsilon = cfg['label_smooth']

        input_dim = embed_dim
        output_dim = 2
        self.itm_head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim),
        )

        def list_meta(model):
            metas = []
            for name, p in model.named_parameters():
                if getattr(p, 'is_meta', False):
                    metas.append(('param', name, p.shape))
            for name, b in model.named_buffers():
                if getattr(b, 'is_meta', False):
                    metas.append(('buffer', name, b.shape))
            return metas

        metas = list_meta(self)
        if metas:
            print('Found meta tensors:')
            for kind, name, shape in metas:
                print(f'  {kind}: {name} {shape}')
            breakpoint()
        else:
            print('No meta tensors detected.')

    @staticmethod
    def build_model(config, device: torch.device | str):
        sigmodel, proc, tokenizer = siglip2.build_model(device)
        model = SigLIP2CMP(sigmodel, config).to(device)
        return model, proc, tokenizer

    @staticmethod
    def from_pretrained(siglip2_ckpt: Path):
        raise NotImplementedError()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        idx=None,
    ):
        sigout = self.siglip2.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            output_hidden_states=True,
        )
        loss = sigout.loss
        dev = self.siglip2.device

        image_embeds = sigout.vision_model_output.hidden_states[-1]
        image_feat = sigout.image_embeds
        image_atts = torch.ones(image_embeds.shape[:2], device=dev)
        text_embeds = sigout.text_model_output.hidden_states[-1]
        text_feat = sigout.text_embeds
        text_atts = torch.ones(text_embeds.shape[:2], device=dev)

        loss += self.get_matching_loss(
            image_embeds=image_embeds,
            image_atts=image_atts,
            image_feat=image_feat,
            text_embeds=text_embeds,
            text_atts=text_atts,
            text_feat=text_feat,
            idx=idx,
        )

        output = SigLIP2CMPOutput(
            loss=loss,
        )
        return output

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
        idx=None,
    ):
        """
        Matching Loss with in-batch hard negatives
        """
        bs = image_embeds.size(0)

        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp
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


def build_bert(config, vision_width):
    config_text = BertConfig.from_json_file(config['text_config'])
    config_text.encoder_width = vision_width
    bert, msg = BertForMaskedLM.from_pretrained(
        config['text_encoder'],
        config=config_text,
        output_loading_info=True,
        local_files_only=True,
    )
    if config['load_params']:
        print('build_text_encoder: load bert ====>')
        for k, v in msg.items():
            print(f'{k}: {sorted(v)}')

    p = bert.cls.predictions.decoder.bias
    bert.cls.predictions.decoder.bias = nn.Parameter(torch.zeros(p.shape))

    return bert


__all__ = [
    'SigLIP2CMP',
    'ITMOutput',
]
