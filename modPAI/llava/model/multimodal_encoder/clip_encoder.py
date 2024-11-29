import logging

import torch
import torch.nn as nn
import numpy as np

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .eva_clip.configuration_evaclip  import EvaCLIPVisionConfig
from .eva_clip.modeling_evaclip import EvaCLIPVisionModel
from .intern_vit_6b.configuration_intern_vit import InternVisionConfig
from .intern_vit_6b.modeling_intern_vit import InternVisionModel
from .internvl_14b.configuration_internvl import InternVLConfig
from .internvl_14b.modeling_internvl import InternVLModel


def get_top_ratio(attn):
    attn_np = attn.cpu().numpy().flatten()

    Q1 = np.percentile(attn_np, 25)
    Q3 = np.percentile(attn_np, 75)
    IQR = Q3 - Q1

    # lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = np.where((attn_np > upper_bound))[0]

    ratio = len(outlier_indices) / len(attn_np)
    return ratio

def update_modpai_alpha(vit_attn, shared_dict, num_vision_tokens=576):
    top_attn = vit_attn[0][0, :, 0, -num_vision_tokens:].sum(dim=0)
    bottom_attn = vit_attn[-2][0, :, 0, -num_vision_tokens:].sum(dim=0)
    _, top_tokens = torch.topk(top_attn, int(num_vision_tokens*0.25), largest=True)
    _, bottom_tokens = torch.topk(bottom_attn, int(num_vision_tokens*get_top_ratio(bottom_attn)), largest=True)
    top_tokens_set = set(top_tokens.tolist())
    bottom_tokens_set = set(bottom_tokens.tolist())
    top_tokens = torch.tensor(list(top_tokens_set - bottom_tokens_set), dtype=top_tokens.dtype, device=top_tokens.device)

    shared_dict['top_tokens'] = top_tokens
    shared_dict['bottom_tokens'] = bottom_tokens

    # print(f"top_tokens: {len(top_tokens)}, bottom_tokens: {len(bottom_tokens)}")



def is_intern_vit_6b_model(vision_tower_name):
    model_names = ["intern_vit_6b", "internvit_6b", "InternViT-6B", "internvit6b"]
    return any(name in vision_tower_name for name in model_names)


def is_internvl_14b_model(vision_tower_name):
    model_names = ["internvl_14b", "intern_vl_14b", "InternVL-14B", "internvl14b"]
    return any(name in vision_tower_name for name in model_names)


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            if "EVA" in self.vision_tower_name or "eva" in self.vision_tower_name:
                self.cfg_only = EvaCLIPVisionConfig.from_pretrained(self.vision_tower_name)
            elif is_intern_vit_6b_model(self.vision_tower_name):
                self.cfg_only = InternVisionConfig.from_pretrained(self.vision_tower_name)
            elif is_internvl_14b_model(self.vision_tower_name):
                self.cfg_only = InternVLConfig.from_pretrained(self.vision_tower_name)
            else:
                self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if "EVA" in self.vision_tower_name or "eva" in self.vision_tower_name:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = EvaCLIPVisionModel.from_pretrained(self.vision_tower_name)
        elif is_intern_vit_6b_model(self.vision_tower_name):
            crop_size = 448 if "448" in self.vision_tower_name else 336
            self.image_processor = CLIPImageProcessor(
                crop_size=crop_size, do_center_crop=True, do_normalize=True, do_resize=True,
                image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size=crop_size
            )
            # self.vision_tower = InternVisionModel.from_pretrained(self.vision_tower_name)
            self.vision_tower = InternVisionModel.from_pretrained("/root/autodl-tmp/InternViT-6B-224px")
        elif is_internvl_14b_model(self.vision_tower_name):
            self.image_processor = CLIPImageProcessor(
                crop_size=336, do_center_crop=True, do_normalize=True, do_resize=True,
                image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size=336
            )
            self.vision_tower = InternVLModel.from_pretrained(self.vision_tower_name)
            self.vision_tower.eval()
        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                if is_internvl_14b_model(self.vision_tower_name):
                    image_forward_out, query_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append([image_feature, query_out])
                else:
                    image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
        else:
            if is_internvl_14b_model(self.vision_tower_name):
                image_forward_outs, query_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)
                image_features = [image_features, query_outs]
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)
                update_modpai_alpha(image_forward_outs.attentions, self.shared_dict)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        if is_internvl_14b_model(self.vision_tower_name):
            return (self.config.image_size // self.config.patch_size) ** 2 + 96
        else:
            return (self.config.image_size // self.config.patch_size) ** 2
