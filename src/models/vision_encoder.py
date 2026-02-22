from pathlib import Path

import timm
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from loguru import logger
from vint_train.models.vint.vint import ViNT

from cfg.CFG import VINT_MODEL_WEIGHTS


class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            "efficientvit_m3.r224_in1k",
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
            cache_dir=Path("~/.cache/torch/hub/timm").expanduser(),
        ).eval()
        data_config = timm.data.resolve_model_data_config(self.vit)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        logger.info("ViT Encoder initialized")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: model output is not normalized, consider adding normalization
        return self.vit(self.transforms(x))


class ViNTVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vint_full_model: nn.Module = ViNT(mha_num_attention_heads=4, mha_num_attention_layers=4)
        checkpoint = torch.load(VINT_MODEL_WEIGHTS, weights_only=False)["model"]
        vint_full_model.load_state_dict(checkpoint.state_dict())

        self.encoder: EfficientNet = vint_full_model.obs_encoder.eval()
        self.encoder_head = vint_full_model.compress_obs_enc.eval()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        obs_encoding = self.encoder.extract_features(img)
        # currently the size is [batch_size, 1280, H/32, W/32]
        obs_encoding = self.encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size, 1280, 1, 1]
        if self.encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.encoder._dropout(obs_encoding)

        obs_encoding = self.encoder_head(obs_encoding)
        # currently the size is [batch_size, self.obs_encoding_size = 512]
        return obs_encoding


if __name__ == "__main__":
    encoder = ViNTVisionEncoder()
    sample_input = torch.randn((2, 3, 224, 224))  # batch of 2 RGB images of size 224x224
    output = encoder(sample_input)
    print("Output shape:", output.shape)  # should be [2, 512]
