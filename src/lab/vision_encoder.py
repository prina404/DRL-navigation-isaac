from pathlib import Path
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from loguru import logger   

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            'efficientvit_m3.r224_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
            cache_dir=Path('~/.cache/torch/hub/timm').expanduser()
        ).eval()
        data_config = timm.data.resolve_model_data_config(self.vit)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        logger.info("ViT Encoder initialized")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: model output is not normalized, consider adding normalization
        return self.vit(self.transforms(x))
