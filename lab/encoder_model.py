import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = transformers.ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k", add_pooling_layer=True
        )
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.vit(x).pooler_output
