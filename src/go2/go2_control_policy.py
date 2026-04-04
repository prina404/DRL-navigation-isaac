import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from cfg.CFG import CHECKPOINT_DIR, DEVICE


class Go2CtrlPolicy(nn.Module):
	"""PyTorch equivalent of the JAX/Flax Policy network """

	def __init__(self, action_space: int = 12, policy_mean_abs_clip: float = 10.0):
		super().__init__()
		self.policy_mean_abs_clip = policy_mean_abs_clip

		# Input size is fixed to 48 to match the observation vector used by the JAX model.
		self.dense0 = nn.Linear(48, 512)
		self.layer_norm = nn.LayerNorm(512, eps=1e-6)
		self.dense1 = nn.Linear(512, 256)
		self.dense2 = nn.Linear(256, 128)
		self.dense3 = nn.Linear(128, action_space)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.dense0(x)
		x = self.layer_norm(x)
		x = torch.nn.functional.elu(x)
		x = self.dense1(x)
		x = torch.nn.functional.elu(x)
		x = self.dense2(x)
		x = torch.nn.functional.elu(x)
		x = self.dense3(x)
		x = torch.clamp(x, -self.policy_mean_abs_clip, self.policy_mean_abs_clip)
		return x	

	def load_ckpts(self, path: str | Path):
		with open(path, "rb") as f:
			state_dict = torch.load(f, map_location="cpu")
		self.load_state_dict(state_dict)


def get_mpc_policy() -> Go2CtrlPolicy:
	policy = Go2CtrlPolicy()
	ckpts = CHECKPOINT_DIR / "unitree_go2_mpc/go2_control_jax_to_torch.pt"
	policy.load_ckpts(ckpts)
	return policy.eval().to(DEVICE)