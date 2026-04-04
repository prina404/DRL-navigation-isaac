import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from go2.go2_control_policy import Go2CtrlPolicy


def _to_numpy(array_like: Any) -> np.ndarray:
	return np.asarray(array_like)


def _get_param_tree(jax_params: Mapping[str, Any]) -> Mapping[str, Any]:
	"""Normalizes Flax parameter trees to the direct params subtree."""
	if "params" in jax_params:
		return jax_params["params"]
	return jax_params


@torch.no_grad()
def load_jax_weights_into_torch(model: Go2CtrlPolicy, jax_params: Mapping[str, Any]) -> Go2CtrlPolicy:
	"""
	Loads Flax policy parameters into the equivalent PyTorch model.

	Mapping:
	- Dense_0/1/2/3 kernel -> dense0/1/2/3 weight (transposed)
	- Dense_0/1/2/3 bias   -> dense0/1/2/3 bias
	- LayerNorm_0 scale/bias -> layer_norm weight/bias
	"""
	params = _get_param_tree(jax_params)

	dense_map = [
		("Dense_0", model.dense0),
		("Dense_1", model.dense1),
		("Dense_2", model.dense2),
		("Dense_3", model.dense3),
	]

	for jax_name, torch_layer in dense_map:
		jax_kernel = _to_numpy(params[jax_name]["kernel"])
		jax_bias = _to_numpy(params[jax_name]["bias"])

		# Flax Dense kernel shape: [in_features, out_features]
		# Torch Linear weight shape: [out_features, in_features]
		torch_layer.weight.copy_(torch.from_numpy(jax_kernel.T).to(torch_layer.weight.dtype))
		torch_layer.bias.copy_(torch.from_numpy(jax_bias).to(torch_layer.bias.dtype))

	ln_params = params["LayerNorm_0"]
	model.layer_norm.weight.copy_(torch.from_numpy(_to_numpy(ln_params["scale"])).to(model.layer_norm.weight.dtype))
	model.layer_norm.bias.copy_(torch.from_numpy(_to_numpy(ln_params["bias"])).to(model.layer_norm.bias.dtype))

	return model


def build_torch_policy_from_jax_params(
	jax_params: Mapping[str, Any],
	action_space: int,
	policy_mean_abs_clip: float,
	device: str | torch.device = "cpu",
) -> Go2CtrlPolicy:
	"""Builds the equivalent Torch policy and loads weights from a Flax param tree."""
	model = Go2CtrlPolicy(action_space=action_space, policy_mean_abs_clip=policy_mean_abs_clip)
	model.to(device)
	load_jax_weights_into_torch(model, jax_params)
	model.eval()
	return model
