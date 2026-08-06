"""Navigation policy for ``rsl-rl-lib >= 5.0`` (IsaacSim 6.x / IsaacLab >= 3.0).

This is the port of :class:`policy.NavPolicy.ActorCriticWithEncoders` (``rsl-rl-lib==3.3.0``) to the model-based API
introduced in ``rsl-rl-lib>=4.0``. The learned architecture is unchanged: every observation group listed in the
``encoders_hidden_dims`` mapping is first compressed by its own MLP encoder, the encoder outputs (and the raw tensors
of the groups without an encoder) are concatenated into a single latent, and that latent feeds the actor/critic head.

What changed between the two rsl-rl versions:ĸ

- A single ``ActorCritic`` module holding both networks was replaced by two independent models (``cfg["actor"]`` and
  ``cfg["critic"]``), each an :class:`~rsl_rl.models.MLPModel` subclass. Hence one class here is instantiated twice.
- The observation set previously named ``"policy"`` is now named ``"actor"``. The observations themselves are
  untouched: the model still receives the environment's ``TensorDict`` with one entry per observation group, exactly
  as before. :func:`resolve_legacy_obs_groups` maps the legacy key so the existing task configurations keep working.
- The action distribution moved out of the model into a ``distribution_cfg`` sub-config, so ``init_noise_std``,
  ``noise_std_type`` and ``state_dependent_std`` are now ``init_std``, ``std_type`` and the choice between
  ``GaussianDistribution`` and ``HeteroscedasticGaussianDistribution``.
- ``act()`` / ``act_inference()`` / ``evaluate()`` were replaced by a single ``forward()`` whose ``stochastic_output``
  flag selects sampling or the deterministic output; the runner's ``get_inference_policy()`` now returns the actor
  model itself, which stays call-compatible with ``policy(obs)``.

Both the ``tanh`` output activation of the actor head and the ``sigmoid`` output activation of the encoders are kept,
but note that ``rsl_rl.models.MLPModel`` builds a *linear* head, so ``last_activation`` must be set explicitly in the
configuration (:data:`go2_policy_cfg` below does so). Checkpoints of the old class can be converted with
:func:`convert_legacy_checkpoint`.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict

# Observation set used by the actor, renamed from "policy" in rsl-rl >= 4.0
LEGACY_ACTOR_OBS_SET = "policy"
ACTOR_OBS_SET = "actor"


def resolve_legacy_obs_groups(obs_groups: dict[str, list[str]] | None) -> dict[str, list[str]]:
    """Map the legacy ``"policy"`` observation set to the ``"actor"`` set expected by rsl-rl >= 4.0.

    The task configurations declare ``obs_groups = {"policy": [...], "critic": [...]}``. Renaming the key here keeps
    them untouched while satisfying :func:`rsl_rl.utils.resolve_obs_groups`, which requires an ``"actor"`` set.

    Args:
        obs_groups: Dictionary mapping observation sets to lists of observation groups.

    Returns:
        A copy of the dictionary where the ``"policy"`` set has been renamed to ``"actor"``.
    """
    if obs_groups is None:
        raise ValueError("obs_groups is None, set it before policy initialization")

    resolved = {set_name: list(groups) for set_name, groups in obs_groups.items()}
    if ACTOR_OBS_SET not in resolved and LEGACY_ACTOR_OBS_SET in resolved:
        resolved[ACTOR_OBS_SET] = resolved.pop(LEGACY_ACTOR_OBS_SET)
    return resolved


class MLPModelWithEncoders(MLPModel):
    """MLP model with one dedicated MLP encoder per observation group.

    Each observation group of the model's observation set is passed through its own encoder (an MLP with a ``sigmoid``
    output activation) when the group appears in ``encoders_hidden_dims``, and forwarded untouched otherwise. The
    encoder outputs are concatenated into the model latent, which is then optionally normalized and fed to the head.

    .. note::
        As in the ``rsl-rl-lib==3.3.0`` implementation, the empirical normalization is applied to the *encoded*
        latent, not to the raw observations.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoders_hidden_dims: dict[str, tuple[int, ...] | list[int]] | None = None,
        encoder_activation: str | None = None,
        encoder_last_activation: str | None = "sigmoid",
        last_activation: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model ("actor" or "critic"). The legacy name "policy" is
                accepted as an alias of "actor".
            output_dim: Dimension of the output (number of actions for the actor, 1 for the critic).
            hidden_dims: Hidden dimensions of the head MLP.
            activation: Activation function of the head MLP.
            obs_normalization: Whether to normalize the encoded latent before feeding it to the head MLP.
            distribution_cfg: Configuration of the output distribution. If provided, the model output is stochastic.
            encoders_hidden_dims: Mapping from observation group to the dimensions of its encoder. The last entry is
                the encoder output dimension, the preceding ones are its hidden dimensions. Groups missing from this
                mapping are forwarded to the head without encoding.
            encoder_activation: Activation function of the encoders. Defaults to ``activation``.
            encoder_last_activation: Output activation of the encoders. ``None`` results in a linear encoder output.
            last_activation: Output activation of the head MLP. ``None`` results in a linear output. The actor of the
                legacy implementation used "tanh", the critic used ``None``.
        """
        if kwargs:
            print(
                "MLPModelWithEncoders.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )

        # Resolve the observation set name and the encoder configuration. These are plain attributes set before
        # nn.Module.__init__() runs, as they are needed by _get_obs_dim(), which the parent constructor calls.
        obs_groups = resolve_legacy_obs_groups(obs_groups)
        if obs_set == LEGACY_ACTOR_OBS_SET:
            obs_set = ACTOR_OBS_SET
        self.encoders_hidden_dims = {group: list(dims) for group, dims in (encoders_hidden_dims or {}).items()}
        self.encoder_activation = encoder_activation if encoder_activation is not None else activation
        self.encoder_last_activation = encoder_last_activation

        # Initialize the parent MLP model. It creates the observation normalizer, the output distribution and the head
        # MLP, whose input dimension is the encoded latent dimension returned by _get_obs_dim().
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        # Encoders, registered once the parent constructor has initialized the module
        self.encoders = nn.ModuleDict()
        for group in self.obs_groups:
            self.encoders[group] = self._build_encoder(group)

        # Output activation of the head. It is appended after the parent constructor so that the distribution-specific
        # weight initialization, which indexes the last linear layer of the head, operates on the unmodified MLP.
        if last_activation is not None:
            self.mlp.add_module(str(len(self.mlp)), resolve_nn_activation(last_activation))

    """
    Operations
    """

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor:
        """Build the model latent by encoding and concatenating the observation groups."""
        latent = torch.cat([self.encoders[group](obs[group]) for group in self.obs_groups], dim=-1)
        return self.obs_normalizer(latent)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update the normalization statistics of the encoded latent."""
        if self.obs_normalization:
            with torch.no_grad():
                latent = torch.cat([self.encoders[group](obs[group]) for group in self.obs_groups], dim=-1)
            self.obs_normalizer.update(latent)  # type: ignore

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchMLPModelWithEncoders(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxMLPModelWithEncoders(self, verbose)

    """
    Helper functions
    """

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select the active observation groups and compute the dimension of the encoded latent.

        The returned dimension is the one of the *encoded* latent, so that the head MLP and the observation
        normalizer created by the parent class are both sized for the encoder outputs.
        """
        active_obs_groups = obs_groups[obs_set]
        self.encoder_input_dims: dict[str, int] = {}
        latent_dim = 0
        for group in active_obs_groups:
            if len(obs[group].shape) != 2:
                raise ValueError(f"The model only supports 1D observations, got shape {obs[group].shape} for '{group}'.")
            self.encoder_input_dims[group] = obs[group].shape[-1]
            latent_dim += self._get_encoder_output_dim(group)
        return active_obs_groups, latent_dim

    def _get_encoder_output_dim(self, group: str) -> int:
        """Return the output dimension of the encoder of an observation group."""
        if group in self.encoders_hidden_dims:
            return self.encoders_hidden_dims[group][-1]
        return self.encoder_input_dims[group]

    def _build_encoder(self, group: str) -> nn.Module:
        """Create the encoder of an observation group, or an identity if the group is not encoded."""
        if group not in self.encoders_hidden_dims:
            return nn.Identity()
        hidden_dims = self.encoders_hidden_dims[group]
        return MLP(
            self.encoder_input_dims[group],
            hidden_dims[-1],
            hidden_dims[:-1],
            self.encoder_activation,
            last_activation=self.encoder_last_activation,
        )


class _TorchMLPModelWithEncoders(nn.Module):
    """Exportable model for JIT."""

    def __init__(self, model: MLPModelWithEncoders) -> None:
        """Create a TorchScript-friendly copy of a MLPModelWithEncoders."""
        super().__init__()
        # Convert ModuleDict to ModuleList for ordered iteration
        self.encoders = nn.ModuleList([copy.deepcopy(model.encoders[group]) for group in model.obs_groups])
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference on the observation groups, ordered as in ``model.obs_groups``."""
        latent_list = []
        for i, encoder in enumerate(self.encoders):
            latent_list.append(encoder(obs[i]))
        latent = self.obs_normalizer(torch.cat(latent_list, dim=-1))
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        """Reset the export state (no-op for this model)."""
        pass


class _OnnxMLPModelWithEncoders(nn.Module):
    """Exportable model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: MLPModelWithEncoders, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a MLPModelWithEncoders."""
        super().__init__()
        self.verbose = verbose
        # Convert ModuleDict to ModuleList for ordered iteration
        self.encoders = nn.ModuleList([copy.deepcopy(model.encoders[group]) for group in model.obs_groups])
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.obs_group_names = list(model.obs_groups)
        self.obs_input_dims = [model.encoder_input_dims[group] for group in model.obs_groups]

    def forward(self, *obs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent_list = []
        for i, encoder in enumerate(self.encoders):
            latent_list.append(encoder(obs[i]))
        latent = self.obs_normalizer(torch.cat(latent_list, dim=-1))
        return self.deterministic_output(self.mlp(latent))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        return tuple(torch.zeros(1, dim) for dim in self.obs_input_dims)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return list(self.obs_group_names)

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]


"""
Checkpoint conversion
"""


LEGACY_STATE_DICT_KEY = "model_state_dict"
"""Key under which the legacy runner stored the single actor-critic state dict."""


def convert_legacy_checkpoint(
    legacy_checkpoint: dict[str, Any] | str,
    map_location: str | None = "cpu",
) -> dict[str, Any]:
    """Convert a checkpoint of the legacy ``ActorCriticWithEncoders`` to the rsl-rl >= 5.0 format.

    The legacy runner stored both networks in a single ``"model_state_dict"``, while the new one stores an
    ``"actor_state_dict"`` and a ``"critic_state_dict"``. The parameters themselves are identical, only their names
    change (``actor_encoders.<group>`` -> ``encoders.<group>``, ``actor`` -> ``mlp``, ``log_std`` ->
    ``distribution.log_std_param``, ...). The standard deviation parameterization is taken from the legacy key name,
    which is ``"std"`` for ``noise_std_type="scalar"`` and ``"log_std"`` for ``noise_std_type="log"``.

    Args:
        legacy_checkpoint: Checkpoint dictionary saved by the legacy runner, or a path to it.
        map_location: Device the checkpoint is loaded onto when a path is given.

    Returns:
        A checkpoint dictionary that :meth:`rsl_rl.runners.OnPolicyRunner.load` accepts. The optimizer state is
        dropped, as the new runner uses one optimizer over two models with a different parameter ordering; loading it
        is done with ``load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True}``.
    """
    if isinstance(legacy_checkpoint, str):
        legacy_checkpoint = torch.load(legacy_checkpoint, weights_only=False, map_location=map_location)
    legacy_state_dict = legacy_checkpoint[LEGACY_STATE_DICT_KEY]  # type: ignore

    actor_state_dict: dict[str, torch.Tensor] = {}
    critic_state_dict: dict[str, torch.Tensor] = {}
    for key, value in legacy_state_dict.items():
        if key == "std":
            actor_state_dict["distribution.std_param"] = value
        elif key == "log_std":
            actor_state_dict["distribution.log_std_param"] = value
        elif key.startswith("actor_encoders."):
            actor_state_dict[key.replace("actor_encoders.", "encoders.", 1)] = value
        elif key.startswith("critic_encoders."):
            critic_state_dict[key.replace("critic_encoders.", "encoders.", 1)] = value
        elif key.startswith("actor_obs_normalizer."):
            actor_state_dict[key.replace("actor_obs_normalizer.", "obs_normalizer.", 1)] = value
        elif key.startswith("critic_obs_normalizer."):
            critic_state_dict[key.replace("critic_obs_normalizer.", "obs_normalizer.", 1)] = value
        elif key.startswith("actor."):
            actor_state_dict[key.replace("actor.", "mlp.", 1)] = value
        elif key.startswith("critic."):
            critic_state_dict[key.replace("critic.", "mlp.", 1)] = value
        else:
            raise KeyError(f"Unexpected key '{key}' in the legacy checkpoint.")

    return {
        "actor_state_dict": actor_state_dict,
        "critic_state_dict": critic_state_dict,
        "iter": legacy_checkpoint.get("iter", 0),  # type: ignore
        "infos": legacy_checkpoint.get("infos", None),  # type: ignore
    }


def load_policy_checkpoint(
    runner: Any,
    path: str,
    load_cfg: dict[str, bool] | None = None,
    strict: bool = True,
) -> dict | None:
    """Load a checkpoint into an :class:`~rsl_rl.runners.OnPolicyRunner`, converting legacy checkpoints on the fly.

    Behaves like :meth:`rsl_rl.runners.OnPolicyRunner.load`, except that checkpoints written by the ``rsl-rl-lib<4.0``
    runner are recognized by their ``"model_state_dict"`` entry and converted with :func:`convert_legacy_checkpoint`.
    Legacy checkpoints carry no reusable optimizer state, so it is skipped for them.

    Args:
        runner: The runner to load the checkpoint into.
        path: Path of the checkpoint.
        load_cfg: What to load, see :meth:`rsl_rl.algorithms.PPO.load`. Defaults to everything available.
        strict: Whether the state dict keys must match exactly.

    Returns:
        The ``"infos"`` entry of the checkpoint.
    """
    loaded_dict = torch.load(path, weights_only=False, map_location=runner.device)
    if LEGACY_STATE_DICT_KEY in loaded_dict:
        print(f"Converting legacy (rsl-rl < 4.0) checkpoint '{path}' to the current format.")
        loaded_dict = convert_legacy_checkpoint(loaded_dict)
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": True}

    if runner.alg.load(loaded_dict, load_cfg, strict):
        runner.current_learning_iteration = loaded_dict["iter"]
    return loaded_dict["infos"]


"""
Configuration
"""

# Encoder of every observation group that is compressed before being fed to the actor/critic head. The last entry of
# each list is the encoder output dimension. Groups that are absent are forwarded to the head unencoded.
ENCODERS_HIDDEN_DIMS: dict[str, list[int]] = {
    "velocity_buffer": [128, 128],
    "action_buffer": [128, 128],
    "global_plan": [128, 128],
    "depth": [1024, 1024, 512],
    "vision": [1024, 1024, 512],
}

go2_policy_cfg: dict[str, Any] = {
    # Training loop parameters
    "num_steps_per_env": math.ceil((15 / (16 * 0.005))),  # 15 seconds per episode, converted to steps (with decimation)
    "save_interval": 50,  # Save a checkpoint every 50 iterations
    "max_iterations": 1000,
    "device": "cuda:0",
    # Observation routing
    "obs_groups": None,  # must be assigned depending on task configuration
    # Actor architecture
    "actor": {
        "class_name": MLPModelWithEncoders,
        "hidden_dims": [256, 128, 128],
        "activation": "elu",
        "obs_normalization": False,
        "last_activation": "tanh",  # the legacy actor head ended with a tanh
        "encoders_hidden_dims": ENCODERS_HIDDEN_DIMS,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",  # "HeteroscedasticGaussianDistribution" for a state-dependent std
            "init_std": 1.0,
            "std_type": "log",
        },
    },
    # Critic architecture
    "critic": {
        "class_name": MLPModelWithEncoders,
        "hidden_dims": [256, 128, 128],
        "activation": "elu",
        "obs_normalization": False,
        "last_activation": None,  # the legacy critic head was linear
        "encoders_hidden_dims": ENCODERS_HIDDEN_DIMS,
        "distribution_cfg": None,  # deterministic value output
    },
    # PPO algorithm parameters
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
    "load_run": "unitree_go2_nav",
    "load_checkpoint": "last_ckpt_vis.pt",
}


def make_go2_policy_cfg(obs_groups: dict[str, list[str]], **overrides: Any) -> dict[str, Any]:
    """Build a runner configuration for the given observation groups.

    :class:`~rsl_rl.runners.OnPolicyRunner` consumes its configuration destructively (it pops ``"class_name"`` from
    the model and algorithm sub-configs), so a deep copy of :data:`go2_policy_cfg` is returned on every call.

    Args:
        obs_groups: Observation groups of the task, either with the legacy ``"policy"`` set or the new ``"actor"`` one.
        **overrides: Top-level entries to override, e.g. ``device="cuda:1"`` or ``max_iterations=500``.

    Returns:
        The runner configuration dictionary.
    """
    cfg = copy.deepcopy(go2_policy_cfg)
    cfg["obs_groups"] = resolve_legacy_obs_groups(obs_groups)
    cfg.update(overrides)
    return cfg
