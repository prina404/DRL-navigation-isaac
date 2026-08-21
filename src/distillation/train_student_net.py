"""Offline distillation of the per-map navigation teachers into a single student actor.

The student is the teacher's actor -- :class:`policy.NavPolicyv2.MLPModelWithEncoders` built from
``go2_policy_cfg["actor"]`` -- without the critic, and with a heteroscedastic distribution head, so it predicts both
the 3-D action mean and a per-observation std. It is fitted by minimizing the forward KL between the recorded
teacher distribution and the student one, ``KL(N(mu_t, sigma_t) || N(mu_s(o), sigma_s(o)))``, which is the cross
entropy of the two minus the constant teacher entropy. Early stopping on the validation loss is the only regularizer.

.. note::
    The recorded ``sigma_t`` is state-independent and constant per map (0.25 to 1.11 across the collection), so the
    std head has no within-map heteroscedasticity to fit: what it can learn is which map the observation came from.
    ``{train,val}/std_error`` is the metric that says whether it does.

Usage::

    env_isaaclab/bin/python src/distillation/train_student_net.py --batch_size=4096 --init_std=0.5 --wandb
"""

from __future__ import annotations

import argparse
import copy
import os
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from loguru import logger
from tensordict import TensorDict
from torch.utils.data import DataLoader

from cfg.CFG import CHECKPOINT_DIR, DISTILLATION_DIR, ROOT_DIR
from distillation.dataset_utils import get_distillation_dataloader
from policy.NavPolicyv2 import ACTOR_OBS_SET, MLPModelWithEncoders, go2_policy_cfg

# Observation groups of the student, the order MUST match the one in``go2_lidar_full`` task.
STUDENT_OBS_GROUPS = ["velocity_buffer", "action_buffer", "global_plan", "lidar"]



class MeanTanhHead(nn.Module):
    """
    Output activation of a heteroscedastic head: ``tanh`` on the mean slice, a clamp on the log-std slice.
    """

    def __init__(self, min_log_std: float = -3.0, max_log_std: float = 1.0) -> None:
        super().__init__()
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, log_std = x[..., 0, :], x[..., 1, :]
        return torch.stack((mean.tanh(), log_std.clamp(self.min_log_std, self.max_log_std)), dim=-2)


class StudentActorModule(L.LightningModule):
    """Fits the teacher's actor architecture to the recorded teacher distributions."""

    def __init__(
        self,
        sample_obs: TensorDict,
        init_std: float = 1.0,
        learning_rate: float = 1.0e-3,
        weight_decay: float = 0.0,
    ) -> None:
        """
        Args:
            sample_obs: One batch of observations, used only to size the network.
            init_std: Std the head starts at, before it learns to predict its own.
            learning_rate: AdamW learning rate.
            weight_decay: AdamW weight decay.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["sample_obs"])

        # the teacher config, with a state-dependent std: the head emits (B, 2, 3), the mean and the log std
        actor_cfg = copy.deepcopy(go2_policy_cfg["actor"])
        actor_cfg.pop("class_name", None)
        actor_cfg["distribution_cfg"] = {
            "class_name": "HeteroscedasticGaussianDistribution",
            "init_std": init_std,
            "std_type": "log",
        }
        actor_cfg["last_activation"] = None  # applied per slice below instead, see MeanTanhHead
        self.actor = MLPModelWithEncoders(
            obs=sample_obs,
            obs_groups={ACTOR_OBS_SET: STUDENT_OBS_GROUPS},
            obs_set=ACTOR_OBS_SET,
            output_dim=3,
            **actor_cfg,
        )
        self.actor.mlp.add_module(str(len(self.actor.mlp)), MeanTanhHead())

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict the action mean for a batch of observation groups."""
        groups = {group: obs[group] for group in STUDENT_OBS_GROUPS}
        return self.actor(TensorDict(groups, batch_size=obs["lidar"].shape[:1]))

    def predict_distribution(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the predicted ``(mean, std)``; ``forward`` gives the mean alone, as inference only needs that."""
        groups = {group: obs[group] for group in STUDENT_OBS_GROUPS}
        latent = self.actor.get_latent(TensorDict(groups, batch_size=obs["lidar"].shape[:1]))
        self.actor.distribution.update(self.actor.mlp(latent))
        return self.actor.distribution.mean, self.actor.distribution.std

    def training_step(self, batch: tuple[dict, dict], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[dict, dict], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

    def _shared_step(self, batch: tuple[dict, dict], stage: str) -> torch.Tensor:
        """KL(N(mu_t, sigma_t) || N(mu_s, sigma_s)) summed over the action dims, averaged over the batch."""
        obs, target = batch
        mean, std = self.predict_distribution(obs)
        # the loader is built with with_std=True, so a missing action_std is a bug worth crashing on
        teacher = (target["action_mean"], target["action_std"])
        loss = self.actor.get_kl_divergence(teacher, (mean, std)).mean()

        squared_error = (mean - target["action_mean"]).square()
        batch_size = squared_error.shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/mse", squared_error.mean(), batch_size=batch_size)
        # how far the std head is from the (per-map constant) teacher std, i.e. whether it learned anything at all
        self.log(f"{stage}/std_error", (std - target["action_std"]).abs().mean(), batch_size=batch_size)
        # per-channel, because the yaw command is the one that is hard to imitate
        for i, name in enumerate(("vx", "vy", "wz")):
            self.log(f"{stage}/mse_{name}", squared_error[:, i].mean(), batch_size=batch_size)
            self.log(f"{stage}/std_{name}", std[:, i].mean(), batch_size=batch_size)
        return loss


def export_actor_checkpoint(module: StudentActorModule, path: Path, iteration: int = 0) -> None:
    """Write the student weights in the ``actor_state_dict`` layout the rsl-rl runner loads.

    The heteroscedastic distribution owns no parameters, so the actor state dict is the whole model. It does **not**
    load into a runner built from the stock :data:`policy.NavPolicyv2.go2_policy_cfg`: that config declares a
    state-independent ``GaussianDistribution`` (different head shape) and a ``last_activation`` of ``"tanh"``, which
    on a heteroscedastic head would squash the log std too. Reproduce the ``infos`` entries below on the actor
    config first, and load with ``load_cfg={"critic": False}``, as no critic is distilled.
    """
    state_dict = {key: value.detach().cpu().clone() for key, value in module.actor.state_dict().items()}
    torch.save(
        {
            "actor_state_dict": state_dict,
            "iter": int(iteration),
            "infos": {
                "distribution_cfg": {
                    "class_name": "HeteroscedasticGaussianDistribution",
                    "init_std": module.hparams.init_std,
                    "std_type": "log",
                },
                "last_activation": "MeanTanhHead",  # not the config's "tanh", see the class
                "obs_groups": {ACTOR_OBS_SET: STUDENT_OBS_GROUPS},
            },
        },
        path,
    )
    logger.info(f"Exported student actor to {path}")


def peek_sample_obs(loader: DataLoader) -> TensorDict:
    """Pull one batch off a loader and keep a single row of it, to size the network before training starts."""
    obs, _ = next(iter(loader))
    missing = [group for group in STUDENT_OBS_GROUPS if group not in obs]
    if missing:
        raise KeyError(f"The dataloader does not provide the observation groups {missing}, got {sorted(obs)}")
    return TensorDict({group: obs[group][:1].float() for group in STUDENT_OBS_GROUPS}, batch_size=[1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill the navigation teachers into a single student actor.")
    # data
    parser.add_argument("--data_dir", type=Path, default=DISTILLATION_DIR, help="Root of the recorded rollouts.")
    parser.add_argument("--batch_size", type=int, default=4096, help="Rows per batch, must be a multiple of rows_per_yield.")
    parser.add_argument("--rows_per_yield", type=int, default=64, help="Rows a map yields at once, see dataset_utils.")
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="Training batches per epoch; the stream is infinite.")
    parser.add_argument("--val_batches", type=int, default=50, help="Batches per validation check.")
    parser.add_argument("--stochastic", action="store_true", help="Use the *_stochastic.pt recordings instead of the mean ones.")
    parser.add_argument("--keep_collided", action="store_true", help="Keep the episodes that bumped into something.")
    # optimization
    parser.add_argument("--init_std", type=float, default=1.0, help="Std the head starts at, before it predicts its own.")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient norm clipping, 0 disables it.")
    parser.add_argument("--max_epochs", type=int, default=200, help="Upper bound; early stopping usually fires first.")
    parser.add_argument("--patience", type=int, default=20, help="Validation checks without improvement before stopping.")
    parser.add_argument("--val_every_n_steps", type=int, default=None, help="Validate every N batches instead of every epoch.")
    # runtime
    parser.add_argument("--seed", type=int, default=0, help="Seed for torch and for the map sampling of the loader.")
    parser.add_argument("--accelerator", type=str, default="auto", help="Lightning accelerator, e.g. gpu or cpu.")
    parser.add_argument("--out_dir", type=Path, default=CHECKPOINT_DIR / "distillation", help="Checkpoint directory.")
    parser.add_argument("--run_name", type=str, default=None, help="Run name, defaults to student_<timestamp>.")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B (needs WANDB_PROJECT in .env).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> float:
    """Train the student and return the best validation loss."""
    args = parse_args(argv)
    if args.val_every_n_steps is not None and args.val_every_n_steps > args.steps_per_epoch:
        raise ValueError(f"--val_every_n_steps={args.val_every_n_steps} exceeds --steps_per_epoch={args.steps_per_epoch}")
    L.seed_everything(args.seed, workers=True)

    run_name = args.run_name or f"student_{datetime.now().strftime('%m-%d_%H-%M')}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_distillation_dataloader(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        use_deterministic=not args.stochastic,
        drop_collided=not args.keep_collided,
        steps_per_epoch=args.steps_per_epoch,
        with_std=True,
        seed=args.seed,
        rows_per_yield=args.rows_per_yield,
    )
    sample_obs = peek_sample_obs(train_loader)
    module = StudentActorModule(sample_obs, args.init_std, args.lr, args.weight_decay)
    logger.info(f"Student actor with {sum(p.numel() for p in module.parameters()):,} parameters")

    if args.wandb:
        load_dotenv()
        train_logger = WandbLogger(project="LFP-student-training", name=run_name, save_dir=str(out_dir))
        # wandb cannot serialize the Path arguments
        train_logger.log_hyperparams({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    else:
        train_logger = CSVLogger(save_dir=str(ROOT_DIR / "logs" / "distillation"), name=run_name)

    # `best_checkpoint` keeps the weights of the minimum val/loss, `early_stopping` holds that value in `best_score`
    best_checkpoint = ModelCheckpoint(
        dirpath=str(out_dir), filename="student-{epoch:03d}-{step:06d}", monitor="val/loss", mode="min", save_top_k=1
    )
    early_stopping = EarlyStopping(monitor="val/loss", mode="min", patience=args.patience, verbose=True)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=1,
        logger=train_logger,
        callbacks=[best_checkpoint, early_stopping],
        # the loaders stream restarting iterable datasets, so the epoch ends have to be imposed from here as well
        limit_train_batches=args.steps_per_epoch,
        limit_val_batches=args.val_batches,
        val_check_interval=args.val_every_n_steps,
        gradient_clip_val=args.grad_clip or None,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_loss = float(early_stopping.best_score) if early_stopping.best_score is not None else float("nan")
    logger.info(f"Best validation loss: {best_loss:.6f} ({best_checkpoint.best_model_path or 'no checkpoint written'})")

    if best_checkpoint.best_model_path:
        best = StudentActorModule.load_from_checkpoint(best_checkpoint.best_model_path, sample_obs=sample_obs)
        export_actor_checkpoint(best, out_dir / "student_actor.pt", iteration=trainer.global_step)
    if args.wandb:
        train_logger.experiment.summary["best_val_loss"] = best_loss

    return best_loss


if __name__ == "__main__":
    main()
