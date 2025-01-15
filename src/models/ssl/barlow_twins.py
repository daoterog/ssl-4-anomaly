import os
from typing import Any, List, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig

import wandb
from src.models.ssl.base import BaseSSL


def barlow_loss_func(
    z1: torch.Tensor, z2: torch.Tensor, lamb: float = 5e-3, scale_loss: float = 0.025
) -> torch.Tensor:
    """Computes Barlow Twins' loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        lamb (float, optional): off-diagonal scaling factor for the cross-covariance matrix.
            Defaults to 5e-3.
        scale_loss (float, optional): final scaling factor of the loss. Defaults to 0.025.

    Returns:
        torch.Tensor: Barlow Twins' loss.
    """

    N, D = z1.size()

    # to match the original code
    bn = torch.nn.BatchNorm1d(D, affine=False).to(z1.device)
    z1 = bn(z1)
    z2 = bn(z2)

    corr = torch.einsum("bi, bj -> ij", z1, z2) / N

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(corr)
        world_size = dist.get_world_size()
        corr /= world_size

    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    cdif[~diag.bool()] *= lamb
    loss = scale_loss * cdif.sum()
    return loss


class BarlowTwins(BaseSSL):
    """Implements Barlow Twins (https://arxiv.org/abs/2103.03230)

    Extra cfg settings:
        ssl_settings:
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            proj_output_dim (int): number of dimensions of projected features.
            lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
            scale_loss (float): scaling factor of the loss.
    """

    def __init__(self, conf_dict: DictConfig, criterion: nn.Module):
        super().__init__(conf_dict, criterion)

        self.lamb: float = conf_dict.ssl_settings.lamb
        self.scale_loss: float = conf_dict.ssl_settings.scale_loss

        self.proj_hidden_dim: int = conf_dict.ssl_settings.proj_hidden_dim
        self.proj_output_dim: int = conf_dict.ssl_settings.proj_output_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, self.proj_hidden_dim),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
        )

    def log_params_to_wandb(self):
        """Log the parameters of the model to wandb."""
        super().log_params_to_wandb()
        wandb.config.update(
            {
                "ssl_settings": {
                    "ssl_method": "barlow_twins",
                    "proj_hidden_dim": self.proj_hidden_dim,
                    "proj_output_dim": self.proj_output_dim,
                    "lambda": self.lamb,
                    "scale_loss": self.scale_loss,
                }
            }
        )

    def load_individual_components(
        self, individual_components_path: str, load_classifier: bool
    ):
        """If a trained model is provided, load the individual components."""

        super().load_individual_components(individual_components_path, load_classifier)

        self.projector.load_state_dict(
            torch.load(
                os.path.join(individual_components_path, "projector.pth"),
                weights_only=True,
            )
        )

    def save_model(self, path: str) -> None:
        """Saves the model to the specified path.

        Args:
            path (str): path to save the model.
        """
        super().save_model(path)
        torch.save(self.projector.state_dict(), os.path.join(path, "projector.pth"))

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()}
        ]
        return super().learnable_params + extra_learnable_params

    def ssl_forward(self, X):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        z1, z2 = out["z"]

        # ------- barlow twins loss -------
        final_loss = barlow_loss_func(
            z1, z2, lamb=self.lamb, scale_loss=self.scale_loss
        )

        # TODO: Figure out how to get Z metric
        # self.compute_and_log_ssl_metrics(feats=out["feats"], z=Z)
        if self.only_log_on_epoch_end:
            self.log("ssl_loss", final_loss, on_step=True, sync_dist=False)
        self.log("ssl_loss_epoch", final_loss, on_epoch=True, sync_dist=True)

        # We sum the classification loss to the BYOL loss so the optimizer is able to
        # update the weights of the classifier (w.r.t. the supervised loss) and the backbone
        # (w.r.t. the BYOL loss) in the same loop.
        avg_classif_loss = out["classif_loss"]

        # Check for nan or inf values in the loss
        if torch.isnan(final_loss) or torch.isinf(  # pylint: disable=no-member
            final_loss
        ):
            print("SSL loss is NaN or Inf")
            self.trainer.should_stop = True
            return None

        if torch.isnan(avg_classif_loss) or torch.isinf(  # pylint: disable=no-member
            avg_classif_loss
        ):
            print("Classification loss is NaN or Inf")
            self.trainer.should_stop = True
            return None

        return final_loss + avg_classif_loss
