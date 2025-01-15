"""SimCLR implementation."""

import os
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

import wandb
from src.models.ssl.base import BaseSSL
from src.utils.misc import gather, get_rank


def simclr_loss_func(
    z: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)

    # Compute dot-product between every pair of views.
    # Result is z.dim(0) x gathered_z.dim(0)
    sim = torch.exp(  # pylint: disable=no-member
        torch.einsum("if, jf -> ij", z, gathered_z) / temperature
    )

    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)

    # positives
    pos_mask = indexes.t() == gathered_indexes
    # Remove the diagonal of positives from the mask, we only want to consider the off-diagonal
    # positives, since these correspond to the different views of the same image.
    # We use boolean masking so the portion of the mask we select start at the index we want to
    # remove. If not, the mask will be shifted and the diagonal will not be removed.
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)  # pylint: disable=no-member
    neg = torch.sum(sim * neg_mask, 1)  # pylint: disable=no-member
    loss = -(torch.mean(torch.log(pos / (pos + neg))))  # pylint: disable=no-member
    return loss


class SimCLR(BaseSSL):
    """Implements SimCLR (https://arxiv.org/abs/2002.05709).

    Extra cfg settings:
        ssl_settings:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
    """

    def __init__(self, conf_dict: DictConfig, criterion: nn.Module):
        super().__init__(conf_dict, criterion)

        self.proj_hidden_dim: int = conf_dict.ssl_settings.proj_hidden_dim
        self.proj_output_dim: int = conf_dict.ssl_settings.proj_output_dim

        self.temperature: float = conf_dict.ssl_settings.temperature

        # Instantiate Model
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
        )

    def log_params_to_wandb(self):
        """Log the parameters of the model to wandb."""
        super().log_params_to_wandb()
        wandb.config.update(
            {
                "ssl_settings": {
                    "ssl_method": "simclr",
                    "proj_hidden_dim": self.proj_hidden_dim,
                    "proj_output_dim": self.proj_output_dim,
                    "temperature": self.temperature,
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
            ),
        )

    def save_model(self, path: str) -> None:
        """Saves the model to the specified path.

        Args:
            path (str): path to save the model.
        """
        super().save_model(path)
        torch.save(self.projector.state_dict(), os.path.join(path, "projector.pth"))

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Returns the learnable parameters of the model."""
        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()}
        ]
        return super().learnable_params + extra_learnable_params

    def ssl_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        z = F.normalize(z, dim=-1)
        out.update({"z": z})
        return out

    def multicrop_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        z = F.normalize(z, dim=-1)
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes, batch = batch

        out = super().training_step(batch, batch_idx)
        z = out["z"]

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)

        final_loss = simclr_loss_func(
            torch.cat(z, dim=0),  # pylint: disable=no-member
            indexes=indexes,
            temperature=self.temperature,
        )

        self.compute_and_log_ssl_metrics(feats=out["feats"], z=out["z"])
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
