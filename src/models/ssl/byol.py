"""This module contains the BYOL model for training the SewerNet model. It provides
the necessary functionality for training the model using the BYOL method."""

import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

import wandb
from src.models.ssl.base import BaseMomentumMethod
from src.utils.momentum import initialize_momentum_params


def byol_loss_func(
    p: torch.Tensor, z: torch.Tensor, simplified: bool = True
) -> torch.Tensor:
    """Computes BYOL's loss given batch of predicted features p and projected momentum features z.

    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        simplified (bool): faster computation, but with same result. Defaults to True.

    Returns:
        torch.Tensor: BYOL's loss.
    """

    if simplified:
        cos_sim = F.cosine_similarity(  # pylint: disable=not-callable
            p, z.detach(), dim=-1
        )
        return 2 - 2 * cos_sim.mean()

    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)

    return 2 - 2 * (p * z.detach()).sum(dim=1).mean()


class BYOL(BaseMomentumMethod):
    """Implements BYOL (https://arxiv.org/abs/2006.07733).

    Extra cfg settings:
        ssl_settings:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
    """

    def __init__(self, conf_dict: DictConfig, criterion: nn.Module):
        super().__init__(conf_dict, criterion)

        self.proj_hidden_dim: int = conf_dict.ssl_settings.proj_hidden_dim
        self.proj_output_dim: int = conf_dict.ssl_settings.proj_output_dim
        self.pred_hidden_dim: int = conf_dict.ssl_settings.predictor_hidden_dim
        self.normalize_projector: bool = conf_dict.ssl_settings.normalize_projector

        # Instantiate Model
        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, self.proj_hidden_dim),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, self.proj_hidden_dim),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.proj_output_dim, self.pred_hidden_dim),
            nn.BatchNorm1d(self.pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.pred_hidden_dim, self.proj_output_dim),
        )

    def log_params_to_wandb(self):
        """Log the parameters of the model to wandb."""
        super().log_params_to_wandb()

        wandb.config.update(
            {
                "ssl_settings": {
                    "ssl_method": "byol",
                    "proj_hidden_dim": self.proj_hidden_dim,
                    "proj_output_dim": self.proj_output_dim,
                    "pred_hidden_dim": self.pred_hidden_dim,
                    "normalize_projector": self.normalize_projector,
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
        self.momentum_projector.load_state_dict(
            torch.load(
                os.path.join(individual_components_path, "momentum_projector.pth"),
                weights_only=True,
            )
        )
        self.predictor.load_state_dict(
            torch.load(
                os.path.join(individual_components_path, "predictor.pth"),
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
        torch.save(self.predictor.state_dict(), os.path.join(path, "predictor.pth"))

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [
            ("projector", (self.projector, self.momentum_projector))
        ]
        return super().momentum_pairs + extra_momentum_pairs

    def ssl_forward(
        self, X: torch.Tensor
    ) -> Dict[str, Any]:  # pylint: disable=invalid-name
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        out = super().ssl_forward(X)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        if self.normalize_projector:
            z = F.normalize(z, dim=-1)
        out.update({"z": z, "p": p})
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
        p = self.predictor(z)
        if self.normalize_projector:
            z = F.normalize(z, dim=-1)
        out.update({"z": z, "p": p})
        return out

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:  # pylint: disable=invalid-name
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of
                the parent and the momentum projected features.
        """

        out = super().momentum_forward(X)
        z = self.momentum_projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        Z = out["z"]  # pylint: disable=invalid-name
        P = out["p"]  # pylint: disable=invalid-name
        Z_momentum = out["momentum_z"]  # pylint: disable=invalid-name

        # ------- negative consine similarity loss -------
        final_loss = 0
        # Compute loss comparing all view with every other view (except itself)
        for v1 in range(self.num_crops):
            # Use np.delete to remove the current view from the list of views
            for v2 in np.delete(range(self.num_crops), v1):
                final_loss += byol_loss_func(P[v2], Z_momentum[v1])
        final_loss = final_loss / (self.num_crops * (self.num_crops - 1))

        self.compute_and_log_ssl_metrics(feats=out["feats"], z=Z)
        self.compute_and_log_ssl_metrics(
            feats=out["momentum_feats"], z=Z_momentum, prefix="momentum_"
        )
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
