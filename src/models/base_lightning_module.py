"""This module contains the base LightningModule for training defect detection models."""

import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from timm.models.convnext import (convnext_base, convnext_large,
                                  convnext_small, convnext_tiny)
from timm.models.efficientnet import efficientnetv2_s
from timm.models.swin_transformer import swin_base_patch4_window7_224
from torch import nn
from torch.optim import Optimizer
from torchvision.models import (ResNet18_Weights, ResNet50_Weights, resnet18,
                                resnet50)

import wandb
from src.base_classes.pl_module import BaseLightningModule
from src.models.sewer_ml import Xie2019, load_xie_model
from src.models.vit import mae_vit_small, mae_vit_tiny, vit_small, vit_tiny
from src.utils.lars import LARS
from src.utils.misc import remove_bias_and_norm_from_weight_decay
from src.utils.schedulers import LinearWarmupCosineAnnealingLR


class SewerNet(BaseLightningModule):
    """Base LightningModule for training defect detection models.

    Args:
        optimization_cfg (DictConfig): Configuration for optimization.
        model_settings_cfg (DictConfig): Configuration for model settings.
        experiment_settings_cfg (DictConfig): Configuration for experiment settings.
        criterion (nn.Module): Loss criterion.

    Attributes:
        is_binary (bool): Flag indicating if the task is binary.
        model_type (str): Type of the model.
        lr (float): Learning rate for optimization.
        momentum (float): Momentum for optimization.
        weight_decay (float): Weight decay for optimization.
        img_size (int): Size of the input image.
        n_unfrozzen_layers (int): Number of unfrozen layers in the model.
        criterion (nn.Module): Loss criterion.
        model (nn.Module): The model instance.
        train_metrics (nn.ModuleDict): Metrics for training.
        val_metrics (nn.ModuleDict): Metrics for validation.
        test_metrics (nn.ModuleDict): Metrics for testing.

    """

    _BACKBONES = {
        "convnext-base": convnext_base,
        "convnext-small": convnext_small,
        "convnext-tiny": convnext_tiny,
        "convnext-large": convnext_large,
        "swin": swin_base_patch4_window7_224,
        "efficientnet": efficientnetv2_s,
        "xie": Xie2019,
        "resnet18": resnet18,
        "resnet50": resnet50,
        "vit-tiny": vit_tiny,
        "mae-vit-tiny": mae_vit_tiny,
        "vit-small": vit_small,
        "mae-vit-small": mae_vit_small,
    }

    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }

    def __init__(
        self,
        conf_dict: DictConfig,
        criterion: nn.Module,
    ):
        super().__init__()

        # Logging settings
        self.only_log_on_epoch_end: bool = conf_dict.only_log_on_epoch_end

        # Model settings
        self.task_type: str = conf_dict.model_settings.task_type
        self.is_binary: bool = self.task_type == "binary"
        self.img_size: int = conf_dict.model_settings.img_size
        self.model_type: str = conf_dict.model_settings.model_type
        self.pretrained: bool = conf_dict.model_settings.pretrained

        # Instantiate model
        self.backbone: nn.Module = None
        self.classifier: nn.Module = None
        self.num_classes = 1 if self.is_binary else 17
        self.features_dim: int = None
        # if self.model_type == "xie":
        #     # TODO: Check with Randall if we will need to use this to assess if we need to
        #     # fix it
        #     self.backbone = load_xie_model(self.load_pretrained_model)
        #     self.classifier = nn.Linear(512, self.num_classes)

        assert (
            self.model_type in self._BACKBONES
        ), f"model_type must be one of {list(self._BACKBONES.keys())}, found {self.model_type}."
        if self.model_type.startswith("resnet"):
            weights = None
            if self.pretrained:
                if self.model_type == "resnet18":
                    weights = ResNet18_Weights
                else:
                    weights = ResNet50_Weights
            self.backbone = self._BACKBONES[self.model_type](weights=weights)
            self.features_dim = self.backbone.inplanes
            self.backbone.fc = nn.Identity()
        else:
            model_kwargs = {"num_classes": 0, "pretrained": self.pretrained}
            if "vit" in self.model_type:
                model_kwargs["img_size"] = self.img_size
            self.backbone = self._BACKBONES[self.model_type](**model_kwargs)
            self.features_dim = self.backbone.num_features
        self.classifier = nn.Linear(self.features_dim, self.num_classes)

        # Optimization settings
        self.criterion: nn.Module = criterion
        self.channels_last: bool = conf_dict.optimization.channels_last
        self.optimizer: str = conf_dict.optimization.optimizer
        self.scheduler: str = conf_dict.optimization.scheduler
        self.lr_steps: List[int] = conf_dict.optimization.lr_steps
        self.lr: float = conf_dict.optimization.lr
        self.momentum: float = conf_dict.optimization.momentum
        self.weight_decay: float = conf_dict.optimization.weight_decay
        self.batch_size: int = conf_dict.optimization.batch_size
        self.exclude_bias_and_norm: bool = conf_dict.optimization.exclude_bias_and_norm
        self.warmup_start_lr: float = conf_dict.optimization.warmup_start_lr
        self.warmup_epochs: int = conf_dict.optimization.warmup_epochs
        self.eta_min: float = conf_dict.optimization.eta_min
        self.scheduler_interval: str = conf_dict.optimization.scheduler_interval
        self.detach_backbone: bool = conf_dict.optimization.detach_backbone

        if self.detach_backbone:
            self.freeze_backbone()

        # Instantiate metrics
        self.instantiate_metrics(
            "binary" if self.is_binary else "multilabel", self.num_classes
        )

    def log_params_to_wandb(self):
        """Log params to wandb."""
        param_dict = {
            "optimization": {
                "optimizer": self.optimizer,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
                "exclude_bias_and_norm": self.exclude_bias_and_norm,
                "channels_last": self.channels_last,
                "detach_backbone": self.detach_backbone,
                "scheduler": self.scheduler,
            }
        }

        if self.scheduler == "multistep":
            param_dict["optimization"]["lr_steps"] = self.lr_steps

        if self.scheduler == "cosine":
            param_dict["optimization"]["warmup_start_lr"] = self.warmup_start_lr
            param_dict["optimization"]["scheduler_interval"] = self.scheduler_interval
            param_dict["optimization"]["warmup_epochs"] = self.warmup_epochs
            param_dict["optimization"]["eta_min"] = self.eta_min

        wandb.config.update(param_dict)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _scale_lr(self, lr: float) -> float:
        """Scale learning rate based on batch size.

        Args:
            lr (float): Learning rate.

        Returns:
            float: Scaled learning rate.

        """
        return lr * self.batch_size / 256

    def load_individual_components(
        self, individual_components_path: str, load_classifier: bool
    ):
        """If a trained model is provided, load the individual components."""

        self.backbone.load_state_dict(
            torch.load(
                os.path.join(individual_components_path, "backbone.pth"),
                weights_only=True,
            )
        )

        if self.detach_backbone:
            self.freeze_backbone()

        if load_classifier:
            self.classifier.load_state_dict(
                torch.load(
                    os.path.join(individual_components_path, "classifier.pth"),
                    weights_only=True,
                )
            )

    def save_model(self, path: str):
        """Save the model to the specified path."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.backbone.state_dict(), os.path.join(path, "backbone.pth"))
        torch.save(self.classifier.state_dict(), os.path.join(path, "classifier.pth"))

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "backbone", "params": self.backbone.parameters()},
            {"name": "classifier", "params": self.classifier.parameters()},
        ]

    def forward(  # pylint: disable=arguments-differ
        self, X: torch.Tensor, detach_backbone: Optional[bool] = False
    ) -> Dict[str, torch.Tensor]:
        """Basic forward method. Children methods should call this function,
        modify the ouputs (without deleting anything) and return it.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            detach_backbone (Optional[bool]): Flag indicating whether to detach the backbone.

        Returns:
            Dict: dict of logits and features.
        """

        if self.channels_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        if detach_backbone:
            # Detach the backbone from the computation graph
            # Used for linear probing
            feats = feats.detach()
        logits = self.classifier(feats)
        return {"logits": logits, "feats": feats}

    def _base_classif_step(
        self,
        X: torch.Tensor,  # pylint: disable=invalid-name
        targets: torch.Tensor,
        normals: torch.Tensor = None,
        strata: torch.Tensor = None,
        no_additional_info: bool = False,
        detach_backbone: Optional[bool] = False,
    ) -> Dict[str, torch.Tensor]:
        """Base classification step.

        Args:
            X (torch.Tensor): Input batch.
            targets (torch.Tensor): Target labels.
            no_additional_info (bool): Flag indicating whether to include additional info.
            detach_backbone (Optional[bool]): Flag indicating whether to detach the backbone.

        Returns:
            Dict[str, torch.Tensor]: Output dictionary.

        """

        # Make forward pass
        outputs = self.forward(X, detach_backbone=detach_backbone)

        classif_loss = self.criterion(outputs["logits"], targets)

        outputs.update(
            {
                "classif_loss": classif_loss,
                "pred": self.get_pred(outputs["logits"]),
            }
        )

        if not no_additional_info:
            add_info = {"targets": targets}
            if normals is not None:
                add_info["normals"] = normals
            if strata is not None:
                add_info["strata"] = strata
            outputs.update(add_info)

        return outputs

    def _base_shared_step(
        self,
        batch: Dict[str, Union[torch.Tensor, np.ndarray]],
        detach_backbone: Optional[bool] = False,
    ) -> Dict:
        """_base_classif_step wrapper that unpacks batch dict and handles multiple inputs
        in the case of NCropAugmentation.

        Args:
            batch (Dict[str, Union[torch.Tensor, np.ndarray]]): Input batch.

        Returns:
            Dict: Output dictionary.

        """
        # Unpack batch
        X, targets = batch["img"], batch["targets"]  # pylint: disable=invalid-name
        normals = batch["normals"] if not self.is_binary else None
        strata = batch["strata"] if self.is_binary else None

        if isinstance(X, torch.Tensor):
            # If X is a tensor, make forward pass and get prediction
            return self._base_classif_step(
                X, targets, normals, strata, detach_backbone=detach_backbone
            )

        # If X is a list, make forward pass and get prediction for each element
        classif_outputs = [
            self._base_classif_step(
                x,
                targets,
                normals,
                strata,
                detach_backbone=detach_backbone,
            )
            for x in X
        ]

        # Stack outputs
        outputs = {}
        for key in classif_outputs[0].keys():
            if classif_outputs[0][key].dim() == 0:
                outputs[key] = torch.stack(  # pylint: disable=no-member
                    [out[key] for out in classif_outputs]
                ).mean()
            else:
                outputs[key] = torch.cat(  # pylint: disable=no-member
                    [out[key] for out in classif_outputs]
                )

        return outputs

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configure optimizers.

        Returns:
            Tuple[List, List]: Optimizers and schedulers.

        """
        learnable_params = self.learnable_params
        # exclude bias and norm for weight decay
        if self.exclude_bias_and_norm:
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        assert (
            self.optimizer in self._OPTIMIZERS
        ), f"Optimizer {self.optimizer} not found. Please choose one of {list(self._OPTIMIZERS.keys())}."
        optimizer = self._OPTIMIZERS[self.optimizer]

        if self.optimizer.startswith("adam"):
            optim = optimizer(
                learnable_params,
                lr=self._scale_lr(self.lr),
                betas=(self.momentum, 0.999),
                weight_decay=self.weight_decay,
            )
        else:
            optim = optimizer(
                learnable_params,
                lr=self._scale_lr(self.lr),
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        scheduler = None
        if self.scheduler == "none":
            return [optim]
        elif self.scheduler == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optim,
                milestones=self.lr_steps,
            )
        elif self.scheduler == "cosine":
            max_warmup_steps = (
                (self.warmup_epochs * self.trainer.estimated_stepping_batches)
                / self.trainer.max_epochs
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.trainer.max_epochs
            )

            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optim,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=(
                        self._scale_lr(self.warmup_start_lr)
                        if self.warmup_epochs > 0
                        else self._scale_lr(self.lr)
                    ),
                    eta_min=self.eta_min,
                ),
                "interval": "step",
                "frequency": 1,
            }

        return [optim], [scheduler]

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer
    ) -> None:
        """
        This improves performance marginally. It should be fine
        since we are not affected by any of the downsides descrited in
        https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad

        Implemented as in here
        https://lightning.ai/docs/pytorch/latest/advanced/speed.html?highlight=set%20grads%20none
        """
        try:
            optimizer.zero_grad(set_to_none=True)
        except:  # pylint: disable=bare-except
            optimizer.zero_grad()
