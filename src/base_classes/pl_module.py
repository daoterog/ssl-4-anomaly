"""Base class for LightningModules used in defect detection models."""

import copy
from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import F1Score, FBetaScore, Precision, Recall

from src.data.datasets import DEFECT_LABELS
from src.utils.metrics import CustomFBetaScore, CustomRecall, F2CIWScore


class BaseLightningModule(pl.LightningModule):
    """
    Base class for LightningModules used in defect detection models.

    Attributes:
        is_binary (bool): Flag indicating whether the model is binary or multi-class.
        train_metrics (nn.ModuleDict): Metrics for training.
        val_metrics (nn.ModuleDict): Metrics for validation.
        test_metrics (nn.ModuleDict): Metrics for testing.

    Methods:
        instantiate_metrics: Instantiate metrics.
        _get_stage_metrics: Get metrics for a given stage.
        compute_supervised_metrics: Compute supervised metrics.
        log_supervised_metrics: Log supervised metrics.
        get_pred: Get prediction from model output.
    """

    def __init__(self):
        super().__init__()
        self.detach_backbone: bool = None
        self.is_binary: bool = None
        self.only_log_on_epoch_end: bool = None
        self.train_metrics: nn.ModuleDict = None
        self.val_metrics: nn.ModuleDict = None
        self.test_metrics: nn.ModuleDict = None

    def instantiate_metrics(
        self,
        task: str,
        num_labels: Optional[int] = None,
    ) -> Tuple[nn.ModuleDict, nn.ModuleDict, nn.ModuleDict]:
        """Instantiate metrics.

        Args:
            task (str): Task type.
            num_labels (Optional[int]): Number of labels.

        Returns:
            Tuple[nn.ModuleDict, nn.ModuleDict, nn.ModuleDict]: Train, validation, and
                test metrics.

        """

        train_metrics = nn.ModuleDict(
            dict(
                f1_score=F1Score(task=task, num_labels=num_labels),
                precision_score=Precision(task=task, num_labels=num_labels),
                recall_score=Recall(task=task, num_labels=num_labels),
                f2ciw_score=F2CIWScore(is_binary=self.is_binary),
                per_class_f2_score=CustomFBetaScore(is_binary=self.is_binary),
                per_class_recall_score=CustomRecall(is_binary=self.is_binary),
                normals_f1_score=F1Score(task="binary", num_labels=1),
            )
        )

        val_metrics = copy.deepcopy(train_metrics)
        test_metrics = copy.deepcopy(train_metrics)

        setattr(self, "train_metrics", train_metrics)
        setattr(self, "val_metrics", val_metrics)
        setattr(self, "test_metrics", test_metrics)

    def _get_stage_metrics(self, stage: str) -> nn.ModuleDict:
        """Get metrics for a given stage.

        Args:
            stage (str): Stage of the computation.

        Returns:
            nn.ModuleDict: Metrics for the given stage.

        """

        # Check if stage is valid
        assert stage in [
            "train",
            "val",
            "test",
        ], "The stage must be either 'train', 'val' or 'test'"

        # Get metric attribute using stage
        return getattr(self, f"{stage}_metrics")

    def compute_supervised_metrics(
        self, outputs: Dict[str, Union[torch.Tensor, np.ndarray]], stage: str
    ):
        """Compute metrics.

        Args:
            outputs: Model outputs.
            stage (str): Stage of the computation.

        """

        # Get stage metrics
        stage_metrics = self._get_stage_metrics(stage)

        if outputs["targets"].dtype == torch.float16:
            # This is required because torchmetrics does not support float16
            outputs["targets"] = outputs["targets"].to(torch.float32)

        # Update Metrics
        stage_metrics["f1_score"].update(outputs["pred"], outputs["targets"])
        stage_metrics["precision_score"].update(outputs["pred"], outputs["targets"])
        stage_metrics["recall_score"].update(outputs["pred"], outputs["targets"])

        metric_kwargs = {"preds": outputs["pred"], "targets": outputs["targets"]}
        if not self.is_binary:
            stage_metrics["f2ciw_score"].update(**metric_kwargs)
            stage_metrics["per_class_f2_score"].update(**metric_kwargs)
            stage_metrics["per_class_recall_score"].update(**metric_kwargs)
            binary_prediction = (outputs["pred"] == 1).any(dim=1).int()
            stage_metrics["normals_f1_score"].update(
                1 - binary_prediction, 1 - outputs["normals"]
            )
        else:
            metric_kwargs["strata"] = outputs["strata"]
            stage_metrics["f2ciw_score"].update(**metric_kwargs)
            stage_metrics["per_class_f2_score"].update(**metric_kwargs)
            stage_metrics["per_class_recall_score"].update(**metric_kwargs)
            stage_metrics["normals_f1_score"].update(
                1 - outputs["pred"], 1 - outputs["targets"]
            )

    def log_supervised_metrics(
        self, stage: str, classif_loss: Optional[torch.Tensor] = None
    ):
        """Log metrics.

        Args:
            stage (str): Stage of the computation.
            classif_loss (Optional[torch.Tensor]): Classification loss.
        """

        # Get stage metrics
        stage_metrics = self._get_stage_metrics(stage)

        # Log metrics
        for metric_name, metric in stage_metrics.items():
            if metric_name.startswith("per_class_"):
                continue
            self.log(
                f"{stage}_{metric_name}",
                metric,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

        # Log per class metrics
        per_class_recall = stage_metrics["per_class_recall_score"].compute()
        per_class_f2 = stage_metrics["per_class_f2_score"].compute()
        for i, defect_label in enumerate(DEFECT_LABELS):
            self.log(
                f"{stage}_recall_{defect_label}",
                per_class_recall[i].item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_f2_{defect_label}",
                per_class_f2[i].item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

        if classif_loss is not None and stage != "test":
            self.log(
                f"{stage}_classif_loss",
                classif_loss,
                on_step=not self.only_log_on_epoch_end,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

    def get_pred(self, output) -> torch.Tensor:
        """Get prediction from model output.

        Args:
            output: Model output.

        Returns:
            torch.Tensor: Model prediction.

        """
        return (output > 0).float()

    @abstractmethod
    def _base_shared_step(
        self, batch: Dict[str, torch.Tensor], detach_backbone: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Base shared step for training and validation steps.

        Args:
            batch: Batch data.

        Returns:
            Dict[str, torch.Tensor]: Output dictionary.

        """

    def _base_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
    ):
        """Base step for training and validation steps.

        Args:
            batch: Batch data.
            stage: Stage of the computation.

        Returns:
            Dict[str, torch.Tensor]: Output dictionary.

        """
        # Compute classification loss, get predictions and labels
        outputs = self._base_shared_step(
            batch, detach_backbone=self.detach_backbone if stage == "train" else False
        )

        # Compute and log metrics
        if not self.trainer.sanity_checking:
            self.compute_supervised_metrics(outputs, stage)
            self.log_supervised_metrics(stage, classif_loss=outputs["classif_loss"])

        if stage != "train":
            return outputs
        return outputs["classif_loss"]

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Base method for supervised training.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            Dict[str, torch.Tensor]: Output dictionary.

        """

        return self._base_step(batch, "train")

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            Dict[str, torch.Tensor]: Output dictionary.

        """

        return self._base_step(batch, "val")

    def test_step(  # pylint: disable=arguments-differ
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            Dict[str, torch.Tensor]: Output dictionary.

        """

        return self._base_step(batch, "test")
