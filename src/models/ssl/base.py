"""This module contains the base class for the SSL Methods."""

import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

import wandb
from src.models.base_lightning_module import SewerNet
from src.utils.metrics import WeightedKNNClassifier, rankme
from src.utils.misc import MeanPenaltyTerm
from src.utils.momentum import MomentumUpdater, initialize_momentum_params


class BaseSSL(SewerNet):
    """
    Base class for Semi-Supervised Learning (SSL) models in defect detection.
    Inherits from the SewerNet class.

    Attributes:
        _BACKBONES (dict): Dictionary mapping backbone names to corresponding functions.
        backbone (nn.Module): Backbone module of the SSL model.
        classifier (nn.Module): Classifier module of the SSL model.
        model_type (str): Type of the model backbone.
        num_classes (int): Number of classes for classification.
        classifier_lr (float): Learning rate for the classifier module.
        no_channel_last (bool): Flag indicating whether the input tensor has channel last
            format.
        num_large_crops (int): Number of large crops used in training.
        multicrop (bool): Flag indicating whether to use multicrop forward method.
        num_crops (int): Total number of crops used in training.
        use_knn_eval (bool): Flag indicating whether to perform k-nearest neighbors evaluation.
        new_metric (str): Name of the new metric to be logged.

    Methods:
        __init__(*args, **kwargs): Initializes the BaseSSL object.
        instantiate_model(): Instantiates the model by initializing the backbone and
            classifier.
        learnable_params() -> List[Dict[str, Any]]: Defines learnable parameters for the
            base class.
        optimizer_zero_grad(epoch, batch_idx, optimizer, optimizer_idx): Sets gradients
            to zero for optimization.
        forward(X) -> Dict: Performs the forward pass of the model.
        multicrop_forward(X: torch.tensor) -> Dict[str, Any]: Performs the forward pass
            for multicrop views.
        _base_shared_step(X: torch.Tensor, targets: torch.Tensor) -> Dict: Computes
            classification loss, logits, features, acc@1, and acc@5.
        training_step(batch: List[Any], batch_idx: int) -> Dict[str, Any]: Training step
            for pytorch lightning.
        validation_step(batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None) -> Dict[str, Any]:
            Validation step for pytorch lightning.
        validation_epoch_end(outs: List[Dict[str, Any]]): Averages the losses and
            accuracies of all the validation batches.
    """

    def __init__(
        self,
        conf_dict: DictConfig,
        criterion: nn.Module,
    ):
        super().__init__(
            conf_dict=conf_dict,
            criterion=criterion,
        )

        self.classifier_lr: float = conf_dict.ssl_optimization.linear_probe_lr
        self.classifier_momentum: float = (
            conf_dict.ssl_optimization.linear_probe_momentum
        )
        self.classifier_weight_decay: float = (
            conf_dict.ssl_optimization.linear_probe_weight_decay
        )
        self.use_supervised_signal: bool = (
            conf_dict.ssl_optimization.use_supervised_signal
        )

        # Instantiate KNN accuracy metric
        self.use_knn_eval: bool = conf_dict.knn_eval.use_knn_eval
        if self.use_knn_eval:
            self.k: int = conf_dict.knn_eval.k
            self.T: int = conf_dict.knn_eval.T  # pylint: disable=invalid-name
            self.distance_fx: str = conf_dict.knn_eval.distance_fx
            self.sample_train_size: int = conf_dict.knn_eval.sample_train_size
            self.knn_acc_metric = WeightedKNNClassifier(
                k=self.k,
                T=self.T,
                distance_fx=self.distance_fx,
                sample_train_size=self.sample_train_size,
            )

        self.num_large_crops: int = conf_dict.ssl_optimization.num_large_crops
        self.num_small_crops: int = conf_dict.ssl_optimization.num_small_crops
        self.num_crops: int = self.num_large_crops + self.num_small_crops
        # turn on multicrop if there are small crops
        self.multicrop: bool = self.num_small_crops != 0

    def log_params_to_wandb(self):
        """Log params to wandb."""
        super().log_params_to_wandb()
        params = {
            "ssl_optimization": {
                "classifier_lr": self.classifier_lr,
                "classifier_momentum": self.classifier_momentum,
                "classifier_weight_decay": self.classifier_weight_decay,
                "use_supervised_signal": self.use_supervised_signal,
                "num_large_crops": self.num_large_crops,
                "num_small_crops": self.num_small_crops,
            },
            "knn_eval": {"use_knn_eval": self.use_knn_eval},
        }

        if self.use_knn_eval:
            params["knn_eval"]["k"] = self.k
            params["knn_eval"]["T"] = self.T
            params["knn_eval"]["distance_fx"] = self.distance_fx
            params["knn_eval"]["sample_train_size"] = self.sample_train_size

        wandb.config.update(params)

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Wrapper around base learnable_params that adds classifier parameters for
        training.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """
        learnable_params = super().learnable_params

        for param in learnable_params:
            if param["name"] != "classifier":
                continue

            param["lr"] = self._scale_lr(self.classifier_lr)
            param["weight_decay"] = self.classifier_weight_decay
            # Deactivate second momentum for classifier. This is an attempt to make the
            # the optimizer more similar to an SGD optimizer.
            # NOTE: this needs to be adapted when using a different optimizer.
            param["betas"] = (self.classifier_momentum, 0)

        return learnable_params

    def ssl_forward(
        self, X: torch.Tensor  # pylint: disable=invalid-name
    ) -> Dict[str, torch.Tensor]:
        """Performs forward pass of the backbone.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the backbone.
        """
        if self.channels_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        return {"feats": feats}

    def multicrop_forward(
        self, X: torch.Tensor  # pylint: disable=invalid-name
    ) -> Dict[str, Any]:
        """Basic multicrop forward method that performs the forward pass
        for the multicrop views. Children classes can override this method to
        add new outputs but should still call this function. Make sure
        that this method and its overrides always return a dict.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of features.
        """

        if not self.channels_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        return {"feats": feats}

    def compute_and_log_ssl_metrics(
        self, feats: List[torch.Tensor], z: List[torch.Tensor], prefix: str = ""
    ):
        """Computes and logs the SSL metrics."""

        # Compute RankMe metric over the features and obtain mean across views
        rankme_mean = torch.mean(  # pylint: disable=no-member
            torch.tensor(  # pylint: disable=no-member
                [rankme(Z) for Z in feats], dtype=torch.float16
            )
        )

        ideal_z_std = 1 / math.sqrt(z[0].size(1))
        z_std = (
            F.normalize(
                torch.stack(z[: self.num_large_crops]),  # pylint: disable=no-member
                dim=-1,
            )
            .std(dim=1)
            .mean()
        )
        z_mean = (
            F.normalize(
                torch.stack(z[: self.num_large_crops]),  # pylint: disable=no-member
                dim=-1,
            )
            .mean(dim=1)
            .mean()
        )
        ideal_feats_std = 1 / math.sqrt(feats[0].size(1))
        feats_std = (
            F.normalize(
                torch.stack(feats[: self.num_large_crops]),  # pylint: disable=no-member
                dim=-1,
            )
            .std(dim=1)
            .mean()
        )
        feats_mean = (
            F.normalize(
                torch.stack(feats[: self.num_large_crops]),  # pylint: disable=no-member
                dim=-1,
            )
            .mean(dim=1)
            .mean()
        )

        metrics = {
            f"{prefix}rankme": rankme_mean,
            f"{prefix}train_z_std": z_std / ideal_z_std,
            f"{prefix}train_z_mean": z_mean,
            f"{prefix}train_feats_std": feats_std / ideal_feats_std,
            f"{prefix}train_feats_mean": feats_mean,
        }
        if not self.only_log_on_epoch_end:
            self.log_dict(metrics, on_step=True, sync_dist=False)
        self.log_dict(
            {k + "_epoch": v for k, v in metrics.items()},
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    def on_train_epoch_start(self) -> None:
        """Resets the k-nearest neighbors metric at the beginning of the training epoch.
        This avoids accumulating statistics from previous epochs and saves memory."""
        if self.use_knn_eval:
            self.knn_acc_metric.reset()

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits.
        """

        # Get classification outputs
        # Compute classification loss, get predictions and labels
        outputs = self._base_shared_step(
            batch, detach_backbone=not self.use_supervised_signal
        )

        # Turn it into a list if no cropping is done
        X = batch["img"]  # pylint: disable=invalid-name
        X = [X] if isinstance(X, torch.Tensor) else X  # pylint: disable=invalid-name

        # Compute feature for each crop and group them in same dictionary
        ssl_outputs = [self.ssl_forward(x) for x in X]
        outputs.update(
            {k: [out[k] for out in ssl_outputs] for k in ssl_outputs[0].keys()}
        )

        if self.multicrop:
            multicrop_outs = [
                self.multicrop_forward(x) for x in X[self.num_large_crops :]
            ]
            for k in multicrop_outs[0].keys():
                outputs[k] = outputs.get(k, []) + [out[k] for out in multicrop_outs]

        # Compute and log metrics
        self.compute_supervised_metrics(outputs, "train")
        self.log_supervised_metrics("train", classif_loss=outputs["classif_loss"])

        if self.use_knn_eval:
            # Update k-nearest neighbors metric
            feats = outputs["feats"]
            targets = batch["targets"]
            self.knn_acc_metric.update(
                train_features=torch.cat(feats).detach(),  # pylint: disable=no-member
                train_targets=targets.repeat(len(feats), 1),
            )

        return outputs

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics. In addition, it
        updates the k-nearest neighbors metric to be computed when the validation epoch ends.
        """

        # Perform classification step and log metrics, obtain features from it
        outputs = super().validation_step(batch, batch_idx)

        # Update k-nearest neighbors metric
        if not self.trainer.sanity_checking and self.use_knn_eval:
            feats = outputs["feats"]
            targets = batch["targets"]
            self.knn_acc_metric.update(
                test_features=feats.detach(), test_targets=targets.detach()
            )

        return outputs

    def on_validation_epoch_end(self):
        """Computes and logs the k-nearest neighbors accuracy at the end of the validation
        epoch."""

        # Compute and log k-nearest neighbors accuracy
        if not self.trainer.sanity_checking and self.use_knn_eval:
            knn_acc, knn_prec, knn_rec, knn_f1 = self.knn_acc_metric.compute()
            self.log_dict(
                {
                    "val_knn_acc": knn_acc,
                    "val_knn_prec": knn_prec,
                    "val_knn_rec": knn_rec,
                    "val_knn_f1": knn_f1,
                },
                on_epoch=True,
                sync_dist=True,
            )


class BaseMomentumMethod(BaseSSL):
    """Base momentum model that implements all basic operations for all self-supervised methods
    that use a momentum backbone. It adds shared momentum arguments, adds basic learnable
    parameters, implements basic training and validation steps for the momentum backbone and
    classifier. Also implements momentum update using exponential moving average and cosine
    annealing of the weighting decrease coefficient.

    Extra cfg settings:
        momentum:
            base_tau (float): base value of the weighting decrease coefficient in [0,1].
            final_tau (float): final value of the weighting decrease coefficient in [0,1].
            classifier (bool): whether or not to train a classifier on top of the momentum backbone.
    """

    def __init__(
        self,
        conf_dict: DictConfig,
        criterion: nn.Module,
    ):
        super().__init__(
            conf_dict=conf_dict,
            criterion=criterion,
        )

        # momentum updater
        self.base_tau: float = conf_dict.momentum_encoder.base_tau
        self.final_tau: float = conf_dict.momentum_encoder.final_tau
        self.momentum_updater = MomentumUpdater(self.base_tau, self.final_tau)

        # Initialized once training starts
        self.last_step: int = None

        # Instantiate momentum backbone and initialize its parameters
        if self.model_type.startswith("resnet"):
            self.momentum_backbone = self._BACKBONES[self.model_type]()
            self.momentum_backbone.fc = nn.Identity()
        else:
            self.momentum_backbone = self._BACKBONES[self.model_type](num_classes=0)
        initialize_momentum_params(self.backbone, self.momentum_backbone)

    def log_params_to_wandb(self):
        """Log params to wandb."""
        super().log_params_to_wandb()
        wandb.config.update(
            {
                "momentum_encoder": {
                    "base_tau": self.base_tau,
                    "final_tau": self.final_tau,
                }
            }
        )

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Defines base momentum pairs that will be updated using exponential moving average.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs (two element tuples).
        """

        return [("backbone", (self.backbone, self.momentum_backbone))]

    def load_individual_components(
        self, individual_components_path: str, load_classifier: bool
    ):
        """If a trained model is provided, load the individual components."""

        super().load_individual_components(individual_components_path, load_classifier)

        self.momentum_backbone.load_state_dict(
            torch.load(
                os.path.join(individual_components_path, "momentum_backbone.pth"),
                weights_only=True,
            )
        )

    def save_model(self, path: str) -> None:
        """Saves the model to the specified path.

        Args:
            path (str): path to save the model.
        """
        super().save_model(path)
        momentum_pairs = self.momentum_pairs
        for name, (_, mp) in momentum_pairs:
            torch.save(mp.state_dict(), os.path.join(path, f"momentum_{name}.pth"))

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0

    @torch.no_grad()
    def momentum_forward(
        self, X: torch.Tensor  # pylint: disable=invalid-name
    ) -> Dict[str, Any]:
        """Momentum forward method. Children methods should call this function,
        modify the ouputs (without deleting anything) and return it.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """

        if self.channels_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.momentum_backbone(X)
        return {"feats": feats}

    def _get_X(  # pylint: disable=invalid-name
        self, batch: Dict[str, Any]
    ) -> List[torch.Tensor]:
        """Get X from batch and turn it into a list if no cropping is done."""
        X = batch["img"]  # pylint: disable=invalid-name
        return [X] if isinstance(X, torch.Tensor) else X  # pylint: disable=invalid-name

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It performs all the shared operations for the
        momentum backbone and classifier, such as forwarding the crops in the momentum backbone
        and classifier, and computing statistics.
        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: a dict with the features of the momentum backbone and the classification
                loss and logits of the momentum classifier.
        """

        # Get classification outputs and backbone features for each crop
        outputs = super().training_step(batch, batch_idx)

        # Turn it into a list if no cropping is done
        X = self._get_X(batch)  # pylint: disable=invalid-name

        # Compute momentum features for each crop
        momentum_outputs = [self.momentum_forward(x) for x in X]
        outputs.update(
            {
                f"momentum_{k}": [out[k] for out in momentum_outputs]
                for k in momentum_outputs[0].keys()
            }
        )

        return outputs

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int
    ):  # pylint: disable=unused-argument
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.

        Args:
            outputs (Dict[str, Any]): the outputs of the training step.
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.
        """

        if self.trainer.global_step > self.last_step:
            # update momentum backbone and projector
            momentum_pairs = self.momentum_pairs
            for _, mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            # log tau momentum
            self.log(
                "tau",
                self.momentum_updater.cur_tau,
                on_epoch=True,
                on_step=False,
                rank_zero_only=True,
            )
            # update tau
            self.momentum_updater.update_tau(
                cur_step=self.trainer.global_step,
                max_steps=self.trainer.estimated_stepping_batches,
            )
        self.last_step = self.trainer.global_step
