"""DinoV1 implementation"""

import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

import wandb
from src.models.ssl.base import BaseMomentumMethod
from src.utils.momentum import initialize_momentum_params


class DinoLoss(nn.Module):
    """Auxiliary module to compute DINO's loss.

    Args:
        num_prototypes (int): number of prototypes.
        warmup_teacher_temp (float): base temperature for the temperature schedule
            of the teacher.
        teacher_temp (float): final temperature for the teacher.
            Comment from DINO repo
            For most experiments, anything above 0.07 is unstable. We recommend
            starting with the default value of 0.04 and increase this slightly if needed.
        warmup_teacher_temp_epochs (float): number of epochs for the cosine annealing schedule.
            Comment from DINO repo
            Initial value for the teacher temperature: 0.04 works well in most cases.
            Try decreasing it if the training loss does not decrease.
        num_epochs (int): total number of epochs.
        student_temp (float, optional): temperature for the student. Defaults to 0.1.
        num_large_crops (int, optional): number of crops/views. Defaults to 2.
        center_momentum (float, optional): momentum for the EMA update of the center of
            mass of the teacher. Defaults to 0.9.
    """

    def __init__(
        self,
        num_prototypes: int,
        warmup_teacher_epochs: int,
        num_epochs: int,
        num_large_crops: int,
        center_momentum: float = 0.9,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.04,
        warmup_teacher_temperature: float = 0.04,
    ):
        super().__init__()
        self.epoch = 0
        self.student_temperature = student_temperature
        self.center_momentum = center_momentum
        self.num_large_crops = num_large_crops
        self.register_buffer(
            "center", torch.zeros(1, num_prototypes)  # pylint: disable=no-member
        )
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training unstable at the beginning
        warmup_teacher_epochs = min(warmup_teacher_epochs, num_epochs)
        self.teacher_temp_schedule = np.linspace(
            warmup_teacher_temperature,
            teacher_temperature,
            warmup_teacher_epochs,
        )
        if num_epochs > warmup_teacher_epochs:
            self.teacher_temp_schedule = np.concatenate(
                (
                    self.teacher_temp_schedule,
                    np.ones(num_epochs - warmup_teacher_epochs) * teacher_temperature,
                )
            )

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Computes DINO's loss given a batch of logits of the student and a batch of logits of the
        teacher.

        Args:
            student_output (torch.Tensor): NxP Tensor containing student logits for all views.
            teacher_output (torch.Tensor): NxP Tensor containing teacher logits for all views.

        Returns:
            torch.Tensor: DINO loss.
        """

        student_logits = student_logits / self.student_temperature
        # Chunk is just a trick to turn a tensor into a list of tensors, we are basically
        # undoing the cat operation done before the forward pass
        # TODO: replace with num_crops when using global and local crops
        student_outs = student_logits.chunk(self.num_large_crops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[self.epoch]
        teacher_outs = F.softmax((teacher_logits - self.center) / temp, dim=-1)
        # Chunk is just a trick to turn a tensor into a list of tensors, we are basically
        # undoing the cat operation done before the forward pass
        # TODO: replace with global_crops when using global and local crops
        teacher_outs = teacher_outs.detach().chunk(self.num_large_crops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_outs):
            for iv, v in enumerate(student_outs):
                if iv == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                # Compute cross-entropy loss
                loss = torch.sum(  # pylint: disable=no-member
                    -q * F.log_softmax(v, dim=-1), dim=-1
                )
                # Here we take the mean of the loss because the loss is computed on an
                # element-wise basis, so we need to average the loss across all elements.
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_logits)
        return total_loss

    def update_center(self, teacher_logits: torch.Tensor):
        """Updates the center for DINO's loss using exponential moving average.

        Args:
            teacher_output (torch.Tensor): NxP Tensor containing teacher logits of all views.
        """
        batch_center = torch.sum(  # pylint: disable=no-member
            teacher_logits, dim=0, keepdim=True  # pylint: disable=no-member
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()
        batch_center = batch_center / len(teacher_logits)

        # EMA update
        self.center = (  # pylint: disable=attribute-defined-outside-init
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )


class DINOHead(nn.Module):
    """DINO head that takes as input the features of the backbone, projects them in a lower
    dimensional space and multiplies with the prototypes.

    Args:
        in_dim (int): number of dimensions of the input (aka backbone features).
        num_prototypes (int): number of prototypes.
        use_bn (bool, optional): whether to use batch norm in projector. Defaults to True.
        norm_prototyper (bool, optional): whether to l2-norm the last layer. Defaults to True.
            DINO repo comment:
            Not normalizing leads to better performance but can make the training unstable.
            In our experiments, we typically set this paramater to False with vit_small and True with vit_base.
        num_layers (int, optional): number of layers in projector. Defaults to 3.
        hidden_dim (int, optional): number of dimension in hidden layers. Defaults to 2048.
        bottleneck_dim (int, optional): number of dimensions in bottleneck. Defaults to 256.
    """

    def __init__(
        self,
        in_dim: int,
        num_prototypes: int,
        use_bn: bool = True,
        norm_prototyper: bool = True,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        proj_output_dim: int = 256,
    ):
        super().__init__()

        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.projector = nn.Linear(in_dim, proj_output_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers += [nn.ReLU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, proj_output_dim))
            self.projector = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.prototyper = nn.utils.weight_norm(
            nn.Linear(proj_output_dim, num_prototypes, bias=False)
        )
        self.prototyper.weight_g.data.fill_(1)

        if norm_prototyper:
            self.prototyper.weight_g.requires_grad = False

    def _init_weights(self, m: nn.Module):
        """Initializes weights with truncated normal and biases with zeros.

        Args:
            m (nn.Module): a layer of the DINO head.
        """

        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of  the projector and the prototyper.

        Args:
            x (torch.Tensor): a batch of features.

        Returns:
            torch.Tensor: a batch of logits.
        """
        x = self.projector(x)
        x = F.normalize(x, dim=-1)
        return self.prototyper(x)


class DINO(BaseMomentumMethod):
    """Adds DINO head to the student and momentum DINO head to the teacher.

    Extra cfg settings:
        method_kwargs:
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            proj_output_dim (int): number of output neurons in the projector.
            num_prototypes (int): number of prototypes.
            use_bn_in_head (bool): whether or not to use bn in the head.
            norm_prototyper (bool): whether or not to normalize the last layer (prototypes).
            clip_grad (float): threshold for gradient clipping.
            freeze_prototyper (bool): whether or not to freeze the last layer (prototypes).
            student_temperature (float): temperature for the student.
            teacher_temperature (float): temperature for the teacher.
            warmup_teacher_temperature (float): base temperature for the teacher.
            warmup_teacher_temperature_epochs (int): number of epochs of cosine annealing
                scheduling for teacher temperature.
    """

    def __init__(self, conf_dict: DictConfig, criterion: nn.Module):
        super().__init__(conf_dict, criterion)

        # Architecture settings
        self.proj_hidden_dim: int = conf_dict.ssl_settings.proj_hidden_dim
        self.proj_output_dim: int = conf_dict.ssl_settings.proj_output_dim
        self.use_bn_in_head: bool = conf_dict.ssl_settings.use_bn_in_head
        self.num_prototypes: int = conf_dict.ssl_settings.num_prototypes
        self.norm_prototyper: bool = conf_dict.ssl_settings.norm_prototyper

        # Optimization settings
        self.clip_grad: bool = conf_dict.ssl_settings.clip_grad
        self.freeze_prototyper: int = conf_dict.ssl_settings.freeze_prototyper
        self.student_temperature: float = conf_dict.ssl_settings.student_temperature
        self.warmup_teacher_temperature: float = (
            conf_dict.ssl_settings.warmup_teacher_temperature
        )
        self.teacher_temperature: float = conf_dict.ssl_settings.teacher_temperature
        self.warmup_temperature_epochs: int = (
            conf_dict.ssl_settings.warmup_temperature_epochs
        )

        # Instantiate Model
        head_params = {
            "in_dim": self.features_dim,
            "num_prototypes": self.num_prototypes,
            "use_bn": self.use_bn_in_head,
            "hidden_dim": self.proj_hidden_dim,
            "proj_output_dim": self.proj_output_dim,
        }

        self.head = DINOHead(norm_prototyper=self.norm_prototyper, **head_params)

        self.momentum_head = DINOHead(**head_params)
        initialize_momentum_params(self.head, self.momentum_head)

        self.dino_loss_fn = DinoLoss(
            num_prototypes=self.num_prototypes,
            num_large_crops=self.num_crops,
            warmup_teacher_epochs=self.warmup_temperature_epochs,
            num_epochs=conf_dict.pl_trainer_settings.max_epochs,
            student_temperature=self.student_temperature,
            teacher_temperature=self.teacher_temperature,
            warmup_teacher_temperature=self.warmup_teacher_temperature,
        )

    def log_params_to_wandb(self):
        """Log the parameters of the model to wandb."""
        super().log_params_to_wandb()
        wandb.config.update(
            {
                "ssl_settings": {
                    "ssl_method": "dino",
                    "proj_hidden_dim": self.proj_hidden_dim,
                    "proj_output_dim": self.proj_output_dim,
                    "num_prototypes": self.num_prototypes,
                    "clip_grad": self.clip_grad,
                    "freeze_prototyper": self.freeze_prototyper,
                    "use_bn_in_head": self.use_bn_in_head,
                    "norm_prototyper": self.norm_prototyper,
                    "student_temperature": self.student_temperature,
                    "warmup_teacher_temperature": self.warmup_teacher_temperature,
                    "teacher_temperature": self.teacher_temperature,
                    "warmup_temperature_epochs": self.warmup_temperature_epochs,
                }
            }
        )

    def load_individual_components(
        self, individual_components_path: str, load_classifier: bool
    ):
        """If a trained model is provided, load the individual components."""

        super().load_individual_components(individual_components_path, load_classifier)

        self.head.load_state_dict(
            torch.load(os.path.join(individual_components_path, "head.pth"))
        )
        self.momentum_head.load_state_dict(
            torch.load(os.path.join(individual_components_path, "momentum_head.pth"))
        )
        self.dino_loss_fn.load_state_dict(
            torch.load(os.path.join(individual_components_path, "dino_loss.pth"))
        )

    def save_model(self, path: str) -> None:
        """Saves the model to the specified path.

        Args:
            path (str): path to save the model.
        """
        super().save_model(path)
        torch.save(self.head.state_dict(), os.path.join(path, "head.pth"))
        torch.save(
            self.momentum_head.state_dict(), os.path.join(path, "momentum_head.pth")
        )
        torch.save(self.dino_loss_fn.state_dict(), os.path.join(path, "dino_loss.pth"))

    @property
    def learnable_params(self) -> List[dict]:
        """Adds DINO head parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "head", "params": self.head.parameters()}]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (head, momentum_head) to the parent's momentum pairs.

        Returns:
            List[dict]: list of momentum pairs.
        """

        extra_momentum_pairs = [("head", (self.head, self.momentum_head))]
        return super().momentum_pairs + extra_momentum_pairs

    def dino_clip_gradients(self, clip: float):
        """Clips gradients after backward pass.

        Args:
            clip (float): threshold for gradient clipping.
        """

        for p in self.backbone.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)

    def on_train_epoch_start(self):
        """Updates the current epoch in DINO's loss object."""
        self.dino_loss_fn.epoch = self.current_epoch

    def ssl_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the student (backbone and head).

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the logits of the head.
        """

        out = super().ssl_forward(X)
        z = self.head(out["feats"])
        out.update({"z": z})
        return out

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the key.
        """

        out = super().momentum_forward(X)
        z = self.momentum_head(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for DINO reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where [X]
                is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of DINO loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        # We concatenate the logits of the student and the momentum backbone so we can
        # perform global operations inside the loss function. No loops needed.
        prototypes = torch.cat(out["z"])  # pylint: disable=no-member
        momentum_prototypes = torch.cat(out["momentum_z"])  # pylint: disable=no-member

        # ------- cross entropy loss -------
        final_loss = self.dino_loss_fn(prototypes, momentum_prototypes)

        self.compute_and_log_ssl_metrics(feats=out["feats"], z=out["z"])
        self.compute_and_log_ssl_metrics(
            feats=out["momentum_feats"], z=out["momentum_z"], prefix="momentum_"
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

    def on_after_backward(self):
        """Performs gradient clipping and zeros the gradients on the last layer (prototypes)."""

        # clip gradients
        if self.clip_grad:
            self.dino_clip_gradients(self.clip_grad)
        # zero gradients on last layer
        if self.current_epoch < self.freeze_prototyper:
            for p in self.head.prototyper.parameters():
                p.grad = None
