"""Module that contains utility functions for training the defect detection model."""

from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


def get_rank():
    """
    Get the rank of the current process in the distributed environment.

    Returns:
        int: The rank of the current process. Returns 0 if distributed environment is not available or not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):  # pylint: disable=abstract-method
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):  # pylint: disable=arguments-differ
        if dist.is_available() and dist.is_initialized():
            output = [
                torch.zeros_like(x)  # pylint: disable=no-member
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)  # pylint: disable=no-member
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)  # pylint: disable=no-member


class MeanPenaltyTerm(torch.nn.Module):
    """Computes the mean penalty term loss value.

    Args:
        loss_type: Type of loss to use for the mean penalty term.
        mean_penalty_weight: Weight to apply to the mean penalty term.
        num_crops: Number of crops used in the model.
    """

    def __init__(self, loss_type: str, mean_penalty_weight: float):
        super().__init__()
        self.loss_type = loss_type
        self.mean_penalty_weight = mean_penalty_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the mean penalty term.

        Args:
            x: Mean latent representation across views.

        Returns:
            torch.Tensor: Mean penalty term loss value
        """

        if self.loss_type == "mse":
            mean_loss = F.mse_loss(x, torch.zeros_like(x))  # pylint: disable=no-member
        elif self.loss_type == "cossim":
            mean_loss = 1 - F.cosine_similarity(  # pylint: disable=not-callable
                x, torch.zeros_like(x)  # pylint: disable=no-member
            )
        elif self.loss_type == "sum":
            mean_loss = x.sum()

        return self.mean_penalty_weight * mean_loss


def remove_bias_and_norm_from_weight_decay(parameter_groups: List[Dict]):
    """
    Removes bias and normalization parameters from weight decay calculation.

    Args:
        parameter_groups (List[Dict]): A list of parameter groups.

    Returns:
        List[Dict]: A modified list of parameter groups with bias and normalization parameters removed from weight decay calculation.
    """
    out = []
    for group in parameter_groups:
        # parameters with weight decay
        decay_group = {k: v for k, v in group.items() if k != "params"}

        # parameters without weight decay
        no_decay_group = {k: v for k, v in group.items() if k != "params"}
        no_decay_group["weight_decay"] = 0
        group_name = group.get("name", None)
        if group_name:
            no_decay_group["name"] = group_name + "_no_decay"

        # split parameters into the two lists
        decay_params = []
        no_decay_params = []
        for param in group["params"]:
            if param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # add groups back
        if decay_params:
            decay_group["params"] = decay_params
            out.append(decay_group)
        if no_decay_params:
            no_decay_group["params"] = no_decay_params
            out.append(no_decay_group)
    return out


def broadcast_error_to_all_workers():
    """
    Broadcasts an error signal to all workers in a distributed setting.
    """
    if torch.distributed.is_initialized():
        # Create a tensor to signal an error has occurred.
        error_signal = torch.tensor([1.0], device="cuda")  # pylint: disable=no-member
        # Broadcast this error signal to all workers.
        dist.broadcast(error_signal, src=0)


def generate_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Adapted from https://github.com/facebookresearch/mae.
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def generate_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # Adapted from https://github.com/facebookresearch/mae.

    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = generate_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = generate_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def generate_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Adapted from https://github.com/facebookresearch/mae.
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or
        [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = generate_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def make_contiguous(module):
    """Make the model contigous in order to comply with some distributed strategies.
    https://github.com/lucidrains/DALLE-pytorch/issues/330
    """

    with torch.no_grad():
        for param in module.parameters():
            param.set_(param.contiguous())
