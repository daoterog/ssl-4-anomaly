import os
from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from omegaconf import DictConfig
from timm.models.vision_transformer import Block

import wandb
from src.models.ssl.base import BaseSSL
from src.utils.misc import generate_2d_sincos_pos_embed


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Patchifies an image according to some patch size.
    Adapted from https://github.com/facebookresearch/mae.

    Args:
        imgs (torch.Tensor): [N, 3, H, W] Tensor containing the original images.
        patch_size (int): size of each patch.

    Returns:
        torch.Tensor: [N, Tokens, pixels * pixels * 3] Tensor containing the patchified images.
    """

    if imgs.ndim < 4:
        imgs = imgs.unsqueeze(0)

    assert imgs.size(2) == imgs.size(3) and imgs.size(2) % patch_size == 0

    h = w = imgs.size(2) // patch_size
    x = imgs.reshape(shape=(imgs.size(0), 3, h, patch_size, w, patch_size))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.size(0), h * w, patch_size**2 * 3))
    return x


def mae_loss_func(
    imgs: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    norm_pix_loss: bool = True,
) -> torch.Tensor:
    """Computes MAE's loss given batch of images, the decoder predictions, the input mask and respective patch size.
    Adapted from https://github.com/facebookresearch/mae.

    Args:
        imgs (torch.Tensor): [N, 3, H, W] Tensor containing the original images.
        pred (torch.Tensor): [N, Tokens, pixels * pixels * 3] Tensor containing the predicted patches.
        mask (torch.Tensor): [N, Tokens] Tensor representing a binary mask, where value 1 means masked.
        patch_size (int): size of each patch.
        norm_pix_loss (bool): whether to normalize the pixels of each patch with their respective mean and std.

    Returns:
        torch.Tensor: MAE's loss.
    """

    target = patchify(imgs, patch_size)

    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss


class MAEDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim,
        depth,
        num_heads,
        num_patches,
        patch_size,
        mlp_ratio=4.0,
    ) -> None:
        super().__init__()

        self.num_patches = num_patches

        self.decoder_embed = nn.Linear(in_dim, embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * 3, bias=True)

        # init all weights according to MAE's repo
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        decoder_pos_embed = generate_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class MAE(BaseSSL):
    """Implements MAE (https://arxiv.org/abs/2111.06377).

    Extra cfg settings:
        ssl_settings:
            mask_ratio (float): percentage of image to mask.
            decoder_embed_dim (int): number of dimensions for the embedding in the decoder
            decoder_depth (int) depth of the decoder
            decoder_num_heads (int) number of heads for the decoder
            norm_pix_loss (bool): whether to normalize the pixels of each patch with their
                respective mean and std for the loss. Defaults to False.
    """

    def __init__(self, conf_dict: DictConfig, criterion: nn.Module):
        super().__init__(conf_dict, criterion)

        self.backbone_name = conf_dict.model_settings.model_type

        assert "vit" in self.backbone_name, "MAE only supports ViT as backbone."
        assert (
            "mae" in self.backbone_name
        ), "You must select the `mae` variant of the ViT."

        self.mask_ratio: float = conf_dict.ssl_settings.mask_ratio
        self.norm_pix_loss: bool = conf_dict.ssl_settings.norm_pix_loss

        # gather backbone info from timm
        self._vit_embed_dim: int = self.backbone.pos_embed.size(-1)
        # if patch size is not available, defaults to 16 or 14 depending on backbone
        default_patch_size = 14 if self.backbone_name == "vit_huge" else 16
        self._vit_patch_size: int = (
            self.backbone.patch_embed.patch_size[0] or default_patch_size
        )
        self._vit_num_patches: int = self.backbone.patch_embed.num_patches

        self.decoder_embed_dim: int = conf_dict.ssl_settings.decoder_embed_dim
        self.decoder_depth: int = conf_dict.ssl_settings.decoder_depth
        self.decoder_num_heads: int = conf_dict.ssl_settings.decoder_num_heads

        # decoder
        self.decoder = MAEDecoder(
            in_dim=self.features_dim,
            embed_dim=self.decoder_embed_dim,
            depth=self.decoder_depth,
            num_heads=self.decoder_num_heads,
            num_patches=self._vit_num_patches,
            patch_size=self._vit_patch_size,
            mlp_ratio=4.0,
        )

    def log_params_to_wandb(self):
        """Log the parameters of the model to wandb."""
        super().log_params_to_wandb()
        wandb.config.update(
            {
                "ssl_settings": {
                    "ssl_method": "mae",
                    "mask_ratio": self.mask_ratio,
                    "norm_pix_loss": self.norm_pix_loss,
                    "decoder_embed_dim": self.decoder_embed_dim,
                    "decoder_depth": self.decoder_depth,
                    "decoder_num_heads": self.decoder_num_heads,
                }
            }
        )

    def load_individual_components(
        self, individual_components_path: str, load_classifier: bool
    ):
        """If a trained model is provided, load the individual components."""

        super().load_individual_components(individual_components_path, load_classifier)

        self.decoder.load_state_dict(
            torch.load(
                os.path.join(individual_components_path, "decoder.pth"),
                weights_only=True,
            ),
        )

    def save_model(self, path: str) -> None:
        """Saves the model to the specified path.

        Args:
            path (str): path to save the model.
        """
        super().save_model(path)
        torch.save(self.decoder.state_dict(), os.path.join(path, "decoder.pth"))

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(MAE, MAE).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "ssl_settings.decoder_embed_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "ssl_settings.decoder_depth")
        assert not omegaconf.OmegaConf.is_missing(cfg, "ssl_settings.decoder_num_heads")

        cfg.ssl_settings.mask_ratio = omegaconf_select(
            cfg, "ssl_settings.mask_ratio", 0.75
        )
        cfg.ssl_settings.norm_pix_loss = omegaconf_select(
            cfg,
            "ssl_settings.norm_pix_loss",
            False,
        )

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "decoder", "params": self.decoder.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor, detach_backbone: bool) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        # modified base forward
        if self.channels_last:
            X = X.to(memory_format=torch.channels_last)

        out = {}
        if self.training:
            feats, patch_feats, mask, ids_restore = self.backbone(X, self.mask_ratio)
            pred = self.decoder(patch_feats, ids_restore)
            out.update({"mask": mask, "pred": pred})
        else:
            feats = self.backbone(X)

        logits = self.classifier(feats.detach())
        out.update({"logits": logits, "feats": feats})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for MAE reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MAE and classification loss.
        """

        out = super().training_step(batch, batch_idx)

        patch_size = self._vit_patch_size
        imgs = batch["img"][1]
        final_loss = 0
        for i in range(self.num_large_crops):
            final_loss += mae_loss_func(
                imgs[i],
                out["pred"][i],
                out["mask"][i],
                patch_size,
                norm_pix_loss=self.norm_pix_loss,
            )
        final_loss /= self.num_large_crops

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
