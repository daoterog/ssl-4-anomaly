import os
from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl

import wandb
from src.data.datasets import DEFECT_LABELS


def _add_prefix_to_list(prefix: str, list_: list) -> list:
    return [f"{prefix}_{item}" for item in list_]


class SavePredictionsCallback(pl.Callback):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.output_filename = "predictions-epoch={}.csv"
        self.paths = []
        self.logits = []
        self.preds = []
        self.targets = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.paths = []
        self.logits = []
        self.preds = []
        self.targets = []

    def on_test_epoch_start(self, trainer, pl_module):
        self.on_validation_epoch_start(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if trainer.sanity_checking:
            return

        self.paths.extend(batch["path"])
        self.targets.append(batch["targets"].cpu().numpy())

        # Make sure to get first n_obs samples because when working with multiple views
        # the number of samples in the outputs can be different
        n_obs = len(batch["path"])
        self.logits.append(outputs["logits"].cpu().numpy()[:n_obs])
        self.preds.append(outputs["pred"].cpu().numpy()[:n_obs])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_epoch_end(self, trainer, pl_module):

        if trainer.sanity_checking:
            return

        # Concatenate all predictions
        logits = np.concatenate(self.logits, axis=0).squeeze()
        preds = np.concatenate(self.preds, axis=0).squeeze()
        targets = np.concatenate(self.targets, axis=0).squeeze()

        # Check logits shape to determine if it is a binary or multi-class classification
        if logits.ndim == 1:
            df = pd.DataFrame(
                {
                    "Filename": self.paths,
                    "Logit": logits,
                    "Prediction": preds,
                    "Target": targets,
                }
            )
        else:
            df_logits = pd.DataFrame(
                logits, columns=_add_prefix_to_list("Logit", DEFECT_LABELS)
            )
            df_preds = pd.DataFrame(
                preds, columns=_add_prefix_to_list("Prediction", DEFECT_LABELS)
            )
            df_targets = pd.DataFrame(
                targets, columns=_add_prefix_to_list("Target", DEFECT_LABELS)
            )
            df = pd.concat(
                [
                    pd.DataFrame(self.paths, columns=["Filename"]),
                    df_logits,
                    df_preds,
                    df_targets,
                ],
                axis=1,
            )

        # Save predictions in the output directory and format the epoch number
        output_filename = self.output_filename.format(trainer.current_epoch)
        output_path = os.path.join(self.output_dir, output_filename)
        df.to_csv(output_path, index=False)
        print(f"Saved predictions to {self.output_dir}")

    def on_test_epoch_end(self, trainer, pl_module):
        self.on_validation_epoch_end(trainer, pl_module)


class ArbitraryEpochCheckpoint(pl.Callback):
    def __init__(self, percentages: List[float], checkpoint_dirpath: str):
        # Check that every epoch checkpoint is in (0, 1)
        assert all(0 < epoch < 1 for epoch in percentages)

        super().__init__()
        self.checkpoint_dirpath = checkpoint_dirpath
        self.percentages = percentages
        self.save_epochs: List[int] = None

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Calculate the epochs to save
        self.save_epochs = [
            int(epoch * trainer.max_epochs) for epoch in self.percentages
        ]
        return super().on_train_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch in self.save_epochs:
            print(f"Saving checkpoint at epoch {trainer.current_epoch}")
            save_path = os.path.join(
                self.checkpoint_dirpath,
                f"id={wandb.run.id}-epoch={trainer.current_epoch}.pth",
            )
            trainer.save_checkpoint(save_path)
            print(f"Checkpoint saved at {save_path}")
