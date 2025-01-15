"""This module will contain the logic for training the defect detection model."""

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from torch import nn

import wandb
from parse_arguments import get_training_parser
from src.data.data_module import SewerDataModule
from src.models import BYOL, DINO, MAE, BarlowTwins, SewerNet, SimCLR
from src.utils.callbacks import (ArbitraryEpochCheckpoint,
                                 SavePredictionsCallback)
from src.utils.misc import broadcast_error_to_all_workers, make_contiguous

_MODELS = {
    "sewer_net": SewerNet,
    "byol": BYOL,
    "mae": MAE,
    "barlow_twins": BarlowTwins,
    "simclr": SimCLR,
    "dino": DINO,
}


def build_conf_dict(
    my_parser: Any,
    my_args: Dict[str, Any],
) -> Dict[str, DictConfig]:
    """Builds configuration dictionaries from parser groups."""

    conf_dict = DictConfig({})
    for action in my_parser._action_groups:  # pylint: disable=protected-access
        group_dict = {}
        for arg in action._group_actions:  # pylint: disable=protected-access
            if arg.dest in my_args.keys():
                arg_value = my_args[arg.dest]

                if arg_value in ["True", "False"]:
                    # Resolve boolean values
                    arg_value = arg_value == "True"

                elif arg_value == "None":
                    # Resolve None values
                    arg_value = None

                group_dict[arg.dest] = arg_value
        conf_dict[action.title] = group_dict

    conf_dict["learning_strategy"] = my_args["learning_strategy"]
    conf_dict["only_log_on_epoch_end"] = my_args["only_log_on_epoch_end"]

    return conf_dict


def get_wandb_run(entity: str, project: str, run_id: str):
    """Wrapper to get a wandb run."""
    return wandb.Api().run(f"{entity}/{project}/{run_id}")


def load_model_config(run) -> Tuple[DictConfig, str]:
    """Load model config from a run."""
    print("Loading model config...")
    model_config_path = run.file("model_config.json").download(replace=True).name
    print(f"Downloaded {model_config_path}")

    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    return OmegaConf.create(model_config), model_config_path


def load_ckpt_to_resume_run(
    config: DictConfig,
) -> Tuple[DictConfig, List[str], Dict[str, Any]]:
    """
    Resumes a run by loading a checkpoint and updating the model configuration.
    Args:
        config (DictConfig): The configuration settings for resuming the run.
    Raises:
        AssertionError: If `load_model` is not set to True when providing a wandb run `id`.
    Returns:
        config (DictConfig): The updated configuration settings.
        loaded_files (List[str]): The paths to the loaded files.
    """

    assert (
        config.load_model_settings.load_model
    ), "If you provide a wandb run `id`, you must set `load_model=True`."

    # We are resuming a run
    run = get_wandb_run(
        config.wandb_settings.entity,
        config.wandb_settings.project,
        config.wandb_settings.id,
    )

    # Download the checkpoint and get the local path
    print("Loading checkpoint...")
    ckpt_local_path = (
        run.file(config.load_model_settings.ckpt_path).download(replace=True).name
    )
    print(f"Downloaded {ckpt_local_path}")

    # Overwrite the model config
    config, local_model_config_path = load_model_config(run)
    config.wandb_settings.id = run.id

    # Upload the model settings
    load_model_settings = {
        "load_model_settings": {
            "load_model": True,
            "ckpt_path": config.load_model_settings.ckpt_path,
        }
    }

    return config, [ckpt_local_path, local_model_config_path], load_model_settings


def load_individual_component_to_load_model(
    config: DictConfig,
) -> Tuple[DictConfig, List[str], Dict[str, Any]]:
    """
    Load the model with the specified configuration.
    Args:
        config (DictConfig): The configuration settings for loading the model.
    Returns:
        DictConfig: The updated configuration with the loaded model settings.
        List[str]: The paths to the loaded files.
    """

    load_params_to_wandb = {
        "load_model": True,
        "wandb_id": config.load_model_settings.wandb_id,
        "individual_components_path": config.load_model_settings.individual_components_path,
        "load_classifier": config.load_model_settings.load_classifier,
    }

    if config.load_model_settings.wandb_entity is not None:
        load_params_to_wandb["wandb_entity"] = config.load_model_settings.wandb_entity

    if config.load_model_settings.wandb_project is not None:
        load_params_to_wandb["wandb_project"] = config.load_model_settings.wandb_project

    # Load the model without resuming the run
    run = get_wandb_run(
        config.load_model_settings.wandb_entity or config.wandb_settings.entity,
        config.load_model_settings.wandb_project or config.wandb_settings.project,
        config.load_model_settings.wandb_id,
    )

    # Filter individual components
    files = run.files()
    individual_component_filenames = [
        file.name
        for file in files
        if file.name.startswith(config.load_model_settings.individual_components_path)
    ]

    # List to store path of downloaded files
    loaded_files = []

    for filename in individual_component_filenames:
        # Download the checkpoint
        print("Loading individual component:", filename)
        ind_comp_path = run.file(filename).download(replace=True).name
        print(f"Downloaded {ind_comp_path}")
        loaded_files.append(ind_comp_path)

    if config.load_model_settings.ckpt_path is not None:
        assert (
            config.wandb_settings.job_type == "train"
        ), "You cannot load `ckpt_path` when `job_type`=='train'."

        # TODO: make sure that the new optimization settings are not being overwritten
        # by the ones embedded in the checkpoint path
        print("Loading checkpoint...")
        ckpt_path = (
            run.file(config.load_model_settings.ckpt_path).download(replace=True).name
        )
        print(f"Downloaded {ckpt_path}")
        loaded_files.append(ckpt_path)

        load_params_to_wandb["ckpt_path"] = config.load_model_settings.ckpt_path

    # Overwrite the model config
    old_config, local_model_config_path = load_model_config(run)
    loaded_files.append(local_model_config_path)

    config.model_settings = old_config.model_settings

    if config.load_model_settings.load_optimization_settings:
        config.optimization = old_config.optimization

        load_params_to_wandb["load_optimization_settings"] = True

    if config.learning_strategy == "ssl":
        config.ssl_settings = old_config.ssl_settings
        config.momentum_settings = old_config.momentum_encoder

        if config.load_model_settings.load_optimization_settings:
            config.ssl_optimization = old_config.ssl_optimization

    return config, loaded_files, {"load_model_settings": load_params_to_wandb}


def save_model_config(config: DictConfig, path: str) -> None:
    """Save model config to a file."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(config, resolve=True), f)


def get_model(pos_weights: torch.Tensor, conf_dict: DictConfig) -> nn.Module:
    """Instantiate the model based on the configuration dictionary."""

    # Define supervised loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Define model
    if conf_dict.learning_strategy == "supervised":
        model_name = "sewer_net"
    else:
        model_name = conf_dict.ssl_settings.ssl_method

    model_class = _MODELS[model_name]

    if conf_dict.wandb_settings.id is not None:
        # Resume run
        model = model_class.load_from_checkpoint(
            conf_dict=conf_dict,
            criterion=criterion,
            checkpoint_path=conf_dict.load_model_settings.ckpt_path,
        )
    elif conf_dict.load_model_settings.load_model:
        model = model_class(conf_dict=conf_dict, criterion=criterion)

        # Build complete individual components path
        cur_path = os.path.dirname(os.path.abspath(__file__))
        individual_components_path = os.path.join(
            cur_path, conf_dict.load_model_settings.individual_components_path
        )

        # Load individual components
        model.load_individual_components(
            individual_components_path,
            conf_dict.load_model_settings.load_classifier,
        )
    else:
        model = model_class(conf_dict=conf_dict, criterion=criterion)

    make_contiguous(model)

    if conf_dict.optimization.channels_last:
        # can provide up to ~20% speed up
        model.backbone = model.backbone.to(memory_format=torch.channels_last)
        if hasattr(model, "momentum_backbone"):
            model.momentum_backbone = model.momentum_backbone.to(
                memory_format=torch.channels_last
            )

    return model


def get_save_predictions_callback() -> Callback:
    """Get callback to save predictions."""

    predictions_path = os.path.join(wandb.run.dir, "predictions")
    os.makedirs(predictions_path, exist_ok=True)
    wandb.save(predictions_path + "/*", base_path=wandb.run.dir)
    return SavePredictionsCallback(predictions_path)


def get_callbacks(percentages: List[float]) -> List[Callback]:
    """Get callbacks for the training process."""

    # Define checkpoint path
    ckpt_path = os.path.join(wandb.run.dir, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    wandb.save(ckpt_path + "/*", base_path=wandb.run.dir)

    model_ckpt = ModelCheckpoint(
        dirpath=ckpt_path,
        filename=f"id={wandb.run.id}" + "-{epoch}",
        save_last=True,
        save_top_k=0,
        verbose=True,
    )
    model_ckpt.FILE_EXTENSION = ".pth"
    model_ckpt.CHECKPOINT_NAME_LAST = f"id={wandb.run.id}-last"

    lr_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True)

    save_predictions = get_save_predictions_callback()

    arbitrary_epoch_checkpoint = ArbitraryEpochCheckpoint(
        percentages=percentages, checkpoint_dirpath=ckpt_path
    )

    return [model_ckpt, lr_monitor, save_predictions, arbitrary_epoch_checkpoint]


def train_model(
    dm: SewerDataModule,
    conf_dict: DictConfig,
    is_main_node: bool,
    model: nn.Module,
    trainer_kwargs: Dict[str, Any],
):
    """Train the model."""
    trainer_kwargs["logger"].watch(model, log="gradients", log_freq=100)

    if is_main_node:
        model.log_params_to_wandb()

        wandb.config.update(
            {
                "train-pos-proportion": np.round(
                    dm.train_dataset.proportion_of_positives, 3
                ),
                "val-pos-proporttion": np.round(
                    dm.valid_dataset.proportion_of_positives, 3
                ),
            }
        )

    # Define trainer
    trainer = Trainer(
        callbacks=get_callbacks(conf_dict.arbitrary_epoch_checkpoint.percentages),
        accelerator="cuda",
        **trainer_kwargs,
    )

    # Train model
    trainer.fit(model, datamodule=dm, ckpt_path=conf_dict.load_model_settings.ckpt_path)

    if is_main_node:
        # Save individual components
        ind_comps_save_path = os.path.join(wandb.run.dir, "individual_components")
        os.makedirs(ind_comps_save_path, exist_ok=True)
        model.save_model(ind_comps_save_path)
        wandb.save(ind_comps_save_path + "/*", base_path=wandb.run.dir)


def test_model(
    dm: SewerDataModule,
    is_main_node: bool,
    model: nn.Module,
    trainer_kwargs: Dict[str, Any],
):
    """Test the model."""

    # Get proportion of positive samples in the test dataset and set them as tags
    test_pos_proportion = dm.test_dataset.proportion_of_positives
    if is_main_node:
        wandb.config.update({"test-pos-proportion": np.round(test_pos_proportion, 3)})

    # Initialize a new Trainer object with one GPU
    trainer = Trainer(
        callbacks=get_save_predictions_callback(), accelerator="cuda", **trainer_kwargs
    )

    trainer.test(model, datamodule=dm)


def main(is_main_node: bool, conf_dict: DictConfig):
    """Run training and testing workflow."""
    # Define datamodule
    dm = SewerDataModule(conf_dict)

    if is_main_node:
        dm.log_params_to_wandb()

    # Define logger
    logger = WandbLogger(log_model="last")

    # Define Trainer Kwargs
    trainer_kwargs = OmegaConf.to_container(conf_dict.pl_trainer_settings, resolve=True)
    trainer_kwargs["logger"] = logger

    if conf_dict.wandb_settings.job_type == "train":
        # Create train and val datasets
        dm.setup("fit")

        train_model(
            dm=dm,
            conf_dict=conf_dict,
            is_main_node=is_main_node,
            model=get_model(dm.train_dataset.pos_weights, conf_dict),
            trainer_kwargs=trainer_kwargs,
        )
    else:
        # Create test dataset
        dm.setup("test")

        test_model(
            dm=dm,
            is_main_node=is_main_node,
            model=get_model(None, conf_dict),
            trainer_kwargs=trainer_kwargs,
        )


def remove_loaded_files(loaded_files: List[str]) -> None:
    """Remove the loaded files."""

    if not loaded_files:
        return

    print("Removing loaded files...")
    for file in loaded_files:
        print(f"Removing {file}")
        os.remove(file)
        # Remove empty directories
        dir_path = os.path.dirname(file)
        if not os.listdir(dir_path):
            os.rmdir(dir_path)
            print(f"Removed empty directory: {dir_path}")
    print("Files and empty directories removed.")


if __name__ == "__main__":
    # Set environment variables
    os.environ["NCCL_DEBUG"] = "WARN"
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(42)

    # Parse Arguments
    parser = get_training_parser()
    args = vars(parser.parse_args())

    # Load model checkpoint if provided
    CONFIG = build_conf_dict(parser, args)

    loaded_files = []
    if CONFIG.wandb_settings.id is not None:
        CONFIG, loaded_files, load_model_settings = load_ckpt_to_resume_run(CONFIG)

    elif CONFIG.load_model_settings.load_model:
        CONFIG, loaded_files, load_model_settings = (
            load_individual_component_to_load_model(CONFIG)
        )

    else:
        load_model_settings = {"load_model_settings": {"load_model": False}}

    try:
        # Initialize multinode training
        # TODO: check with Randall how to handle this on his GPUs
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        node_rank = int(os.environ.get("NODE_RANK", 0))
        IS_MAIN_NODE = local_rank == 0 and node_rank == 0

        if IS_MAIN_NODE:
            # Get wandb settings
            wandb_kwargs = OmegaConf.to_container(CONFIG.wandb_settings, resolve=True)

            run_config = OmegaConf.to_container(CONFIG, resolve=True)
            run_config.pop("ssl_settings")
            run_config.pop("ssl_optimization")
            run_config.pop("knn_eval")
            run_config.pop("momentum_encoder")
            run_config.pop("data_module_settings")
            run_config.pop("optimization")
            run_config.pop("load_model_settings")
            run_config.pop("augmentations")

            wandb.init(
                config=run_config,
                resume="allow",
                **wandb_kwargs,
            )

            wandb.config.update(load_model_settings)

            save_model_config(CONFIG, os.path.join(wandb.run.dir, "model_config.json"))

        # Run main
        main(is_main_node=IS_MAIN_NODE, conf_dict=CONFIG)

        remove_loaded_files(loaded_files)

        # End run properly
        wandb.finish()
    except Exception as e:
        print(f"Error encountered: {e}")

        remove_loaded_files(loaded_files)
        # If an error occurs, broadcast it to all workers to ensure a unified shutdown.
        broadcast_error_to_all_workers()
        raise e  # Re-raise the exception to handle it normally or to stop the program.
