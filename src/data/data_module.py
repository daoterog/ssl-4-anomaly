"""Contains the data module for training the SewerNet model."""

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import wandb
from src.data.datasets import DatasetWithIndex, SewerDataset
from src.data.transforms import NCropAugmentation, instantiate_transforms


class SewerDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the SewerNet model.

    This module contains the data module for training the SewerNet model. It provides
    the necessary data loading and preprocessing functionality for training, validation,
    and testing.

    Attributes:
        train_data_roots_and_annots (list): A list of tuples containing the paths to the
            training data and annotation files.
        valid_data_roots_and_annots (list): A list of tuples containing the paths to the
            validation data and annotation files.
        test_data_roots_and_annots (list): A list of tuples containing the paths to the
            test data and annotation files.
        is_binary (bool): A flag indicating whether the task is binary classification.
        is_weighted (bool): A flag indicating whether to use weighted sampling.
        sampling_strategy (str): The sampling strategy for the dataset.
        batch_size (int): The batch size for training.
        num_workers (int): The number of workers for data loading.
        img_size (int): The size of the input images.
        train_transform (torchvision.transforms.Compose): The transformation pipeline for
            training data.
        val_test_transform (torchvision.transforms.Compose): The transformation pipeline
            for validation and test data.
        train_dataset (SewerDataset): The training dataset.
        valid_dataset (SewerDataset): The validation dataset.
        test_dataset (SewerDataset): The test dataset.
    """

    def __init__(self, conf_dict: DictConfig):
        super().__init__()

        # Set attributes
        self.train_data_roots_and_annots = (
            list(
                zip(
                    conf_dict.data_inputs.train_data, conf_dict.data_inputs.train_annots
                )
            )
            if conf_dict.wandb_settings.job_type == "train"
            else []
        )
        self.valid_data_roots_and_annots = (
            list(
                zip(
                    conf_dict.data_inputs.valid_data, conf_dict.data_inputs.valid_annots
                )
            )
            if conf_dict.wandb_settings.job_type == "train"
            else []
        )
        self.test_data_roots_and_annots = list(
            zip(conf_dict.data_inputs.test_data, conf_dict.data_inputs.test_annots)
        )

        self.is_binary: bool = conf_dict.model_settings.task_type == "binary"
        self.is_weighted: bool = (
            conf_dict.data_module_settings.sampling_strategy != "random"
        )
        self.sampling_strategy = conf_dict.data_module_settings.sampling_strategy
        self.batch_size: int = conf_dict.optimization.batch_size
        self.num_workers: int = conf_dict.data_module_settings.num_workers
        self.img_size: int = conf_dict.model_settings.img_size
        self.is_simclr: bool = conf_dict.ssl_settings.ssl_method == "simclr"
        self.effective_beta: float = conf_dict.data_module_settings.effective_beta
        self.use_f2ciw_pos_weights: bool = (
            conf_dict.data_module_settings.use_f2ciw_pos_weights
        )
        self.imbalance_data: bool = conf_dict.data_module_settings.imbalance_dataset
        self.defect_proportion: float = conf_dict.data_module_settings.defect_proportion
        self.use_sample: bool = conf_dict.data_module_settings.use_sample
        self.sample_percentage: float = conf_dict.data_module_settings.sample_percentage
        self.dl_kwargs = dict(num_workers=self.num_workers, pin_memory=True)

        # Define transformations
        self.transforms = instantiate_transforms(
            conf_dict.augmentations, conf_dict.model_settings.img_size
        )

        if conf_dict.learning_strategy == "ssl":
            self.transforms = NCropAugmentation(
                self.transforms, conf_dict.ssl_optimization.num_large_crops
            )

        self.train_dataset: SewerDataset = None
        self.valid_dataset: SewerDataset = None
        self.test_dataset: SewerDataset = None

    def log_params_to_wandb(self):
        """Log params to wandb."""
        param_dict = {
            "data_module_settings": {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "sampling_strategy": self.sampling_strategy,
                "imbalance_dataset": self.imbalance_data,
                "use_sample": self.use_sample,
            }
        }

        if self.imbalance_data:
            param_dict["data_module_settings"][
                "defect_proportion"
            ] = self.defect_proportion

        if self.use_sample:
            param_dict["data_module_settings"][
                "sample_percentage"
            ] = self.sample_percentage

        if not self.is_binary:
            param_dict["data_module_settings"]["effective_beta"] = self.effective_beta
            param_dict["data_module_settings"][
                "use_f2ciw_pos_weights"
            ] = self.use_f2ciw_pos_weights

        wandb.config.update(param_dict)

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Define dataset
        if stage == "fit":
            # Instantiate correct dataset class
            if self.is_simclr:
                dataset_class = DatasetWithIndex
            else:
                dataset_class = SewerDataset

            beta = None
            use_f2ciw_pos_weights = False
            if not self.is_binary:
                beta = self.effective_beta
                use_f2ciw_pos_weights = self.use_f2ciw_pos_weights

            self.train_dataset = dataset_class(
                data_roots_and_annots=self.train_data_roots_and_annots,
                stage="train",
                is_binary=self.is_binary,
                transform=self.transforms,
                sampling_strategy=self.sampling_strategy,
                beta=beta,
                use_f2ciw_pos_weights=use_f2ciw_pos_weights,
                defect_proportion=(
                    self.defect_proportion if self.imbalance_data else None
                ),
                sample_percentage=(self.sample_percentage if self.use_sample else None),
            )
            self.valid_dataset = SewerDataset(
                data_roots_and_annots=self.valid_data_roots_and_annots,
                stage="val",
                is_binary=self.is_binary,
                transform=self.transforms,
                sampling_strategy=self.sampling_strategy,
                sample_percentage=(self.sample_percentage if self.use_sample else None),
            )

        if stage == "test":
            self.test_dataset = SewerDataset(
                data_roots_and_annots=self.test_data_roots_and_annots,
                stage="test",
                is_binary=self.is_binary,
                transform=self.transforms,
                sampling_strategy=self.sampling_strategy,
                defect_proportion=(
                    self.defect_proportion if self.imbalance_data else None
                ),
                sample_percentage=(self.sample_percentage if self.use_sample else None),
            )

    def train_dataloader(self):
        # Define sampler
        sampler = None
        if self.is_weighted:
            sampler = WeightedRandomSampler(
                weights=self.train_dataset.class_weights,
                num_samples=len(self.train_dataset),
                replacement=True,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if sampler is None else False,
            sampler=sampler,
            # persistent_workers=True,
            **self.dl_kwargs,
        )

    def create_val_test_dataloader(
        self, dataset: Dataset, batch_size: int
    ) -> DataLoader:
        """Create a dataloader for validation or test dataset."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            **self.dl_kwargs,
        )

    def val_dataloader(self):
        return self.create_val_test_dataloader(
            self.valid_dataset, int(self.batch_size * 2)
        )

    def test_dataloader(self):
        return self.create_val_test_dataloader(
            self.test_dataset, int(self.batch_size / 2)
        )
