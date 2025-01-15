"""This module contains the dataset for training the SewerNet model."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from src.base_classes.dataset import BaseDataset
from src.utils.sampling import split_data, stratified_sampling

DEFECT_LABELS = [
    "RB",
    "OB",
    "PF",
    "DE",
    "FS",
    "IS",
    "RO",
    "IN",
    "AF",
    "BE",
    "FO",
    "GR",
    "PH",
    "PB",
    "OS",
    "OP",
    "OK",
]


class SewerDataset(BaseDataset):
    """Dataset that loads pipe images.

    Args:
        data_roots_and_annots (List[Tuple[str, str]]): A list of tuples containing the
            data root path and annotation file path for each dataset.
        stage (str): The stage of the dataset, either "train" or "test".
        is_binary (bool): Indicates whether the dataset is for binary classification or
            multi-class classification.
        transform (transforms.Compose): A composition of image transformations to apply
            to the dataset.
        sampling_strategy (str): The sampling strategy to use for class balancing, options
            are "random" or "binary-weighted".

    Attributes:
        stage (str): The stage of the dataset, either "train", "val" or "test".
        transform (transforms.Compose): A composition of image transformations to apply
            to the dataset.
        sampling_strategy (str): The sampling strategy used for class balancing.
        label_names (List[str]): The names of the labels in the dataset.
        filenames (np.ndarray): An array of image filenames.
        img_types (np.ndarray): An array of image types.
        clusters (np.ndarray): An array of cluster values.
        clients (np.ndarray): An array of client values.
        labels (torch.Tensor): A tensor of label values.
        class_weights (torch.Tensor): The class weights for WeightedRandomSampler.
        pos_weights (torch.Tensor): The positive weights for the CrossEntropy loss function.
        proportion_of_positives (float): The proportion of positive samples in the dataset.
    """

    _DEFECT_LABELS = DEFECT_LABELS

    def __init__(
        self,
        data_roots_and_annots: List[Tuple[str, str]],
        stage: str,
        is_binary: bool,
        transform: transforms.Compose,
        sampling_strategy: str,
        defect_proportion: Optional[float] = None,
        beta: Optional[float] = None,
        use_f2ciw_pos_weights: Optional[bool] = False,
        sample_percentage: Optional[int] = None,
    ):
        """Initialize the dataset."""
        super().__init__()
        self.is_binary = is_binary
        self.stage = stage
        self.transform = transform
        self.sampling_strategy = sampling_strategy
        self.beta = beta
        self.use_f2ciw_pos_weights = use_f2ciw_pos_weights
        self.defect_proportion = defect_proportion
        self.sample_percentage = sample_percentage

        # Set label names according to the experiment setting
        if is_binary:
            self.label_names = ["Defect"]
        else:
            self.label_names = self._DEFECT_LABELS

        # Load annotations from CSV file

        # Used in ITpipes and SewerML
        self.filenames: np.ndarray = None
        self.targets: torch.Tensor = None
        self.strata: np.ndarray = None

        # Used only in SewerML when task_type is multi-label
        self.normals: np.ndarray = None

        self.load_data(data_roots_and_annots)

        if self.defect_proportion is not None:
            self.imbalance_data()

        # Get class weights and positive weights
        if self.stage == "train":
            self.get_class_weights()
            self.get_pos_weights()

        # Compute proportion of positives
        if not self.is_binary:
            defects = self.normals
        else:
            defects = self.targets
        self.proportion_of_positives = defects.to(torch.float32).sum() / len(defects)

    def get_annotations_from_folder(
        self, data_root: str, annot_path: str
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Reads the CSV file with the annotations located in the data root. If there are
        multiple CSV files in the data root, an exception is raised. If there are no CSV
        files in the data root, an exception is raised. The CSV file is read and the
        filenames and labels are returned."""

        annotations = pd.read_csv(annot_path)

        if self.sample_percentage is not None:
            annotations, _ = split_data(
                annotations, self.sample_percentage, strata=self._DEFECT_LABELS
            )

        df_component_dict = {}
        df_component_dict["filenames"] = (
            str(data_root) + "/" + annotations["Filename"]
        ).values
        df_component_dict["targets"] = torch.tensor(  # pylint: disable=no-member
            annotations[self.label_names].values, dtype=torch.float16
        )

        if not self.is_binary:
            df_component_dict["normals"] = torch.tensor(  # pylint: disable=no-member
                annotations["Defect"].values, dtype=torch.float16
            )
        else:
            df_component_dict["strata"] = torch.tensor(
                annotations[self._DEFECT_LABELS].values, dtype=torch.float16
            )

        return df_component_dict

    def load_data(
        self, data_roots_and_annots: List[Tuple[str, str]]
    ):  # pylint: disable=arguments-differ
        """Loads annotations iteratively from the given data roots and annotations."""
        for root, annot_path in data_roots_and_annots:
            # Iterate through data roots and get annotations
            df_component_dict = self.get_annotations_from_folder(root, annot_path)

            # Concatenate annotations
            if self.filenames is None:
                self.filenames = df_component_dict["filenames"]
                self.targets = df_component_dict["targets"]
                if not self.is_binary:
                    self.normals = df_component_dict["normals"]
                else:
                    self.strata = df_component_dict["strata"]
            else:
                self.filenames = np.concatenate(
                    [self.filenames, df_component_dict["filenames"]], axis=0
                )
                self.targets = torch.cat(  # pylint: disable=no-member
                    [self.targets, df_component_dict["targets"]], dim=0
                )
                if not self.is_binary:
                    self.normals = torch.cat(  # pylint: disable=no-member
                        [self.normals, df_component_dict["normals"]], dim=0
                    )
                else:
                    self.strata = torch.cat(
                        [self.strata, df_component_dict["strata"]], dim=0
                    )

    def sample_data(self):
        """Takes a sample of the dataset."""

        # Create a np array of indices
        indices = np.arange(len(self.filenames))

        # Split the data according to the sample percentage
        sampled_indices, _ = split_data(
            indices, self.sample_percentage, strata=self._DEFECT_LABELS
        )

    def imbalance_data(self):
        """Removes samples from the dataset to create class imbalance."""

        if self.is_binary:
            # Create DataFrame from filenames and targets
            df = pd.DataFrame(
                {
                    "Filename": self.filenames,
                    "Defect": self.targets.numpy().squeeze().astype(int),
                }
            )
            df_strata = pd.DataFrame(
                self.strata.numpy().squeeze().astype(int), columns=self._DEFECT_LABELS
            )
            df = pd.concat([df, df_strata], axis=1)
        else:
            # Create DataFrame from filenames, targets and normals
            df = pd.DataFrame(
                {
                    "Filename": self.filenames,
                    "Defect": self.normals.numpy().squeeze().astype(int),
                }
            )
            df_targets = pd.DataFrame(
                self.targets.numpy().squeeze().astype(int), columns=self.label_names
            )
            df = pd.concat([df, df_targets], axis=1)

        imbalanced_df = stratified_sampling(
            df, desired_proportion=self.defect_proportion, strata=self._DEFECT_LABELS
        )

        # Update attributes
        self.filenames = imbalanced_df["Filename"].values
        self.targets = torch.tensor(  # pylint: disable=no-member
            imbalanced_df[self.label_names].values, dtype=torch.float16
        )
        if not self.is_binary:
            self.normals = torch.tensor(  # pylint: disable=no-member
                imbalanced_df["Defect"].values, dtype=torch.float16
            )
        else:
            self.strata = torch.tensor(
                imbalanced_df[self._DEFECT_LABELS].values, dtype=torch.float16
            )

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.filenames)

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        # Get paths from index
        path = self.filenames[index]

        outs = dict(
            img=self.transform(default_loader(path)),
            targets=self.targets[index],
        )

        if not self.is_binary:
            outs["normals"] = self.normals[index]
        else:
            outs["strata"] = self.strata[index]

        if self.stage != "train":
            outs["path"] = path

        return outs


class DatasetWithIndex(SewerDataset):
    """Dataset that loads pipe images with their index."""

    def __getitem__(self, index):
        data = super().__getitem__(index)
        return (index, data)
