"""
This module contains the `BaseDataset` class and helper functions for dataset handling
in defect detection models.

Classes:
    BaseDataset: Base class for datasets used in defect detection models.

Functions:
    _get_empirical_pos_weights: Get class weights and simulate 5 epochs of training to get
        the mean occurrence of each class and compute the empirical positive weights with it.
    _compute_binary_pos_weights: Computes the ratio of non-defects to defects.
"""

from abc import abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset

from src.utils.metrics import SEWERML_CLASS_IMPORTANCE_WEIGHTS


def _compute_f2ciw_pos_weights() -> torch.Tensor:
    """Computes positive weights based on the class importance weights defined in the
    Sewer-ML paper."""

    # Build class importance weights tensor
    class_importance_weights = torch.tensor(  # pylint: disable=no-member
        list(SEWERML_CLASS_IMPORTANCE_WEIGHTS.values()),
        dtype=torch.float32,
    )
    norm_val = torch.mean(class_importance_weights)  # pylint: disable=no-member

    # Normalize class importance weights
    return (1 + class_importance_weights / norm_val) * 2.0


def _compute_binary_pos_weights(labels: torch.Tensor) -> torch.Tensor:
    """Computes the ratio of non-defects to defects."""
    # Get number of positives and negatives
    num_positives = torch.sum(labels, dim=0).type(  # pylint: disable=no-member
        torch.int32
    )
    num_negatives = len(labels) - num_positives

    # Get class weights
    pos_weights = num_negatives / num_positives

    # Filter out Infs
    return torch.where(  # pylint: disable=no-member
        torch.isinf(pos_weights),  # pylint: disable=no-member
        torch.zeros_like(pos_weights),  # pylint: disable=no-member
        pos_weights,
    )


class BaseDataset(Dataset):
    """
    Base class for datasets used in defect detection models.

    Attributes:
        sampling_strategy (str): The sampling strategy used for the dataset.
        targets (torch.Tensor): The target labels for the dataset.
        class_weights (torch.Tensor): The class weights computed for the dataset.
        pos_weights (torch.Tensor): The positive weights computed for the dataset.

    Methods:
        load_data: Abstract method to load the data for the dataset.
        get_class_weights: Compute class weights to use in WeightedRandomSampler.
        get_pos_weights: Get the proportion of negative samples per positive samples to
            use within CrossEntropy loss function.
    """

    def __init__(self):
        self.beta: float = None
        self.is_binary: bool = None
        self.use_f2ciw_pos_weights: bool = None
        self.sampling_strategy: str = None
        self.targets: torch.Tensor = None

        self.class_weights: torch.Tensor = None
        self.pos_weights: torch.Tensor = None

    def load_data(self):
        """
        Optional template to load the data for the dataset.
        Subclasses that load data internally should implement this method.
        Make sure to set the following attributes:
        - self.targets: torch.Tensor (dtype=torch.long)
        """

    def get_class_weights(self):
        """Compute class weights to use in WeightedRandomSampler. In the case of multi-label
        classification on Sewer-ML, the class weights are used to compute the effective
        weights for each class to use within the BCELoss."""

        # Count number of samples per class
        num_samples_per_class = torch.sum(  # pylint: disable=no-member
            self.targets, dim=0
        )

        assert (
            num_samples_per_class > 0
        ).all(), "There are classes with no samples in the dataset"

        if len(num_samples_per_class.shape) > 1:
            raise NotImplementedError(
                "Class weights for multi-label classification not implemented"
            )

        if not self.is_binary:
            # Compute number of positives per class
            num_positives = torch.sum(self.targets, dim=0)  # pylint: disable=no-member

            # Compute effective weights and normalize them
            effective_weights = (1 - self.beta) / (
                1 - torch.pow(self.beta, num_positives)
            )
            class_weights = effective_weights / effective_weights.mean()

        elif self.sampling_strategy == "random":
            return

        elif self.sampling_strategy == "binary-weighted":
            samples_per_class = torch.bincount(  # pylint: disable=no-member
                self.targets.squeeze().long()
            )
            label_proportion = 1 / samples_per_class
            class_weights = label_proportion[self.targets.long()].squeeze()

        else:
            raise NotImplementedError(
                f"Sampling strategy {self.sampling_strategy} not implemented"
            )

        self.class_weights = class_weights

    def get_pos_weights(self) -> torch.Tensor:
        """Get the proportion of negative samples per positive samples to use within
        CrossEntropy loss function."""

        if not self.is_binary and self.use_f2ciw_pos_weights:
            pos_weights = _compute_f2ciw_pos_weights()
        else:
            pos_weights = _compute_binary_pos_weights(self.targets)

        self.pos_weights = pos_weights

    @abstractmethod
    def __getitem__(self, index) -> Any:
        """This is just a formality to avoid python to complain about the missing
        implementation of the abstract method."""
