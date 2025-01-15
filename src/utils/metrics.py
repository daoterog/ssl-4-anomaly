"""This module will contain custom metrics used to evaluate the defect detection model."""

from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import (F1Score, MultilabelFBetaScore,
                                         MultilabelRecall, Precision, Recall)
from torchmetrics.metric import Metric

from src.utils.misc import get_rank

SEWERML_CLASS_IMPORTANCE_WEIGHTS = {
    "RB": 1.0000,
    "OB": 0.5518,
    "PF": 0.2896,
    "DE": 0.1622,
    "FS": 0.6419,
    "IS": 0.1847,
    "RO": 0.3559,
    "IN": 0.3131,
    "AF": 0.0811,
    "BE": 0.2275,
    "FO": 0.2477,
    "GR": 0.0901,
    "PH": 0.4167,
    "PB": 0.4167,
    "OS": 0.9009,
    "OP": 0.3829,
    "OK": 0.4396,
}


class CustomRecall(torchmetrics.Metric):
    """This is a wrapper of the MultilabelRecall metric. We provide an interface that allows
    us to compute the metric even on the binary case."""

    def __init__(self, is_binary: bool = False):
        super().__init__()
        self.is_binary = is_binary
        self.recall = MultilabelRecall(num_labels=17, average="none")

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        strata: Optional[torch.Tensor] = None,
    ):  # pylint: disable=arguments-differ
        """Wrapper to update the F2 score."""
        if self.is_binary:
            expanded_predictions = torch.where(
                preds == 1, strata, torch.zeros_like(strata)
            )

            self.recall.update(expanded_predictions, strata)
        else:
            self.recall.update(preds, targets)

    def compute(self):
        return self.recall.compute()


class CustomFBetaScore(torchmetrics.Metric):
    """This is a wrapper of the MultilabelFBetaScore metric. We provide an interface that allows
    us to compute the metric even on the binary case."""

    def __init__(self, is_binary: bool = False):
        super().__init__()
        self.is_binary = is_binary
        self.f2_score = MultilabelFBetaScore(beta=2.0, num_labels=17, average="none")

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        strata: Optional[torch.Tensor] = None,
    ):  # pylint: disable=arguments-differ
        """Wrapper to update the F2 score."""
        if self.is_binary:
            expanded_predictions = torch.where(
                preds == 1, strata, torch.zeros_like(strata)
            )

            self.f2_score.update(expanded_predictions, strata)
        else:
            self.f2_score.update(preds, targets)

    def compute(self):
        return self.f2_score.compute()


class F2CIWScore(CustomFBetaScore):
    """Implements the Multilabel F2 score with Class Importance Weights (CIW)."""

    def __init__(self, is_binary: bool = False):
        super().__init__(is_binary)
        if torch.cuda.is_available():
            device = f"cuda:{get_rank()}"
        else:
            device = torch.device("cpu")
        self.class_importance_weights = torch.tensor(  # pylint: disable=no-member
            list(SEWERML_CLASS_IMPORTANCE_WEIGHTS.values()),
            device=device,
            dtype=torch.float32,
        )
        self.norm_val = torch.sum(  # pylint: disable=no-member
            self.class_importance_weights
        )

    def compute(self):
        f2_score = super().compute()

        if self.class_importance_weights.device != f2_score.device:
            self.class_importance_weights = self.class_importance_weights.to(
                f2_score.device
            )
            self.norm_val = self.norm_val.to(f2_score.device)

        weighted_f2_score = torch.sum(  # pylint: disable=no-member
            f2_score * self.class_importance_weights
        )
        return weighted_f2_score / self.norm_val


class WeightedF1Score(torchmetrics.Metric):
    def __init__(self, num_classes, average="macro", dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.average = average
        zeros_tensor = torch.zeros((num_classes,))  # pylint: disable=no-member
        self.add_state(
            "num_samples", default=deepcopy(zeros_tensor), dist_reduce_fx="sum"
        )
        self.add_state(
            "true_positives", default=deepcopy(zeros_tensor), dist_reduce_fx="sum"
        )
        self.add_state(
            "false_positives", default=deepcopy(zeros_tensor), dist_reduce_fx="sum"
        )
        self.add_state(
            "false_negatives", default=deepcopy(zeros_tensor), dist_reduce_fx="sum"
        )

    def update(
        self, preds, target, sample_weight=None
    ):  # pylint: disable=arguments-differ
        if preds.dim() == 2:
            preds = torch.argmax(  # pylint: disable=no-member
                preds, dim=1
            )  # Convert probabilities to class predictions

        for class_idx in range(self.num_classes):
            true_positives = torch.logical_and(  # pylint: disable=no-member
                preds == class_idx, target == class_idx
            )
            false_positives = torch.logical_and(  # pylint: disable=no-member
                preds == class_idx, target != class_idx
            )
            false_negatives = torch.logical_and(  # pylint: disable=no-member
                preds != class_idx, target == class_idx
            )

            if sample_weight is not None:
                true_positives = true_positives.float() * sample_weight
                false_positives = false_positives.float() * sample_weight
                false_negatives = false_negatives.float() * sample_weight

            self.true_positives[class_idx] += true_positives.sum()
            self.false_positives[class_idx] += false_positives.sum()
            self.false_negatives[class_idx] += false_negatives.sum()
            self.num_samples[class_idx] += (true_positives + false_negatives).sum()

    def compute(self):
        precision = self.true_positives / (
            self.true_positives + self.false_positives + 1e-7
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + 1e-7
        )

        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        if self.average == "macro":
            weighted_f1 = f1.mean()
        elif self.average == "weighted":
            class_weights = self.num_samples / self.num_samples.sum()
            weighted_f1 = (f1 * class_weights).sum()
        else:
            raise ValueError("Invalid 'average' parameter. Use 'macro' or 'weighted'.")

        return weighted_f1


def rankme(Z: torch.Tensor, eps: float = 1e-6) -> float:  # pylint: disable=invalid-name
    r"""Our implementation of the RankMe metric proposed in:
    https://arxiv.org/pdf/2210.02885v2.pdf

    The RankMe score is defined by the formula:

    .. math::
        \text{RankMe}(Z) = \exp\left( - \sum_{k=1}^{\min(N, K)} p_k \log p_k \right),

    where :math:`p_k` is given by:

    .. math::
        p_k = \frac{\sigma_k(Z)}{\|\sigma(Z)\|_1} + \epsilon,

    with :math:`\sigma_k(Z)` being the :math:`k`-th singular value of :math:`Z`,
    :math:`\|\sigma(Z)\|_1` is the 1-norm of the singular values,
    and :math:`\epsilon` is a small constant to avoid division by zero.

    Parameters
    ----------
    Z : array_like
        The input matrix for which to compute the RankMe score.

    Returns
    -------
    float
        The RankMe score of the matrix.
    """

    if Z.dtype != torch.float32:
        # svdvals cannot be computed over half precision
        Z = Z.to(torch.float32)

    # Handle the case where the input matrix is not square and the SVD cannot be computed
    try:
        # Compute Singluar Values of the feature matrix
        S = torch.linalg.svdvals(Z)  # pylint: disable=invalid-name, not-callable

        # Compute the norm-1 of the singular values vector
        S_norm = torch.linalg.norm(  # pylint: disable=invalid-name, not-callable
            S, ord=1
        )

        # Compute p_k
        p_k = (S / S_norm) + eps

        # Compute Shannon's entropy
        entropy = -torch.sum(p_k * torch.log(p_k))  # pylint: disable=no-member

        rank_me_score = torch.exp(entropy).item()  # pylint: disable=no-member
    except:  # pylint: disable=bare-except
        rank_me_score = -1

    return rank_me_score


def lidar(Z: torch.Tensor, eps: float = 1e-6) -> float:  # pylint: disable=invalid-name
    """Our implementation of the LiDAR metric proposed in:
    https://arxiv.org/pdf/2312.04000.pdf
    """
    raise NotImplementedError("LiDAR metric is not implemented yet.")


class WeightedKNNClassifier(Metric):
    """Implements the weighted k-NN classifier used for evaluation. Taken from:
    https://github.com/aarashfeizi/gps-ssl/blob/main/solo/utils/knn.py

    Args:
        k (int, optional): number of neighbors. Defaults to 20.
        T (float, optional): temperature for the exponential. Only used with cosine
            distance. Defaults to 0.07.
        max_distance_matrix_size (int, optional): maximum number of elements in the
            distance matrix. Defaults to 5e6.
        distance_fx (str, optional): Distance function. Accepted arguments: "cosine" or
            "euclidean". Defaults to "cosine".
        epsilon (float, optional): Small value for numerical stability. Only used with
            euclidean distance. Defaults to 0.00001.
        dist_sync_on_step (bool, optional): whether to sync distributed values at every
            step. Defaults to False.
    """

    def __init__(
        self,
        k: int = 20,
        T: float = 0.07,
        max_distance_matrix_size: int = int(5e6),
        distance_fx: str = "cosine",
        epsilon: float = 0.00001,
        dist_sync_on_step: bool = False,
        sample_train_size: float = 1.0,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.k = k
        self.T = T  # pylint: disable=invalid-name
        self.max_distance_matrix_size = max_distance_matrix_size
        self.distance_fx = distance_fx
        self.sample_train_size = sample_train_size

        if self.distance_fx not in ["cosine", "euclidean"]:
            raise ValueError(
                f"Distance function {self.distance_fx} not supported. Only 'cosine' and "
                "'euclidean' are supported."
            )

        self.epsilon = epsilon

        self.add_state("train_features", default=[], persistent=False)
        self.add_state("train_targets", default=[], persistent=False)
        self.add_state("test_features", default=[], persistent=False)
        self.add_state("test_targets", default=[], persistent=False)

    def _check_targets_shape(self, targets: torch.Tensor):
        """Check the shape of the targets tensor is the correct one."""
        if targets.ndim == 1:
            return

        if targets.ndim == 2 and (targets.size(1) == 1 or targets.size(0) == 1):
            return

        raise ValueError(
            f"Targets tensor must be of shape (n,), (1, n) or (n, 1). Got {targets.shape}."
        )

    def update(
        self,
        train_features: torch.Tensor = None,
        train_targets: torch.Tensor = None,
        test_features: torch.Tensor = None,
        test_targets: torch.Tensor = None,
    ):  # pylint: disable=arguments-differ
        """Updates the memory banks. If train (test) features are passed as input, the
        corresponding train (test) targets must be passed as well.

        Args:
            train_features (torch.Tensor, optional): a batch of train features. Defaults to None.
            train_targets (torch.Tensor, optional): a batch of train targets. Defaults to None.
            test_features (torch.Tensor, optional): a batch of test features. Defaults to None.
            test_targets (torch.Tensor, optional): a batch of test targets. Defaults to None.
        """

        # Make sure that tuples are complete
        assert (train_features is None) == (train_targets is None)
        assert (test_features is None) == (test_targets is None)

        if train_features is not None:
            assert train_features.size(0) == train_targets.size(0)
            self._check_targets_shape(train_targets)

            if self.sample_train_size < 1.0:
                sample_size = int(train_features.size(0) * self.sample_train_size)
                indices = torch.randperm(  # pylint: disable=no-member
                    train_features.size(0)
                )[:sample_size]
                train_features = train_features[indices]
                train_targets = train_targets[indices]

            self.train_features.append(train_features.detach())
            self.train_targets.append(train_targets.detach())

        if test_features is not None:
            assert test_features.size(0) == test_targets.size(0)
            self._check_targets_shape(test_targets)
            self.test_features.append(test_features.detach())
            self.test_targets.append(test_targets.detach())

    @torch.no_grad()
    def compute(self) -> Tuple[float, float, float, float]:
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.

        Returns:
            Tuple[float, float, float, float]: accuracy, precision, recall and F1 score.
        """

        # Concatenate features in memory banks
        train_features = torch.cat(self.train_features)  # pylint: disable=no-member
        train_targets = torch.cat(self.train_targets)  # pylint: disable=no-member
        test_features = torch.cat(self.test_features)  # pylint: disable=no-member
        test_targets = torch.cat(self.test_targets)  # pylint: disable=no-member

        if self.distance_fx == "cosine":
            train_features = F.normalize(train_features)
            test_features = F.normalize(test_features)

        # Smart way to compute chunk size
        num_train_images = train_targets.size(0)
        num_test_images = test_targets.size(0)
        chunk_size = min(
            max(1, self.max_distance_matrix_size // num_train_images),
            num_test_images,
        )

        # Recompute k if necessary
        k = min(self.k, num_train_images)

        # Initialize metrics
        acc, total = 0.0, 0

        # Create one hot tensor for the retrieval
        num_classes = torch.unique(test_targets).numel()
        retrieval_one_hot = torch.zeros(k, num_classes).to(  # pylint: disable=no-member
            train_features.device
        )
        # Set arguments for the metrics based on the number of classes
        metric_kwargs = dict(
            task="multiclass" if num_classes > 2 else "binary",
            num_classes=num_classes,
        )
        # Instantiate classification metrics
        prec_metric = Precision(**metric_kwargs)
        rec_metric = Recall(**metric_kwargs)
        f1_metric = F1Score(**metric_kwargs)

        if prec_metric.device != train_features.device:  # pylint: disable=no-member
            prec_metric = prec_metric.to(  # pylint: disable=no-member
                train_features.device
            )
            rec_metric = rec_metric.to(  # pylint: disable=no-member
                train_features.device
            )
            f1_metric = f1_metric.to(train_features.device)  # pylint: disable=no-member

        for idx in range(0, num_test_images, chunk_size):
            # get the features for test images
            features = test_features[idx : min((idx + chunk_size), num_test_images), :]
            targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
            batch_size = targets.size(0)

            # calculate the dot product and compute top-k neighbors
            if self.distance_fx == "cosine":
                similarities = torch.mm(  # pylint: disable=no-member
                    features, train_features.t()
                )
            elif self.distance_fx == "euclidean":
                similarities = 1 / (
                    torch.cdist(features, train_features) + self.epsilon
                )
            else:
                raise NotImplementedError

            # get the top-k predictions and return topk similarities values and their
            # indices. Both outputs are of shape (batch_size, k).
            similarities, indices = similarities.topk(k, largest=True, sorted=True)

            # Use view to reshape tensor to be 1xn.
            # Then use expand to reshape tensor to be (batch_size, n).
            # In summary, we are repeating the targets tensor batch_size times across the rows.

            # Make sure that train targets are of shape (n,), (1, n) or (n, 1) for the gather
            # operation to make sense. If we had a tensor shaped (n, num_classes), then
            # the indices tensor would not point to the correct class.
            candidates = train_targets.view(1, -1).expand(batch_size, -1)

            # Candidates is shape (batch_size, n) and indices is shape (batch_size, k)
            # torch.gather will use the indices tensor to select the k neighbors for each
            # sample in the batch along the 1st dimension (cols) of the candidates tensor.
            # Output will be of shape (batch_size, k).
            retrieved_neighbors = torch.gather(  # pylint: disable=no-member
                candidates, 1, indices
            )

            # Initially, the retrieval_one_hot tensor is a zero tensor of shape
            # (k, num_classes).
            # Resize the tensor to be of shape (batch_size * k, num_classes) and fill it
            # with zeros.
            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()

            if retrieved_neighbors.dtype != torch.int64:
                # This is necessary for the scatter_ function to work
                retrieved_neighbors = retrieved_neighbors.to(torch.int64)

            # Along side the 1st dimension (cols) of the retrieval_one_hot tensor, we
            # will scatter 1s at the indices given by the retrieved_neighbors tensor.
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            if self.distance_fx == "cosine":
                if similarities.dtype != torch.float32:
                    # This is necessary for numerical stability of the exponential
                    similarities = similarities.to(torch.float32)
                similarities = similarities.clone().div_(self.T).exp_()

            # Perform element-wise multiplication of the retrieval_one_hot tensor and
            # the similarities tensor. This will give us the weighted similarity scores
            # (similarities that do not belong to the retrieved neighbors will be zeroed).
            # Then sum the neighbors similarity scores across the 1st dimension (cols) will
            # give us the weighted scores for each class.
            probs = torch.sum(  # pylint: disable=no-member
                torch.mul(  # pylint: disable=no-member
                    # Reshape the retrieval_one_hot tensor to be of shape (batch_size, k, num_classes)
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    # Reshape the similarities tensor to be of shape (batch_size, k, 1)
                    similarities.view(batch_size, -1, 1),
                ),
                1,
            )

            # Sort predictions across classes in descending order
            predictions = probs.argmax(dim=1, keepdim=True)

            # Sum the correct predictions across the 1st dimension (cols). .narrow is used
            # to select the first x columns of the tensor.
            acc += (predictions == targets).sum().item()
            # sum over classes
            total += targets.size(0)

            # Update the precision, recall and F1 score metrics for each batch
            prec_metric.update(predictions, targets)
            rec_metric.update(predictions, targets)
            f1_metric.update(predictions, targets)

        # Compute multi-class precision, recall and F1 score after all batches have been processed
        prec = prec_metric.compute()
        rec = rec_metric.compute()
        f1 = f1_metric.compute()

        acc = acc / total

        self.reset()
        prec_metric.reset()
        rec_metric.reset()
        f1_metric.reset()

        return acc, prec, rec, f1
