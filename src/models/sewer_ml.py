"""This module contians the Xie2019 model from the SewerML paper for defect detection."""

from collections import OrderedDict

import torch
from torch import nn


class Xie2019(nn.Module):
    """Xie's architecture taken from SewerML codebase.
    Link to original implementation:
    https://bitbucket.org/aauvap/sewer-ml/src/master/sewer_models.py"""

    def __init__(self, dropout_rate=0.6):
        super(Xie2019, self).__init__()
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, padding=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):  # pylint: disable=missing-function-docstring
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # pylint: disable=no-member
        x = self.classifier(x)
        return x


def load_xie_model(path_to_model: str) -> nn.Module:
    """Load Xie's set of weights and return the model"""

    print("#" * 80)
    print("Loading model weights...")

    # Load model weights
    xie_model_weights = torch.load(path_to_model)

    # Create model
    xie_model = Xie2019()

    # Get weights
    model_weights = xie_model_weights["state_dict"]

    # Strip "model." from the keys
    updated_state_dict = OrderedDict()
    for k, v in model_weights.items():
        name = k.replace("model.", "")
        if "criterion" in name:
            continue

        updated_state_dict[name] = v

    # Load weights
    xie_model.load_state_dict(updated_state_dict)

    print("Model weights loaded.")
    print("#" * 80)

    return xie_model
