"""CIFAR-10 CNN architectures.

Includes:
- cifar_cnn_2conv: compact 2-conv baseline.
- cifar_cnn_3conv: deeper 3-block CNN with BN/Dropout.
- cifar_cnn_3conv_shared / _specific: split variant (shared extractor + task head).
All models return unnormalized class logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class cifar_cnn_2conv(nn.Module):
    """Compact 2-conv CNN for 32×32 RGB inputs."""

    def __init__(self, output_dim, inter_dim):
        super(cifar_cnn_2conv, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride=1)
        self.conv2 = nn.Conv2d(64, 64, 5, stride=1)
        self.fc1 = nn.Linear(5 * 5 * 64, inter_dim)
        self.fc2 = nn.Linear(inter_dim, output_dim)

    def forward(self, x):
        """Return logits of shape (N, output_dim)."""
        x = torch.reshape(x, (-1, 3, 32, 32))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class cifar_cnn_3conv(nn.Module):
    """Three conv blocks + FC head (suitable for 32×32 RGB inputs)."""

    def __init__(self, input_channels, output_channels):
        super(cifar_cnn_3conv, self).__init__()

        # Convolutional feature extractor
        self.conv_layer = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully-connected classifier
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10),  # logits for 10 classes
        )

    def forward(self, x):
        """Return logits of shape (N, 10)."""
        x = self.conv_layer(x)          # (N, 256, 4, 4)
        x = x.view(x.size(0), -1)       # (N, 4096)
        x = self.fc_layer(x)            # (N, 10)
        return x


class cifar_cnn_3conv_shared(nn.Module):
    """Shared feature extractor (conv blocks) for split models."""

    def __init__(self, input_channels):
        super(cifar_cnn_3conv_shared, self).__init__()
        self.conv_layer = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        """Return flattened feature tensor of shape (N, 4096)."""
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return x

    def feature_out_dim(self):
        """Feature dimension after flatten: 4096 (for 32×32 CIFAR-10)."""
        return 4096


class cifar_cnn_3conv_specific(nn.Module):
    """Task-specific head to pair with cifar_cnn_3conv_shared."""

    def __init__(self, input_channels, output_channels):
        super(cifar_cnn_3conv_specific, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """Return logits of shape (N, 10)."""
        return self.fc_layer(x)
