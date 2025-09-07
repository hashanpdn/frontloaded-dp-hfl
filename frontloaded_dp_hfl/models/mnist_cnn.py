"""MNIST CNN (LeNet-style).

A compact convolutional network for 28×28 grayscale inputs, used as the shared
model in experiments. The network produces unnormalized class logits.
"""

import torch.nn as nn
import torch.nn.functional as F


class mnist_lenet(nn.Module):
    """Two conv layers + two fully connected layers (LeNet-style)."""

    def __init__(self, input_channels, output_channels):
        super(mnist_lenet, self).__init__()
        # Conv stack: 5×5 convs with ReLU and 2×2 max-pooling; Dropout2d after conv2.
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        # FC stack: 320 → 50 → output_channels; dropout between fc1 and fc2.
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, output_channels)

    def forward(self, x):
        """Return unnormalized logits of shape (N, output_channels)."""
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # Flatten to (N, 320) given MNIST input and this conv/pool stack.
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc2(x)
        return x
