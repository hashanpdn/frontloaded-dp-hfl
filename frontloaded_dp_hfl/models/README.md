# Models

This directory contains the neural network architectures and the factory that initializes them for use in hierarchical federated learning (HFL).

## Contents
- `mnist_cnn.py`
  - **mnist_lenet**: LeNet-style CNN for MNIST (input `1×28×28`, output `10` classes).
- `cifar_cnn.py`
  - **cifar_cnn_2conv**: Lightweight 2-conv CNN for CIFAR-10 (input `3×32×32`, output `10`).
  - **cifar_cnn_3conv**: Deeper 3-block CNN for CIFAR-10 (input `3×32×32`, output `10`).
  - **cifar_cnn_3conv_shared / cifar_cnn_3conv_specific**: Split variants used by the optional MTL setting.
- `initialize_model.py`
  - Builds the training wrapper (`MTL_Model`) with:
    - `shared_layers` (and optional `specific_layers` for MTL),
    - SGD optimizer (lr/momentum/weight_decay from CLI),
    - Cross-entropy loss,
    - Utility methods: `optimize_model`, `test_model`, `update_model`.

## Expected Inputs & Transforms
- **MNIST**: grayscale `1×28×28`, normalized with mean `0.1307`, std `0.3081`.
- **CIFAR-10**: RGB `3×32×32`, normalized with standard CIFAR-10 stats.  
  (See `datasets/README.md` for exact transforms.)

## Selecting a Model (CLI)
- `--dataset mnist --model lenet` → `mnist_lenet`
- `--dataset cifar10 --model cnn_complex_2` → `cifar_cnn_2conv`
- `--dataset cifar10 --model cnn_complex_3` → `cifar_cnn_3conv`

## Global vs. MTL
- `--global_model 1` (default) uses a single `shared_layers` model for all clients.
- `--mtl_model 1` builds shared/specific splits (`*_shared`, `*_specific`) for multi-task learning.
- Exactly one of `--global_model` and `--mtl_model` should be enabled.

## Extending
To add a new architecture:
1. Implement the `nn.Module` in a new or existing file under `models/`.
2. Import and wire it in `initialize_model.py` under the appropriate dataset branch.
3. Expose it via `--model <name>` in `options.py`.

> Note: Model choices must be compatible with dataset input shape and number of classes used by the experiments.
