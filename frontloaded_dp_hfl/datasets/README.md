# Datasets

This repository uses the torchvision **MNIST** and **CIFAR-10** datasets. By default, datasets are automatically downloaded to `data/` on first run.

## Sources & Licenses
- **MNIST** (LeCun et al., 1998): distributed via torchvision.
- **CIFAR-10** (Krizhevsky et al., 2009): distributed via torchvision.
  
Please refer to the original dataset licenses when reusing the data.

## Directory Layout
- MNIST: `data/mnist/`
- CIFAR-10: `data/cifar10/`

You can change the root via `--dataset_root`.

## Transforms (summary)
- MNIST: `ToTensor` + normalization `(mean=0.1307, std=0.3081)`.
- CIFAR-10: train uses random crop + horizontal flip; both train/test use standard CIFAR-10 normalization.

## Partitioning Schemes (`--iid`)
- `1`  — **IID, equal size**: each client receives an equal number of IID samples.
- `0`  — **Non-IID, equal size**: label-skewed shards; client sizes equal.
- `-1` — **IID, imbalanced size**: IID labels; client sizes vary (imbalanced).
- `-2` — **Non-IID, one-class**: each client receives samples from a single label.

Related flags: `--num_clients`, `--classes_per_client`, `--batch_size`.

## Reproducibility
Random seeds are set via `--seed`. cuDNN determinism is configured in `hierfavg.py`.

## Offline / Air-gapped Use
If auto-download is unavailable, manually place the torchvision MNIST/CIFAR-10 files under `data/mnist/` and `data/cifar10/` (matching torchvision’s expected layout), or point `--dataset_root` to an existing cache.

## Approximate Disk Usage
- MNIST: ~12 MB
- CIFAR-10: ~170 MB
