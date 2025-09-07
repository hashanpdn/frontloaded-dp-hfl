"""Dataset interface for hierarchical FL.

This module exposes:
  - get_dataloaders(args): returns per-client train/test loaders and
    concatenated validation loaders.
  - show_distribution: re-exported helper to inspect class distributions.

Currently supports: MNIST and CIFAR-10.
"""

from datasets.cifar_mnist import get_dataset, show_distribution


def get_dataloaders(args):
    """Build dataloaders for the requested dataset.

    Returns:
        (train_loaders, test_loaders, v_train_loader, v_test_loader)
    """
    if args.dataset in ["mnist", "cifar10"]:
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataset(
            dataset_root="data",
            dataset=args.dataset,
            args=args,
        )
    else:
        raise ValueError("This dataset is not implemented yet.")
    return train_loaders, test_loaders, v_train_loader, v_test_loader
