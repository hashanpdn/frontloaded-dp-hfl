"""Utility helpers for the project.

Currently includes:
  - gaussian_noise: draw i.i.d. N(0, σ²) noise with explicit shape/device/dtype.
"""

import torch


@torch.no_grad()
def gaussian_noise(
    shape,
    sigma: float,
    device=None,
    dtype=torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw i.i.d. Gaussian noise N(0, σ²) with the given shape.

    Args:
        shape: Tensor shape (tuple or torch.Size).
        sigma: Standard deviation of the Gaussian.
        device: Destination device (defaults to CPU if None).
        dtype: Tensor dtype (defaults to torch.float32).
        generator: Optional RNG for deterministic sampling.

    Returns:
        torch.Tensor of shape `shape` on `device` with dtype `dtype`.
    """
    if device is None:
        device = torch.device("cpu")
    return torch.empty(*shape, device=device, dtype=dtype).normal_(0.0, sigma, generator=generator)
