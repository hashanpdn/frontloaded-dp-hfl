import torch

@torch.no_grad()
def gaussian_noise(shape, sigma: float, device=None, dtype=torch.float32,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Draw i.i.d. N(0, sigma^2) with exact shape, directly on device/dtype.
    Deterministic if 'generator' is provided.
    """
    if device is None:
        device = torch.device("cpu")
    return torch.empty(*shape, device=device, dtype=dtype).normal_(0.0, sigma, generator=generator)
