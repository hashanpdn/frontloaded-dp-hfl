"""
Averaging utilities for hierarchical FL.

- average_weights: FedAvg over full state_dicts (edge/cloud).
- average_weights_edge: Apply averaged client deltas to an edge model (CP-NP).
"""

import torch


def _check_inputs(w, s_num):
    """Validate list lengths, positive total weight, and matching keys/shapes."""
    if not isinstance(w, (list, tuple)) or len(w) == 0:
        raise ValueError("'w' must be a non-empty list of state_dicts.")
    if not isinstance(s_num, (list, tuple)) or len(s_num) != len(w):
        raise ValueError("'s_num' must match the length of 'w'.")
    if sum(s_num) <= 0:
        raise ValueError("Total sample count must be > 0.")

    ref = w[0]
    ref_keys = set(ref.keys())
    for i, sd in enumerate(w[1:], start=1):
        keys = set(sd.keys())
        if keys != ref_keys:
            missing = ref_keys - keys
            extra = keys - ref_keys
            raise ValueError(
                f"Key mismatch at index {i}. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )
        for k in ref_keys:
            if sd[k].shape != ref[k].shape:
                raise ValueError(
                    f"Shape mismatch for '{k}' at index {i}: {sd[k].shape} vs {ref[k].shape}"
                )


@torch.no_grad()
def average_weights(w, s_num):
    """
    Weighted FedAvg over FULL weights.
    Floating tensors are averaged; non-floating are copied from the first model.
    """
    _check_inputs(w, s_num)
    total = float(sum(s_num))
    ref = w[0]
    out = {}

    for k, ref_t in ref.items():
        if ref_t.is_floating_point():
            acc = torch.zeros_like(ref_t)
            for idx, sd in enumerate(w):
                t = sd[k]
                if t.dtype != ref_t.dtype or t.device != ref_t.device:
                    t = t.to(device=ref_t.device, dtype=ref_t.dtype)
                acc.add_(t, alpha=s_num[idx] / total)
            out[k] = acc
        else:
            out[k] = ref_t.clone()
    return out


@torch.no_grad()
def average_weights_edge(e_model, w, s_num, eta: float = 1.0):
    """
    Edge-side aggregation for CP-NP:
    new_state = e_model + eta * sum_i (s_i / sum_j s_j) * delta_i
    """
    if not isinstance(e_model, dict):
        raise ValueError("'e_model' must be a state_dict (dict).")
    _check_inputs(w, s_num)

    # Ensure delta keys match e_model
    model_keys = set(e_model.keys())
    delta_keys = set(w[0].keys())
    if delta_keys != model_keys:
        missing = model_keys - delta_keys
        extra = delta_keys - model_keys
        raise ValueError(
            "Key mismatch between e_model and deltas. "
            f"Missing in deltas: {sorted(missing)}; Extra in deltas: {sorted(extra)}"
        )

    total = float(sum(s_num))

    # Average deltas per key (dtype/device aligned to e_model)
    delta_avg = {}
    for k, base in e_model.items():
        if base.is_floating_point():
            acc = torch.zeros_like(base)
            for idx, sd in enumerate(w):
                dt = sd[k]
                if dt.dtype != base.dtype or dt.device != base.device:
                    dt = dt.to(device=base.device, dtype=base.dtype)
                acc.add_(dt, alpha=s_num[idx] / total)
            delta_avg[k] = acc
        else:
            delta_avg[k] = None

    # Apply averaged delta
    new_state = {}
    for k, base in e_model.items():
        if base.is_floating_point() and (delta_avg[k] is not None):
            new_state[k] = base + eta * delta_avg[k]
        else:
            new_state[k] = base.clone()
    return new_state
