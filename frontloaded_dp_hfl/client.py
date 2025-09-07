"""
Client-side logic for hierarchical federated learning with front-loaded DP.

Supports four modes:
  - "baseline": standard local SGD (no DP)
  - "CG-NG"   : per-batch gradient clipping, accumulate per epoch, add noise to
                the epoch-averaged gradient only in the final local epoch
  - "CG-NP"   : per-batch gradient clipping, accumulate per epoch, then add noise
                to model parameters after local training (no noise to grads)
  - "CP-NP"   : standard local training, compute parameter delta once at the end,
                clip global L2 norm of the delta, add noise, and send only delta
"""

import copy
import torch
from models.initialize_model import initialize_model
from utils import gaussian_noise


class Client:
    """
    Unified client that executes local training and communicates either full
    model weights (baseline, CG-NG, CG-NP) or noisy parameter deltas (CP-NP).
    """

    def __init__(self, id, train_loader, test_loader, args, device):
        """Initialize dataloaders, model, and per-round buffers."""
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        self.receiver_buffer = {}
        self.batch_size = args.batch_size
        self.noisy_params_state_dict = None  # set only in CP-NP
        self.epoch = 0
        self.clock = []
        self.device = device

    def local_update(self, num_iter, device, mode="baseline", clip=None, sigma=None):
        """
        Run local training for `num_iter` epochs under the selected DP mode.

        Args:
            num_iter (int): number of local epochs.
            device (torch.device): compute device.
            mode (str): "baseline" | "CG-NG" | "CG-NP" | "CP-NP".
            clip (float): L2 clipping bound for DP modes.
            sigma (float): Gaussian noise std for DP modes.

        Returns:
            float: average training loss over all local mini-batches.
        """
        total_loss = 0.0
        self.noisy_params_state_dict = None  # clear any stale delta at the start

        # CP-NP requires a pre-training snapshot to form a delta.
        if mode == "CP-NP":
            params_before_training = copy.deepcopy(self.model.shared_layers.state_dict())

        # Enter training mode for shared (and optionally specific) layers.
        self.model.shared_layers.train(True)
        if getattr(self.model, "specific_layers", None):
            self.model.specific_layers.train(True)

        for epoch in range(num_iter):
            # CG-* modes: accumulate clipped grads across batches, one step at epoch end.
            if mode in ("CG-NG", "CG-NP"):
                self.model.optimizer.zero_grad()
                clipped_grads = {
                    name: torch.zeros_like(param, device=param.device)
                    for name, param in self.model.shared_layers.named_parameters()
                }

            for inputs, labels in self.train_loader:
                # Baseline / CP-NP: standard per-batch step.
                if mode in ("baseline", "CP-NP"):
                    self.model.optimizer.zero_grad(set_to_none=True)

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Forward pass through shared (+ specific) layers.
                if getattr(self.model, "specific_layers", None):
                    outputs = self.model.specific_layers(self.model.shared_layers(inputs))
                else:
                    outputs = self.model.shared_layers(inputs)

                # Loss can be vector-valued for some criteria; reduce to scalar.
                loss = self.model.criterion(outputs, labels)
                if hasattr(loss, "dim") and loss.dim() > 0:
                    loss = loss.mean()

                total_loss += float(loss.item())
                loss.backward()

                if mode in ("baseline", "CP-NP"):
                    self.model.optimizer.step()

                if mode in ("CG-NG", "CG-NP"):
                    # Clip L2 norm and accumulate gradients.
                    torch.nn.utils.clip_grad_norm_(self.model.shared_layers.parameters(), max_norm=clip)
                    for name, param in self.model.shared_layers.named_parameters():
                        if param.grad is not None:
                            clipped_grads[name] += param.grad
                    # Clear only shared-layer grads before next mini-batch.
                    self.model.shared_layers.zero_grad(set_to_none=True)

            if mode in ("CG-NG", "CG-NP"):
                # Average accumulated gradients over the number of mini-batches.
                denom = max(1, len(self.train_loader))
                for name in clipped_grads:
                    clipped_grads[name] /= denom

                # CG-NG: add noise to the averaged gradient only in the final local epoch.
                if mode == "CG-NG" and (epoch == num_iter - 1) and (sigma is not None):
                    for name in clipped_grads:
                        # Match noise dtype to grad to avoid dtype/device mismatches.
                        noise = gaussian_noise(clipped_grads[name].shape, sigma, device=device).to(clipped_grads[name].dtype)
                        clipped_grads[name] += noise

                # Apply averaged (and possibly noised) gradient once and step.
                for name, param in self.model.shared_layers.named_parameters():
                    param.grad = clipped_grads[name]
                self.model.optimizer.step()

        # CG-NP: add noise to parameters after all local epochs (no gradient noise).
        if mode == "CG-NP" and (sigma is not None):
            for _, param in self.model.shared_layers.named_parameters():
                noise = gaussian_noise(param.data.shape, sigma, device=device).to(param.data.dtype)
                param.data += noise

        # CP-NP: construct, clip (global L2), and noise the parameter delta.
        if mode == "CP-NP":
            params_after = self.model.shared_layers.state_dict()
            delta_param = {k: (params_after[k] - params_before_training[k]) for k in params_before_training}

            # Global L2 norm of concatenated delta (computed via sum of squared elements).
            total_norm_sq = 0.0
            for t in delta_param.values():
                total_norm_sq += t.pow(2).sum().item()
            total_norm = (total_norm_sq ** 0.5)

            # Clip and add noise to the delta; keep tensors on the target device.
            clip_coef = min(1.0, float(clip) / (total_norm + 1e-6))
            noisy_delta = {}
            for name, t in delta_param.items():
                d = t.mul(clip_coef)
                if sigma is not None:
                    noise = gaussian_noise(d.shape, sigma, device=device).to(d.dtype)
                    d = d + noise
                noisy_delta[name] = d.to(device)

            self.noisy_params_state_dict = noisy_delta  # sent instead of full weights

        average_loss = total_loss / (num_iter * max(1, len(self.train_loader)))
        return average_loss

    @torch.no_grad()
    def test_model(self, device):
        """Evaluate the current client model on the local test loader."""
        self.model.shared_layers.train(False)
        if getattr(self.model, "specific_layers", None):
            self.model.specific_layers.train(False)

        correct, total = 0, 0
        for inputs, labels in self.test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = self.model.test_model(input_batch=inputs)
            _, predict = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
        return correct, total

    def send_to_edgeserver(self, edgeserver):
        """
        Send updates to the edge:
          - CP-NP: send only the noisy parameter delta (privacy friendly).
          - Otherwise: send the full shared model weights.
        """
        if self.noisy_params_state_dict:
            edgeserver.receive_from_client(
                client_id=self.id,
                cparams_state_dict=copy.deepcopy(self.noisy_params_state_dict)
            )
            # Avoid resending stale deltas in the next round.
            self.noisy_params_state_dict = None
        else:
            edgeserver.receive_from_client(
                client_id=self.id,
                cshared_state_dict=copy.deepcopy(self.model.shared_layers.state_dict())
            )
        return None

    def receive_from_edgeserver(self, shared_state_dict):
        """Store the latest edge model snapshot for synchronization."""
        self.receiver_buffer = shared_state_dict
        return None

    def sync_with_edgeserver(self):
        """Overwrite local shared layers with the edge-provided state dict."""
        if not self.receiver_buffer:
            raise RuntimeError("receiver_buffer is empty; call receive_from_edgeserver() first.")
        self.model.update_model(self.receiver_buffer)
        self.receiver_buffer = {}
        return None
