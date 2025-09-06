import copy
import torch
from models.initialize_model import initialize_model
from utils import gaussian_noise


class Client:
    """
    Unified client with modes:
      - "baseline" : non-private SGD
      - "CG-NG"    : clip per-batch grads; add noise to averaged grad in the final epoch; one step per epoch
      - "CG-NP"    : clip per-batch grads; one step per epoch; add noise to params at the end
      - "CP-NP"    : train normally; build param delta once; clip L2; add noise; send delta only
    """

    def __init__(self, id, train_loader, test_loader, args, device):
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
        Args:
            num_iter: local epochs
            device: torch.device
            mode: "baseline" | "CG-NG" | "CG-NP" | "CP-NP"
            clip: L2 clipping bound (DP modes)
            sigma: Gaussian noise std (DP modes)
        """
        total_loss = 0.0
        self.noisy_params_state_dict = None  # clear any stale delta at the start

        # CP-NP needs a snapshot before training
        if mode == "CP-NP":
            params_before_training = copy.deepcopy(self.model.shared_layers.state_dict())

        # train mode
        self.model.shared_layers.train(True)
        if getattr(self.model, "specific_layers", None):
            self.model.specific_layers.train(True)

        for epoch in range(num_iter):
            # CG-* accumulate clipped grads per epoch; single optimizer step at epoch end
            if mode in ("CG-NG", "CG-NP"):
                self.model.optimizer.zero_grad()
                clipped_grads = {
                    name: torch.zeros_like(param, device=param.device)
                    for name, param in self.model.shared_layers.named_parameters()
                }

            for inputs, labels in self.train_loader:
                if mode in ("baseline", "CP-NP"):
                    self.model.optimizer.zero_grad(set_to_none=True)

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # forward
                if getattr(self.model, "specific_layers", None):
                    outputs = self.model.specific_layers(self.model.shared_layers(inputs))
                else:
                    outputs = self.model.shared_layers(inputs)

                # scalar loss (robust if criterion returns vector)
                loss = self.model.criterion(outputs, labels)
                if hasattr(loss, "dim") and loss.dim() > 0:
                    loss = loss.mean()

                total_loss += float(loss.item())
                loss.backward()

                if mode in ("baseline", "CP-NP"):
                    self.model.optimizer.step()

                if mode in ("CG-NG", "CG-NP"):
                    # clip grads and accumulate
                    torch.nn.utils.clip_grad_norm_(self.model.shared_layers.parameters(), max_norm=clip)
                    for name, param in self.model.shared_layers.named_parameters():
                        if param.grad is not None:
                            clipped_grads[name] += param.grad
                    # reset just the shared layers (matches your originals)
                    self.model.shared_layers.zero_grad(set_to_none=True)

            if mode in ("CG-NG", "CG-NP"):
                # average accumulated grads over batches
                denom = max(1, len(self.train_loader))
                for name in clipped_grads:
                    clipped_grads[name] /= denom

                # CG-NG: add noise to the averaged grads ONLY in the final epoch
                if mode == "CG-NG" and (epoch == num_iter - 1) and (sigma is not None):
                    for name in clipped_grads:
                        # cast noise to the grad dtype to avoid dtype mismatch
                        noise = gaussian_noise(clipped_grads[name].shape, sigma, device=device).to(clipped_grads[name].dtype)
                        clipped_grads[name] += noise

                # apply once and step
                for name, param in self.model.shared_layers.named_parameters():
                    param.grad = clipped_grads[name]
                self.model.optimizer.step()

        # CG-NP: noise to parameters after all epochs
        if mode == "CG-NP" and (sigma is not None):
            for _, param in self.model.shared_layers.named_parameters():
                noise = gaussian_noise(param.data.shape, sigma, device=device).to(param.data.dtype)
                param.data += noise

        # CP-NP: build, clip, noise the delta (send later)
        if mode == "CP-NP":
            params_after = self.model.shared_layers.state_dict()
            delta_param = {k: (params_after[k] - params_before_training[k]) for k in params_before_training}

            # global L2 norm of concatenated delta
            total_norm_sq = 0.0
            for t in delta_param.values():
                total_norm_sq += t.pow(2).sum().item()
            total_norm = (total_norm_sq ** 0.5)

            clip_coef = min(1.0, float(clip) / (total_norm + 1e-6))
            noisy_delta = {}
            for name, t in delta_param.items():
                d = t.mul(clip_coef)
                if sigma is not None:
                    noise = gaussian_noise(d.shape, sigma, device=device).to(d.dtype)
                    d = d + noise
                noisy_delta[name] = d.to(device)

            self.noisy_params_state_dict = noisy_delta  # will be sent instead of full weights

        average_loss = total_loss / (num_iter * max(1, len(self.train_loader)))
        return average_loss

    @torch.no_grad()
    def test_model(self, device):
        # eval mode for stable metrics
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
        # CP-NP: send ONLY the noisy delta (privacy-safe)
        if self.noisy_params_state_dict:
            edgeserver.receive_from_client(
                client_id=self.id,
                cparams_state_dict=copy.deepcopy(self.noisy_params_state_dict)
            )
            # prevent accidental resend next round
            self.noisy_params_state_dict = None
        else:
            # Baseline / CG-NG / CG-NP: send full shared weights
            edgeserver.receive_from_client(
                client_id=self.id,
                cshared_state_dict=copy.deepcopy(self.model.shared_layers.state_dict())
            )
        return None

    def receive_from_edgeserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict
        return None

    def sync_with_edgeserver(self):
        if not self.receiver_buffer:
            raise RuntimeError("receiver_buffer is empty; call receive_from_edgeserver() first.")
        self.model.update_model(self.receiver_buffer)
        self.receiver_buffer = {}
        return None