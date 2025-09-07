# Privacy-Preserving Hierarchical Federated Learning with Front-Loaded Differential Privacy Mechanism

This repository implements **front-loaded differential privacy (DP)** for **hierarchical federated learning (HFL)** in a **client → edge → cloud** topology. It extends a HierFAVG-style framework with three DP mechanisms and a non-private baseline, using unified Client/Edge/Cloud components in PyTorch.

> **Paper status:** *Privacy-Preserving Hierarchical Federated Learning with Front-Loaded Differential Privacy Mechanism* — under review.
> This repository provides the implementation and scripts required for independent reproduction.

---

## Mechanisms and Aggregation

### Client modes (select via `--mode`)

* **baseline** — standard local SGD (no DP).
* **CG-NG — Differential Privacy through Clipping and Noise Addition to Local Gradients**
  Per-batch gradient clipping during local training; within each local epoch, gradients are accumulated/averaged, and **only in the final local epoch** Gaussian noise is added to the **averaged local gradient**, followed by a single optimizer step for that epoch.
* **CG-NP — Differential Privacy through Clipping of Gradients and Noise Addition to Model Parameters**
  Per-batch gradient clipping during local training; one optimizer step at the end of each local epoch; **after** local epochs complete, Gaussian noise is added **directly to model parameters** prior to upload.
* **CP-NP — Differential Privacy through Clipping and Noise Addition to Model Parameter Differences**
  Local training proceeds normally; compute the parameter difference Δw = w\_after − w\_before once per client; compute a **global L2 norm**, **clip** Δw by `min(1, clip / (‖Δw‖₂ + 1e−6))`, add Gaussian **noise** to the clipped Δw, and **send only the noisy Δw** to the edge (no full weights).

### Aggregation

* **Edge server (baseline / CG-NG / CG-NP):** weighted FedAvg over received full model weights.
* **Edge server (CP-NP):** apply the **weighted average of client deltas** to the current edge model. An edge learning-rate **η** (`--eta`, default `1.0`) scales the averaged delta before application (η = 1.0 reproduces FedAvg on deltas).
* **Cloud server:** weighted FedAvg over edge models (full weights).
* Weights are proportional to each participant’s registered training sample count in that round.

---

## Repository Layout

```
frontloaded-dp-hfl/
├─ frontloaded_dp_hfl/
│  ├─ hierfavg.py          # main training loop (client→edge→cloud)
│  ├─ client.py            # unified client (baseline, CG-NG, CG-NP, CP-NP)
│  ├─ edge.py              # receives full weights OR deltas; aggregates accordingly
│  ├─ cloud.py             # FedAvg across edges (full weights)
│  ├─ average.py           # average_weights(), average_weights_edge(..., eta)
│  ├─ options.py           # CLI flags (incl. --mode, --clip, --sigma, --eta)
│  ├─ utils.py             # gaussian_noise(...)
│  ├─ models/              # initialize_model.py, MNIST/CIFAR CNNs
│  └─ datasets/            # dataloaders, partitioning helpers
├─ requirements-cpu.txt
└─ requirements-gpu-cu118.txt
```

---

## Installation

Select **one** environment: CPU **or** GPU (CUDA 11.8). Do **not** install both requirements files in the same environment.

### CPU (no NVIDIA GPU required)

```bash
python -m venv .venv
# Windows (PowerShell):
#   .\.venv\Scripts\Activate.ps1
# macOS/Linux:
#   source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-cpu.txt
```

### GPU (CUDA 11.8)

Requires an NVIDIA driver compatible with CUDA 11.8 and a CUDA-capable GPU.

```bash
python -m venv .venv
# Windows (PowerShell):
#   .\.venv\Scripts\Activate.ps1
# macOS/Linux:
#   source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-gpu-cu118.txt
```

### Verify installation

```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
```

For other CUDA versions, install the matching `torch`/`torchvision` wheels from the official PyTorch site first, then install the remaining packages from `requirements-cpu.txt` (excluding torch/torchvision).

---

## Datasets

* **MNIST** and **CIFAR-10** are automatically downloaded via `torchvision` to `--dataset_root` (default: `data/`).
* IID and non-IID partitions at both client and edge levels are supported via `--iid`, `--edgeiid`, and `--classes_per_client`.

---

## Quick Start

Run from the repository root using the package entry point:

### Baseline (MNIST, LeNet)

```bash
python -m frontloaded_dp_hfl.hierfavg \
  --dataset mnist --model lenet \
  --num_clients 100 --num_edges 5 \
  --cfrac 0.2 --efrac 1.0 \
  --num_communication 200 --num_edge_aggregation 2 --num_local_update 6 \
  --batch_size 20 --lr 0.1 --seed 1 \
  --mode baseline
```

### CG-NG (noise to averaged local gradients in the final local epoch)

```bash
python -m frontloaded_dp_hfl.hierfavg \
  --dataset mnist --model lenet \
  --num_clients 100 --num_edges 5 \
  --cfrac 0.2 --efrac 1.0 \
  --num_communication 200 --num_edge_aggregation 2 --num_local_update 6 \
  --batch_size 20 --lr 0.1 --seed 1 \
  --mode CG-NG --clip 2.0 --sigma 0.01
```

### CG-NP (noise to parameters after local epochs)

```bash
python -m frontloaded_dp_hfl.hierfavg \
  --dataset mnist --model lenet \
  --num_clients 100 --num_edges 5 \
  --cfrac 0.2 --efrac 1.0 \
  --num_communication 200 --num_edge_aggregation 2 --num_local_update 6 \
  --batch_size 20 --lr 0.1 --seed 1 \
  --mode CG-NP --clip 2.0 --sigma 0.01
```

### CP-NP (send clipped, noisy parameter differences)

```bash
python -m frontloaded_dp_hfl.hierfavg \
  --dataset mnist --model lenet \
  --num_clients 100 --num_edges 5 \
  --cfrac 0.2 --efrac 1.0 \
  --num_communication 200 --num_edge_aggregation 2 --num_local_update 6 \
  --batch_size 20 --lr 0.1 --seed 1 \
  --mode CP-NP --clip 2.0 --sigma 0.01 --eta 1.0
```

For CIFAR-10, set `--dataset cifar10`, choose `--model cnn_complex_2` or `cnn_complex_3`, and ensure `--input_channels 3`.

---

## Command-Line Options (selected)

Run `python -m frontloaded_dp_hfl.hierfavg --help` for the full list. Key flags include:

* **Mechanism / DP**

  * `--mode {baseline,CG-NG,CG-NP,CP-NP}`
  * `--clip` — L2 clipping bound (DP modes)
  * `--sigma` — Gaussian noise standard deviation (DP modes)
  * `--eta` — edge learning rate for applying averaged deltas in CP-NP (η = 1.0 reproduces FedAvg on deltas)
* **Topology / Participation**

  * `--num_clients`, `--num_edges`, `--cfrac`, `--efrac`
  * `--num_communication` (cloud rounds), `--num_edge_aggregation` (τ₂), `--num_local_update` (τ₁)
* **Optimization**

  * `--batch_size`, `--lr`, `--momentum`, `--lr_decay`, `--lr_decay_epoch`, `--weight_decay`
* **Data / Splits**

  * `--dataset {mnist,cifar10}`, `--dataset_root`
  * `--iid`:

    * `1`  → iid by label and size
    * `0`  → non-iid by label (equal client sizes)
    * `-1` → iid by label with **unequal client sizes** (imbalanced)
    * `-2` → **one-class per client**; with `--edgeiid {0,1}` controlling per-edge class coverage
  * `--edgeiid` (when `--iid -2`): `1` edge-iid, `0` edge-non-iid
  * `--classes_per_client` (used by certain non-iid splits)
* **System**

  * `--gpu` (device index), `--seed`, `--verbose`

---

## Reproducibility

Seeds and cuDNN flags are configured **before** building models/dataloaders:

```python
import random, numpy as np, torch
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Enforcing determinism can reduce throughput and may restrict kernel selection; remove these flags if exact reproducibility is not required.

---

## Differential Privacy Notes

* **CG-NG / CG-NP** implement **per-batch** gradient clipping as used in the experiments. For strict example-level DP, per-example gradients (or micro-batching) would be required.
* **CP-NP** realizes **client-level DP** when `σ` is calibrated via an appropriate privacy accountant reflecting your sampling and composition; post-processing by the edge/cloud (including scaling by **η**) does **not** degrade DP guarantees.
* The utility `gaussian_noise(shape, sigma, device, dtype)` draws IID Gaussian entries with the correct device/dtype and is applied to the averaged gradient (CG-NG), to parameters (CG-NP), or to the clipped Δw (CP-NP) as specified.

---

## Logging

Training logs are written under `runs/`. To visualize:

```bash
tensorboard --logdir runs
```

---

## Acknowledgments

This codebase customizes and extends a public HierFAVG implementation of hierarchical FL to integrate front-loaded DP mechanisms (CG-NG, CG-NP, CP-NP) with a unified Client/Edge/Cloud design.

* Upstream codebase: **Client-Edge-Cloud Hierarchical Federated Learning with PyTorch (HierFL)** — [https://github.com/LuminLiu/HierFL](https://github.com/LuminLiu/HierFL)

---

## Citation

If you use this repository, please cite the paper once publicly available. Until then:

```
@misc{frontloaded-dp-hfl,
  title        = {Privacy-Preserving Hierarchical Federated Learning with Front-Loaded Differential Privacy Mechanism},
  author       = {Hashan Ratnayake and Lin Chen and Xiaofeng Ding},
  year         = {2025},
  note         = {Code: https://github.com/hashanpdn/frontloaded-dp-hfl}
}
```

---

## License

**MIT License.** See `LICENSE` for details.
