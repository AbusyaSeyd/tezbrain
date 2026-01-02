import json
import os
import time
from datetime import datetime
from contextlib import nullcontext
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

# --- REQUIRED IMPORTS ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import GNNPruner, Br35HDataset, BrainTumorGNN


# -----------------------------
# Fixed experiment configuration
# -----------------------------
BASELINE_WEIGHTS = "br35h_baseline_best.pth"
DATA_ROOT = "data"  # expects data/br35h/{yes,no}

# CRITICAL FIXES
N_SEGMENTS = 50
BATCH_SIZE = 32
NUM_WORKERS = 0

PRUNING_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Baseline architecture params from Step 2
HIDDEN_DIM = 128
DROPOUT = 0.129
NUM_CLASSES = 2

# Outputs
RESULTS_JSON = "pruning_comparison_results.json"
PLOT_PATH = "pruning_comparison.png"
TQDM_DISABLE = True  # prevents terminal output issues in some environments

# Research-grade tweaks (recommended for thesis-quality results)
EXCLUDE_BATCHNORM_FROM_PRUNING = True
GLOBAL_UNSTRUCTURED = True  # global sparsity allocation (more stable than layerwise)


class CachedDataset(torch.utils.data.Dataset):
    """In-process cache to avoid rebuilding graphs repeatedly (critical on Windows/CPU bottleneck)."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self._cache: List[object] = [None] * len(base_dataset)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self._cache[idx]
        if item is None:
            item = self.base_dataset[idx]
            self._cache[idx] = item
        return item


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> float:
    model.eval()
    correct = 0
    total = 0

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )

    for batch in tqdm(loader, desc="eval", leave=False, disable=TQDM_DISABLE):
        batch = batch.to(device, non_blocking=(device.type == "cuda"))
        with amp_ctx:
            logits = model(batch.x, batch.edge_index, batch.batch)
        y = batch.y.view(-1)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())

    return correct / total if total > 0 else 0.0


def load_clean_model(device: torch.device, input_dim: int) -> BrainTumorGNN:
    if not os.path.exists(BASELINE_WEIGHTS):
        raise FileNotFoundError(
            f"Baseline weights not found: {BASELINE_WEIGHTS}. "
            f"Expected in: {os.path.abspath(os.getcwd())}"
        )

    state_dict = torch.load(BASELINE_WEIGHTS, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Unexpected checkpoint format: expected a state_dict dict.")

    model = BrainTumorGNN(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _is_batchnorm(m: nn.Module) -> bool:
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))


def _named_prunable_weight_params(
    model: nn.Module, *, exclude_batchnorm: bool
) -> List[tuple[nn.Module, str]]:
    """
    Collect (module, 'weight') pairs usable for torch pruning utilities.
    We only include true Parameters (not buffers) and optionally exclude BatchNorm.
    """
    params: List[tuple[nn.Module, str]] = []
    for _, module in model.named_modules():
        if exclude_batchnorm and _is_batchnorm(module):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        # only if 'weight' is a Parameter (not a buffer)
        if "weight" not in dict(module.named_parameters(recurse=False)):
            continue
        params.append((module, "weight"))
    return params


def _remove_masks_from_modules(modules: List[nn.Module]) -> None:
    for module in modules:
        for name in ("weight", "bias"):
            try:
                if hasattr(module, f"{name}_mask"):
                    prune.remove(module, name)
            except Exception:
                pass


def apply_global_unstructured_pruning(model: nn.Module, amount: float) -> Dict:
    """
    Global unstructured L1 pruning across all eligible weights (excluding BatchNorm by default).
    """
    prunable = _named_prunable_weight_params(
        model, exclude_batchnorm=EXCLUDE_BATCHNORM_FROM_PRUNING
    )
    if len(prunable) == 0:
        return {"method": "global_unstructured", "amount": amount, "sparsity": 0.0, "n_params_groups": 0}

    prune.global_unstructured(
        prunable,
        pruning_method=prune.L1Unstructured,
        amount=float(amount),
    )
    _remove_masks_from_modules([m for (m, _) in prunable])

    # Estimate sparsity over the same prunable set
    total = 0
    zeros = 0
    for module, _ in prunable:
        w = module.weight
        total += w.numel()
        zeros += int((w == 0).sum().item())
    sparsity = zeros / total if total > 0 else 0.0
    return {"method": "global_unstructured", "amount": amount, "sparsity": sparsity, "n_params_groups": len(prunable)}


def apply_structured_linear_pruning(model: nn.Module, amount: float) -> Dict:
    """
    Structured pruning (neuron/channel-level) applied ONLY to Linear layers' weights.
    BatchNorm is excluded. This better matches 'structured' meaning in practice.
    """
    linear_layers: List[nn.Linear] = []
    for _, module in model.named_modules():
        if EXCLUDE_BATCHNORM_FROM_PRUNING and _is_batchnorm(module):
            continue
        if isinstance(module, nn.Linear) and "weight" in dict(module.named_parameters(recurse=False)):
            linear_layers.append(module)

    if len(linear_layers) == 0:
        return {"method": "structured_linear", "amount": amount, "sparsity": 0.0, "n_linear_layers": 0}

    # dim=0 => prune output neurons (rows)
    for lin in linear_layers:
        try:
            prune.ln_structured(lin, name="weight", amount=float(amount), n=2, dim=0)
        except Exception:
            # fallback to unstructured for safety
            prune.l1_unstructured(lin, name="weight", amount=float(amount))

    _remove_masks_from_modules(list(linear_layers))

    total = 0
    zeros = 0
    for lin in linear_layers:
        total += lin.weight.numel()
        zeros += int((lin.weight == 0).sum().item())
    sparsity = zeros / total if total > 0 else 0.0
    return {"method": "structured_linear", "amount": amount, "sparsity": sparsity, "n_linear_layers": len(linear_layers)}


def run_sensitivity(
    *,
    structured: bool,
    input_dim: int,
    test_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict:
    results = {
        "structured": structured,
        "rates": [],
        "accuracies": [],
        "effective_sparsities": [],
        "wall_time_sec": [],
    }

    for rate in tqdm(PRUNING_RATES, desc=("structured" if structured else "unstructured"), disable=TQDM_DISABLE):
        t0 = time.time()

        model = load_clean_model(device=device, input_dim=input_dim)
        if structured:
            # Research variant: structured pruning applied to Linear layers only (exclude BatchNorm)
            stats = apply_structured_linear_pruning(model, amount=float(rate))
        else:
            # Research variant: global unstructured pruning (exclude BatchNorm)
            if GLOBAL_UNSTRUCTURED:
                stats = apply_global_unstructured_pruning(model, amount=float(rate))
            else:
                pruner = GNNPruner(model, pruning_method="magnitude")
                stats = pruner.magnitude_prune(amount=float(rate), structured=False)
                pruner.remove_pruning_masks()

        acc = evaluate_accuracy(model, test_loader, device=device, use_amp=use_amp)
        dt = time.time() - t0

        results["rates"].append(float(rate))
        results["accuracies"].append(float(acc))
        results["effective_sparsities"].append(float(stats.get("sparsity", 0.0)))
        results["wall_time_sec"].append(float(dt))

    return results


def plot_comparison(
    *,
    rates: List[float],
    acc_unstructured: List[float],
    acc_structured: List[float],
    out_path: str,
) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(10.5, 6.5))

    x = np.array(rates)
    y1 = np.array(acc_unstructured)
    y2 = np.array(acc_structured)

    plt.plot(x, y1, color="blue", marker="o", linewidth=2.5, markersize=7, label="Unstructured (Magnitude)")
    plt.plot(x, y2, color="red", marker="o", linewidth=2.5, markersize=7, label="Structured (Neuron/Channel)")

    plt.title("Pruning Sensitivity: Unstructured vs Structured (BR35H)")
    plt.xlabel("Sparsity (Pruning Rate)")
    plt.ylabel("Accuracy")
    plt.xticks(x, [f"{int(v*100)}%" for v in x])
    plt.ylim(0.0, 1.0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    # Windows safety (even with num_workers=0)
    torch.multiprocessing.freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # Data
    br35h_root = os.path.join(DATA_ROOT, "br35h")
    test_base = Br35HDataset(root=br35h_root, split="test", n_segments=N_SEGMENTS)

    # Cache graphs once (critical: we evaluate many times)
    test_dataset = CachedDataset(test_base)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    # Infer input dim
    input_dim = test_base[0].x.shape[1]

    # Baseline accuracy (no pruning)
    baseline_model = load_clean_model(device=device, input_dim=input_dim)
    baseline_acc = evaluate_accuracy(baseline_model, test_loader, device=device, use_amp=use_amp)

    # Sensitivity analysis
    unstructured = run_sensitivity(
        structured=False,
        input_dim=input_dim,
        test_loader=test_loader,
        device=device,
        use_amp=use_amp,
    )

    structured = run_sensitivity(
        structured=True,
        input_dim=input_dim,
        test_loader=test_loader,
        device=device,
        use_amp=use_amp,
    )

    summary = {
        "dataset": "br35h",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "config": {
            "baseline_weights": BASELINE_WEIGHTS,
            "n_segments": N_SEGMENTS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pruning_rates": PRUNING_RATES,
            "model": {
                "input_dim": int(input_dim),
                "hidden_dim": int(HIDDEN_DIM),
                "dropout": float(DROPOUT),
                "num_classes": int(NUM_CLASSES),
            },
        },
        "baseline_accuracy": float(baseline_acc),
        "unstructured": unstructured,
        "structured": structured,
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_comparison(
        rates=PRUNING_RATES,
        acc_unstructured=unstructured["accuracies"],
        acc_structured=structured["accuracies"],
        out_path=PLOT_PATH,
    )

    print("=" * 70)
    print("Step 3 pruning comparison complete")
    print(f"Device: {device}")
    print(f"Baseline acc: {baseline_acc:.4f}")
    print(f"Saved JSON: {os.path.abspath(RESULTS_JSON)}")
    print(f"Saved plot: {os.path.abspath(PLOT_PATH)}")


if __name__ == "__main__":
    main()
