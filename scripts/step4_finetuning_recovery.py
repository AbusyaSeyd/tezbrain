import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

# --- Existing codebase imports (required by task) ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import GNNPruner, Br35HDataset, BrainTumorGNN, GraphClassifier


# -----------------------------
# Step 4: Fine-tuning Recovery
# -----------------------------
BASELINE_WEIGHTS = "br35h_baseline_best.pth"
DATA_ROOT = "data"  # expects data/br35h/{yes,no}

# Critical system fixes
N_SEGMENTS = 50

# ===== PERFORMANCE TUNING =====
# Increase NUM_WORKERS to use more CPU for data loading (parallel graph construction)
# Safe values for Windows: 2-4 (test incrementally; 0 is safest but slowest)
NUM_WORKERS = 4  # Try 2, 4, or up to multiprocessing.cpu_count() // 2

BATCH_SIZE = 32

# Fine-tuning hyperparameters
FT_EPOCHS = 10
FT_LR = 1e-4

# Sparsity levels to recover
SPARSITIES = [0.3, 0.4, 0.5]

# Output
PLOT_PATH = "finetuning_recovery.png"


class CachedDataset(torch.utils.data.Dataset):
    """In-process cache for graphs (critical when num_workers=0)."""

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


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_baseline_model(device: torch.device, input_dim: int) -> BrainTumorGNN:
    if not os.path.exists(BASELINE_WEIGHTS):
        raise FileNotFoundError(
            f"Baseline weights not found: {BASELINE_WEIGHTS}. "
            f"Expected in: {os.path.abspath(os.getcwd())}"
        )

    state_dict = torch.load(BASELINE_WEIGHTS, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Unexpected checkpoint format: expected raw state_dict dict.")

    model = BrainTumorGNN(
        input_dim=input_dim,
        hidden_dim=128,  # Step 2 best
        num_classes=2,
        dropout=0.129,  # Step 2 best
    )
    model.load_state_dict(state_dict)
    return model.to(device)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=(device.type == "cuda"))
        logits = model(batch.x, batch.edge_index, batch.batch)
        y = batch.y.view(-1)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())

    return correct / total if total > 0 else 0.0


def fine_tune(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> float:
    """Fine-tune for a small number of epochs and return final test accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    classifier = GraphClassifier(model, device, scaler=scaler if use_amp else None)

    for ep in range(1, epochs + 1):
        train_loss, train_acc = classifier.train_epoch(train_loader, optimizer, criterion)
        test_loss, test_acc, _, _ = classifier.evaluate(test_loader, criterion)
        print(
            f"    FT epoch {ep:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    return float(test_acc)


def make_bar_plot(
    sparsities: List[float],
    acc_before: List[float],
    acc_after: List[float],
    out_path: str,
) -> None:
    sns.set_style("whitegrid")
    x = np.arange(len(sparsities))
    width = 0.38

    plt.figure(figsize=(10.5, 6.5))
    plt.bar(x - width / 2, acc_before, width=width, label="Pruned (before FT)")
    plt.bar(x + width / 2, acc_after, width=width, label="Recovered (after FT)")

    plt.xticks(x, [f"{int(s * 100)}%" for s in sparsities])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy")
    plt.title("Fine-tuning Recovery after Pruning (BR35H)")
    plt.legend(loc="best")

    # Annotate bars
    for i, v in enumerate(acc_before):
        plt.text(i - width / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    for i, v in enumerate(acc_after):
        plt.text(i + width / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    torch.multiprocessing.freeze_support()
    set_seed(42)

    # ===== GPU OPTIMIZATION =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        # Enable cuDNN autotuner for optimal convolution algorithms (small overhead, then faster)
        torch.backends.cudnn.benchmark = True
        # Optionally force full GPU utilization
        torch.cuda.set_device(0)

    print("=" * 80)
    print("Step 4: Fine-tuning Recovery")
    print(f"Time: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"NUM_WORKERS: {NUM_WORKERS} (CPU parallelism for data loading)")
    print(f"BATCH_SIZE: {BATCH_SIZE}")

    # Data
    br35h_root = os.path.join(DATA_ROOT, "br35h")
    train_base = Br35HDataset(root=br35h_root, split="train", n_segments=N_SEGMENTS)
    test_base = Br35HDataset(root=br35h_root, split="test", n_segments=N_SEGMENTS)

    # Cache graphs in-process to avoid repeated CPU work
    train_dataset = CachedDataset(train_base)
    test_dataset = CachedDataset(test_base)

    # Use persistent_workers to keep worker processes alive between epochs (faster if num_workers > 0)
    use_persistent = NUM_WORKERS > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,  # Prefetch 2 batches per worker
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    input_dim = train_base[0].x.shape[1]
    print(f"Input dim detected: {input_dim}")

    acc_before: List[float] = []
    acc_after: List[float] = []

    # Baseline (optional reference)
    baseline_model = load_baseline_model(device=device, input_dim=input_dim)
    baseline_acc = evaluate_accuracy(baseline_model, test_loader, device=device)
    print(f"Baseline accuracy (no pruning): {baseline_acc:.4f}")

    for s in SPARSITIES:
        print("-" * 80)
        print(f"Sparsity = {s:.1%}")

        # 1) Load clean baseline
        model = load_baseline_model(device=device, input_dim=input_dim)

        # 2) Prune (unstructured magnitude-style using existing GNNPruner)
        pruner = GNNPruner(model, pruning_method="magnitude")
        _ = pruner.magnitude_prune(amount=float(s), structured=False)
        pruner.remove_pruning_masks()

        # 3) Evaluate before FT
        pre = evaluate_accuracy(model, test_loader, device=device)
        acc_before.append(float(pre))
        print(f"  Accuracy BEFORE fine-tuning: {pre:.4f}")

        # 4) Fine-tune
        print(f"  Fine-tuning for {FT_EPOCHS} epochs (Adam, lr={FT_LR})...")
        post = fine_tune(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=FT_EPOCHS,
            lr=FT_LR,
        )
        acc_after.append(float(post))
        print(f"  Accuracy AFTER  fine-tuning: {post:.4f}")

    # Plot
    make_bar_plot(SPARSITIES, acc_before, acc_after, PLOT_PATH)

    print("=" * 80)
    print("FINAL RESULTS")
    for s, pre, post in zip(SPARSITIES, acc_before, acc_after):
        print(f"Sparsity {s:.1%} | before: {pre:.4f} | after: {post:.4f} | delta: {post - pre:+.4f}")
    print(f"Saved plot: {os.path.abspath(PLOT_PATH)}")


if __name__ == "__main__":
    main()








