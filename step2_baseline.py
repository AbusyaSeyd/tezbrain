import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- REQUIRED IMPORTS ---
from data_loader import Br35HDataset
from gnn_model import BrainTumorGNN


# -----------------------------
# Step 1: Best hyperparameters
# -----------------------------
BEST_PARAMS = {
    "learning_rate": 0.00215,
    "hidden_channels": 128,
    "dropout": 0.129,
    "weight_decay": 9.67e-05,
    "optimizer": "AdamW",
}


# -----------------------------
# Step 2: Training configuration
# -----------------------------
EPOCHS = 150
BATCH_SIZE = 32
PATIENCE = 20

# CRITICAL CPU FIX: must be 50 (NOT 100)
N_SEGMENTS = 50

# Windows fix
NUM_WORKERS = 0

# Paths / outputs
DATA_ROOT = "."  # expects dataset at ./br35h/{yes,no}
BEST_WEIGHTS_PATH = "br35h_baseline_best.pth"
LOG_PATH = "step2_baseline.log"

ACC_PLOT_PATH = "br35h_baseline_accuracy_curve.png"
LOSS_PLOT_PATH = "br35h_baseline_loss_curve.png"
CM_PLOT_PATH = "br35h_baseline_confusion_matrix.png"


class CachedDataset(torch.utils.data.Dataset):
    """In-process (num_workers=0) cache to avoid recomputing graphs every epoch."""

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


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("step2_baseline")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_batch(batch, device: torch.device):
    # torch_geometric Batch supports .to(device)
    return batch.to(device, non_blocking=(device.type == "cuda"))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch.x, batch.edge_index, batch.batch)
            y = batch.y.view(-1)
            loss = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())

        pbar.set_postfix(loss=float(loss.item()))

    return total_loss / max(1, len(loader)), correct / max(1, total)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = _move_batch(batch, device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch.x, batch.edge_index, batch.batch)
            y = batch.y.view(-1)
            loss = criterion(logits, y)

        total_loss += float(loss.item())
        preds = logits.argmax(dim=1)

        correct += int((preds == y).sum().item())
        total += int(y.numel())

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())

    return total_loss / max(1, len(loader)), correct / max(1, total), all_preds, all_labels


def save_accuracy_curve(train_acc: List[float], val_acc: List[float], out_path: str) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_loss_curve(train_loss: List[float], val_loss: List[float], out_path: str) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_confusion_matrix(y_true: List[int], y_pred: List[int], out_path: str) -> None:
    class_names = ["No Tumor", "Tumor"]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    opt_name = BEST_PARAMS["optimizer"].lower()
    lr = float(BEST_PARAMS["learning_rate"])
    wd = float(BEST_PARAMS["weight_decay"])

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    raise ValueError(f"Unsupported optimizer: {BEST_PARAMS['optimizer']}")


def main() -> None:
    logger = setup_logger(LOG_PATH)

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger.info("=" * 80)
    logger.info("Step 2 - Baseline Training (br35h)")
    logger.info(f"Start time: {datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("Best hyperparameters (Step 1):")
    for k, v in BEST_PARAMS.items():
        logger.info(f"  {k}: {v}")

    logger.info("Training config (Step 2):")
    logger.info(f"  epochs: {EPOCHS}")
    logger.info(f"  batch_size: {BATCH_SIZE}")
    logger.info(f"  patience: {PATIENCE}")
    logger.info(f"  n_segments: {N_SEGMENTS}")
    logger.info(f"  num_workers: {NUM_WORKERS}")
    logger.info(f"  pin_memory: True")
    logger.info(f"  mixed_precision: {use_amp}")

    # -----------------------------
    # Dataset / loaders
    # -----------------------------
    br35h_root = os.path.join(DATA_ROOT, "br35h")

    logger.info(f"Loading datasets from: {os.path.abspath(br35h_root)}")
    train_base = Br35HDataset(root=br35h_root, split="train", n_segments=N_SEGMENTS)
    test_base = Br35HDataset(root=br35h_root, split="test", n_segments=N_SEGMENTS)

    # Cache graphs in-process to avoid recomputing every epoch.
    train_dataset = CachedDataset(train_base)
    test_dataset = CachedDataset(test_base)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
    )

    # -----------------------------
    # Model / opt / loss
    # -----------------------------
    input_dim = train_base[0].x.shape[1]
    logger.info(f"Detected input_dim: {input_dim}")

    model = BrainTumorGNN(
        input_dim=input_dim,
        hidden_dim=int(BEST_PARAMS["hidden_channels"]),
        num_classes=2,
        dropout=float(BEST_PARAMS["dropout"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)

    # Optional scheduler (safe default)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    # -----------------------------
    # Train loop (early stopping)
    # -----------------------------
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0

    t0 = time.time()
    logger.info("Starting training...")

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler if use_amp else None,
            use_amp=use_amp,
        )

        val_loss, val_acc, _, _ = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        # Scheduler on val accuracy (same convention as your existing step2 script)
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(float(new_lr))

        epoch_s = time.time() - epoch_t0
        logger.info(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={new_lr:.6g} | {epoch_s:.1f}s"
        )

        if new_lr != old_lr:
            logger.info(f"LR changed: {old_lr:.6g} -> {new_lr:.6g}")

        # Early stopping on val_acc (patience)
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = float(val_acc)
            best_epoch = epoch
            patience_counter = 0

            # Save best weights (required output name)
            torch.save(model.state_dict(), BEST_WEIGHTS_PATH)
            logger.info(f"Saved new best weights: {BEST_WEIGHTS_PATH} (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered (no val_acc improvement for {PATIENCE} epochs).")
                break

    total_min = (time.time() - t0) / 60.0
    logger.info(f"Training finished in {total_min:.2f} min")
    logger.info(f"Best val_acc: {best_val_acc:.4f} @ epoch {best_epoch}")

    # -----------------------------
    # Final eval with best weights
    # -----------------------------
    if os.path.exists(BEST_WEIGHTS_PATH):
        model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location=device))
        logger.info(f"Loaded best weights from: {BEST_WEIGHTS_PATH}")

    final_loss, final_acc, y_pred, y_true = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )
    logger.info(f"Final evaluation (best weights) | loss={final_loss:.4f} acc={final_acc:.4f}")

    # -----------------------------
    # Plots (required outputs)
    # -----------------------------
    logger.info("Saving plots...")
    save_accuracy_curve(history["train_acc"], history["val_acc"], ACC_PLOT_PATH)
    save_loss_curve(history["train_loss"], history["val_loss"], LOSS_PLOT_PATH)
    save_confusion_matrix(y_true=y_true, y_pred=y_pred, out_path=CM_PLOT_PATH)

    logger.info(f"Saved: {ACC_PLOT_PATH}")
    logger.info(f"Saved: {LOSS_PLOT_PATH}")
    logger.info(f"Saved: {CM_PLOT_PATH}")
    logger.info("Done.")


if __name__ == "__main__":
    # Windows safety (even though num_workers=0)
    torch.multiprocessing.freeze_support()
    main()
