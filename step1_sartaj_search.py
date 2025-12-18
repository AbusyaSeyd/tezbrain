"""
Optuna hyperparameter search for the Sartaj multiclass brain tumor dataset.

Key differences vs the generic search:
- Optimizes macro F1 (not accuracy) to treat all four classes fairly.
- Uses class-balanced CrossEntropy via dynamically computed class weights.
- Expands model capacity search space for the harder multiclass setting.
- Applies Windows/GTX 1650 friendly defaults (num_workers=0, n_segments=50).
"""
import argparse
import json
import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
from optuna.visualization import matplotlib as optuna_vis
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from sklearn.metrics import f1_score, classification_report
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

from data_loader import SartajDataset
from gnn_model import BrainTumorGNN
from paths import prepare_artifact_dirs

# Global caches to avoid reloading between trials
_TRAIN_DS = None
_VAL_DS = None
_INPUT_DIM = None
_CLASS_WEIGHTS = None

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Logging initialized. File: %s", log_file)


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    # Balanced weighting: total/(num_classes * count_c)
    weights = counts.sum() / (counts + 1e-6) / num_classes
    return weights


def load_sartaj(data_root: str, n_segments: int = 50) -> Tuple[SartajDataset, SartajDataset, int, torch.Tensor]:
    """
    Load Sartaj train/val once and compute input dimension + class weights.
    """
    global _TRAIN_DS, _VAL_DS, _INPUT_DIM, _CLASS_WEIGHTS
    if _TRAIN_DS is not None:
        return _TRAIN_DS, _VAL_DS, _INPUT_DIM, _CLASS_WEIGHTS

    root = os.path.join(data_root, "sartaj")
    _TRAIN_DS = SartajDataset(root=root, split="train", n_segments=n_segments)
    _VAL_DS = SartajDataset(root=root, split="test", n_segments=n_segments)

    sample = _TRAIN_DS[0]
    _INPUT_DIM = sample.x.shape[1]
    _CLASS_WEIGHTS = compute_class_weights(_TRAIN_DS.labels, num_classes=4)

    print(
        f"[DATA] Sartaj loaded | train={len(_TRAIN_DS)} val={len(_VAL_DS)} "
        f"| input_dim={_INPUT_DIM} | class_weights={_CLASS_WEIGHTS.tolist()}"
    )
    logging.info(
        "[DATA] Sartaj loaded | train=%d val=%d | input_dim=%d | class_weights=%s",
        len(_TRAIN_DS),
        len(_VAL_DS),
        _INPUT_DIM,
        _CLASS_WEIGHTS.tolist(),
    )
    return _TRAIN_DS, _VAL_DS, _INPUT_DIM, _CLASS_WEIGHTS


def create_loaders(train_ds, val_ds, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )
    return train_loader, val_loader


def override_gat_heads(
    model: BrainTumorGNN,
    heads: int,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> BrainTumorGNN:
    """
    Replace GAT layers with the requested head count while keeping
    the rest of the BrainTumorGNN architecture intact.
    """
    convs = nn.ModuleList()
    bns = nn.ModuleList()

    convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=False))
    bns.append(nn.BatchNorm1d(hidden_dim))

    for _ in range(num_layers - 2):
        convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False))
        bns.append(nn.BatchNorm1d(hidden_dim))

    # Keep final layer single-head to preserve hidden_dim width
    convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False))
    bns.append(nn.BatchNorm1d(hidden_dim))

    model.convs = convs
    model.batch_norms = bns
    return model


def build_model(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    use_gat: bool,
    heads: int,
) -> BrainTumorGNN:
    model = BrainTumorGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=4,
        num_layers=num_layers,
        dropout=dropout,
        use_gat=use_gat,
    )
    if use_gat:
        model = override_gat_heads(model, heads=heads, input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    grad_clip: float = 2.0,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda"):
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.squeeze())

        scaler.scale(loss).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    preds: List[int] = []
    labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.squeeze())

            total_loss += loss.item()
            preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(batch.y.squeeze().cpu().numpy().tolist())

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return total_loss / max(1, len(loader)), macro_f1, np.array(preds), np.array(labels)


def train_and_validate(
    params: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
) -> Tuple[float, Dict[str, Any]]:
    model = build_model(
        input_dim=input_dim,
        hidden_dim=params["hidden_channels"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        use_gat=params["use_gat"],
        heads=params["heads"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_macro = 0.0
    best_snapshot = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_macro, preds, labels = evaluate(model, val_loader, criterion, device)

        if val_macro > best_macro:
            best_macro = val_macro
            best_snapshot = {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "preds": preds,
                "labels": labels,
                "val_loss": val_loss,
            }

        logging.info(
            "[EPOCH %02d] train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f",
            epoch + 1,
            train_loss,
            val_loss,
            val_macro,
        )

    return best_macro, best_snapshot


def objective_factory(
    train_ds,
    val_ds,
    input_dim: int,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
):
    def objective(trial: optuna.Trial) -> float:
        batch_size = trial.suggest_categorical("batch_size", [8, 12, 16, 24])
        hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.2, 0.6)
        use_gat = trial.suggest_categorical("use_gat", [True, False])
        heads = trial.suggest_categorical("heads", [4, 8]) if use_gat else 4
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        train_loader, val_loader = create_loaders(train_ds, val_ds, batch_size)

        model = build_model(
            input_dim=input_dim,
            hidden_dim=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_gat=use_gat,
            heads=heads,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

        best_macro = 0.0
        for epoch in range(epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_loss, val_macro, _, _ = evaluate(model, val_loader, criterion, device)

            trial.report(val_macro, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_macro > best_macro:
                best_macro = val_macro

            logging.info(
                "[TRIAL %d | epoch %02d] macro_f1=%.4f best_macro=%.4f params(hidden=%d, heads=%d, bs=%d, dropout=%.2f, lr=%.5f, wd=%.5f, gat=%s)",
                trial.number,
                epoch + 1,
                val_macro,
                best_macro,
                hidden_channels,
                heads,
                batch_size,
                dropout,
                learning_rate,
                weight_decay,
                use_gat,
            )

        return best_macro

    return objective


def evaluate_best_params(
    best_params: Dict[str, Any],
    train_ds,
    val_ds,
    input_dim: int,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
) -> Dict[str, Any]:
    train_loader, val_loader = create_loaders(train_ds, val_ds, best_params["batch_size"])
    macro_f1, snapshot = train_and_validate(
        params=best_params,
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        class_weights=class_weights,
        device=device,
        epochs=epochs,
    )

    per_class_f1 = f1_score(snapshot["labels"], snapshot["preds"], average=None, zero_division=0)
    report = classification_report(
        snapshot["labels"],
        snapshot["preds"],
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    logging.info("\n[BEST TRIAL] Per-class F1:")
    for cls, score in zip(CLASS_NAMES, per_class_f1):
        logging.info("  %s: %.4f", cls, score)
    logging.info("\n[CLASSIFICATION REPORT]\n%s", report)

    return {
        "macro_f1": float(macro_f1),
        "per_class_f1": {cls: float(score) for cls, score in zip(CLASS_NAMES, per_class_f1)},
        "report": report,
    }


def save_study_artifacts(study: optuna.Study, plots_dir: Path, metrics_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Optimization history
    try:
        fig = optuna_vis.plot_optimization_history(study)
        hist_path = plots_dir / "optuna_sartaj_optimization_history.png"
        fig.savefig(hist_path, dpi=300, bbox_inches="tight")
        logging.info("Saved plot: %s", hist_path)
    except Exception as e:
        logging.warning("Failed to save optimization history plot: %s", e)

    # Parameter importances
    try:
        fig = optuna_vis.plot_param_importances(study)
        imp_path = plots_dir / "optuna_sartaj_param_importances.png"
        fig.savefig(imp_path, dpi=300, bbox_inches="tight")
        logging.info("Saved plot: %s", imp_path)
    except Exception as e:
        logging.warning("Failed to save parameter importances plot: %s", e)

    # Trials dataframe
    try:
        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        csv_path = metrics_dir / "optuna_sartaj_trials.csv"
        df.to_csv(csv_path, index=False)
        logging.info("Saved trials dataframe: %s", csv_path)
    except Exception as e:
        logging.warning("Failed to save trials dataframe: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna search for Sartaj multiclass GNN (macro F1 optimized)")
    parser.add_argument("--data_root", type=str, default=".", help="Path containing the 'sartaj' folder")
    parser.add_argument("--artifact_dir", type=str, default="artifacts", help="Where to store metrics/plots")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file (default: artifacts/logs/step1_sartaj_search.log)")
    args = parser.parse_args()

    set_seed(args.seed)
    artifact_dirs = prepare_artifact_dirs(args.artifact_dir)
    logs_dir: Path = artifact_dirs["logs"]
    metrics_dir: Path = artifact_dirs["metrics"]
    plots_dir: Path = artifact_dirs["plots"]

    log_file = Path(args.log_file) if args.log_file else logs_dir / "step1_sartaj_search.log"
    setup_logging(log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("[SETUP] device=%s | cuda_available=%s", device, torch.cuda.is_available())

    train_ds, val_ds, input_dim, class_weights = load_sartaj(args.data_root, n_segments=50)

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)

    objective = objective_factory(
        train_ds=train_ds,
        val_ds=val_ds,
        input_dim=input_dim,
        class_weights=class_weights,
        device=device,
        epochs=args.epochs,
    )

    logging.info(
        "[SEARCH] Trials=%d | Epochs/Trial=%d | Optimize=Macro F1 | MedianPruner active",
        args.n_trials,
        args.epochs,
    )
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1, show_progress_bar=True)

    logging.info("\n[RESULTS] Best trial:")
    logging.info("  Trial #: %d", study.best_trial.number)
    logging.info("  Macro F1: %.4f", study.best_value)
    for k, v in study.best_params.items():
        logging.info("  %s: %s", k, v)

    # Re-train best params to report per-class F1 clearly
    best_metrics = evaluate_best_params(
        best_params=study.best_params,
        train_ds=train_ds,
        val_ds=val_ds,
        input_dim=input_dim,
        class_weights=class_weights,
        device=device,
        epochs=args.epochs,
    )

    # Persist best params + metrics
    best_payload = {
        "best_value_macro_f1": study.best_value,
        "best_params": study.best_params,
        "per_class_f1": best_metrics["per_class_f1"],
        "classification_report": best_metrics["report"],
        "n_trials": len(study.trials),
        "epochs_per_trial": args.epochs,
    }
    out_path = metrics_dir / "optuna_sartaj_best_params.json"
    with open(out_path, "w") as f:
        json.dump(best_payload, f, indent=2)
    logging.info("[SAVED] Best params + metrics -> %s", out_path)

    save_study_artifacts(study, plots_dir=plots_dir, metrics_dir=metrics_dir)


if __name__ == "__main__":
    main()
