#!/usr/bin/env python3
"""
Optuna-based Hyperparameter Search for Brain Tumor Detection GNN
Optimized for execution speed and visibility with parallel processing.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import optuna
from optuna.pruners import MedianPruner
from optuna.visualization import matplotlib as optuna_vis
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import signal
import argparse
from typing import Dict, Any
from pathlib import Path

# Import existing modules
from data_loader import Br35HDataset, SartajDataset
from gnn_model import BrainTumorGNN, GraphClassifier
from paths import prepare_artifact_dirs

# Global variables for dataset (loaded once per process to avoid reloading in each trial)
_global_train_loader = None
_global_val_loader = None
_global_num_classes = None
_global_input_dim = None
_global_dataset_name = None
_global_epochs = 15  # Default epochs per trial

# Configuration (set before optimization, accessible in all processes)
_config_data_root = '.'
_config_dataset_name = 'br35h'


def setup_optuna_logging():
    """Configure Optuna logging to show all trial completions."""
    optuna.logging.set_verbosity(optuna.logging.INFO)


def load_datasets_once(dataset_name: str, data_root: str):
    """
    Load datasets once per process and store globally to avoid reloading in each trial.
    This significantly speeds up parallel execution.
    """
    global _global_train_loader, _global_val_loader, _global_num_classes, _global_input_dim, _global_dataset_name
    
    # If already loaded in this process, skip
    if _global_train_loader is not None and _global_dataset_name == dataset_name:
        return
    
    _global_dataset_name = dataset_name
    
    print(f"[INFO] Loading {dataset_name.upper()} dataset (process {os.getpid()})...")
    
    # Load datasets
    if dataset_name == 'br35h':
        train_dataset = Br35HDataset(
            root=os.path.join(data_root, 'br35h'),
            split='train',
            n_segments=100  # Fixed for consistency
        )
        val_dataset = Br35HDataset(
            root=os.path.join(data_root, 'br35h'),
            split='test',
            n_segments=100
        )
        _global_num_classes = 2
    else:  # sartaj
        train_dataset = SartajDataset(
            root=os.path.join(data_root, 'sartaj'),
            split='train',
            n_segments=100
        )
        val_dataset = SartajDataset(
            root=os.path.join(data_root, 'sartaj'),
            split='test',
            n_segments=100
        )
        _global_num_classes = 4
    
    # Get input dimension from sample
    sample = train_dataset[0]
    _global_input_dim = sample.x.shape[1]
    
    # Store datasets (loaders will be created per trial with different batch sizes)
    _global_train_loader = train_dataset
    _global_val_loader = val_dataset
    
    print(f"[INFO] Dataset loaded (process {os.getpid()}): input_dim={_global_input_dim}, num_classes={_global_num_classes}")


def create_data_loaders(batch_size: int):
    """Create data loaders with specified batch size."""
    global _global_train_loader, _global_val_loader
    
    train_loader = DataLoader(
        _global_train_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for multiprocessing safety
        pin_memory=False
    )
    
    val_loader = DataLoader(
        _global_val_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.squeeze())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += pred.eq(batch.y.squeeze()).sum().item()
        total += batch.y.size(0)
    
    return total_loss / len(train_loader), correct / total


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.squeeze())
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += pred.eq(batch.y.squeeze()).sum().item()
            total += batch.y.size(0)
    
    return total_loss / len(val_loader), correct / total


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    Returns validation accuracy.
    """
    global _global_input_dim, _global_num_classes, _global_train_loader, _global_val_loader
    global _global_dataset_name, _global_epochs, _config_data_root, _config_dataset_name
    
    # Load datasets if not already loaded in this process
    if _global_train_loader is None:
        load_datasets_once(_config_dataset_name, _config_data_root)
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    
    # Fixed batch size for consistency (can be made tunable if needed)
    batch_size = 32
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(batch_size)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = BrainTumorGNN(
        input_dim=_global_input_dim,
        hidden_dim=hidden_channels,
        num_classes=_global_num_classes,
        num_layers=3,  # Fixed
        dropout=dropout,
        use_gat=False  # Fixed to GCN for speed
    ).to(device)
    
    # Optimizer
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:  # AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    epochs = _global_epochs
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Report intermediate value for pruning
        trial.report(val_acc, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Track best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Print progress (visible in terminal)
        print(f"Trial #{trial.number} | Epoch {epoch+1}/{epochs} | "
              f"Params: lr={learning_rate:.6f}, hidden={hidden_channels}, "
              f"dropout={dropout:.3f}, wd={weight_decay:.6f}, opt={optimizer_name} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
              f"Best Val Acc: {best_val_acc:.4f}")
    
    # Final print statement
    print(f"Trial #{trial.number} COMPLETED | Params: {{"
          f"'learning_rate': {learning_rate:.6f}, "
          f"'hidden_channels': {hidden_channels}, "
          f"'dropout': {dropout:.3f}, "
          f"'weight_decay': {weight_decay:.6f}, "
          f"'optimizer': '{optimizer_name}'}} | "
          f"Final Accuracy: {best_val_acc:.4f}")
    
    return best_val_acc


def create_visualizations(study: optuna.Study, output_dir: str = '.'):
    """Create and save optimization visualizations."""
    print("\n[INFO] Creating visualizations...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Optimization history
    try:
        fig = optuna_vis.plot_optimization_history(study)
        fig.savefig(output_path / 'optimization_history.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved: optimization_history.png")
    except Exception as e:
        print(f"[WARNING] Failed to create optimization_history.png: {e}")
    
    # Parameter importances
    try:
        fig = optuna_vis.plot_param_importances(study)
        fig.savefig(output_path / 'param_importances.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved: param_importances.png")
    except Exception as e:
        print(f"[WARNING] Failed to create param_importances.png: {e}")


def save_best_params(study: optuna.Study, output_file: str = 'best_params.json'):
    """Save best hyperparameters to JSON file."""
    best_params = study.best_params.copy()
    best_params['best_value'] = study.best_value
    best_params['n_trials'] = len(study.trials)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"[INFO] Best parameters saved to: {output_path}")
    print(f"[INFO] Best accuracy: {study.best_value:.4f} ({study.best_value*100:.2f}%)")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n[INFO] Interrupted by user. Saving current progress...")
    sys.exit(0)


def main():
    """Main function for hyperparameter search."""
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Search for GNN')
    parser.add_argument('--dataset', type=str, choices=['br35h', 'sartaj'],
                       default='br35h', help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='.',
                       help='Root directory containing dataset folders')
    parser.add_argument('--n_trials', type=int, default=75,
                       help='Number of trials (default: 75)')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Epochs per trial (default: 15)')
    parser.add_argument('--storage', type=str, default=None,
                       help='SQLite storage path (optional, for thread safety)')
    parser.add_argument('--study_name', type=str, default=None,
                       help='Study name (for resuming)')
    parser.add_argument('--artifact_dir', type=str, default='artifacts',
                       help='Base directory to store models, metrics, and plots')
    
    args = parser.parse_args()
    artifact_dirs = prepare_artifact_dirs(args.artifact_dir)
    metrics_dir = artifact_dirs["metrics"]
    plots_dir = artifact_dirs["plots"]
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Setup Optuna logging
    setup_optuna_logging()
    
    # Load datasets once (before parallel execution)
    print(f"\n{'='*80}")
    print(f"OPTUNA HYPERPARAMETER SEARCH")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs per trial: {args.epochs}")
    print(f"Parallel execution: ENABLED (all CPU cores)")
    print(f"{'='*80}\n")
    
    # Set module-level configuration (accessible in all processes)
    global _config_data_root, _config_dataset_name, _global_epochs
    _config_data_root = args.data_root
    _config_dataset_name = args.dataset
    _global_epochs = args.epochs
    
    # Load datasets in main process first (will also load in each worker process)
    load_datasets_once(args.dataset, args.data_root)
    
    # Create study with MedianPruner
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=1
    )
    
    # Use SQLite storage for thread safety in parallel execution
    if args.storage:
        storage = f"sqlite:///{args.storage}"
        study = optuna.create_study(
            direction='maximize',
            pruner=pruner,
            storage=storage,
            study_name=args.study_name or f"gnn_{args.dataset}",
            load_if_exists=True
        )
    else:
        # In-memory storage (works but less safe for parallel execution)
        study = optuna.create_study(
            direction='maximize',
            pruner=pruner
        )
    
    print(f"[INFO] Starting optimization with {args.n_trials} trials...")
    print(f"[INFO] Pruner: MedianPruner (startup_trials=5, warmup_steps=5)")
    print(f"[INFO] Press Ctrl+C to stop and save progress\n")
    
    try:
        # Optimize with parallel execution
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=-1,  # Use all available CPU cores
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[INFO] Optimization interrupted by user.")
    
    # Results summary
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f} ({study.best_value*100:.2f}%)")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Save results
    output_prefix = f"optuna_{args.dataset}"
    best_params_path = metrics_dir / f"{output_prefix}_best_params.json"
    save_best_params(study, best_params_path)
    create_visualizations(study, plots_dir)
    
    # Rename visualization files with prefix
    hist_path = plots_dir / 'optimization_history.png'
    param_path = plots_dir / 'param_importances.png'
    if hist_path.exists():
        hist_path.rename(plots_dir / f"{output_prefix}_optimization_history.png")
    if param_path.exists():
        param_path.rename(plots_dir / f"{output_prefix}_param_importances.png")
    
    print(f"\n[INFO] All results saved with prefix: {output_prefix}_")


if __name__ == '__main__':
    main()

