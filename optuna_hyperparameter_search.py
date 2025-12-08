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

# Global logger (will be initialized in main)
_logger = None


def setup_logging(dataset_name: str, log_dir: str = 'logs') -> logging.Logger:
    """
    Setup logging to both file and console.
    Returns logger instance.
    """
    # Create logs directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f'optuna_search_{dataset_name}')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'optuna_search_{dataset_name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler (for visibility)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def setup_optuna_logging():
    """Configure Optuna logging to show all trial completions."""
    optuna.logging.set_verbosity(optuna.logging.INFO)


def load_datasets_once(dataset_name: str, data_root: str):
    """
    Load datasets once per process and store globally to avoid reloading in each trial.
    This significantly speeds up parallel execution.
    """
    global _global_train_loader, _global_val_loader, _global_num_classes, _global_input_dim, _global_dataset_name, _logger
    
    # If already loaded in this process, skip
    if _global_train_loader is not None and _global_dataset_name == dataset_name:
        return
    
    _global_dataset_name = dataset_name
    
    log_msg = f"Loading {dataset_name.upper()} dataset (process {os.getpid()})..."
    if _logger:
        _logger.info(log_msg)
    else:
        print(f"[INFO] {log_msg}")
    
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
    
    log_msg = f"Dataset loaded (process {os.getpid()}): input_dim={_global_input_dim}, num_classes={_global_num_classes}"
    if _logger:
        _logger.info(log_msg)
    else:
        print(f"[INFO] {log_msg}")


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
    global _global_dataset_name, _global_epochs, _config_data_root, _config_dataset_name, _logger
    
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
            if _logger:
                _logger.info(f"Trial #{trial.number} PRUNED at epoch {epoch+1}")
            raise optuna.TrialPruned()
        
        # Track best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Log progress (both file and console)
        log_msg = (f"Trial #{trial.number} | Epoch {epoch+1}/{epochs} | "
                   f"Params: lr={learning_rate:.6f}, hidden={hidden_channels}, "
                   f"dropout={dropout:.3f}, wd={weight_decay:.6f}, opt={optimizer_name} | "
                   f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                   f"Best Val Acc: {best_val_acc:.4f}")
        if _logger:
            _logger.info(log_msg)
        else:
            print(log_msg)
    
    # Final log statement
    final_msg = (f"Trial #{trial.number} COMPLETED | Params: {{"
                 f"'learning_rate': {learning_rate:.6f}, "
                 f"'hidden_channels': {hidden_channels}, "
                 f"'dropout': {dropout:.3f}, "
                 f"'weight_decay': {weight_decay:.6f}, "
                 f"'optimizer': '{optimizer_name}'}} | "
                 f"Final Accuracy: {best_val_acc:.4f}")
    if _logger:
        _logger.info(final_msg)
    else:
        print(final_msg)
    
    return best_val_acc


def create_visualizations(study: optuna.Study, output_dir: str = '.'):
    """Create and save optimization visualizations."""
    global _logger
    
    msg = "\nCreating visualizations..."
    if _logger:
        _logger.info(msg)
    else:
        print(f"[INFO] {msg}")
    
    # Optimization history
    try:
        fig = optuna_vis.plot_optimization_history(study)
        fig.savefig(os.path.join(output_dir, 'optimization_history.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        msg = "Saved: optimization_history.png"
        if _logger:
            _logger.info(msg)
        else:
            print(f"[INFO] {msg}")
    except Exception as e:
        msg = f"Failed to create optimization_history.png: {e}"
        if _logger:
            _logger.warning(msg)
        else:
            print(f"[WARNING] {msg}")
    
    # Parameter importances
    try:
        fig = optuna_vis.plot_param_importances(study)
        fig.savefig(os.path.join(output_dir, 'param_importances.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        msg = "Saved: param_importances.png"
        if _logger:
            _logger.info(msg)
        else:
            print(f"[INFO] {msg}")
    except Exception as e:
        msg = f"Failed to create param_importances.png: {e}"
        if _logger:
            _logger.warning(msg)
        else:
            print(f"[WARNING] {msg}")


def save_best_params(study: optuna.Study, output_file: str = 'best_params.json'):
    """Save best hyperparameters to JSON file."""
    global _logger
    
    best_params = study.best_params.copy()
    best_params['best_value'] = study.best_value
    best_params['n_trials'] = len(study.trials)
    
    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    msg1 = f"Best parameters saved to: {output_file}"
    msg2 = f"Best accuracy: {study.best_value:.4f} ({study.best_value*100:.2f}%)"
    if _logger:
        _logger.info(msg1)
        _logger.info(msg2)
    else:
        print(f"[INFO] {msg1}")
        print(f"[INFO] {msg2}")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global _logger
    msg = "\nInterrupted by user. Saving current progress..."
    if _logger:
        _logger.info(msg)
    else:
        print(f"[INFO] {msg}")
    sys.exit(0)


def main():
    """Main function for hyperparameter search."""
    global _logger
    
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
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for log files (default: logs)')
    
    args = parser.parse_args()
    
    # Setup logging (file + console)
    _logger = setup_logging(args.dataset, args.log_dir)
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Setup Optuna logging
    setup_optuna_logging()
    
    # Log startup information
    header = f"\n{'='*80}\nOPTUNA HYPERPARAMETER SEARCH\n{'='*80}"
    _logger.info(header)
    _logger.info(f"Dataset: {args.dataset.upper()}")
    _logger.info(f"Trials: {args.n_trials}")
    _logger.info(f"Epochs per trial: {args.epochs}")
    _logger.info(f"Parallel execution: ENABLED (all CPU cores)")
    _logger.info(f"{'='*80}\n")
    
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
    
    _logger.info(f"Starting optimization with {args.n_trials} trials...")
    _logger.info(f"Pruner: MedianPruner (startup_trials=5, warmup_steps=5)")
    _logger.info(f"Press Ctrl+C to stop and save progress\n")
    
    try:
        # Optimize with parallel execution
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=-1,  # Use all available CPU cores
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        _logger.info("\nOptimization interrupted by user.")
    
    # Results summary
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    
    summary = (f"\n{'='*80}\n"
               f"OPTIMIZATION COMPLETE\n"
               f"{'='*80}\n"
               f"Number of completed trials: {completed_trials}\n"
               f"Number of pruned trials: {pruned_trials}\n"
               f"Best trial: {study.best_trial.number}\n"
               f"Best value: {study.best_value:.4f} ({study.best_value*100:.2f}%)\n"
               f"Best params:\n")
    for key, value in study.best_params.items():
        summary += f"  {key}: {value}\n"
    summary += f"{'='*80}\n"
    
    _logger.info(summary)
    
    # Save results
    output_prefix = f"optuna_{args.dataset}"
    save_best_params(study, f"{output_prefix}_best_params.json")
    create_visualizations(study, '.')
    
    # Rename visualization files with prefix
    if os.path.exists('optimization_history.png'):
        os.rename('optimization_history.png', f"{output_prefix}_optimization_history.png")
    if os.path.exists('param_importances.png'):
        os.rename('param_importances.png', f"{output_prefix}_param_importances.png")
    
    _logger.info(f"\nAll results saved with prefix: {output_prefix}_")


if __name__ == '__main__':
    main()

