import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import argparse
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import logging
from datetime import datetime
import json
import time
from pathlib import Path

# Импорты ваших модулей
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import Br35HDataset, SartajDataset, BrainTumorGNN, GraphClassifier, prepare_artifact_dirs

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(dataset_name: str, log_dir: str = 'logs') -> logging.Logger:
    """Setup logger with file and console handlers."""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f'train_{dataset_name}')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir_path / f'train_{dataset_name}_{timestamp}.log'
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def train_generic(args, dataset_name, dataset_class, num_classes, class_names):
    """
    Generic training function for any GNN dataset.
    """
    # Prepare directories
    artifact_dirs = prepare_artifact_dirs(args.artifact_dir)
    models_dir = artifact_dirs["models"]
    metrics_dir = artifact_dirs["metrics"]
    plots_dir = artifact_dirs["plots"]

    logger = setup_logger(dataset_name, args.log_dir)
    
    logger.info("=" * 50)
    logger.info(f"Training on {dataset_name} Dataset ({num_classes} classes)")
    logger.info("=" * 50)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Mixed precision
    use_amp = args.mixed_precision and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Load Datasets
    logger.info("Loading datasets...")
    # Construct paths properly based on data_root provided in main
    train_dataset = dataset_class(root=args.data_root, split='train', n_segments=args.n_segments)
    test_dataset = dataset_class(root=args.data_root, split='test', n_segments=args.n_segments)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Data Loaders
    num_workers = args.num_workers
    pin_memory = device.type == 'cuda'
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=pin_memory)
    
    # Model Init
    input_dim = train_dataset[0].x.shape[1]
    logger.info(f"Input dimension: {input_dim}")
    
    model = BrainTumorGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_gat=args.use_gat
    )
    
    classifier = GraphClassifier(model, device, scaler=scaler, grad_clip=args.grad_clip)
    
    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    early_stopping = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta
    ) if args.early_stop_patience > 0 else None
    
    # Training Loop Variables
    train_losses, train_accs, test_losses, test_accs, learning_rates = [], [], [], [], []
    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train & Eval step
        train_loss, train_acc = classifier.train_epoch(train_loader, optimizer, criterion)
        test_loss, test_acc, _, _ = classifier.evaluate(test_loader, criterion)
        
        # Store metrics
        train_losses.append(train_loss); train_accs.append(train_acc)
        test_losses.append(test_loss); test_accs.append(test_acc)
        
        # Scheduler & LR
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_model_path = models_dir / f'best_model_{dataset_name}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'args': vars(args)
            }, best_model_path)
        
        # Logging
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.2f}s) | "
                        f"Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f} | Best: {best_acc:.4f}")
            
        # Early Stopping
        if early_stopping and early_stopping(test_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    logger.info(f"Training finished in {total_time/60:.2f} mins")
    
    # --- Final Evaluation ---
    # Load best model
    best_model_path = models_dir / f'best_model_{dataset_name}.pth'
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best model from epoch {checkpoint.get('epoch')}")
    
    test_loss, test_acc, preds, labels = classifier.evaluate(test_loader, criterion)
    
    # Save Metrics JSON
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    metrics = {
        'dataset': dataset_name,
        'best_epoch': best_epoch,
        'best_test_acc': float(best_acc),
        'final_test_acc': float(test_acc),
        'training_time_min': total_time / 60,
        'classification_report': report,
        'hyperparameters': vars(args)
    }
    
    with open(metrics_dir / f'training_metrics_{dataset_name}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Plot Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.savefig(plots_dir / f'confusion_matrix_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Training Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Loss
    axes[0].plot(train_losses, label='Train'); axes[0].plot(test_losses, label='Test')
    axes[0].set_title('Loss'); axes[0].legend()
    # Acc
    axes[1].plot(train_accs, label='Train'); axes[1].plot(test_accs, label='Test')
    axes[1].axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.4f}')
    axes[1].set_title('Accuracy'); axes[1].legend()
    # LR
    axes[2].plot(learning_rates, color='green')
    axes[2].set_title('Learning Rate'); axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'training_curves_{dataset_name}.png', dpi=300)
    plt.close()
    
    logger.info("Training complete. Artifacts saved.")

def main():
    parser = argparse.ArgumentParser(description='Train GNN for Brain Tumor Classification')
    parser.add_argument('--dataset', type=str, choices=['br35h', 'sartaj', 'both'], default='both')
    parser.add_argument('--data_root', type=str, default='data', help='Root directory containing dataset folders')
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_segments', type=int, default=100)
    parser.add_argument('--use_gat', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Optimization & System
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0)
    parser.add_argument('--artifact_dir', type=str, default='artifacts')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    
    args = parser.parse_args()
    
    # Set seed globally
    set_seed(args.seed)

    if args.log_dir is None:
        args.log_dir = str(Path(args.artifact_dir) / "logs")
    
    # Training Logic
    if args.dataset in ['br35h', 'both']:
        # Create a copy of args specifically for this dataset to safely modify paths
        br35h_args = argparse.Namespace(**vars(args))
        br35h_args.data_root = os.path.join(args.data_root, 'br35h')
        
        train_generic(
            br35h_args, 
            dataset_name='br35h', 
            dataset_class=Br35HDataset, 
            num_classes=2, 
            class_names=['No Tumor', 'Tumor']
        )
    
    if args.dataset in ['sartaj', 'both']:
        sartaj_args = argparse.Namespace(**vars(args))
        sartaj_args.data_root = os.path.join(args.data_root, 'sartaj')
        
        train_generic(
            sartaj_args, 
            dataset_name='sartaj', 
            dataset_class=SartajDataset, 
            num_classes=4, 
            class_names=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        )

if __name__ == '__main__':
    main()