import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import logging
from datetime import datetime
import json
import time
from pathlib import Path

from data_loader import Br35HDataset, SartajDataset
from gnn_model import BrainTumorGNN, GraphClassifier


def setup_logger(dataset_name: str, log_dir: str = 'logs') -> logging.Logger:
    """Setup logger with file and console handlers."""
    # Create logs directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f'train_{dataset_name}')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{dataset_name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
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
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint."""
        self.best_weights = model.state_dict().copy()


def train_br35h(args):
    """Train GNN on br35h dataset (binary classification)."""
    # Setup logger
    logger = setup_logger('br35h', args.log_dir)
    
    logger.info("=" * 50)
    logger.info("Training on br35h Dataset (Binary Classification)")
    logger.info("=" * 50)
    
    # Log hyperparameters
    logger.info(f"Hyperparameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  Hidden Dim: {args.hidden_dim}")
    logger.info(f"  Num Layers: {args.num_layers}")
    logger.info(f"  Dropout: {args.dropout}")
    logger.info(f"  Weight Decay: {args.weight_decay}")
    logger.info(f"  N Segments: {args.n_segments}")
    logger.info(f"  Use GAT: {args.use_gat}")
    logger.info(f"  Gradient Clipping: {args.grad_clip}")
    logger.info(f"  Early Stopping Patience: {args.early_stop_patience}")
    logger.info(f"  Mixed Precision: {args.mixed_precision}")
    
    start_time = time.time()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Mixed precision training
    use_amp = args.mixed_precision and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Using mixed precision training (AMP)")
    
    # Datasets
    logger.info("Loading datasets...")
    train_dataset = Br35HDataset(
        root=args.data_root,
        split='train',
        n_segments=args.n_segments
    )
    
    test_dataset = Br35HDataset(
        root=args.data_root,
        split='test',
        n_segments=args.n_segments
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Data loaders with optimizations
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 0
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Get input dimension from first sample
    sample = train_dataset[0]
    input_dim = sample.x.shape[1]
    logger.info(f"Input dimension: {input_dim}")
    
    # Model
    model = BrainTumorGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_gat=args.use_gat
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    classifier = GraphClassifier(model, device, scaler=scaler, grad_clip=args.grad_clip)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Better scheduler: ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        restore_best_weights=True
    ) if args.early_stop_patience > 0 else None
    
    # Training
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    learning_rates = []
    
    best_acc = 0.0
    best_epoch = 0
    
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = classifier.train_epoch(train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate
        test_loss, test_acc, _, _ = classifier.evaluate(test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Early stopping
        if early_stopping:
            if early_stopping(test_loss, model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                logger.info(f"Best validation loss: {early_stopping.best_loss:.4f}")
                break
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'args': vars(args)
            }, 'best_model_br35h.pth')
            logger.debug(f"Saved best model at epoch {epoch+1} with accuracy {best_acc:.4f}")
        
        epoch_time = time.time() - epoch_start
        
        # Log every epoch or every N epochs
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.2f}s)")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            logger.info(f"  Best Acc: {best_acc:.4f} (epoch {best_epoch})")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # Log to file every epoch
        logger.debug(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, lr={current_lr:.6f}")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.2f} minutes")
    
    # Final evaluation
    checkpoint = torch.load('best_model_br35h.pth', map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # Old format: just state dict
        model.load_state_dict(checkpoint)
        logger.info("Loaded best model (old format)")
    
    test_loss, test_acc, preds, labels = classifier.evaluate(test_loader, criterion)
    
    logger.info("=" * 50)
    logger.info("Final Results")
    logger.info("=" * 50)
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    # Classification report
    report = classification_report(labels, preds, target_names=['No Tumor', 'Tumor'], output_dict=True)
    logger.info("\nClassification Report:")
    logger.info(classification_report(labels, preds, target_names=['No Tumor', 'Tumor']))
    
    # Save metrics to JSON
    metrics = {
        'dataset': 'br35h',
        'best_epoch': best_epoch,
        'best_test_acc': float(best_acc),
        'final_test_acc': float(test_acc),
        'final_test_loss': float(test_loss),
        'total_training_time_minutes': total_time / 60,
        'total_epochs': len(train_losses),
        'classification_report': report,
        'hyperparameters': vars(args)
    }
    
    with open('training_metrics_br35h.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to training_metrics_br35h.json")
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'])
    plt.title('Confusion Matrix - br35h Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_br35h.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Confusion matrix saved to confusion_matrix_br35h.png")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(test_losses, label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss Curves')
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(train_accs, label='Train Acc', linewidth=2)
    axes[1].plot(test_accs, label='Test Acc', linewidth=2)
    axes[1].axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Accuracy Curves')
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate curve
    axes[2].plot(learning_rates, label='Learning Rate', linewidth=2, color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_br35h.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Training curves saved to training_curves_br35h.png")
    
    logger.info("=" * 50)
    logger.info("Training completed successfully!")
    logger.info("=" * 50)


def train_sartaj(args):
    """Train GNN on sartaj dataset (multi-class classification)."""
    # Setup logger
    logger = setup_logger('sartaj', args.log_dir)
    
    logger.info("=" * 50)
    logger.info("Training on sartaj Dataset (Multi-class Classification)")
    logger.info("=" * 50)
    
    # Log hyperparameters
    logger.info(f"Hyperparameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  Hidden Dim: {args.hidden_dim}")
    logger.info(f"  Num Layers: {args.num_layers}")
    logger.info(f"  Dropout: {args.dropout}")
    logger.info(f"  Weight Decay: {args.weight_decay}")
    logger.info(f"  N Segments: {args.n_segments}")
    logger.info(f"  Use GAT: {args.use_gat}")
    logger.info(f"  Gradient Clipping: {args.grad_clip}")
    logger.info(f"  Early Stopping Patience: {args.early_stop_patience}")
    logger.info(f"  Mixed Precision: {args.mixed_precision}")
    
    start_time = time.time()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Mixed precision training
    use_amp = args.mixed_precision and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Using mixed precision training (AMP)")
    
    # Datasets
    logger.info("Loading datasets...")
    train_dataset = SartajDataset(
        root=args.data_root,
        split='train',
        n_segments=args.n_segments
    )
    
    test_dataset = SartajDataset(
        root=args.data_root,
        split='test',
        n_segments=args.n_segments
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Data loaders with optimizations
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 0
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Get input dimension from first sample
    sample = train_dataset[0]
    input_dim = sample.x.shape[1]
    logger.info(f"Input dimension: {input_dim}")
    
    # Model
    model = BrainTumorGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=4,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_gat=args.use_gat
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    classifier = GraphClassifier(model, device, scaler=scaler, grad_clip=args.grad_clip)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Better scheduler: ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        restore_best_weights=True
    ) if args.early_stop_patience > 0 else None
    
    # Training
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    learning_rates = []
    
    best_acc = 0.0
    best_epoch = 0
    
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = classifier.train_epoch(train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate
        test_loss, test_acc, _, _ = classifier.evaluate(test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Early stopping
        if early_stopping:
            if early_stopping(test_loss, model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                logger.info(f"Best validation loss: {early_stopping.best_loss:.4f}")
                break
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'args': vars(args)
            }, 'best_model_sartaj.pth')
            logger.debug(f"Saved best model at epoch {epoch+1} with accuracy {best_acc:.4f}")
        
        epoch_time = time.time() - epoch_start
        
        # Log every epoch or every N epochs
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.2f}s)")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            logger.info(f"  Best Acc: {best_acc:.4f} (epoch {best_epoch})")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # Log to file every epoch
        logger.debug(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, lr={current_lr:.6f}")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.2f} minutes")
    
    # Final evaluation
    checkpoint = torch.load('best_model_sartaj.pth', map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # Old format: just state dict
        model.load_state_dict(checkpoint)
        logger.info("Loaded best model (old format)")
    
    test_loss, test_acc, preds, labels = classifier.evaluate(test_loader, criterion)
    
    logger.info("=" * 50)
    logger.info("Final Results")
    logger.info("=" * 50)
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    # Classification report
    class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    logger.info("\nClassification Report:")
    logger.info(classification_report(labels, preds, target_names=class_names))
    
    # Save metrics to JSON
    metrics = {
        'dataset': 'sartaj',
        'best_epoch': best_epoch,
        'best_test_acc': float(best_acc),
        'final_test_acc': float(test_acc),
        'final_test_loss': float(test_loss),
        'total_training_time_minutes': total_time / 60,
        'total_epochs': len(train_losses),
        'classification_report': report,
        'hyperparameters': vars(args)
    }
    
    with open('training_metrics_sartaj.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to training_metrics_sartaj.json")
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - sartaj Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_sartaj.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Confusion matrix saved to confusion_matrix_sartaj.png")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(test_losses, label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss Curves')
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(train_accs, label='Train Acc', linewidth=2)
    axes[1].plot(test_accs, label='Test Acc', linewidth=2)
    axes[1].axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Accuracy Curves')
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate curve
    axes[2].plot(learning_rates, label='Learning Rate', linewidth=2, color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_sartaj.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Training curves saved to training_curves_sartaj.png")
    
    logger.info("=" * 50)
    logger.info("Training completed successfully!")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Train GNN for Brain Tumor Classification')
    parser.add_argument('--dataset', type=str, choices=['br35h', 'sartaj', 'both'],
                       default='both', help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='.',
                       help='Root directory of datasets')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--n_segments', type=int, default=100,
                       help='Number of superpixels')
    parser.add_argument('--use_gat', action='store_true',
                       help='Use GAT instead of GCN')
    
    # Training optimizations
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (AMP) for faster training on GPU')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value (0 to disable)')
    parser.add_argument('--early_stop_patience', type=int, default=20,
                       help='Early stopping patience (0 to disable)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                       help='Minimum change in validation loss to qualify as improvement')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save log files')
    parser.add_argument('--log_interval', type=int, default=5,
                       help='Log training progress every N epochs')
    
    # Data loading optimizations
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers (0 for main process only)')
    
    args = parser.parse_args()
    
    if args.dataset == 'br35h' or args.dataset == 'both':
        br35h_args = argparse.Namespace(**vars(args))
        br35h_args.data_root = os.path.join(args.data_root, 'br35h')
        train_br35h(br35h_args)
    
    if args.dataset == 'sartaj' or args.dataset == 'both':
        sartaj_args = argparse.Namespace(**vars(args))
        sartaj_args.data_root = os.path.join(args.data_root, 'sartaj')
        train_sartaj(sartaj_args)


if __name__ == '__main__':
    main()

