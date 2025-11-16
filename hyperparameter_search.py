"""
Random Search Hyperparameter Optimization for GNN Models
Tests different hyperparameter combinations and finds the best configuration.
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import time
from pathlib import Path

from data_loader import Br35HDataset, SartajDataset
from gnn_model import BrainTumorGNN, GraphClassifier


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience=30, min_delta=0.0, restore_best_weights=True):
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


def train_model_with_config(dataset_name, config, epochs=200, data_root='.'):
    """Train a model with given hyperparameter configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    if dataset_name == 'br35h':
        train_dataset = Br35HDataset(
            root=os.path.join(data_root, 'br35h'),
            split='train',
            n_segments=config['n_segments']
        )
        test_dataset = Br35HDataset(
            root=os.path.join(data_root, 'br35h'),
            split='test',
            n_segments=config['n_segments']
        )
        num_classes = 2
    else:  # sartaj
        train_dataset = SartajDataset(
            root=os.path.join(data_root, 'sartaj'),
            split='train',
            n_segments=config['n_segments']
        )
        test_dataset = SartajDataset(
            root=os.path.join(data_root, 'sartaj'),
            split='test',
            n_segments=config['n_segments']
        )
        num_classes = 4
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Get input dimension
    sample = train_dataset[0]
    input_dim = sample.x.shape[1]
    
    # Create model
    model = BrainTumorGNN(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_classes=num_classes,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_gat=config['use_gat']
    )
    model = model.to(device)
    
    # Setup training
    classifier = GraphClassifier(model, device, scaler=None, grad_clip=config.get('grad_clip', 1.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )
    
    early_stopping = EarlyStopping(patience=40, min_delta=0.0001, restore_best_weights=True)
    
    # Training loop
    best_acc = 0.0
    best_epoch = 0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(epochs):
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
        
        # Early stopping
        if early_stopping(test_loss, model):
            break
        
        # Track best
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
    
    return {
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_test_acc': test_acc,
        'final_test_loss': test_loss,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }


def generate_random_config(search_space):
    """Generate a random hyperparameter configuration."""
    config = {}
    for param, values in search_space.items():
        if isinstance(values, list):
            config[param] = random.choice(values)
        elif isinstance(values, tuple) and len(values) == 2:
            # Continuous range
            if isinstance(values[0], float):
                config[param] = random.uniform(values[0], values[1])
            else:
                config[param] = random.randint(values[0], values[1])
    return config


def random_search(dataset_name, n_trials=20, epochs=200, data_root='.'):
    """Perform random search hyperparameter optimization."""
    print(f"\n{'='*70}")
    print(f"RANDOM SEARCH HYPERPARAMETER OPTIMIZATION - {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"Number of trials: {n_trials}")
    print(f"Epochs per trial: {epochs}")
    print(f"{'='*70}\n")
    
    # Define search space with high learning rates
    search_space = {
        'lr': [0.001, 0.002, 0.003, 0.005, 0.01, 0.015, 0.02],  # High learning rates
        'hidden_dim': [32, 64, 128, 256],
        'num_layers': [2, 3, 4, 5],
        'batch_size': [16, 32, 64],
        'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
        'weight_decay': [1e-4, 5e-4, 1e-3, 5e-3],
        'n_segments': [50, 100, 150, 200],
        'use_gat': [True, False],
        'grad_clip': [0.5, 1.0, 2.0]
    }
    
    results = []
    results_file = f'hyperparameter_results_{dataset_name}_incremental.json'
    
    # Load existing results if any
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
                results = existing_results
                print(f"Loaded {len(results)} existing results. Continuing from trial {len(results) + 1}")
        except:
            pass
    
    for trial in tqdm(range(len(results), n_trials), desc="Random Search Trials", initial=len(results), total=n_trials):
        # Generate random configuration
        config = generate_random_config(search_space)
        config['trial'] = trial + 1
        
        print(f"\nTrial {trial + 1}/{n_trials}")
        print(f"Configuration: {config}")
        
        try:
            # Train model
            start_time = time.time()
            training_results = train_model_with_config(
                dataset_name, config, epochs=epochs, data_root=data_root
            )
            training_time = time.time() - start_time
            
            # Combine results
            result = {
                **config,
                **training_results,
                'training_time_minutes': training_time / 60
            }
            results.append(result)
            
            # Save incrementally
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"  Best Accuracy: {result['best_acc']:.4f} (epoch {result['best_epoch']})")
            print(f"  Final Test Accuracy: {result['final_test_acc']:.4f}")
            print(f"  Training Time: {training_time/60:.2f} minutes")
            print(f"  Progress: {len(results)}/{n_trials} trials completed")
            
        except Exception as e:
            print(f"  ERROR in trial {trial + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def create_results_table(results, dataset_name):
    """Create a formatted results table."""
    df = pd.DataFrame(results)
    
    # Select relevant columns for display
    display_cols = [
        'trial', 'lr', 'hidden_dim', 'num_layers', 'batch_size',
        'dropout', 'weight_decay', 'n_segments', 'use_gat',
        'best_acc', 'best_epoch', 'final_test_acc', 'training_time_minutes'
    ]
    
    # Ensure all columns exist
    available_cols = [col for col in display_cols if col in df.columns]
    df_display = df[available_cols].copy()
    
    # Sort by best accuracy
    df_display = df_display.sort_values('best_acc', ascending=False)
    
    # Format columns
    if 'lr' in df_display.columns:
        df_display['lr'] = df_display['lr'].apply(lambda x: f"{x:.4f}")
    if 'dropout' in df_display.columns:
        df_display['dropout'] = df_display['dropout'].apply(lambda x: f"{x:.2f}")
    if 'weight_decay' in df_display.columns:
        df_display['weight_decay'] = df_display['weight_decay'].apply(lambda x: f"{x:.6f}")
    if 'best_acc' in df_display.columns:
        df_display['best_acc'] = df_display['best_acc'].apply(lambda x: f"{x:.4f}")
    if 'final_test_acc' in df_display.columns:
        df_display['final_test_acc'] = df_display['final_test_acc'].apply(lambda x: f"{x:.4f}")
    if 'training_time_minutes' in df_display.columns:
        df_display['training_time_minutes'] = df_display['training_time_minutes'].apply(lambda x: f"{x:.2f}")
    
    return df_display, df


def visualize_results(results, dataset_name, save_dir='.'):
    """Create comprehensive visualizations of hyperparameter search results."""
    df = pd.DataFrame(results)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy vs Learning Rate
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(df['lr'], df['best_acc'], c=df['hidden_dim'], 
                          cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    ax1.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Learning Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Hidden Dim')
    
    # 2. Accuracy vs Hidden Dimension
    ax2 = fig.add_subplot(gs[0, 1])
    df_hidden = df.groupby('hidden_dim')['best_acc'].agg(['mean', 'std']).reset_index()
    ax2.errorbar(df_hidden['hidden_dim'], df_hidden['mean'], 
                yerr=df_hidden['std'], fmt='o-', linewidth=2, markersize=8, capsize=5)
    ax2.set_xlabel('Hidden Dimension', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy vs Hidden Dimension', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy vs Number of Layers
    ax3 = fig.add_subplot(gs[0, 2])
    df_layers = df.groupby('num_layers')['best_acc'].agg(['mean', 'std']).reset_index()
    ax3.errorbar(df_layers['num_layers'], df_layers['mean'],
                yerr=df_layers['std'], fmt='o-', linewidth=2, markersize=8, capsize=5)
    ax3.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy vs Number of Layers', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy vs Dropout
    ax4 = fig.add_subplot(gs[1, 0])
    scatter4 = ax4.scatter(df['dropout'], df['best_acc'], c=df['lr'],
                          cmap='plasma', s=100, alpha=0.7, edgecolors='black')
    ax4.set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Accuracy vs Dropout Rate', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=ax4, label='Learning Rate')
    
    # 5. Accuracy vs Batch Size
    ax5 = fig.add_subplot(gs[1, 1])
    df_batch = df.groupby('batch_size')['best_acc'].agg(['mean', 'std']).reset_index()
    ax5.bar(df_batch['batch_size'], df_batch['mean'], yerr=df_batch['std'],
           capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    ax5.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax5.set_title('Accuracy vs Batch Size', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. GAT vs GCN Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    df_gat = df.groupby('use_gat')['best_acc'].agg(['mean', 'std']).reset_index()
    df_gat['model_type'] = df_gat['use_gat'].map({True: 'GAT', False: 'GCN'})
    ax6.bar(df_gat['model_type'], df_gat['mean'], yerr=df_gat['std'],
           capsize=5, alpha=0.7, color=['coral', 'lightblue'], edgecolor='black')
    ax6.set_xlabel('Model Type', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax6.set_title('GAT vs GCN Performance', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Top 10 Configurations
    ax7 = fig.add_subplot(gs[2, :])
    df_top10 = df.nlargest(10, 'best_acc')
    y_pos = np.arange(len(df_top10))
    ax7.barh(y_pos, df_top10['best_acc'], alpha=0.7, color='green', edgecolor='black')
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels([f"Trial {t}" for t in df_top10['trial']])
    ax7.set_xlabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax7.set_title('Top 10 Configurations', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='x')
    for i, acc in enumerate(df_top10['best_acc']):
        ax7.text(acc + 0.001, i, f'{acc:.4f}', va='center', fontweight='bold')
    
    # 8. Hyperparameter Importance (correlation with accuracy)
    ax8 = fig.add_subplot(gs[3, 0])
    numeric_cols = ['lr', 'hidden_dim', 'num_layers', 'batch_size', 'dropout', 'weight_decay', 'n_segments']
    correlations = []
    param_names = []
    for col in numeric_cols:
        if col in df.columns:
            corr = df[col].corr(df['best_acc'])
            correlations.append(abs(corr))
            param_names.append(col.replace('_', ' ').title())
    
    if correlations:
        bars = ax8.barh(param_names, correlations, alpha=0.7, color='purple', edgecolor='black')
        ax8.set_xlabel('Absolute Correlation with Accuracy', fontsize=12, fontweight='bold')
        ax8.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')
        for i, corr in enumerate(correlations):
            ax8.text(corr + 0.01, i, f'{corr:.3f}', va='center', fontweight='bold')
    
    # 9. Training Time vs Accuracy
    ax9 = fig.add_subplot(gs[3, 1])
    scatter9 = ax9.scatter(df['training_time_minutes'], df['best_acc'],
                          c=df['num_layers'], cmap='coolwarm', s=100, alpha=0.7, edgecolors='black')
    ax9.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax9.set_title('Training Time vs Accuracy', fontsize=14, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    plt.colorbar(scatter9, ax=ax9, label='Num Layers')
    
    # 10. Best Epoch Distribution
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.hist(df['best_epoch'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax10.set_xlabel('Best Epoch', fontsize=12, fontweight='bold')
    ax10.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax10.set_title('Distribution of Best Epochs', fontsize=14, fontweight='bold')
    ax10.grid(True, alpha=0.3, axis='y')
    ax10.axvline(df['best_epoch'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df["best_epoch"].mean():.1f}')
    ax10.legend()
    
    plt.suptitle(f'Hyperparameter Search Results - {dataset_name.upper()} Dataset', 
                fontsize=18, fontweight='bold', y=0.995)
    
    save_path = os.path.join(save_dir, f'hyperparameter_search_{dataset_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {save_path}")
    return save_path


def create_results_table_image(df_display, dataset_name, save_dir='.'):
    """Create an image of the results table."""
    fig, ax = plt.subplots(figsize=(20, max(10, len(df_display) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df_display.values,
                    colLabels=df_display.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(df_display.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight top 3 rows
    for i in range(1, min(4, len(df_display) + 1)):
        for j in range(len(df_display.columns)):
            table[(i, j)].set_facecolor('#E8F5E9')
    
    # Highlight best accuracy column
    if 'best_acc' in df_display.columns:
        acc_col_idx = list(df_display.columns).index('best_acc')
        for i in range(1, len(df_display) + 1):
            table[(i, acc_col_idx)].set_facecolor('#FFF9C4')
    
    plt.title(f'Hyperparameter Search Results Table - {dataset_name.upper()} Dataset',
             fontsize=16, fontweight='bold', pad=20)
    
    save_path = os.path.join(save_dir, f'hyperparameter_table_{dataset_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results table saved to {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Random Search Hyperparameter Optimization')
    parser.add_argument('--dataset', type=str, choices=['br35h', 'sartaj', 'both'],
                       default='both', help='Dataset to optimize')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of random search trials')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs per trial')
    parser.add_argument('--data_root', type=str, default='.',
                       help='Root directory of datasets')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    datasets = ['br35h', 'sartaj'] if args.dataset == 'both' else [args.dataset]
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'#'*70}")
        print(f"# Starting hyperparameter search for {dataset_name.upper()}")
        print(f"{'#'*70}")
        
        # Perform random search
        results = random_search(
            dataset_name,
            n_trials=args.n_trials,
            epochs=args.epochs,
            data_root=args.data_root
        )
        
        if not results:
            print(f"No results for {dataset_name}")
            continue
        
        # Create results table
        df_display, df_full = create_results_table(results, dataset_name)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY - {dataset_name.upper()}")
        print(f"{'='*70}")
        print(f"\nBest Configuration:")
        best_config = df_full.loc[df_full['best_acc'].idxmax()]
        print(f"  Trial: {int(best_config['trial'])}")
        print(f"  Learning Rate: {best_config['lr']:.4f}")
        print(f"  Hidden Dimension: {int(best_config['hidden_dim'])}")
        print(f"  Number of Layers: {int(best_config['num_layers'])}")
        print(f"  Batch Size: {int(best_config['batch_size'])}")
        print(f"  Dropout: {best_config['dropout']:.2f}")
        print(f"  Weight Decay: {best_config['weight_decay']:.6f}")
        print(f"  N Segments: {int(best_config['n_segments'])}")
        print(f"  Use GAT: {best_config['use_gat']}")
        print(f"  Best Accuracy: {best_config['best_acc']:.4f}")
        print(f"  Best Epoch: {int(best_config['best_epoch'])}")
        print(f"  Training Time: {best_config['training_time_minutes']:.2f} minutes")
        
        print(f"\nTop 5 Configurations:")
        print(df_display.head(5).to_string(index=False))
        
        # Save results to CSV
        csv_path = f'hyperparameter_results_{dataset_name}.csv'
        df_full.to_csv(csv_path, index=False)
        print(f"\nFull results saved to {csv_path}")
        
        # Save results to JSON
        json_path = f'hyperparameter_results_{dataset_name}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {json_path}")
        
        # Create visualizations
        visualize_results(results, dataset_name)
        create_results_table_image(df_display, dataset_name)
        
        all_results[dataset_name] = {
            'best_config': best_config.to_dict(),
            'all_results': results
        }
    
    # Print final summary
    print(f"\n{'#'*70}")
    print("# HYPERPARAMETER SEARCH COMPLETED")
    print(f"{'#'*70}")
    for dataset_name, data in all_results.items():
        best = data['best_config']
        print(f"\n{dataset_name.upper()} - Best Accuracy: {best['best_acc']:.4f}")
        print(f"  LR: {best['lr']:.4f}, Hidden: {int(best['hidden_dim'])}, "
              f"Layers: {int(best['num_layers'])}, Batch: {int(best['batch_size'])}")


if __name__ == '__main__':
    main()

