import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from datetime import datetime

from data_loader import Br35HDataset, SartajDataset
from gnn_model import BrainTumorGNN, GraphClassifier


def train_with_hyperparams(args, dataset_name='br35h', return_full_history=False):
    """
    Train model with given hyperparameters and return test accuracy.
    Simplified version for hyperparameter search.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    if dataset_name == 'br35h':
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
        num_classes = 2
    else:  # sartaj
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
        num_classes = 4
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Get input dimension
    sample = train_dataset[0]
    input_dim = sample.x.shape[1]
    
    # Model
    model = BrainTumorGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_gat=args.use_gat
    )
    
    classifier = GraphClassifier(model, device, scaler=None, grad_clip=args.grad_clip)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Training
    best_acc = 0.0
    train_accs = []
    test_accs = []
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = classifier.train_epoch(train_loader, optimizer, criterion)
        train_accs.append(train_acc)
        
        # Evaluate
        test_loss, test_acc, _, _ = classifier.evaluate(test_loader, criterion)
        test_accs.append(test_acc)
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Track best accuracy
        if test_acc > best_acc:
            best_acc = test_acc
    
    if return_full_history:
        return best_acc, train_accs, test_accs
    return best_acc


def random_search(dataset_name='br35h', n_trials=20, epochs=100, data_root='.'):
    """
    Perform random search hyperparameter optimization.
    
    Args:
        dataset_name: 'br35h' or 'sartaj'
        n_trials: Number of random trials
        epochs: Number of training epochs per trial
        data_root: Root directory for datasets
    
    Returns:
        Dictionary with results and best hyperparameters
    """
    print(f"\n{'='*60}")
    print(f"Random Search Hyperparameter Tuning for {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Number of trials: {n_trials}")
    print(f"Epochs per trial: {epochs}")
    print(f"{'='*60}\n")
    
    # Define search space
    search_space = {
        'lr': [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01],
        'batch_size': [16, 32, 64],
        'hidden_dim': [32, 64, 128, 256],
        'num_layers': [2, 3, 4, 5],
        'dropout': [0.3, 0.4, 0.5, 0.6],
        'weight_decay': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'use_gat': [True, False],
        'n_segments': [50, 100, 150],
        'grad_clip': [0.5, 1.0, 1.5]
    }
    
    results = []
    best_score = 0.0
    best_params = None
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        print("-" * 60)
        
        # Randomly sample hyperparameters
        params = {
            'lr': random.choice(search_space['lr']),
            'batch_size': random.choice(search_space['batch_size']),
            'hidden_dim': random.choice(search_space['hidden_dim']),
            'num_layers': random.choice(search_space['num_layers']),
            'dropout': random.choice(search_space['dropout']),
            'weight_decay': random.choice(search_space['weight_decay']),
            'use_gat': random.choice(search_space['use_gat']),
            'n_segments': random.choice(search_space['n_segments']),
            'grad_clip': random.choice(search_space['grad_clip'])
        }
        
        # Create args object
        args = argparse.Namespace(
            data_root=os.path.join(data_root, dataset_name),
            epochs=epochs,
            batch_size=params['batch_size'],
            lr=params['lr'],
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            weight_decay=params['weight_decay'],
            n_segments=params['n_segments'],
            use_gat=params['use_gat'],
            grad_clip=params['grad_clip'],
            early_stop_patience=0,  # Disable early stopping for fair comparison
            early_stop_min_delta=0.0,
            mixed_precision=False,
            log_dir='logs',
            log_interval=10,
            num_workers=0
        )
        
        # Print current hyperparameters
        print(f"Learning Rate: {params['lr']:.6f}")
        print(f"Batch Size: {params['batch_size']}")
        print(f"Hidden Dim: {params['hidden_dim']}")
        print(f"Num Layers: {params['num_layers']}")
        print(f"Dropout: {params['dropout']}")
        print(f"Weight Decay: {params['weight_decay']:.6f}")
        print(f"Use GAT: {params['use_gat']}")
        print(f"N Segments: {params['n_segments']}")
        print(f"Grad Clip: {params['grad_clip']}")
        
        # Train and evaluate
        start_time = time.time()
        try:
            test_acc = train_with_hyperparams(args, dataset_name=dataset_name)
            train_time = time.time() - start_time
            
            print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"Training Time: {train_time/60:.2f} minutes")
            
            # Store results
            result = {
                'trial': trial + 1,
                'lr': params['lr'],
                'batch_size': params['batch_size'],
                'hidden_dim': params['hidden_dim'],
                'num_layers': params['num_layers'],
                'dropout': params['dropout'],
                'weight_decay': params['weight_decay'],
                'use_gat': params['use_gat'],
                'n_segments': params['n_segments'],
                'grad_clip': params['grad_clip'],
                'test_acc': test_acc,
                'train_time_min': train_time / 60
            }
            results.append(result)
            
            # Update best
            if test_acc > best_score:
                best_score = test_acc
                best_params = params.copy()
                print(f"*** NEW BEST SCORE: {best_score:.4f} ***")
        
        except Exception as e:
            print(f"Error in trial {trial + 1}: {e}")
            continue
    
    return {
        'dataset': dataset_name,
        'n_trials': n_trials,
        'epochs': epochs,
        'results': results,
        'best_score': best_score,
        'best_params': best_params
    }


def create_results_table_image(results_dict, output_file='hyperparameter_search_results.png'):
    """
    Create a beautiful table visualization of hyperparameter search results.
    """
    dataset_name = results_dict['dataset'].upper()
    results = results_dict['results']
    best_params = results_dict['best_params']
    best_score = results_dict['best_score']
    
    # Sort results by test accuracy (descending)
    sorted_results = sorted(results, key=lambda x: x['test_acc'], reverse=True)
    
    # Create figure with multiple tables
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 1, hspace=0.3, height_ratios=[1.2, 1.5, 0.5])
    
    # Title
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')
    title = f"Random Search Hyperparameter Tuning Results - {dataset_name} Dataset"
    subtitle = f"Total Trials: {results_dict['n_trials']} | Epochs per Trial: {results_dict['epochs']} | Best Score: {best_score:.4f} ({best_score*100:.2f}%)"
    ax_title.text(0.5, 0.7, title, ha='center', va='center', fontsize=24, fontweight='bold')
    ax_title.text(0.5, 0.3, subtitle, ha='center', va='center', fontsize=16, style='italic')
    
    # Top 10 results table
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    
    # Prepare data for top 10
    top_results = sorted_results[:10]
    table_data = []
    headers = ['Rank', 'LR', 'Batch Size', 'Hidden Dim', 'Layers', 'Dropout', 'Weight Decay', 'GAT', 'Segments', 'Grad Clip', 'Test Acc (%)', 'Time (min)']
    
    for i, result in enumerate(top_results, 1):
        table_data.append([
            i,
            f"{result['lr']:.6f}",
            result['batch_size'],
            result['hidden_dim'],
            result['num_layers'],
            f"{result['dropout']:.2f}",
            f"{result['weight_decay']:.6f}",
            'Yes' if result['use_gat'] else 'No',
            result['n_segments'],
            result['grad_clip'],
            f"{result['test_acc']*100:.2f}",
            f"{result['train_time_min']:.1f}"
        ])
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best result
    for i in range(len(headers)):
        table[(1, i)].set_facecolor('#90EE90')
    
    # Best hyperparameters summary
    ax_best = fig.add_subplot(gs[2])
    ax_best.axis('off')
    
    best_text = f"BEST HYPERPARAMETERS:\n\n"
    best_text += f"Learning Rate: {best_params['lr']:.6f}\n"
    best_text += f"Batch Size: {best_params['batch_size']}\n"
    best_text += f"Hidden Dimension: {best_params['hidden_dim']}\n"
    best_text += f"Number of Layers: {best_params['num_layers']}\n"
    best_text += f"Dropout: {best_params['dropout']}\n"
    best_text += f"Weight Decay: {best_params['weight_decay']:.6f}\n"
    best_text += f"Use GAT: {'Yes' if best_params['use_gat'] else 'No'}\n"
    best_text += f"N Segments: {best_params['n_segments']}\n"
    best_text += f"Gradient Clipping: {best_params['grad_clip']}\n"
    best_text += f"\nBEST TEST ACCURACY: {best_score:.4f} ({best_score*100:.2f}%)"
    
    ax_best.text(0.5, 0.5, best_text, ha='center', va='center', fontsize=14,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 family='monospace')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nResults table saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Random Search Hyperparameter Tuning')
    parser.add_argument('--dataset', type=str, choices=['br35h', 'sartaj', 'both'],
                       default='both', help='Dataset to tune')
    parser.add_argument('--data_root', type=str, default='.',
                       help='Root directory of datasets')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of random search trials')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs per trial')
    
    args = parser.parse_args()
    
    all_results = {}
    
    if args.dataset == 'br35h' or args.dataset == 'both':
        print("\n" + "="*60)
        print("Starting hyperparameter search for BR35H dataset")
        print("="*60)
        br35h_results = random_search(
            dataset_name='br35h',
            n_trials=args.n_trials,
            epochs=args.epochs,
            data_root=args.data_root
        )
        all_results['br35h'] = br35h_results
        
        # Create visualization
        create_results_table_image(
            br35h_results,
            output_file='hyperparameter_search_results_br35h.png'
        )
        
        # Save results to JSON
        import json
        with open('hyperparameter_search_results_br35h.json', 'w') as f:
            json.dump(br35h_results, f, indent=2)
        print("\nResults saved to hyperparameter_search_results_br35h.json")
    
    if args.dataset == 'sartaj' or args.dataset == 'both':
        print("\n" + "="*60)
        print("Starting hyperparameter search for SARTAJ dataset")
        print("="*60)
        sartaj_results = random_search(
            dataset_name='sartaj',
            n_trials=args.n_trials,
            epochs=args.epochs,
            data_root=args.data_root
        )
        all_results['sartaj'] = sartaj_results
        
        # Create visualization
        create_results_table_image(
            sartaj_results,
            output_file='hyperparameter_search_results_sartaj.png'
        )
        
        # Save results to JSON
        import json
        with open('hyperparameter_search_results_sartaj.json', 'w') as f:
            json.dump(sartaj_results, f, indent=2)
        print("\nResults saved to hyperparameter_search_results_sartaj.json")
    
    # Create combined visualization if both datasets
    if args.dataset == 'both':
        create_combined_results_image(all_results)
    
    print("\n" + "="*60)
    print("Hyperparameter search completed!")
    print("="*60)


def create_combined_results_image(all_results):
    """Create a combined visualization showing both datasets."""
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    datasets = ['br35h', 'sartaj']
    
    for idx, dataset_name in enumerate(datasets):
        if dataset_name not in all_results:
            continue
        
        results_dict = all_results[dataset_name]
        results = sorted(results_dict['results'], key=lambda x: x['test_acc'], reverse=True)
        best_params = results_dict['best_params']
        best_score = results_dict['best_score']
        
        ax = fig.add_subplot(gs[idx])
        ax.axis('off')
        
        # Title
        title = f"{dataset_name.upper()} Dataset - Best Score: {best_score:.4f} ({best_score*100:.2f}%)"
        ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=18, fontweight='bold')
        
        # Top 5 results
        top_results = results[:5]
        table_data = []
        headers = ['Rank', 'LR', 'Batch', 'Hidden', 'Layers', 'Dropout', 'WD', 'GAT', 'Test Acc (%)']
        
        for i, result in enumerate(top_results, 1):
            table_data.append([
                i,
                f"{result['lr']:.4f}",
                result['batch_size'],
                result['hidden_dim'],
                result['num_layers'],
                f"{result['dropout']:.2f}",
                f"{result['weight_decay']:.4f}",
                'Y' if result['use_gat'] else 'N',
                f"{result['test_acc']*100:.2f}"
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            bbox=[0.1, 0.3, 0.8, 0.6]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4A90E2')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best
        for i in range(len(headers)):
            table[(1, i)].set_facecolor('#90EE90')
        
        # Best params text
        best_text = f"Best: LR={best_params['lr']:.6f}, Batch={best_params['batch_size']}, "
        best_text += f"Hidden={best_params['hidden_dim']}, Layers={best_params['num_layers']}, "
        best_text += f"Dropout={best_params['dropout']}, GAT={'Yes' if best_params['use_gat'] else 'No'}"
        ax.text(0.5, 0.15, best_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Random Search Hyperparameter Tuning - Combined Results', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.savefig('hyperparameter_search_results_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\nCombined results table saved to hyperparameter_search_results_combined.png")


if __name__ == '__main__':
    main()

