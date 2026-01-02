"""
Pruning and Evaluation Script
Prunes GNN models and compares results.
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from typing import List
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import Br35HDataset, SartajDataset, BrainTumorGNN, GraphClassifier, GNNPruner, compare_models, prune_model_weights, prepare_artifact_dirs

def evaluate_model(model, test_loader, device, criterion):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += pred.eq(batch.y.squeeze()).sum().item()
            total += batch.y.size(0)
    
    return correct / total if total > 0 else 0.0


def evaluate_pruning_rates(model, train_loader, test_loader, device, criterion, 
                           dataset_name: str, pruning_rates: List[float] = None):
    """Evaluate model at different pruning rates."""
    if pruning_rates is None:
        pruning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = {
        'pruning_rates': [],
        'accuracies': [],
        'sparsities': [],
        'model_sizes': [],
        'inference_times': [],
        'compression_ratios': []
    }
    
    original_pruner = GNNPruner(model)
    original_size = original_pruner.get_model_size()
    original_acc = evaluate_model(model, test_loader, device, criterion)
    
    print(f"\nOriginal Model:")
    print(f"  Accuracy: {original_acc:.4f}")
    print(f"  Model Size: {original_size['model_size_mb']:.2f} MB")
    print(f"  Parameters: {original_size['total_params']:,}")
    print(f"\nEvaluating different pruning rates...")
    
    import time
    
    for pruning_rate in tqdm(pruning_rates, desc="Pruning"):
        # Create pruned model
        pruned_model = prune_model_weights(model, pruning_amount=pruning_rate, structured=False)
        pruned_pruner = GNNPruner(pruned_model)
        pruned_size = pruned_pruner.get_model_size()
        
        # Evaluate accuracy
        acc = evaluate_model(pruned_model, test_loader, device, criterion)
        
        # Measure inference time
        pruned_model.eval()
        start_time = time.time()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                _ = pruned_model(batch.x, batch.edge_index, batch.batch)
        inference_time = time.time() - start_time
        
        compression_ratio = original_size['total_params'] / pruned_size['effective_params'] if pruned_size['effective_params'] > 0 else 0
        
        results['pruning_rates'].append(pruning_rate)
        results['accuracies'].append(acc)
        results['sparsities'].append(pruned_size['sparsity'])
        results['model_sizes'].append(pruned_size['model_size_mb'])
        results['inference_times'].append(inference_time)
        results['compression_ratios'].append(compression_ratio)
        
        print(f"\nPruning Rate: {pruning_rate:.1%}")
        print(f"  Accuracy: {acc:.4f} (Drop: {original_acc - acc:.4f})")
        print(f"  Sparsity: {pruned_size['sparsity']:.2%}")
        print(f"  Model Size: {pruned_size['model_size_mb']:.2f} MB")
        print(f"  Compression Ratio: {compression_ratio:.2f}x")
    
    return results, original_acc, original_size


def visualize_pruning_results(results: dict, original_acc: float, original_size: dict, 
                              dataset_name: str, save_path: str = None):
    """Visualize pruning results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'GNN Pruning Results - {dataset_name}', fontsize=16, fontweight='bold')
    
    pruning_rates = np.array(results['pruning_rates'])
    accuracies = np.array(results['accuracies'])
    sparsities = np.array(results['sparsities'])
    model_sizes = np.array(results['model_sizes'])
    inference_times = np.array(results['inference_times'])
    compression_ratios = np.array(results['compression_ratios'])
    
    # 1. Accuracy vs Pruning Rate
    ax = axes[0, 0]
    ax.plot(pruning_rates * 100, accuracies * 100, 'b-o', linewidth=2, markersize=8, label='Pruned Model')
    ax.axhline(y=original_acc * 100, color='r', linestyle='--', linewidth=2, label='Original Model')
    ax.set_xlabel('Pruning Rate (%)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Pruning Rate', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    
    # 2. Accuracy Drop vs Pruning Rate
    ax = axes[0, 1]
    accuracy_drops = (original_acc - accuracies) * 100
    ax.plot(pruning_rates * 100, accuracy_drops, 'r-o', linewidth=2, markersize=8)
    ax.set_xlabel('Pruning Rate (%)', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('Accuracy Drop vs Pruning Rate', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    
    # 3. Model Size Reduction
    ax = axes[0, 2]
    size_reduction = (1 - model_sizes / original_size['model_size_mb']) * 100
    ax.plot(pruning_rates * 100, size_reduction, 'g-o', linewidth=2, markersize=8)
    ax.set_xlabel('Pruning Rate (%)', fontsize=12)
    ax.set_ylabel('Size Reduction (%)', fontsize=12)
    ax.set_title('Model Size Reduction', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    
    # 4. Compression Ratio
    ax = axes[1, 0]
    ax.plot(pruning_rates * 100, compression_ratios, 'm-o', linewidth=2, markersize=8)
    ax.set_xlabel('Pruning Rate (%)', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('Compression Ratio vs Pruning Rate', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    
    # 5. Inference Time
    ax = axes[1, 1]
    ax.plot(pruning_rates * 100, inference_times, 'c-o', linewidth=2, markersize=8, label='Pruned')
    ax.axhline(y=inference_times[0], color='r', linestyle='--', linewidth=2, label='Original (approx)')
    ax.set_xlabel('Pruning Rate (%)', fontsize=12)
    ax.set_ylabel('Inference Time (s)', fontsize=12)
    ax.set_title('Inference Time vs Pruning Rate', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    
    # 6. Accuracy vs Compression Ratio (Trade-off)
    ax = axes[1, 2]
    scatter = ax.scatter(compression_ratios, accuracies * 100, c=pruning_rates * 100, 
                        cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    ax.set_xlabel('Compression Ratio', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Compression Ratio', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Pruning Rate (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nResults saved to {save_path}")
    else:
        plt.savefig(f'pruning_results_{dataset_name}.png', dpi=300, bbox_inches='tight')
        print(f"\nResults saved to pruning_results_{dataset_name}.png")
    
    plt.close()


def fine_tune_and_evaluate(original_model, pruned_model, train_loader, test_loader,
                          device, criterion, epochs: int = 20, lr: float = 0.0001):
    """Fine-tune pruned model and evaluate."""
    print(f"\nFine-tuning pruned model for {epochs} epochs...")
    
    optimizer = torch.optim.Adam(pruned_model.parameters(), lr=lr, weight_decay=5e-4)
    
    best_acc = 0.0
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        # Train
        pruned_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = pruned_model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.squeeze())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += pred.eq(batch.y.squeeze()).sum().item()
            total += batch.y.size(0)
        
        train_acc = correct / total
        
        # Evaluate
        test_acc = evaluate_model(pruned_model, test_loader, device, criterion)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    return pruned_model, best_acc, train_accs, test_accs


def main():
    parser = argparse.ArgumentParser(description='Prune and Evaluate GNN Models')
    parser.add_argument('--dataset', type=str, choices=['br35h', 'sartaj'],
                       default='br35h', help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Root directory of datasets')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model (if None, will train new model)')
    parser.add_argument('--pruning_rates', type=float, nargs='+',
                       default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       help='Pruning rates to evaluate')
    parser.add_argument('--fine_tune', action='store_true',
                       help='Fine-tune pruned models')
    parser.add_argument('--fine_tune_epochs', type=int, default=20,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--n_segments', type=int, default=100,
                       help='Number of superpixels')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--artifact_dir', type=str, default='artifacts',
                       help='Base directory to store models, metrics, and plots')
    
    args = parser.parse_args()
    artifact_dirs = prepare_artifact_dirs(args.artifact_dir)
    models_dir = artifact_dirs["models"]
    metrics_dir = artifact_dirs["metrics"]
    plots_dir = artifact_dirs["plots"]
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    if args.dataset == 'br35h':
        train_dataset = Br35HDataset(root=os.path.join(args.data_root, 'br35h'),
                                    split='train', n_segments=args.n_segments)
        test_dataset = Br35HDataset(root=os.path.join(args.data_root, 'br35h'),
                                   split='test', n_segments=args.n_segments)
        num_classes = 2
    else:
        train_dataset = SartajDataset(root=os.path.join(args.data_root, 'sartaj'),
                                     split='train', n_segments=args.n_segments)
        test_dataset = SartajDataset(root=os.path.join(args.data_root, 'sartaj'),
                                    split='test', n_segments=args.n_segments)
        num_classes = 4
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get input dimension
    sample = train_dataset[0]
    input_dim = sample.x.shape[1]
    
    # Load or create model
    model_path = Path(args.model_path) if args.model_path else None
    if model_path and model_path.exists():
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Detected checkpoint file; using 'model_state_dict'.")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Detected checkpoint file; using 'state_dict'.")
            elif all(isinstance(k, str) and (k.startswith('convs') or k.startswith('batch_norms') or k.startswith('classifier'))
                     for k in checkpoint.keys()):
                state_dict = checkpoint
            else:
                raise RuntimeError(
                    "Unrecognized checkpoint format. Expected raw state_dict or keys "
                    "like 'model_state_dict'/'state_dict'."
                )
            ckpt_args = checkpoint.get('args')
            if ckpt_args is not None and not isinstance(ckpt_args, dict):
                ckpt_args = vars(ckpt_args)
        else:
            state_dict = checkpoint
            ckpt_args = None

        ckpt_hidden_dim = ckpt_args.get('hidden_dim') if isinstance(ckpt_args, dict) else None
        ckpt_num_layers = ckpt_args.get('num_layers') if isinstance(ckpt_args, dict) else None

        hidden_dim = args.hidden_dim
        if ckpt_hidden_dim and ckpt_hidden_dim != hidden_dim:
            print(f"Overriding hidden_dim {hidden_dim} with checkpoint value {ckpt_hidden_dim}.")
            hidden_dim = ckpt_hidden_dim

        num_layers = args.num_layers
        if ckpt_num_layers and ckpt_num_layers != num_layers:
            print(f"Overriding num_layers {num_layers} with checkpoint value {ckpt_num_layers}.")
            num_layers = ckpt_num_layers

        model = BrainTumorGNN(input_dim=input_dim, hidden_dim=hidden_dim,
                             num_classes=num_classes, num_layers=num_layers)
        model.load_state_dict(state_dict)
        model = model.to(device)
    else:
        print("Training new model...")
        model = BrainTumorGNN(input_dim=input_dim, hidden_dim=args.hidden_dim,
                             num_classes=num_classes, num_layers=args.num_layers)
        classifier = GraphClassifier(model, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        
        # Quick training
        print("Training for 30 epochs...")
        for epoch in range(30):
            classifier.train_epoch(train_loader, optimizer, criterion)
            if (epoch + 1) % 10 == 0:
                test_loss, test_acc, _, _ = classifier.evaluate(test_loader, criterion)
                print(f"  Epoch {epoch+1}/30, Test Acc: {test_acc:.4f}")
        
        # Save model
        model_path = models_dir / f'baseline_model_{args.dataset}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate pruning at different rates
    print("\n" + "=" * 60)
    print("PRUNING EVALUATION")
    print("=" * 60)
    
    results, original_acc, original_size = evaluate_pruning_rates(
        model, train_loader, test_loader, device, criterion,
        args.dataset, args.pruning_rates
    )
    
    # Fine-tune if requested
    if args.fine_tune:
        print("\n" + "=" * 60)
        print("FINE-TUNING PRUNED MODELS")
        print("=" * 60)
        
        fine_tuned_results = {
            'pruning_rates': [],
            'accuracies': [],
            'original_accuracies': results['accuracies']
        }
        
        for pruning_rate in args.pruning_rates:
            print(f"\nFine-tuning model pruned at {pruning_rate:.1%}...")
            pruned_model = prune_model_weights(model, pruning_amount=pruning_rate)
            pruned_model = pruned_model.to(device)
            
            fine_tuned_model, best_acc, _, _ = fine_tune_and_evaluate(
                model, pruned_model, train_loader, test_loader,
                device, criterion, epochs=args.fine_tune_epochs
            )
            
            fine_tuned_results['pruning_rates'].append(pruning_rate)
            fine_tuned_results['accuracies'].append(best_acc)
        
        # Update results with fine-tuned accuracies
        results['fine_tuned_accuracies'] = fine_tuned_results['accuracies']
    
    # Visualize results
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    results_plot_path = plots_dir / f'pruning_results_{args.dataset}.png'
    visualize_pruning_results(results, original_acc, original_size, args.dataset, save_path=results_plot_path)
    
    # Save results to JSON
    results_summary = {
        'dataset': args.dataset,
        'original_accuracy': float(original_acc),
        'original_size_mb': float(original_size['model_size_mb']),
        'original_params': int(original_size['total_params']),
        'pruning_results': {
            'rates': [float(r) for r in results['pruning_rates']],
            'accuracies': [float(a) for a in results['accuracies']],
            'sparsities': [float(s) for s in results['sparsities']],
            'model_sizes_mb': [float(s) for s in results['model_sizes']],
            'compression_ratios': [float(c) for c in results['compression_ratios']]
        }
    }
    
    if args.fine_tune:
        results_summary['pruning_results']['fine_tuned_accuracies'] = [
            float(a) for a in results['fine_tuned_accuracies']
        ]
    
    results_json_path = metrics_dir / f'pruning_results_{args.dataset}.json'
    with open(results_json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults summary saved to {results_json_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original Model Accuracy: {original_acc:.4f}")
    print(f"Original Model Size: {original_size['model_size_mb']:.2f} MB")
    print(f"\nBest Pruning Results:")
    
    best_idx = np.argmax(results['accuracies'])
    print(f"  Pruning Rate: {results['pruning_rates'][best_idx]:.1%}")
    print(f"  Accuracy: {results['accuracies'][best_idx]:.4f}")
    print(f"  Accuracy Drop: {original_acc - results['accuracies'][best_idx]:.4f}")
    print(f"  Compression Ratio: {results['compression_ratios'][best_idx]:.2f}x")
    print(f"  Model Size: {results['model_sizes'][best_idx]:.2f} MB")
    print(f"  Size Reduction: {(1 - results['model_sizes'][best_idx] / original_size['model_size_mb']) * 100:.1f}%")


if __name__ == '__main__':
    main()

