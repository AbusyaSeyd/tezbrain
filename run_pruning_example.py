"""
Quick example script to demonstrate GNN pruning and show results.
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import os

from data_loader import Br35HDataset, SartajDataset
from gnn_model import BrainTumorGNN
from gnn_pruning import GNNPruner, prune_model_weights
from prune_and_evaluate import evaluate_pruning_rates, visualize_pruning_results


def quick_pruning_demo(dataset_name='br35h', data_root='.'):
    """Quick demo of pruning on a small subset."""
    print("=" * 70)
    print(f"GNN PRUNING DEMONSTRATION - {dataset_name.upper()} Dataset")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load dataset
    if dataset_name == 'br35h':
        train_dataset = Br35HDataset(root=os.path.join(data_root, 'br35h'), 
                                    split='train', n_segments=50)  # Smaller for demo
        test_dataset = Br35HDataset(root=os.path.join(data_root, 'br35h'), 
                                   split='test', n_segments=50)
        num_classes = 2
    else:
        train_dataset = SartajDataset(root=os.path.join(data_root, 'sartaj'), 
                                     split='train', n_segments=50)
        test_dataset = SartajDataset(root=os.path.join(data_root, 'sartaj'), 
                                    split='test', n_segments=50)
        num_classes = 4
    
    # Use smaller subsets for faster demo
    train_indices = list(range(min(100, len(train_dataset))))
    test_indices = list(range(min(50, len(test_dataset))))
    
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)
    
    print(f"Train samples: {len(train_subset)}")
    print(f"Test samples: {len(test_subset)}")
    
    # Get model
    sample = train_dataset[0]
    input_dim = sample.x.shape[1]
    
    model = BrainTumorGNN(input_dim=input_dim, hidden_dim=32, 
                         num_classes=num_classes, num_layers=2)
    model = model.to(device)
    
    # Quick training (or load existing model)
    print("\nTraining baseline model (quick training for demo)...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(10):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()
    
    # Evaluate baseline
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
    baseline_acc = correct / total
    
    print(f"Baseline Accuracy: {baseline_acc:.4f}\n")
    
    # Get baseline model size
    pruner = GNNPruner(model)
    baseline_size = pruner.get_model_size()
    print(f"Baseline Model Size: {baseline_size['model_size_mb']:.2f} MB")
    print(f"Baseline Parameters: {baseline_size['total_params']:,}\n")
    
    # Test pruning at different rates
    print("=" * 70)
    print("TESTING DIFFERENT PRUNING RATES")
    print("=" * 70)
    
    pruning_rates = [0.2, 0.4, 0.6, 0.8]
    results = []
    
    for rate in pruning_rates:
        print(f"\nPruning at {rate:.0%}...")
        pruned_model = prune_model_weights(model, pruning_amount=rate, structured=False)
        pruned_model = pruned_model.to(device)
        
        # Evaluate pruned model
        pruned_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = pruned_model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += pred.eq(batch.y.squeeze()).sum().item()
                total += batch.y.size(0)
        pruned_acc = correct / total
        
        # Get pruned model size
        pruned_pruner = GNNPruner(pruned_model)
        pruned_size = pruned_pruner.get_model_size()
        
        compression_ratio = baseline_size['total_params'] / pruned_size['effective_params'] if pruned_size['effective_params'] > 0 else 0
        size_reduction = (1 - pruned_size['model_size_mb'] / baseline_size['model_size_mb']) * 100
        
        results.append({
            'rate': rate,
            'accuracy': pruned_acc,
            'size_mb': pruned_size['model_size_mb'],
            'sparsity': pruned_size['sparsity'],
            'compression': compression_ratio,
            'size_reduction': size_reduction
        })
        
        print(f"  Accuracy: {pruned_acc:.4f} (Drop: {baseline_acc - pruned_acc:.4f})")
        print(f"  Sparsity: {pruned_size['sparsity']:.2%}")
        print(f"  Model Size: {pruned_size['model_size_mb']:.2f} MB ({size_reduction:.1f}% reduction)")
        print(f"  Compression: {compression_ratio:.2f}x")
    
    # Print summary
    print("\n" + "=" * 70)
    print("PRUNING SUMMARY")
    print("=" * 70)
    print(f"{'Rate':<8} {'Accuracy':<12} {'Drop':<8} {'Size (MB)':<12} {'Reduction':<12} {'Compression':<12}")
    print("-" * 70)
    print(f"{'Baseline':<8} {baseline_acc:<12.4f} {'-':<8} {baseline_size['model_size_mb']:<12.2f} {'-':<12} {'1.00x':<12}")
    for r in results:
        print(f"{r['rate']:<8.0%} {r['accuracy']:<12.4f} {baseline_acc - r['accuracy']:<8.4f} "
              f"{r['size_mb']:<12.2f} {r['size_reduction']:<12.1f}% {r['compression']:<12.2f}x")
    
    print("\n" + "=" * 70)
    print("Best pruning rate (highest accuracy):")
    best = max(results, key=lambda x: x['accuracy'])
    print(f"  Rate: {best['rate']:.0%}")
    print(f"  Accuracy: {best['accuracy']:.4f}")
    print(f"  Size Reduction: {best['size_reduction']:.1f}%")
    print(f"  Compression: {best['compression']:.2f}x")
    
    return results, baseline_acc, baseline_size


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['br35h', 'sartaj'], default='br35h')
    parser.add_argument('--data_root', type=str, default='.')
    args = parser.parse_args()
    
    quick_pruning_demo(args.dataset, args.data_root)

