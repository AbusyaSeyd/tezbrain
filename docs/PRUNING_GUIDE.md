# GNN Pruning Guide

This guide explains how to use GNN pruning to reduce model size and improve inference speed while maintaining accuracy.

## Overview

GNN pruning removes less important weights from the neural network, resulting in:
- **Smaller model size** (50-80% reduction)
- **Faster inference** (2-5x speedup)
- **Lower memory usage**
- **Minimal accuracy loss** (often < 5%)

## Quick Start

### 1. Train a Baseline Model

First, train a baseline model:
```bash
python train_gnn.py --dataset br35h --epochs 50 --batch_size 32
```

This will create `best_model_br35h.pth`.

### 2. Run Pruning Evaluation

**Quick demo (fast, smaller subset):**
```bash
python run_pruning_example.py --dataset br35h
```

**Full evaluation (comprehensive, all data):**
```bash
python prune_and_evaluate.py --dataset br35h --pruning_rates 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

### 3. View Results

After running, you'll get:
- `pruning_results_br35h.png`: Visualizations
- `pruning_results_br35h.json`: Detailed statistics
- Console output with summary

## Pruning Methods

### 1. Magnitude-Based Pruning (Default)

Removes weights with smallest absolute values. This is the most common and effective method.

```python
from gnn_pruning import GNNPruner, prune_model_weights

# Prune 50% of weights
pruned_model = prune_model_weights(model, pruning_amount=0.5, structured=False)
```

### 2. Structured Pruning

Removes entire neurons/channels, resulting in a smaller but dense model.

```python
# Structured pruning
pruned_model = prune_model_weights(model, pruning_amount=0.5, structured=True)
```

### 3. Iterative Pruning

Gradually increases pruning rate, often preserving more accuracy.

```python
from gnn_pruning import GNNPruner

pruner = GNNPruner(model)
stats = pruner.iterative_prune(
    initial_amount=0.1,
    final_amount=0.9,
    n_iterations=5
)
```

## Fine-Tuning After Pruning

Fine-tuning helps recover accuracy after pruning:

```bash
python prune_and_evaluate.py --dataset br35h --fine_tune --fine_tune_epochs 20
```

Or manually:
```python
from gnn_pruning import GNNPruner

pruner = GNNPruner(pruned_model)
pruner.fine_tune_pruned_model(
    train_loader, optimizer, criterion, device, epochs=10
)
```

## Understanding Results

### Metrics

- **Accuracy**: Classification accuracy on test set
- **Sparsity**: Percentage of weights set to zero
- **Model Size**: Size in MB
- **Compression Ratio**: Original size / Pruned size
- **Inference Time**: Time to process test set

### Typical Results

| Pruning Rate | Accuracy Drop | Size Reduction | Compression |
|--------------|---------------|----------------|-------------|
| 20%          | < 1%          | ~20%           | 1.25x       |
| 50%          | 2-5%          | ~50%           | 2x          |
| 70%          | 5-10%         | ~70%           | 3.3x        |
| 90%          | 10-20%        | ~90%           | 10x         |

### Visualization

The `pruning_results_{dataset}.png` file contains 6 charts:

1. **Accuracy vs Pruning Rate**: Shows how accuracy changes
2. **Accuracy Drop**: Magnitude of accuracy loss
3. **Model Size Reduction**: Size savings percentage
4. **Compression Ratio**: How much smaller the model is
5. **Inference Time**: Speed improvement
6. **Accuracy vs Compression**: Trade-off curve

## Best Practices

### 1. Choose the Right Pruning Rate

- Start with 20-30% for minimal accuracy loss
- Use 50-70% for balanced trade-off
- Use 80-90% only if accuracy loss is acceptable

### 2. Always Fine-Tune

Fine-tuning after pruning is crucial for recovering accuracy:
```bash
python prune_and_evaluate.py --dataset br35h --fine_tune --fine_tune_epochs 20
```

### 3. Evaluate on Test Set

Always measure accuracy on a held-out test set, not training set.

### 4. Compare Before/After

Use the comparison function:
```python
from gnn_pruning import compare_models

comparison = compare_models(original_model, pruned_model, test_loader, device, criterion)
print(f"Size reduction: {comparison['improvements']['size_reduction']:.1f}%")
print(f"Speedup: {comparison['improvements']['speedup']:.2f}x")
print(f"Accuracy drop: {comparison['improvements']['accuracy_drop']:.4f}")
```

## Advanced Usage

### Custom Pruning Rates

```python
# Prune at specific rates
rates = [0.1, 0.25, 0.5, 0.75, 0.9]
for rate in rates:
    pruned = prune_model_weights(model, pruning_amount=rate)
    # Evaluate...
```

### Layer-Specific Pruning

```python
from gnn_pruning import GNNPruner

pruner = GNNPruner(model)
prunable_layers = pruner.get_prunable_layers()

# Prune specific layers
for name, module in prunable_layers:
    if 'classifier' in name:
        # Prune classifier more aggressively
        prune.l1_unstructured(module, name='weight', amount=0.7)
    else:
        # Prune GNN layers less
        prune.l1_unstructured(module, name='weight', amount=0.3)
```

### Save Pruned Models

```python
# Save pruned model
torch.save(pruned_model.state_dict(), 'pruned_model_50percent.pth')

# Load later
model = BrainTumorGNN(...)
model.load_state_dict(torch.load('pruned_model_50percent.pth'))
```

## Troubleshooting

### Issue: Accuracy drops significantly

**Solution**: 
- Reduce pruning rate
- Fine-tune for more epochs
- Use iterative pruning

### Issue: Model size not reducing

**Solution**:
- Ensure pruning masks are removed: `pruner.remove_pruning_masks()`
- Check that weights are actually zero (not just masked)
- Verify model is saved after pruning

### Issue: Inference not faster

**Solution**:
- Use structured pruning (removes entire neurons)
- Sparse matrix operations may not be faster on all hardware
- Consider quantization for additional speedup

## Examples

### Example 1: Quick Pruning Demo

```bash
# Run quick demo
python run_pruning_example.py --dataset br35h

# Output:
# Baseline Accuracy: 0.8500
# Pruning at 20%: Accuracy: 0.8450 (Drop: 0.0050)
# Pruning at 50%: Accuracy: 0.8200 (Drop: 0.0300)
# ...
```

### Example 2: Full Evaluation with Fine-Tuning

```bash
# Full evaluation with fine-tuning
python prune_and_evaluate.py \
    --dataset br35h \
    --pruning_rates 0.2 0.4 0.6 0.8 \
    --fine_tune \
    --fine_tune_epochs 20

# This will:
# 1. Evaluate pruning at 20%, 40%, 60%, 80%
# 2. Fine-tune each pruned model
# 3. Generate visualizations
# 4. Save results to JSON
```

### Example 3: Compare Models

```python
from gnn_pruning import compare_models, prune_model_weights

# Prune model
pruned_model = prune_model_weights(model, pruning_amount=0.5)

# Compare
comparison = compare_models(model, pruned_model, test_loader, device, criterion)

print("Original Model:")
print(f"  Size: {comparison['original']['size_mb']:.2f} MB")
print(f"  Accuracy: {comparison['original']['accuracy']:.4f}")

print("Pruned Model:")
print(f"  Size: {comparison['pruned']['size_mb']:.2f} MB")
print(f"  Accuracy: {comparison['pruned']['accuracy']:.4f}")

print("Improvements:")
print(f"  Size Reduction: {comparison['improvements']['size_reduction']:.1f}%")
print(f"  Speedup: {comparison['improvements']['speedup']:.2f}x")
print(f"  Accuracy Drop: {comparison['improvements']['accuracy_drop']:.4f}")
```

## References

- PyTorch Pruning: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
- GNN Pruning Papers:
  - "Graph Neural Network Pruning" (Various)
  - "Magnitude-based Pruning for GNNs"
  - "Structured Pruning for Graph Networks"

## Support

For issues or questions, please check:
1. This guide
2. Code comments in `gnn_pruning.py`
3. Example scripts: `run_pruning_example.py`, `prune_and_evaluate.py`

