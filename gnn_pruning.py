"""
GNN Pruning Module
Implements various pruning techniques for Graph Neural Networks.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Tuple
import copy


class GNNPruner:
    """Pruning utilities for GNN models."""
    
    def __init__(self, model: nn.Module, pruning_method: str = 'magnitude'):
        """
        Initialize pruner.
        
        Args:
            model: The GNN model to prune
            pruning_method: Pruning method ('magnitude', 'l1', 'ln', 'random')
        """
        self.model = model
        self.pruning_method = pruning_method
        self.pruned_layers = []
        
    def get_prunable_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get list of prunable layers (conv layers, linear layers, and GNN layers)."""
        prunable_layers = []
        for name, module in self.model.named_modules():
            # Skip if it's a container module without direct parameters
            if len(list(module.children())) > 0 and not isinstance(module, (nn.Sequential, nn.ModuleList)):
                continue
            
            # Try to prune if module has weight parameter
            try:
                if hasattr(module, 'weight'):
                    # Check if weight is a parameter (not a buffer)
                    params = dict(module.named_parameters(recurse=False))
                    if 'weight' in params:
                        prunable_layers.append((name, module))
            except:
                continue
        
        return prunable_layers
    
    def magnitude_prune(self, amount: float = 0.2, structured: bool = False) -> Dict:
        """
        Magnitude-based pruning.
        
        Args:
            amount: Fraction of parameters to prune (0.0 to 1.0)
            structured: If True, use structured pruning (remove entire channels/neurons)
        
        Returns:
            Dictionary with pruning statistics
        """
        prunable_layers = self.get_prunable_layers()
        total_params = 0
        pruned_params = 0
        
        for name, module in prunable_layers:
            try:
                # Only prune if module has weight parameter
                if not hasattr(module, 'weight') or module.weight is None:
                    continue
                
                # Check if weight is a parameter (not just a buffer)
                if 'weight' not in dict(module.named_parameters()):
                    continue
                
                if structured:
                    # Structured pruning: remove entire channels/neurons
                    if isinstance(module, nn.Linear):
                        prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                    elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                        prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                    else:
                        # For GNN layers, use unstructured pruning as fallback
                        prune.l1_unstructured(module, name='weight', amount=amount)
                else:
                    # Unstructured pruning: remove individual weights
                    prune.l1_unstructured(module, name='weight', amount=amount)
                
                # Count parameters (after pruning, check the actual weight tensor)
                weight = module.weight
                if weight is not None:
                    total_params += weight.numel()
                    # For pruned weights, we need to check the mask or actual zeros
                    if hasattr(module, 'weight_mask'):
                        # Weight has been masked
                        pruned_params += (module.weight_mask == 0).sum().item()
                    else:
                        # Check for zeros (after remove_masks)
                        pruned_params += (weight == 0).sum().item()
                
                # Add bias pruning if exists and is a parameter
                if hasattr(module, 'bias') and module.bias is not None:
                    if 'bias' in dict(module.named_parameters()):
                        try:
                            if structured and isinstance(module, nn.Linear):
                                prune.ln_structured(module, name='bias', amount=amount, n=2, dim=0)
                            else:
                                prune.l1_unstructured(module, name='bias', amount=amount)
                            
                            total_params += module.bias.numel()
                            if hasattr(module, 'bias_mask'):
                                pruned_params += (module.bias_mask == 0).sum().item()
                            else:
                                pruned_params += (module.bias == 0).sum().item()
                        except:
                            pass  # Skip if bias pruning fails
                
                self.pruned_layers.append(name)
            except Exception as e:
                # Skip layers that can't be pruned
                continue
        
        sparsity = pruned_params / total_params if total_params > 0 else 0.0
        
        return {
            'total_params': total_params,
            'pruned_params': pruned_params,
            'remaining_params': total_params - pruned_params,
            'sparsity': sparsity,
            'pruned_layers': self.pruned_layers
        }
    
    def iterative_prune(self, initial_amount: float = 0.1, final_amount: float = 0.9,
                       n_iterations: int = 5, structured: bool = False) -> List[Dict]:
        """
        Iterative pruning: gradually increase pruning amount.
        
        Args:
            initial_amount: Initial pruning amount
            final_amount: Final pruning amount
            n_iterations: Number of pruning iterations
            structured: If True, use structured pruning
        
        Returns:
            List of pruning statistics for each iteration
        """
        # Create a copy of the model for iterative pruning
        model_copy = copy.deepcopy(self.model)
        original_pruner = GNNPruner(model_copy, self.pruning_method)
        
        pruning_stats = []
        amounts = np.linspace(initial_amount, final_amount, n_iterations)
        
        for i, amount in enumerate(amounts):
            stats = original_pruner.magnitude_prune(amount=amount, structured=structured)
            stats['iteration'] = i + 1
            stats['pruning_amount'] = amount
            pruning_stats.append(stats)
            
            # Restore model to original state for next iteration
            if i < n_iterations - 1:
                model_copy = copy.deepcopy(self.model)
                original_pruner = GNNPruner(model_copy, self.pruning_method)
        
        return pruning_stats
    
    def remove_pruning_masks(self):
        """Remove pruning masks and make pruning permanent."""
        for name, module in self.model.named_modules():
            try:
                # Remove weight mask if exists
                if hasattr(module, 'weight') and hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
                
                # Remove bias mask if exists
                if hasattr(module, 'bias') and module.bias is not None and hasattr(module, 'bias_mask'):
                    prune.remove(module, 'bias')
            except:
                # Skip if removal fails (layer might not have been pruned)
                pass
    
    def get_model_size(self) -> Dict:
        """Calculate model size in parameters and MB."""
        total_params = 0
        trainable_params = 0
        
        for param in self.model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        # Estimate model size in MB (assuming float32, 4 bytes per parameter)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        # Count zero parameters (pruned)
        zero_params = 0
        for param in self.model.parameters():
            zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'zero_params': zero_params,
            'sparsity': sparsity,
            'model_size_mb': model_size_mb,
            'effective_params': total_params - zero_params,
            'compression_ratio': 1.0 / (1.0 - sparsity) if sparsity < 1.0 else float('inf')
        }
    
    def fine_tune_pruned_model(self, train_loader, optimizer, criterion, device, epochs: int = 10):
        """Fine-tune the pruned model."""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y.squeeze())
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += pred.eq(batch.y.squeeze()).sum().item()
                total += batch.y.size(0)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Fine-tuning Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {correct/total:.4f}")


def compare_models(original_model: nn.Module, pruned_model: nn.Module, 
                   test_loader, device, criterion) -> Dict:
    """
    Compare original and pruned models.
    
    Returns:
        Dictionary with comparison metrics
    """
    original_pruner = GNNPruner(original_model)
    pruned_pruner = GNNPruner(pruned_model)
    
    # Model sizes
    original_size = original_pruner.get_model_size()
    pruned_size = pruned_pruner.get_model_size()
    
    # Inference time
    import time
    
    # Original model inference
    original_model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _ = original_model(batch.x, batch.edge_index, batch.batch)
    original_inference_time = time.time() - start_time
    
    # Pruned model inference
    pruned_model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _ = pruned_model(batch.x, batch.edge_index, batch.batch)
    pruned_inference_time = time.time() - start_time
    
    # Accuracy
    original_acc = evaluate_model(original_model, test_loader, device, criterion)
    pruned_acc = evaluate_model(pruned_model, test_loader, device, criterion)
    
    return {
        'original': {
            'size_mb': original_size['model_size_mb'],
            'params': original_size['total_params'],
            'sparsity': original_size['sparsity'],
            'inference_time': original_inference_time,
            'accuracy': original_acc
        },
        'pruned': {
            'size_mb': pruned_size['model_size_mb'],
            'params': pruned_size['total_params'],
            'sparsity': pruned_size['sparsity'],
            'inference_time': pruned_inference_time,
            'accuracy': pruned_acc
        },
        'improvements': {
            'size_reduction': (1 - pruned_size['model_size_mb'] / original_size['model_size_mb']) * 100,
            'speedup': original_inference_time / pruned_inference_time if pruned_inference_time > 0 else 0,
            'accuracy_drop': original_acc - pruned_acc,
            'compression_ratio': original_size['total_params'] / pruned_size['effective_params'] if pruned_size['effective_params'] > 0 else 0
        }
    }


def evaluate_model(model: nn.Module, test_loader, device, criterion) -> float:
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


def prune_model_weights(model: nn.Module, pruning_amount: float = 0.3, 
                       structured: bool = False) -> nn.Module:
    """
    Prune model weights and return pruned model.
    
    Args:
        model: Model to prune
        pruning_amount: Fraction of weights to prune
        structured: Use structured pruning
    
    Returns:
        Pruned model
    """
    pruned_model = copy.deepcopy(model)
    pruner = GNNPruner(pruned_model)
    pruner.magnitude_prune(amount=pruning_amount, structured=structured)
    pruner.remove_pruning_masks()
    return pruned_model

