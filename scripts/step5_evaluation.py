"""
Step 5: Comparative Evaluation
Compares Baseline vs Pruned (40% sparsity) GNN models across:
1. Model Size (MB + zip compression)
2. Computational Complexity (FLOPs, Parameters)
3. Inference Speed (CPU/GPU latency)
"""
import os
import time
import tempfile
import zipfile
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

import matplotlib.pyplot as plt
import seaborn as sns

# --- Existing codebase imports ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import BrainTumorGNN, Br35HDataset, GNNPruner


# ==================== CONFIG ====================
BASELINE_WEIGHTS = "br35h_baseline_best.pth"
PRUNING_RATE = 0.4  # 40% sparsity (optimal from Step 4)

# System fixes (Windows)
N_SEGMENTS = 50
NUM_WORKERS = 0
BATCH_SIZE = 32

# Inference measurement
WARMUP_ITERS = 10
MEASURE_ITERS = 50

# Output
PLOT_PATH = "performance_comparison.png"


def load_baseline_model(device: torch.device, input_dim: int) -> BrainTumorGNN:
    """Load baseline model from checkpoint."""
    if not os.path.exists(BASELINE_WEIGHTS):
        raise FileNotFoundError(f"Baseline weights not found: {BASELINE_WEIGHTS}")

    state_dict = torch.load(BASELINE_WEIGHTS, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Unexpected checkpoint format.")

    model = BrainTumorGNN(
        input_dim=input_dim,
        hidden_dim=128,
        num_classes=2,
        dropout=0.129,
    )
    model.load_state_dict(state_dict)
    return model.to(device)


def create_pruned_model(baseline_model: nn.Module, device: torch.device) -> nn.Module:
    """Create pruned version of baseline model (40% sparsity)."""
    import copy
    
    pruned_model = copy.deepcopy(baseline_model)
    pruner = GNNPruner(pruned_model, pruning_method="magnitude")
    
    # Apply magnitude pruning (unstructured)
    pruner.magnitude_prune(amount=PRUNING_RATE, structured=False)
    pruner.remove_pruning_masks()
    
    return pruned_model.to(device)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total parameters and non-zero (active) parameters.
    Returns: (total_params, nonzero_params)
    """
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total, nonzero


def estimate_flops_manual(model: nn.Module, sample_data: Batch) -> Optional[int]:
    """
    Manual FLOPs estimation for GNN (fallback if thop fails).
    This is approximate - counts MACs in Linear layers.
    """
    try:
        total_flops = 0
        
        # Count FLOPs in GCNConv layers (approximation)
        for name, module in model.named_modules():
            if hasattr(module, 'lin') and isinstance(module.lin, nn.Linear):
                # GCNConv typically has module.lin (Linear transformation)
                in_features = module.lin.in_features
                out_features = module.lin.out_features
                # Approximate: num_edges * (2 * in_features * out_features)
                # Using num_nodes as proxy since we don't know exact computation
                num_nodes = sample_data.x.size(0)
                flops = 2 * num_nodes * in_features * out_features
                total_flops += flops
            
            elif isinstance(module, nn.Linear):
                # Regular Linear layers (classifier)
                in_features = module.in_features
                out_features = module.out_features
                batch_size = sample_data.num_graphs
                flops = 2 * batch_size * in_features * out_features
                total_flops += flops
        
        return total_flops if total_flops > 0 else None
    except Exception as e:
        print(f"  Manual FLOPs estimation failed: {e}")
        return None


def try_compute_flops_thop(model: nn.Module, sample_data: Batch, device: torch.device) -> Optional[int]:
    """
    Try to compute FLOPs using thop library.
    Falls back to manual estimation if thop fails (common with PyG models).
    """
    try:
        from thop import profile
        
        # Create dummy input matching the model signature
        x = sample_data.x.to(device)
        edge_index = sample_data.edge_index.to(device)
        batch = sample_data.batch.to(device)
        
        # Profile the model
        flops, params = profile(
            model,
            inputs=(x, edge_index, batch),
            verbose=False
        )
        return int(flops)
    
    except ImportError:
        print("  Note: thop not installed. Using manual FLOPs estimation.")
        return estimate_flops_manual(model, sample_data)
    
    except Exception as e:
        print(f"  thop failed (common with PyG): {e}")
        print("  Falling back to manual FLOPs estimation...")
        return estimate_flops_manual(model, sample_data)


def measure_model_size(model: nn.Module, name: str) -> Tuple[float, float]:
    """
    Measure model size in MB (raw .pth and compressed .zip).
    Returns: (raw_mb, compressed_mb)
    """
    # Create temp file and close it immediately (Windows fix)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pth')
    os.close(tmp_fd)  # Close file descriptor
    
    try:
        # Save model
        torch.save(model.state_dict(), tmp_path)
        raw_size = os.path.getsize(tmp_path)
        raw_mb = raw_size / (1024 * 1024)
        
        # Compress to zip
        zip_path = tmp_path + '.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(tmp_path, arcname='model.pth')
        
        zip_size = os.path.getsize(zip_path)
        zip_mb = zip_size / (1024 * 1024)
        
        return raw_mb, zip_mb
    
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if os.path.exists(zip_path):
            os.unlink(zip_path)


def measure_inference_latency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup: int = 10,
    measure: int = 50
) -> float:
    """
    Measure average inference latency (ms) per batch.
    Returns: average latency in milliseconds
    """
    model.eval()
    
    # Get first batch for repeated measurements
    sample_batch = next(iter(loader))
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            batch = sample_batch.to(device, non_blocking=True)
            _ = model(batch.x, batch.edge_index, batch.batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Measure
    timings = []
    with torch.no_grad():
        for _ in range(measure):
            batch = sample_batch.to(device, non_blocking=True)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            t0 = time.perf_counter()
            _ = model(batch.x, batch.edge_index, batch.batch)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)  # Convert to ms
    
    return float(np.mean(timings))


def create_comparison_plot(metrics: Dict, out_path: str) -> None:
    """
    Create grouped bar chart comparing Baseline vs Pruned.
    Metrics are normalized to baseline for visualization.
    """
    sns.set_style("whitegrid")
    
    categories = ['Parameters\n(M)', 'Model Size\n(MB)', 'Inference\nLatency (ms)']
    
    baseline_vals = [
        metrics['baseline']['params_m'],
        metrics['baseline']['size_mb'],
        metrics['baseline']['latency_ms']
    ]
    
    pruned_vals = [
        metrics['pruned']['params_m'],
        metrics['pruned']['size_mb'],
        metrics['pruned']['latency_ms']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(11, 6.5))
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#3498db')
    bars2 = ax.bar(x + width/2, pruned_vals, width, label='Pruned (40%)', color='#e74c3c')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Performance Comparison: Baseline vs Pruned Model (40% sparsity)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def print_results_table(metrics: Dict) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("COMPARATIVE EVALUATION RESULTS")
    print("=" * 100)
    
    print(f"\n{'Metric':<30} {'Baseline':<20} {'Pruned (40%)':<20} {'Improvement':<20}")
    print("-" * 100)
    
    # Parameters
    baseline_params = metrics['baseline']['params_m']
    pruned_params = metrics['pruned']['params_m']
    params_reduction = ((baseline_params - pruned_params) / baseline_params) * 100
    print(f"{'Parameters (M)':<30} {baseline_params:<20.3f} {pruned_params:<20.3f} {params_reduction:>18.1f}%")
    
    # Active Parameters (non-zero)
    baseline_active = metrics['baseline']['active_params_m']
    pruned_active = metrics['pruned']['active_params_m']
    print(f"{'Active Parameters (M)':<30} {baseline_active:<20.3f} {pruned_active:<20.3f} {'':<20}")
    
    # Sparsity
    baseline_sparsity = metrics['baseline']['sparsity'] * 100
    pruned_sparsity = metrics['pruned']['sparsity'] * 100
    print(f"{'Sparsity (%)':<30} {baseline_sparsity:<20.1f} {pruned_sparsity:<20.1f} {'':<20}")
    
    print("-" * 100)
    
    # Model Size (raw)
    baseline_size = metrics['baseline']['size_mb']
    pruned_size = metrics['pruned']['size_mb']
    size_reduction = ((baseline_size - pruned_size) / baseline_size) * 100
    print(f"{'Model Size (MB, raw)':<30} {baseline_size:<20.3f} {pruned_size:<20.3f} {size_reduction:>18.1f}%")
    
    # Model Size (compressed)
    baseline_zip = metrics['baseline']['size_zip_mb']
    pruned_zip = metrics['pruned']['size_zip_mb']
    zip_reduction = ((baseline_zip - pruned_zip) / baseline_zip) * 100
    print(f"{'Model Size (MB, zip)':<30} {baseline_zip:<20.3f} {pruned_zip:<20.3f} {zip_reduction:>18.1f}%")
    
    print("-" * 100)
    
    # FLOPs
    if metrics['baseline']['flops_m'] is not None:
        baseline_flops = metrics['baseline']['flops_m']
        pruned_flops = metrics['pruned']['flops_m']
        flops_reduction = ((baseline_flops - pruned_flops) / baseline_flops) * 100
        print(f"{'FLOPs (M, estimated)':<30} {baseline_flops:<20.1f} {pruned_flops:<20.1f} {flops_reduction:>18.1f}%")
    else:
        print(f"{'FLOPs (M, estimated)':<30} {'N/A':<20} {'N/A':<20} {'N/A':<20}")
    
    print("-" * 100)
    
    # Inference Latency (GPU)
    baseline_gpu = metrics['baseline']['latency_ms']
    pruned_gpu = metrics['pruned']['latency_ms']
    speedup_gpu = baseline_gpu / pruned_gpu if pruned_gpu > 0 else 0
    print(f"{'Inference Latency GPU (ms)':<30} {baseline_gpu:<20.2f} {pruned_gpu:<20.2f} {speedup_gpu:>17.2f}x")
    
    # Inference Latency (CPU)
    if 'latency_cpu_ms' in metrics['baseline']:
        baseline_cpu = metrics['baseline']['latency_cpu_ms']
        pruned_cpu = metrics['pruned']['latency_cpu_ms']
        speedup_cpu = baseline_cpu / pruned_cpu if pruned_cpu > 0 else 0
        print(f"{'Inference Latency CPU (ms)':<30} {baseline_cpu:<20.2f} {pruned_cpu:<20.2f} {speedup_cpu:>17.2f}x")
    
    print("=" * 100 + "\n")


def main() -> None:
    torch.multiprocessing.freeze_support()
    
    print("=" * 100)
    print("Step 5: Comparative Evaluation - Baseline vs Pruned (40% sparsity)")
    print("=" * 100)
    
    # Device setup
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    
    print(f"\nGPU Device: {device_gpu}")
    if device_gpu.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Load dataset (single batch for measurements)
    br35h_root = os.path.join("data", "br35h")
    test_dataset = Br35HDataset(root=br35h_root, split="test", n_segments=N_SEGMENTS)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    
    input_dim = test_dataset[0].x.shape[1]
    sample_batch = next(iter(test_loader))
    
    print(f"Input dimension: {input_dim}")
    print(f"Test batch size: {BATCH_SIZE}")
    
    # ==================== BASELINE MODEL ====================
    print("\n" + "-" * 100)
    print("Evaluating BASELINE model...")
    print("-" * 100)
    
    baseline_model = load_baseline_model(device_gpu, input_dim)
    
    baseline_total_params, baseline_active_params = count_parameters(baseline_model)
    baseline_sparsity = 1.0 - (baseline_active_params / baseline_total_params)
    
    print(f"  Total parameters: {baseline_total_params:,}")
    print(f"  Active parameters: {baseline_active_params:,}")
    print(f"  Sparsity: {baseline_sparsity * 100:.2f}%")
    
    baseline_size_mb, baseline_zip_mb = measure_model_size(baseline_model, "baseline")
    print(f"  Model size (raw): {baseline_size_mb:.3f} MB")
    print(f"  Model size (zip): {baseline_zip_mb:.3f} MB")
    
    baseline_flops = try_compute_flops_thop(baseline_model, sample_batch, device_gpu)
    if baseline_flops:
        print(f"  FLOPs (estimated): {baseline_flops / 1e6:.1f} M")
    else:
        print(f"  FLOPs: Could not estimate")
    
    print(f"  Measuring GPU latency ({MEASURE_ITERS} iterations)...")
    baseline_latency_gpu = measure_inference_latency(
        baseline_model, test_loader, device_gpu, WARMUP_ITERS, MEASURE_ITERS
    )
    print(f"  GPU Latency: {baseline_latency_gpu:.2f} ms/batch")
    
    print(f"  Measuring CPU latency ({MEASURE_ITERS} iterations)...")
    baseline_model_cpu = baseline_model.to(device_cpu)
    baseline_latency_cpu = measure_inference_latency(
        baseline_model_cpu, test_loader, device_cpu, WARMUP_ITERS, MEASURE_ITERS
    )
    print(f"  CPU Latency: {baseline_latency_cpu:.2f} ms/batch")
    
    # ==================== PRUNED MODEL ====================
    print("\n" + "-" * 100)
    print(f"Evaluating PRUNED model ({PRUNING_RATE * 100:.0f}% sparsity)...")
    print("-" * 100)
    
    # Reload baseline and apply pruning
    baseline_model = load_baseline_model(device_gpu, input_dim)
    pruned_model = create_pruned_model(baseline_model, device_gpu)
    
    pruned_total_params, pruned_active_params = count_parameters(pruned_model)
    pruned_sparsity = 1.0 - (pruned_active_params / pruned_total_params)
    
    print(f"  Total parameters: {pruned_total_params:,}")
    print(f"  Active parameters: {pruned_active_params:,}")
    print(f"  Sparsity: {pruned_sparsity * 100:.2f}%")
    
    pruned_size_mb, pruned_zip_mb = measure_model_size(pruned_model, "pruned")
    print(f"  Model size (raw): {pruned_size_mb:.3f} MB")
    print(f"  Model size (zip): {pruned_zip_mb:.3f} MB")
    
    pruned_flops = try_compute_flops_thop(pruned_model, sample_batch, device_gpu)
    if pruned_flops:
        print(f"  FLOPs (estimated): {pruned_flops / 1e6:.1f} M")
    else:
        print(f"  FLOPs: Could not estimate")
    
    print(f"  Measuring GPU latency ({MEASURE_ITERS} iterations)...")
    pruned_latency_gpu = measure_inference_latency(
        pruned_model, test_loader, device_gpu, WARMUP_ITERS, MEASURE_ITERS
    )
    print(f"  GPU Latency: {pruned_latency_gpu:.2f} ms/batch")
    
    print(f"  Measuring CPU latency ({MEASURE_ITERS} iterations)...")
    pruned_model_cpu = pruned_model.to(device_cpu)
    pruned_latency_cpu = measure_inference_latency(
        pruned_model_cpu, test_loader, device_cpu, WARMUP_ITERS, MEASURE_ITERS
    )
    print(f"  CPU Latency: {pruned_latency_cpu:.2f} ms/batch")
    
    # ==================== AGGREGATE RESULTS ====================
    metrics = {
        'baseline': {
            'params_m': baseline_total_params / 1e6,
            'active_params_m': baseline_active_params / 1e6,
            'sparsity': baseline_sparsity,
            'size_mb': baseline_size_mb,
            'size_zip_mb': baseline_zip_mb,
            'flops_m': baseline_flops / 1e6 if baseline_flops else None,
            'latency_ms': baseline_latency_gpu,
            'latency_cpu_ms': baseline_latency_cpu,
        },
        'pruned': {
            'params_m': pruned_total_params / 1e6,
            'active_params_m': pruned_active_params / 1e6,
            'sparsity': pruned_sparsity,
            'size_mb': pruned_size_mb,
            'size_zip_mb': pruned_zip_mb,
            'flops_m': pruned_flops / 1e6 if pruned_flops else None,
            'latency_ms': pruned_latency_gpu,
            'latency_cpu_ms': pruned_latency_cpu,
        }
    }
    
    # Print formatted table
    print_results_table(metrics)
    
    # Create visualization
    create_comparison_plot(metrics, PLOT_PATH)
    print(f"Saved comparison plot: {os.path.abspath(PLOT_PATH)}")


if __name__ == "__main__":
    main()








