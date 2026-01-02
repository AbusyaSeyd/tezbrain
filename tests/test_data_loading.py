"""Quick test script to verify data loading works correctly."""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import Br35HDataset, SartajDataset
from torch_geometric.loader import DataLoader

def test_br35h():
    print("Testing br35h dataset...")
    try:
        dataset = Br35HDataset(root='br35h', split='train', n_segments=50)
        print(f"  Loaded {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample node features shape: {sample.x.shape}")
            print(f"  Sample edge index shape: {sample.edge_index.shape}")
            print(f"  Sample label: {sample.y.item()}")
            print("  OK br35h dataset loaded successfully!")
        else:
            print("  NO SAMPLES found in br35h dataset")
    except Exception as e:
        print(f"  ERROR loading br35h dataset: {e}")

def test_sartaj():
    print("Testing sartaj dataset...")
    try:
        dataset = SartajDataset(root='sartaj', split='train', n_segments=50)
        print(f"  Loaded {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample node features shape: {sample.x.shape}")
            print(f"  Sample edge index shape: {sample.edge_index.shape}")
            print(f"  Sample label: {sample.y.item()}")
            print("  OK sartaj dataset loaded successfully!")
        else:
            print("  NO SAMPLES found in sartaj dataset")
    except Exception as e:
        print(f"  ERROR loading sartaj dataset: {e}")

def test_dataloader():
    print("Testing DataLoader...")
    try:
        dataset = Br35HDataset(root='br35h', split='train', n_segments=50)
        if len(dataset) > 0:
            loader = DataLoader(dataset, batch_size=4, shuffle=False)
            batch = next(iter(loader))
            print(f"  Batch node features shape: {batch.x.shape}")
            print(f"  Batch edge index shape: {batch.edge_index.shape}")
            print(f"  Batch labels shape: {batch.y.shape}")
            print(f"  Batch size: {batch.batch.max().item() + 1}")
            print("  OK DataLoader works correctly!")
        else:
            print("  NO SAMPLES available to test DataLoader")
    except Exception as e:
        print(f"  ERROR testing DataLoader: {e}")

if __name__ == '__main__':
    print("=" * 50)
    print("Data Loading Test")
    print("=" * 50)
    test_br35h()
    print()
    test_sartaj()
    print()
    test_dataloader()
    print("=" * 50)

