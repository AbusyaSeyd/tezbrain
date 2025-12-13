"""Quick start script to train GNN on both datasets with default parameters."""
import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("Graph Neural Network for Brain Tumor Classification")
    print("=" * 60)
    print()
    
    # Check if datasets exist
    if not os.path.exists('br35h'):
        print("ERROR: br35h dataset not found!")
        print("Please ensure the br35h folder exists in the current directory.")
        return
    
    if not os.path.exists('sartaj'):
        print("ERROR: sartaj dataset not found!")
        print("Please ensure the sartaj folder exists in the current directory.")
        return
    
    print("Datasets found! Starting training...")
    print()
    
    # Ask user which dataset to train on
    print("Which dataset would you like to train on?")
    print("1. br35h (binary classification)")
    print("2. sartaj (multi-class classification)")
    print("3. both")
    choice = input("Enter choice (1/2/3): ").strip()
    
    dataset_map = {
        '1': 'br35h',
        '2': 'sartaj',
        '3': 'both'
    }
    
    dataset = dataset_map.get(choice, 'both')
    
    print(f"\nTraining on {dataset} dataset...")
    print("This may take a while. Please wait...")
    print()
    
    # Run training
    cmd = [
        sys.executable,
        'train_gnn.py',
        '--dataset', dataset,
        '--epochs', '50',
        '--batch_size', '32',
        '--lr', '0.001',
        '--n_segments', '100'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        print("\nCheck the following files for results:")
        artifact_dir = Path('artifacts')
        models_dir = artifact_dir / 'models'
        plots_dir = artifact_dir / 'plots'
        if dataset in ['br35h', 'both']:
            print(f"  - {models_dir / 'best_model_br35h.pth'}")
            print(f"  - {plots_dir / 'confusion_matrix_br35h.png'}")
            print(f"  - {plots_dir / 'training_curves_br35h.png'}")
        if dataset in ['sartaj', 'both']:
            print(f"  - {models_dir / 'best_model_sartaj.pth'}")
            print(f"  - {plots_dir / 'confusion_matrix_sartaj.png'}")
            print(f"  - {plots_dir / 'training_curves_sartaj.png'}")
    except subprocess.CalledProcessError as e:
        print(f"\nError during training: {e}")
        print("Please check the error messages above.")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

if __name__ == '__main__':
    main()

