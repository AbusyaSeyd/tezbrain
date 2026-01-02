# Graph Neural Network for Brain Tumor Classification

This project implements a Graph Neural Network (GNN) for brain tumor classification on two datasets:
- **br35h**: Binary classification (tumor vs no tumor)
- **sartaj**: Multi-class classification (glioma, meningioma, no tumor, pituitary)

## Project Structure

```
tezbrain/
├── src/                    # Core modules
│   ├── data_loader.py     # Dataset classes and graph conversion
│   ├── gnn_model.py       # GNN model definitions
│   ├── gnn_pruning.py     # Pruning utilities
│   └── paths.py           # Path management utilities
├── scripts/                # Executable scripts
│   ├── train_gnn.py       # Main training script
│   ├── quick_start.py     # Quick start script
│   └── ...                # Other training/evaluation scripts
├── data/                   # Datasets
│   ├── br35h/            # Binary classification dataset
│   └── sartaj/           # Multi-class classification dataset
├── artifacts/             # Training outputs
│   ├── models/           # Saved model checkpoints
│   ├── metrics/          # Training metrics (JSON)
│   ├── plots/            # Visualizations
│   └── logs/             # Log files
├── tests/                 # Test files
├── config/                # Configuration files
└── docs/                  # Detailed documentation
```

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run training:**
```bash
python scripts/quick_start.py
```

Or train directly:
```bash
python scripts/train_gnn.py --dataset br35h --epochs 50 --batch_size 32 --lr 0.001
```

## Documentation

For detailed documentation, see:
- [Full README](docs/README.md) - Complete usage guide
- [Pruning Guide](docs/PRUNING_GUIDE.md) - GNN pruning documentation

## Key Features

- Graph Neural Network architecture for brain tumor classification
- Support for binary (br35h) and multi-class (sartaj) classification
- Model pruning capabilities for model compression
- Hyperparameter search with Optuna
- Comprehensive evaluation and visualization tools

## Requirements

- Python 3.7+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- See `requirements.txt` for full list

