# Graph Neural Network for Brain Tumor Classification

This project implements a Graph Neural Network (GNN) for brain tumor classification on two datasets:
- **br35h**: Binary classification (tumor vs no tumor)
- **sartaj**: Multi-class classification (glioma, meningioma, no tumor, pituitary)

## Dataset Structure

### br35h Dataset
```
br35h/
├── yes/          # Tumor images (1500 images)
├── no/           # No tumor images (1500 images)
└── Br35H-Mask-RCNN/
    ├── TRAIN/
    ├── VAL/
    └── TEST/
```

### sartaj Dataset
```
sartaj/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train on br35h dataset (binary classification):
```bash
python train_gnn.py --dataset br35h --epochs 50 --batch_size 32 --lr 0.001
```

### Train on sartaj dataset (multi-class classification):
```bash
python train_gnn.py --dataset sartaj --epochs 50 --batch_size 32 --lr 0.001
```

### Train on both datasets:
```bash
python train_gnn.py --dataset both --epochs 50 --batch_size 32 --lr 0.001
```

### Advanced options:
```bash
python train_gnn.py \
    --dataset both \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --hidden_dim 128 \
    --num_layers 4 \
    --dropout 0.5 \
    --n_segments 150 \
    --use_gat
```

## Parameters

- `--dataset`: Dataset to use (`br35h`, `sartaj`, or `both`)
- `--data_root`: Root directory of datasets (default: `.`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--hidden_dim`: Hidden dimension (default: 64)
- `--num_layers`: Number of GNN layers (default: 3)
- `--dropout`: Dropout rate (default: 0.5)
- `--weight_decay`: Weight decay (default: 5e-4)
- `--n_segments`: Number of superpixels for graph construction (default: 100)
- `--use_gat`: Use Graph Attention Network instead of GCN
- `--artifact_dir`: Base folder to store models, metrics, plots (default: `artifacts`)
- `--log_dir`: Directory for logs (default: `artifacts/logs`)

## Model Architecture

The GNN model consists of:
1. **Graph Construction**: Images are converted to graphs using superpixels (SLIC algorithm)
2. **Node Features**: Each superpixel region is represented by:
   - Color features (mean and std for each channel)
   - Texture features (Local Binary Pattern)
   - Spatial features (centroid coordinates)
   - Size features
3. **Graph Convolutional Layers**: Multiple GCN or GAT layers with batch normalization
4. **Graph Pooling**: Combines mean, max, and sum pooling
5. **Classifier**: Fully connected layers for classification

## Outputs

After training, artifacts are organized under `artifacts/`:
- Models: `artifacts/models/best_model_{dataset}.pth`
- Metrics: `artifacts/metrics/training_metrics_{dataset}.json`
- Plots: `artifacts/plots/confusion_matrix_{dataset}.png`, `artifacts/plots/training_curves_{dataset}.png`
- Logs: `artifacts/logs/`

## Requirements

- Python 3.7+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- scikit-image
- scikit-learn
- numpy
- matplotlib
- seaborn

## GNN Pruning

The project includes GNN pruning capabilities to reduce model size and improve inference speed.

### Pruning Features

- **Magnitude-based pruning**: Removes weights with smallest magnitudes
- **Unstructured pruning**: Removes individual weights (sparse models)
- **Structured pruning**: Removes entire neurons/channels (dense models)
- **Iterative pruning**: Gradually increases pruning rate
- **Fine-tuning**: Retrain pruned models to recover accuracy

### Run Pruning Evaluation

**Quick pruning demo:**
```bash
python run_pruning_example.py --dataset br35h
```

**Full pruning evaluation with visualization:**
```bash
python prune_and_evaluate.py --dataset br35h --pruning_rates 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

**With fine-tuning:**
```bash
python prune_and_evaluate.py --dataset br35h --fine_tune --fine_tune_epochs 20
```

### Pruning Results

After running pruning evaluation, you'll get:
- `pruning_results_{dataset}.png`: Visualization of pruning results
- `pruning_results_{dataset}.json`: Detailed pruning statistics
- Charts showing:
  - Accuracy vs Pruning Rate
  - Model Size Reduction
  - Compression Ratio
  - Inference Time
  - Accuracy-Compression Trade-off

## Notes

- The model converts images to graphs using superpixels, making it memory-efficient
- Graph structure captures spatial relationships between image regions
- The model uses multiple pooling strategies (mean, max, sum) for robust feature aggregation
- Batch normalization and dropout are used for regularization
- Pruning can reduce model size by 50-80% with minimal accuracy loss
- Fine-tuning after pruning helps recover accuracy

