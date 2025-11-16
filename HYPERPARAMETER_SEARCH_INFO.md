# Hyperparameter Search Information

## Current Status

The hyperparameter search is running in the background with the following configuration:
- **Number of trials**: 20 per dataset
- **Epochs per trial**: 200
- **Learning rates**: High values (0.001 to 0.02)
- **Datasets**: br35h and sartaj

## Expected Duration

With 200 epochs per trial, each trial can take 30-60+ minutes depending on:
- Dataset size
- Model complexity (hidden dimensions, number of layers)
- Hardware (CPU vs GPU)

**Total estimated time**: 10-20+ hours for 20 trials per dataset

## Monitoring Progress

### Check Progress
```bash
python check_hyperparameter_progress.py br35h
python check_hyperparameter_progress.py sartaj
```

### View Logs
```bash
tail -f hyperparameter_search.log
```

### Check Running Processes
```bash
ps aux | grep hyperparameter_search.py
```

## Results Files

Results are saved incrementally after each trial:
- `hyperparameter_results_br35h_incremental.json` - Incremental results for br35h
- `hyperparameter_results_sartaj_incremental.json` - Incremental results for sartaj

Final results will be saved as:
- `hyperparameter_results_br35h.csv` - CSV table
- `hyperparameter_results_br35h.json` - JSON format
- `hyperparameter_results_br35h.png` - Visualization
- `hyperparameter_table_br35h.png` - Results table image

## Hyperparameter Search Space

The random search explores:
- **Learning Rate**: 0.001, 0.002, 0.003, 0.005, 0.01, 0.015, 0.02 (high values as requested)
- **Hidden Dimension**: 32, 64, 128, 256
- **Number of Layers**: 2, 3, 4, 5
- **Batch Size**: 16, 32, 64
- **Dropout**: 0.3, 0.4, 0.5, 0.6, 0.7
- **Weight Decay**: 1e-4, 5e-4, 1e-3, 5e-3
- **N Segments**: 50, 100, 150, 200
- **Use GAT**: True, False
- **Gradient Clipping**: 0.5, 1.0, 2.0

## Quick Test Run (Faster Results)

If you want to see results faster, you can run a smaller test:

```bash
python hyperparameter_search.py --dataset br35h --n_trials 5 --epochs 50
```

This will complete much faster and give you an idea of the results format.

## Output

When complete, you will get:
1. **CSV file** with all results sorted by accuracy
2. **JSON file** with detailed results
3. **Visualization PNG** showing:
   - Accuracy vs each hyperparameter
   - Hyperparameter importance
   - Top 10 configurations
   - Training time analysis
4. **Results table PNG** with formatted table showing all configurations

