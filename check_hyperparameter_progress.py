"""
Quick script to check progress of hyperparameter search and show current best results.
"""
import json
import os
import pandas as pd

def check_progress(dataset_name):
    """Check progress for a specific dataset."""
    results_file = f'hyperparameter_results_{dataset_name}_incremental.json'
    
    if not os.path.exists(results_file):
        print(f"No results file found for {dataset_name}")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if not results:
        print(f"No results yet for {dataset_name}")
        return
    
    df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print(f"PROGRESS - {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"Completed trials: {len(results)}")
    
    if 'best_acc' in df.columns:
        best_idx = df['best_acc'].idxmax()
        best = df.loc[best_idx]
        
        print(f"\nCurrent Best Configuration:")
        print(f"  Trial: {int(best['trial'])}")
        print(f"  Learning Rate: {best['lr']:.4f}")
        print(f"  Hidden Dimension: {int(best['hidden_dim'])}")
        print(f"  Number of Layers: {int(best['num_layers'])}")
        print(f"  Batch Size: {int(best['batch_size'])}")
        print(f"  Dropout: {best['dropout']:.2f}")
        print(f"  Weight Decay: {best['weight_decay']:.6f}")
        print(f"  N Segments: {int(best['n_segments'])}")
        print(f"  Use GAT: {best['use_gat']}")
        print(f"  Best Accuracy: {best['best_acc']:.4f}")
        print(f"  Best Epoch: {int(best['best_epoch'])}")
        
        print(f"\nTop 5 Configurations:")
        top5 = df.nlargest(5, 'best_acc')[['trial', 'lr', 'hidden_dim', 'num_layers', 
                                           'batch_size', 'dropout', 'best_acc', 'best_epoch']]
        print(top5.to_string(index=False))
        
        if 'training_time_minutes' in df.columns:
            total_time = df['training_time_minutes'].sum()
            avg_time = df['training_time_minutes'].mean()
            print(f"\nTime Statistics:")
            print(f"  Total time: {total_time:.2f} minutes ({total_time/60:.2f} hours)")
            print(f"  Average time per trial: {avg_time:.2f} minutes")

if __name__ == '__main__':
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'br35h'
    check_progress(dataset)

