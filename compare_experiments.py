import os
import json
import argparse
import pandas as pd
from glob import glob

def load_experiment_results(experiment_dir):
    config_path = os.path.join(experiment_dir, 'config.json')
    metrics_path = os.path.join(experiment_dir, 'results', 'final_metrics.json')
    
    if not os.path.exists(config_path) or not os.path.exists(metrics_path):
        return None
    
    with open(config_path) as f:
        config = json.load(f)
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    return {
        'experiment': os.path.basename(experiment_dir),
        'model_type': config.get('model_type', 'unknown'),
        'batch_size': config.get('batch_size', 'unknown'),
        'learning_rate': config.get('main_lr', 'unknown'),
        'wav2vec_lr': config.get('wav2vec_lr', 'unknown'),
        'error_accuracy': metrics.get('best_error_accuracy', 0.0),
        'phoneme_accuracy': metrics.get('best_phoneme_accuracy', 0.0),
        'best_val_loss': metrics.get('best_val_loss', float('inf')),
        'per': 1.0 - metrics.get('best_phoneme_accuracy', 0.0)
    }

def compare_experiments(experiment_dirs):
    results = []
    
    for exp_dir in experiment_dirs:
        if os.path.isdir(exp_dir):
            result = load_experiment_results(exp_dir)
            if result:
                results.append(result)
    
    if not results:
        print("No valid experiment results found.")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values('error_accuracy', ascending=False)
    
    print("="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    print(df.to_string(index=False, float_format='%.4f'))
    
    print("\n" + "="*50)
    print("BEST MODELS")
    print("="*50)
    
    best_error = df.loc[df['error_accuracy'].idxmax()]
    best_phoneme = df.loc[df['phoneme_accuracy'].idxmax()]
    best_loss = df.loc[df['best_val_loss'].idxmin()]
    
    print(f"Best Error Detection: {best_error['experiment']} ({best_error['error_accuracy']:.4f})")
    print(f"Best Phoneme Recognition: {best_phoneme['experiment']} ({best_phoneme['phoneme_accuracy']:.4f})")
    print(f"Best Validation Loss: {best_loss['experiment']} ({best_loss['best_val_loss']:.4f})")
    
    comparison_dir = './experiments/comparison_results'
    os.makedirs(comparison_dir, exist_ok=True)
    
    csv_path = os.path.join(comparison_dir, 'performance_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    summary = {
        'total_experiments': len(results),
        'best_error_detection': {
            'experiment': best_error['experiment'],
            'accuracy': best_error['error_accuracy']
        },
        'best_phoneme_recognition': {
            'experiment': best_phoneme['experiment'],
            'accuracy': best_phoneme['phoneme_accuracy']
        },
        'best_validation_loss': {
            'experiment': best_loss['experiment'],
            'loss': best_loss['best_val_loss']
        }
    }
    
    summary_path = os.path.join(comparison_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Compare multiple experiment results')
    parser.add_argument('experiment_dirs', nargs='*', help='Experiment directories to compare')
    parser.add_argument('--pattern', type=str, help='Pattern to match experiment directories')
    
    args = parser.parse_args()
    
    experiment_dirs = []
    
    if args.pattern:
        experiment_dirs.extend(glob(args.pattern))
    elif args.experiment_dirs:
        experiment_dirs.extend(args.experiment_dirs)
    else:
        experiment_dirs = glob('./experiments/*/')
        print(f"No directories specified, comparing all experiments in ./experiments/")
    
    compare_experiments(experiment_dirs)

if __name__ == "__main__":
    main()