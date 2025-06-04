import os
import json
import argparse
import shutil
from datetime import datetime
from glob import glob

def list_experiments():
    experiments_dir = './experiments'
    if not os.path.exists(experiments_dir):
        print("No experiments directory found.")
        return
    
    experiments = []
    for exp_dir in glob(os.path.join(experiments_dir, '*/')):
        exp_name = os.path.basename(exp_dir.rstrip('/'))
        config_path = os.path.join(exp_dir, 'config.json')
        
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            
            checkpoints = glob(os.path.join(exp_dir, 'checkpoints', '*.pth'))
            
            experiments.append({
                'name': exp_name,
                'model_type': config.get('model_type', 'unknown'),
                'checkpoints': len(checkpoints),
                'size_mb': get_dir_size(exp_dir) / (1024 * 1024)
            })
    
    if not experiments:
        print("No experiments found.")
        return
    
    print("="*80)
    print("EXPERIMENTS")
    print("="*80)
    print(f"{'Name':<35} {'Model Type':<12} {'Checkpoints':<12} {'Size (MB)':<10}")
    print("-"*80)
    
    for exp in sorted(experiments, key=lambda x: x['name']):
        print(f"{exp['name']:<35} {exp['model_type']:<12} {exp['checkpoints']:<12} {exp['size_mb']:.1f}")
    
    total_size = sum(exp['size_mb'] for exp in experiments)
    print("-"*80)
    print(f"Total: {len(experiments)} experiments, {total_size:.1f} MB")

def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def cleanup_experiments(keep_best=True, days_old=None, pattern=None):
    experiments_dir = './experiments'
    if not os.path.exists(experiments_dir):
        print("No experiments directory found.")
        return
    
    experiments_to_remove = []
    
    for exp_dir in glob(os.path.join(experiments_dir, '*/')):
        exp_name = os.path.basename(exp_dir.rstrip('/'))
        
        if pattern and pattern not in exp_name:
            continue
        
        if days_old:
            exp_time = datetime.fromtimestamp(os.path.getctime(exp_dir))
            age_days = (datetime.now() - exp_time).days
            if age_days < days_old:
                continue
        
        should_remove = True
        
        if keep_best:
            metrics_path = os.path.join(exp_dir, 'results', 'final_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                
                error_acc = metrics.get('best_error_accuracy', 0.0)
                phoneme_acc = metrics.get('best_phoneme_accuracy', 0.0)
                
                if error_acc > 0.85 or phoneme_acc > 0.85:
                    should_remove = False
        
        if should_remove:
            experiments_to_remove.append((exp_dir, exp_name))
    
    if not experiments_to_remove:
        print("No experiments match the cleanup criteria.")
        return
    
    print("Experiments to remove:")
    total_size = 0
    for exp_dir, exp_name in experiments_to_remove:
        size_mb = get_dir_size(exp_dir) / (1024 * 1024)
        total_size += size_mb
        print(f"  {exp_name} ({size_mb:.1f} MB)")
    
    print(f"\nTotal space to free: {total_size:.1f} MB")
    
    confirm = input("\nProceed with cleanup? (y/N): ")
    if confirm.lower() == 'y':
        for exp_dir, exp_name in experiments_to_remove:
            shutil.rmtree(exp_dir)
            print(f"Removed {exp_name}")
        print(f"Cleanup completed. Freed {total_size:.1f} MB")
    else:
        print("Cleanup cancelled.")

def archive_experiment(experiment_name, archive_dir='./archived_experiments'):
    exp_dir = os.path.join('./experiments', experiment_name)
    if not os.path.exists(exp_dir):
        print(f"Experiment {experiment_name} not found.")
        return
    
    os.makedirs(archive_dir, exist_ok=True)
    
    archive_path = os.path.join(archive_dir, f"{experiment_name}.tar.gz")
    
    import tarfile
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(exp_dir, arcname=experiment_name)
    
    shutil.rmtree(exp_dir)
    
    print(f"Experiment {experiment_name} archived to {archive_path}")

def create_experiment_template(model_type, experiment_name=None):
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"{model_type}_{timestamp}"
    
    template_config = {
        'model_type': model_type,
        'experiment_name': experiment_name,
        'batch_size': 8,
        'num_epochs': 30,
        'wav2vec_lr': 1e-5,
        'main_lr': 1e-4,
        'notes': f'Template for {model_type} model'
    }
    
    exp_dir = os.path.join('./experiments', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    config_path = os.path.join(exp_dir, 'template_config.json')
    with open(config_path, 'w') as f:
        json.dump(template_config, f, indent=2)
    
    print(f"Created experiment template: {experiment_name}")
    print(f"Config saved to: {config_path}")
    print(f"Run training with: python train.py --config model_type={model_type},experiment_name={experiment_name}")

def main():
    parser = argparse.ArgumentParser(description='Experiment Management Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    subparsers.add_parser('list', help='List all experiments')
    
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup old experiments')
    cleanup_parser.add_argument('--keep-best', action='store_true', default=True, help='Keep experiments with good performance')
    cleanup_parser.add_argument('--days-old', type=int, help='Remove experiments older than N days')
    cleanup_parser.add_argument('--pattern', type=str, help='Only remove experiments matching pattern')
    
    archive_parser = subparsers.add_parser('archive', help='Archive an experiment')
    archive_parser.add_argument('experiment_name', help='Name of experiment to archive')
    archive_parser.add_argument('--archive-dir', default='./archived_experiments', help='Archive directory')
    
    template_parser = subparsers.add_parser('template', help='Create experiment template')
    template_parser.add_argument('model_type', choices=['simple', 'transformer', 'cross', 'hierarchical'], help='Model type')
    template_parser.add_argument('--name', help='Experiment name (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_experiments()
    elif args.command == 'cleanup':
        cleanup_experiments(
            keep_best=args.keep_best,
            days_old=args.days_old,
            pattern=args.pattern
        )
    elif args.command == 'archive':
        archive_experiment(args.experiment_name, args.archive_dir)
    elif args.command == 'template':
        create_experiment_template(args.model_type, args.name)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()