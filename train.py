import os
import sys
import json
import logging
import argparse
from pathlib import Path
import torch
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml

from model import MultiTaskBrain, MultiTaskModel
from data_prepare import dataio_prepare, DatasetStats

logger = logging.getLogger(__name__)


def create_experiment_folder(hparams):
    os.makedirs(hparams["output_folder"], exist_ok=True)
    os.makedirs(hparams["save_folder"], exist_ok=True)
    
    log_file = os.path.join(hparams["save_folder"], "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    hparams_file = os.path.join(hparams["save_folder"], "hyperparams.yaml")
    with open(hparams_file, 'w') as f:
        simple_hparams = {k: v for k, v in hparams.items() 
                         if isinstance(v, (str, int, float, bool, list))}
        import yaml
        yaml.dump(simple_hparams, f, default_flow_style=False)


def run_training(hparams_file, run_opts=None, overrides=None):
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    create_experiment_folder(hparams)
    
    logger.info("Starting multi-task training...")
    logger.info(f"Output folder: {hparams['output_folder']}")
    logger.info(f"Task mode: {hparams['task']}")
    
    logger.info("Preparing datasets...")
    train_loader, val_loader, test_loader = dataio_prepare(hparams)
    
    if train_loader:
        train_stats = DatasetStats.compute_stats(
            train_loader, hparams["phoneme_decoder"], hparams["error_decoder"]
        )
        logger.info(f"Training set: {train_stats['total_samples']} samples, "
                   f"{train_stats['total_duration']:.1f}s total, "
                   f"{train_stats['avg_duration']:.2f}s average")
        
        stats_file = os.path.join(hparams["save_folder"], "dataset_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(train_stats, f, indent=2, default=str)
    
    logger.info("Initializing model...")
    model = MultiTaskModel(hparams)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    modules = {
        "model": model,
        "wav2vec2": model.wav2vec2
    }
    
    brain = MultiTaskBrain(
        modules=modules,
        hparams=hparams,
        run_opts=run_opts or {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        checkpointer=hparams["checkpointer"]
    )
    
    brain.test_loader = test_loader
    brain.device = brain.device if hasattr(brain, 'device') else "cuda"
    
    logger.info(f"Starting training for {hparams['number_of_epochs']} epochs...")
    
    try:
        brain.fit(
            hparams["epoch_counter"],
            train_loader,
            val_loader,
            train_loader_kwargs=hparams.get("train_dataloader_opts", {}),
            valid_loader_kwargs=hparams.get("val_dataloader_opts", {})
        )
        
        logger.info("Training completed successfully!")
        
        if test_loader:
            logger.info("Running final evaluation...")
            brain.evaluate(
                test_loader,
                test_loader_kwargs=hparams.get("test_dataloader_opts", {})
            )
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        if hasattr(brain, 'best_error_acc'):
            logger.info(f"Best Error Accuracy: {brain.best_error_acc:.4f}")
        if hasattr(brain, 'best_per'):
            logger.info(f"Best PER: {brain.best_per:.4f}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Train Multi-task Model')
    parser.add_argument('hparams_file', type=str, help='Hyperparameters file')
    parser.add_argument('--data_folder', type=str, help='Data folder path')
    parser.add_argument('--output_folder', type=str, help='Output folder path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    run_opts = {
        "device": args.device,
        "debug": args.debug
    }
    
    overrides = {}
    if args.data_folder:
        overrides["data_folder"] = args.data_folder
    if args.output_folder:
        overrides["output_folder"] = args.output_folder
        overrides["save_folder"] = os.path.join(args.output_folder, "save")
    
    run_training(args.hparams_file, run_opts, overrides)


if __name__ == "__main__":
    main()