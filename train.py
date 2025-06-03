import os
import sys
import json
import logging
import argparse
import torch
import yaml
import speechbrain as sb

from model import SimpleMultiTaskBrain, SimpleMultiTaskModel
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


def run_training(hparams_file, run_opts=None, overrides=None):
    with open(hparams_file, 'r') as f:
        hparams = yaml.safe_load(f)
    
    if overrides:
        hparams.update(overrides)
    
    create_experiment_folder(hparams)
    
    logger.info("Starting SpeechBrain multi-task training...")
    logger.info(f"Output folder: {hparams['output_folder']}")
    logger.info(f"Task mode: {hparams['task']}")
    
    logger.info("Preparing datasets...")
    train_loader, val_loader, test_loader = dataio_prepare(hparams)
    
    if train_loader:
        train_stats = DatasetStats.compute_stats(
            train_loader, hparams["phoneme_decoder"], hparams["error_decoder"]
        )
        logger.info(f"Training set: {train_stats['total_samples']} samples")
    
    logger.info("Initializing model...")
    model = SimpleMultiTaskModel(hparams)
    
    modules = {
        "model": model,
        "wav2vec2": model.wav2vec2
    }
    
    device = run_opts.get("device", "cuda") if run_opts else "cuda"
    
    brain = SimpleMultiTaskBrain(
        modules=modules,
        hparams=hparams,
        run_opts={"device": device}
    )
    
    logger.info(f"Starting training for {hparams['number_of_epochs']} epochs...")
    
    try:
        # Create simple epoch counter
        class SimpleEpochCounter:
            def __init__(self, max_epochs):
                self.max_epochs = max_epochs
                self.current = 0
            
            def __iter__(self):
                return self
            
            def __next__(self):
                if self.current >= self.max_epochs:
                    raise StopIteration
                self.current += 1
                return self.current
        
        epoch_counter = SimpleEpochCounter(hparams["number_of_epochs"])
        
        brain.fit(
            epoch_counter,
            train_loader,
            val_loader,
            train_loader_kwargs={},
            valid_loader_kwargs={}
        )
        
        logger.info("Training completed successfully!")
        
        if test_loader:
            logger.info("Running final evaluation...")
            brain.evaluate(
                test_loader,
                test_loader_kwargs={}
            )
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description='Train SpeechBrain Multi-task Model')
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