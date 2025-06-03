import os
import sys
import json
import logging
import argparse
import torch
import yaml

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
    
    logger.info("Starting simple multi-task training...")
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
    
    wav2vec_params = list(brain.modules.wav2vec2.parameters())
    model_params = list(brain.modules.model.parameters())
    
    brain.wav2vec_optimizer = torch.optim.AdamW(
        wav2vec_params, 
        lr=hparams["lr_wav2vec"], 
        weight_decay=hparams["weight_decay"]
    )
    
    brain.model_optimizer = torch.optim.AdamW(
        model_params, 
        lr=hparams["lr"], 
        weight_decay=hparams["weight_decay"]
    )
    
    logger.info(f"Starting training for {hparams['number_of_epochs']} epochs...")
    
    try:
        for epoch in range(hparams["number_of_epochs"]):
            brain.on_stage_start(sb.Stage.TRAIN, epoch)
            
            train_loss = 0.0
            brain.modules.train()
            
            for batch in train_loader:
                brain.wav2vec_optimizer.zero_grad()
                brain.model_optimizer.zero_grad()
                
                predictions = brain.compute_forward(batch, sb.Stage.TRAIN)
                loss = brain.compute_objectives(predictions, batch, sb.Stage.TRAIN)
                
                loss.backward()
                
                if hparams.get("grad_clipping"):
                    torch.nn.utils.clip_grad_norm_(
                        brain.modules.parameters(), 
                        hparams["grad_clipping"]
                    )
                
                brain.wav2vec_optimizer.step()
                brain.model_optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            brain.on_stage_end(sb.Stage.TRAIN, train_loss, epoch)
            
            if val_loader:
                brain.modules.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        predictions = brain.compute_forward(batch, sb.Stage.VALID)
                        loss = brain.compute_objectives(predictions, batch, sb.Stage.VALID)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                brain.on_stage_end(sb.Stage.VALID, val_loss, epoch)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Train Simple Multi-task Model')
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
    import speechbrain as sb
    main()