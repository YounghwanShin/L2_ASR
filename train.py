#!/usr/bin/env python3
import os
import sys
import logging
import torch
import hyperpyyaml as hpyy
import speechbrain as sb
from pathlib import Path
from transformers import Wav2Vec2Processor

from data_prepare import create_datasets
from model import SimpleMultiTaskBrain

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def run_training(hparams_file, run_opts, overrides):
    with open(hparams_file) as fin:
        hparams = hpyy.load_hyperpyyaml(fin, overrides)
    
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    if hparams.get("sorting", "ascending") != "random":
        hparams["train_dataloader_opts"]["shuffle"] = False
    
    hparams["wav2vec2"] = Wav2Vec2Processor.from_pretrained(hparams["wav2vec2_model"])
    
    logger.info("Preparing datasets...")
    
    train_data, valid_data, test_data, label_encoder = create_datasets(hparams)
    
    logger.info(f"Training set: {len(train_data)} samples")
    
    from model import Wav2Vec2Encoder, MultiTaskHead
    
    if hasattr(hparams, 'wav2vec2_model'):
        wav2vec2_model = hparams.wav2vec2_model
        hidden_dim = hparams.hidden_dim
        num_phonemes = hparams.num_phonemes
        num_errors = hparams.num_errors
        number_of_epochs = hparams.number_of_epochs
    else:
        wav2vec2_model = hparams.get("wav2vec2_model", "facebook/wav2vec2-base")
        hidden_dim = hparams.get("hidden_dim", 768)
        num_phonemes = hparams.get("num_phonemes", 43)
        num_errors = hparams.get("num_errors", 3)
        number_of_epochs = hparams.get("number_of_epochs", 30)
    
    wav2vec2 = Wav2Vec2Encoder(model_name=wav2vec2_model)
    model = MultiTaskHead(
        input_dim=hidden_dim,
        num_phonemes=num_phonemes,
        num_errors=num_errors
    )
    
    epoch_counter = sb.utils.epoch_loop.EpochCounter(limit=number_of_epochs)
    
    modules = {
        "wav2vec2": wav2vec2,
        "model": model
    }
    
    if hasattr(hparams, 'modules'):
        hparams.modules = modules
        hparams.epoch_counter = epoch_counter
    else:
        hparams["modules"] = modules
        hparams["epoch_counter"] = epoch_counter
    
    if hasattr(hparams, 'output_folder'):
        output_folder = hparams.output_folder
    else:
        output_folder = hparams.get("output_folder", "./results")
    
    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=os.path.join(output_folder, "save"),
        recoverables={
            "wav2vec2": wav2vec2,
            "model": model,
            "counter": epoch_counter
        }
    )
    
    logger.info("Initializing model...")
    brain = SimpleMultiTaskBrain(
        modules=modules,
        opt_class=torch.optim.AdamW,
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )
    
    brain.label_encoder = label_encoder
    brain.test_data = test_data
    
    logger.info(f"Starting training for {number_of_epochs} epochs...")
    
    if hasattr(hparams, 'train_dataloader_opts'):
        train_dataloader_opts = hparams.train_dataloader_opts
        valid_dataloader_opts = hparams.valid_dataloader_opts
        test_dataloader_opts = hparams.test_dataloader_opts
    else:
        train_dataloader_opts = hparams.get("train_dataloader_opts", {})
        valid_dataloader_opts = hparams.get("valid_dataloader_opts", {})
        test_dataloader_opts = hparams.get("test_dataloader_opts", {})
    
    try:
        brain.fit(
            epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=train_dataloader_opts,
            valid_loader_kwargs=valid_dataloader_opts,
        )
        
        logger.info("Running final test evaluation...")
        brain.evaluate(
            test_data,
            test_loader_kwargs=test_dataloader_opts
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    if run_opts is None:
        run_opts = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    
    setup_logging()
    logger.info("Starting SpeechBrain multi-task training...")
    logger.info(f"Output folder: {run_opts.get('output_folder', './results')}")
    
    run_training(hparams_file, run_opts, overrides)

if __name__ == "__main__":
    main()