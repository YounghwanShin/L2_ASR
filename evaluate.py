#!/usr/bin/env python3
import os
import sys
import torch
import logging
import hyperpyyaml as hpyy
import speechbrain as sb
from transformers import Wav2Vec2Processor
from data_prepare import create_datasets
from model import SimpleMultiTaskBrain

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def evaluate_model(hparams_file, run_opts, checkpoint_path=None):
    
    with open(hparams_file) as fin:
        hparams = hpyy.load_hyperpyyaml(fin)
    
    if hparams.get("sorting", "ascending") != "random":
        hparams["train_dataloader_opts"]["shuffle"] = False
    
    hparams["wav2vec2"] = Wav2Vec2Processor.from_pretrained(hparams["wav2vec2_model"])
    
    logger.info("Loading datasets...")
    
    _, _, test_data = create_datasets(hparams)
    
    logger.info(f"Test set: {len(test_data)} samples")
    
    from model import Wav2Vec2Encoder, MultiTaskHead
    
    wav2vec2 = Wav2Vec2Encoder(model_name=hparams["wav2vec2_model"])
    model = MultiTaskHead(
        input_dim=hparams["hidden_dim"],
        num_phonemes=hparams["num_phonemes"],
        num_errors=hparams["num_errors"]
    )
    
    modules = {
        "wav2vec2": wav2vec2,
        "model": model
    }
    
    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=os.path.join("./results", "save"),
        recoverables={
            "wav2vec2": wav2vec2,
            "model": model,
        }
    )
    
    brain = SimpleMultiTaskBrain(
        modules=modules,
        opt_class=torch.optim.AdamW,
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=brain.device)
        brain.modules.load_state_dict(checkpoint['model_state_dict'])
        
        if 'best_phoneme_per' in checkpoint:
            logger.info(f"Checkpoint - Best Phoneme PER: {checkpoint['best_phoneme_per']:.4f}")
        if 'best_error_acc' in checkpoint:
            logger.info(f"Checkpoint - Best Error Accuracy: {checkpoint['best_error_acc']:.4f}")
    else:
        logger.warning("No checkpoint specified or found. Using random weights.")
    
    logger.info("Starting evaluation...")
    brain.evaluate(
        test_data,
        test_loader_kwargs=hparams["test_dataloader_opts"]
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <config_file> [checkpoint_path]")
        print("Example: python evaluate.py multitask.yaml ./results/save/best_phoneme_per.ckpt")
        sys.exit(1)
    
    setup_logging()
    
    hparams_file = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    run_opts = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_parallel_count": -1,
        "precision": "fp32"
    }
    
    logger.info("Starting model evaluation...")
    logger.info(f"Config file: {hparams_file}")
    if checkpoint_path:
        logger.info(f"Checkpoint: {checkpoint_path}")
    
    evaluate_model(hparams_file, run_opts, checkpoint_path)

if __name__ == "__main__":
    main()