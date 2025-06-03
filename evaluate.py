#!/usr/bin/env python3
import os
import sys
import torch
import logging
import json
import hyperpyyaml as hpyy
import speechbrain as sb
from data_prepare import create_datasets
from model import SimpleMultiTaskBrain
from mpd_evaluation import mpd_eval_on_dataset

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_mpd_results_json(brain, test_data, test_loader_kwargs, output_path):
    """Create JSON file for MPD evaluation"""
    
    results = {}
    id_to_phoneme = {v: k for k, v in brain.hparams.phoneme_to_id.items()}
    
    test_loader = sb.dataio.dataloader.make_dataloader(test_data, **test_loader_kwargs)
    
    brain.modules.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(brain.device)
            
            predictions = brain.compute_forward(batch, sb.Stage.TEST)
            
            if "phoneme_logits" in predictions:
                phoneme_log_probs = torch.nn.functional.log_softmax(predictions["phoneme_logits"], dim=-1)
                batch_size = phoneme_log_probs.shape[0]
                max_length = phoneme_log_probs.shape[1]
                input_lengths = torch.full((batch_size,), max_length, dtype=torch.long)
                
                batch_predictions = brain.greedy_ctc_decode(phoneme_log_probs, input_lengths)
                
                for i, wav_id in enumerate(batch.id):
                    pred_tokens = batch_predictions[i]
                    pred_phonemes = [id_to_phoneme.get(token, '<unk>') for token in pred_tokens]
                    pred_phonemes = [p for p in pred_phonemes if p != '<blank>']
                    
                    batch_data = test_data.data[wav_id]
                    
                    if 'canonical_aligned' in batch_data and 'perceived_train_target' in batch_data:
                        canonical_phn = batch_data['canonical_aligned']
                        perceived_phn = batch_data['perceived_train_target']
                        
                        results[wav_id] = {
                            "canonical_phn": canonical_phn,
                            "phn": perceived_phn,
                            "hyp": " ".join(pred_phonemes)
                        }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"MPD results saved to: {output_path}")
    return output_path

def evaluate_model(hparams_file, run_opts, checkpoint_path=None):
    """Evaluate trained model on test set with MPD metrics"""
    
    with open(hparams_file) as fin:
        hparams = hpyy.load_hyperpyyaml(fin)
    
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
    
    logger.info("Creating MPD evaluation results...")
    results_dir = os.path.join(hparams.get("output_folder", "./results"), "mpd_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    mpd_json_path = os.path.join(results_dir, "predictions.json")
    create_mpd_results_json(brain, test_data, hparams["test_dataloader_opts"], mpd_json_path)
    
    logger.info("Running MPD evaluation...")
    mpd_results_path = os.path.join(results_dir, "mpd_results.txt")
    per_results_path = os.path.join(results_dir, "per_results.txt")
    
    with open(mpd_json_path, 'r') as f:
        mpd_data = json.load(f)
    
    with open(mpd_results_path, 'w') as mpd_file, open(per_results_path, 'w') as per_file:
        mpd_stats = mpd_eval_on_dataset(mpd_data, mpd_file, per_file)
    
    logger.info("=== MPD EVALUATION RESULTS ===")
    logger.info(f"Precision: {mpd_stats['precision']:.4f}")
    logger.info(f"Recall: {mpd_stats['recall']:.4f}")
    logger.info(f"F1-Score: {mpd_stats['f1']:.4f}")
    logger.info(f"True Accept: {mpd_stats['ta']}")
    logger.info(f"False Rejection: {mpd_stats['fr']}")
    logger.info(f"False Accept: {mpd_stats['fa']}")
    logger.info(f"True Rejection: {mpd_stats['tr']}")
    
    logger.info(f"Detailed MPD results saved to: {mpd_results_path}")
    logger.info(f"PER results saved to: {per_results_path}")

def main():
    """Main evaluation function"""
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