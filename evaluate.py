import torch
import numpy as np
import json
import os
import logging
from tqdm import tqdm
import yaml
import argparse

logger = logging.getLogger(__name__)


def decode_ctc_greedy(log_probs, input_lengths, blank_idx=0):
    greedy_preds = torch.argmax(log_probs, dim=-1).cpu().numpy()
    batch_size = greedy_preds.shape[0]
    decoded_seqs = []
    
    for b in range(batch_size):
        seq = []
        prev = blank_idx
        actual_length = min(input_lengths[b].item(), greedy_preds.shape[1])
        
        for t in range(actual_length):
            pred = greedy_preds[b, t]
            if pred != blank_idx and pred != prev:
                seq.append(int(pred))
            prev = pred
        
        decoded_seqs.append(seq)
    
    return decoded_seqs


def evaluate_simple(brain, data_loader, task="both", device="cuda"):
    brain.modules.eval()
    
    all_error_predictions = []
    all_error_targets = []
    all_phoneme_predictions = []
    all_phoneme_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Evaluation')
        
        for batch in progress_bar:
            batch = batch.to(device)
            predictions = brain.compute_forward(batch, None)
            
            if 'error_logits' in predictions and hasattr(batch, 'error_tokens'):
                error_log_probs = predictions['error_logits'].log_softmax(dim=-1)
                input_lengths = batch.sig[1] * error_log_probs.shape[1]
                input_lengths = input_lengths.long()
                
                batch_error_preds = decode_ctc_greedy(error_log_probs, input_lengths)
                
                batch_error_targets = []
                for tokens, length in zip(batch.error_tokens.data, batch.error_tokens.lengths):
                    target = tokens[:length].cpu().numpy().tolist()
                    batch_error_targets.append(target)
                
                all_error_predictions.extend(batch_error_preds)
                all_error_targets.extend(batch_error_targets)
            
            if 'phoneme_logits' in predictions and hasattr(batch, 'phoneme_tokens'):
                phoneme_log_probs = predictions['phoneme_logits'].log_softmax(dim=-1)
                input_lengths = batch.sig[1] * phoneme_log_probs.shape[1]
                input_lengths = input_lengths.long()
                
                batch_phoneme_preds = decode_ctc_greedy(phoneme_log_probs, input_lengths)
                
                batch_phoneme_targets = []
                for tokens, length in zip(batch.phoneme_tokens.data, batch.phoneme_tokens.lengths):
                    target = tokens[:length].cpu().numpy().tolist()
                    batch_phoneme_targets.append(target)
                
                all_phoneme_predictions.extend(batch_phoneme_preds)
                all_phoneme_targets.extend(batch_phoneme_targets)
    
    results = {}
    
    if all_error_predictions:
        error_acc = sum(1 for pred, target in zip(all_error_predictions, all_error_targets) 
                       if pred == target) / len(all_error_predictions)
        results['error_accuracy'] = error_acc
        logger.info(f"Error Detection Accuracy: {error_acc:.4f}")
    
    if all_phoneme_predictions:
        phoneme_acc = sum(1 for pred, target in zip(all_phoneme_predictions, all_phoneme_targets) 
                         if pred == target) / len(all_phoneme_predictions)
        results['phoneme_accuracy'] = phoneme_acc
        logger.info(f"Phoneme Recognition Accuracy: {phoneme_acc:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Simple Multi-task Model')
    parser.add_argument('hparams_file', type=str, help='Hyperparameters file')
    parser.add_argument('--output_folder', type=str, default='./eval_results', help='Output folder')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    with open(args.hparams_file, 'r') as f:
        hparams = yaml.safe_load(f)
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_folder, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Starting evaluation...")
    
    from model import SimpleMultiTaskBrain, SimpleMultiTaskModel
    from data_prepare import dataio_prepare
    
    train_loader, val_loader, test_loader = dataio_prepare(hparams)
    
    model = SimpleMultiTaskModel(hparams)
    modules = {"model": model, "wav2vec2": model.wav2vec2}
    
    brain = SimpleMultiTaskBrain(
        modules=modules,
        hparams=hparams,
        run_opts={"device": args.device}
    )
    
    results = evaluate_simple(brain, test_loader, hparams["task"], args.device)
    
    results_file = os.path.join(args.output_folder, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {args.output_folder}")
    logger.info("=== SUMMARY ===")
    for key, value in results.items():
        logger.info(f"{key}: {value:.4f}")
    
    return results


if __name__ == "__main__":
    main()