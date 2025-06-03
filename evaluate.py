import torch
import numpy as np
import json
import os
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import speechbrain as sb
from speechbrain.utils.edit_distance import wer_details_for_batch
import argparse
from hyperpyyaml import load_hyperpyyaml

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


def evaluate_error_detection(brain, data_loader, error_decoder, device="cuda"):
    brain.modules.eval()
    
    all_predictions = []
    all_targets = []
    all_ids = []
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Error Detection Evaluation')
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            predictions = brain.compute_forward(batch, sb.Stage.TEST)
            
            if 'error_logits' not in predictions:
                continue
            
            error_log_probs = predictions['error_logits'].log_softmax(dim=-1)
            input_lengths = batch.sig[1] * error_log_probs.shape[1]
            input_lengths = input_lengths.long()
            
            batch_predictions = decode_ctc_greedy(error_log_probs, input_lengths)
            
            if hasattr(batch, 'error_tokens'):
                batch_targets = []
                for tokens, length in zip(batch.error_tokens.data, batch.error_tokens.lengths):
                    target = tokens[:length].cpu().numpy().tolist()
                    batch_targets.append(target)
                
                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)
                all_ids.extend(batch.id)
    
    if not all_predictions:
        logger.warning("No error detection samples found")
        return {}
    
    wer_details = wer_details_for_batch(
        ids=all_ids,
        refs=all_targets,
        hyps=all_predictions,
        compute_alignments=True
    )
    
    total_sequences = len(wer_details)
    correct_sequences = sum(1 for detail in wer_details if detail['WER'] == 0.0)
    sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
    
    total_tokens = sum(detail['num_ref_tokens'] for detail in wer_details)
    total_errors = sum(detail['insertions'] + detail['deletions'] + detail['substitutions'] 
                      for detail in wer_details)
    token_accuracy = 1 - (total_errors / total_tokens) if total_tokens > 0 else 0
    
    total_insertions = sum(detail['insertions'] for detail in wer_details)
    total_deletions = sum(detail['deletions'] for detail in wer_details)
    total_substitutions = sum(detail['substitutions'] for detail in wer_details)
    avg_edit_distance = total_errors / total_sequences if total_sequences > 0 else 0
    
    flat_predictions = [token for pred in all_predictions for token in pred]
    flat_targets = [token for target in all_targets for token in target]
    
    weighted_f1 = macro_f1 = 0
    class_metrics = {}
    
    if len(flat_predictions) > 0 and len(flat_targets) > 0:
        try:
            min_len = min(len(flat_predictions), len(flat_targets))
            flat_predictions = flat_predictions[:min_len]
            flat_targets = flat_targets[:min_len]
            
            weighted_f1 = f1_score(flat_targets, flat_predictions, average='weighted', zero_division=0)
            macro_f1 = f1_score(flat_targets, flat_predictions, average='macro', zero_division=0)
            
            class_report = classification_report(flat_targets, flat_predictions, output_dict=True, zero_division=0)
            
            error_types = {0: 'blank', 1: 'incorrect', 2: 'correct'}
            for class_id, class_name in error_types.items():
                if str(class_id) in class_report:
                    class_metrics[class_name] = {
                        'precision': float(class_report[str(class_id)]['precision']),
                        'recall': float(class_report[str(class_id)]['recall']),
                        'f1': float(class_report[str(class_id)]['f1-score']),
                        'support': int(class_report[str(class_id)]['support'])
                    }
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
    
    results = {
        'sequence_accuracy': float(sequence_accuracy),
        'token_accuracy': float(token_accuracy),
        'avg_edit_distance': float(avg_edit_distance),
        'weighted_f1': float(weighted_f1),
        'macro_f1': float(macro_f1),
        'class_metrics': class_metrics,
        'total_sequences': int(total_sequences),
        'total_tokens': int(total_tokens),
        'total_insertions': int(total_insertions),
        'total_deletions': int(total_deletions),
        'total_substitutions': int(total_substitutions),
        'wer_details': wer_details
    }
    
    return results


def evaluate_phoneme_recognition(brain, data_loader, phoneme_decoder, device="cuda"):
    brain.modules.eval()
    
    all_predictions = []
    all_targets = []
    all_ids = []
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Phoneme Recognition Evaluation')
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            predictions = brain.compute_forward(batch, sb.Stage.TEST)
            
            if 'phoneme_logits' not in predictions:
                continue
            
            phoneme_log_probs = predictions['phoneme_logits'].log_softmax(dim=-1)
            input_lengths = batch.sig[1] * phoneme_log_probs.shape[1]
            input_lengths = input_lengths.long()
            
            batch_predictions = decode_ctc_greedy(phoneme_log_probs, input_lengths)
            
            if hasattr(batch, 'canonical_tokens'):
                batch_targets = []
                for tokens, length in zip(batch.canonical_tokens.data, batch.canonical_tokens.lengths):
                    target = tokens[:length].cpu().numpy().tolist()
                    batch_targets.append(target)
                
                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)
                all_ids.extend(batch.id)
            elif hasattr(batch, 'phoneme_tokens'):
                batch_targets = []
                for tokens, length in zip(batch.phoneme_tokens.data, batch.phoneme_tokens.lengths):
                    target = tokens[:length].cpu().numpy().tolist()
                    batch_targets.append(target)
                
                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)
                all_ids.extend(batch.id)
    
    if not all_predictions:
        logger.warning("No phoneme recognition samples found")
        return {}
    
    pred_phonemes = [[phoneme_decoder(token_id) for token_id in seq] for seq in all_predictions]
    target_phonemes = [[phoneme_decoder(token_id) for token_id in seq] for seq in all_targets]
    
    pred_phonemes = [[p for p in seq if p != "sil"] for seq in pred_phonemes]
    target_phonemes = [[p for p in seq if p != "sil"] for seq in target_phonemes]
    
    per_details = wer_details_for_batch(
        ids=all_ids,
        refs=target_phonemes,
        hyps=pred_phonemes,
        compute_alignments=True
    )
    
    total_phonemes = sum(detail['num_ref_tokens'] for detail in per_details)
    total_errors = sum(detail['insertions'] + detail['deletions'] + detail['substitutions'] 
                      for detail in per_details)
    total_insertions = sum(detail['insertions'] for detail in per_details)
    total_deletions = sum(detail['deletions'] for detail in per_details)
    total_substitutions = sum(detail['substitutions'] for detail in per_details)
    
    per = total_errors / total_phonemes if total_phonemes > 0 else 0
    
    per_sample_metrics = [
        {
            'id': detail['key'],
            'per': detail['WER'],
            'insertions': detail['insertions'],
            'deletions': detail['deletions'],
            'substitutions': detail['substitutions'],
            'true_phonemes': detail['ref_tokens'],
            'pred_phonemes': detail['hyp_tokens']
        }
        for detail in per_details
    ]
    
    results = {
        'per': float(per),
        'accuracy': float(1.0 - per),
        'total_phonemes': int(total_phonemes),
        'total_errors': int(total_errors),
        'insertions': int(total_insertions),
        'deletions': int(total_deletions),
        'substitutions': int(total_substitutions),
        'per_sample': per_sample_metrics,
        'per_details': per_details
    }
    
    return results


def show_sample_predictions(brain, data_loader, phoneme_decoder, error_decoder, num_samples=3, device="cuda"):
    brain.modules.eval()
    
    sample_count = 0
    results = []
    
    with torch.no_grad():
        for batch in data_loader:
            if sample_count >= num_samples:
                break
            
            batch = batch.to(device)
            predictions = brain.compute_forward(batch, sb.Stage.TEST)
            
            for i in range(min(len(batch.id), num_samples - sample_count)):
                sample_result = {
                    'id': batch.id[i],
                    'predictions': {}
                }
                
                if 'error_logits' in predictions and hasattr(batch, 'error_tokens'):
                    error_log_probs = predictions['error_logits'][i:i+1].log_softmax(dim=-1)
                    input_length = torch.tensor([error_log_probs.shape[1]])
                    error_pred = decode_ctc_greedy(error_log_probs, input_length)[0]
                    
                    error_target = batch.error_tokens.data[i][:batch.error_tokens.lengths[i]].cpu().numpy().tolist()
                    
                    sample_result['predictions']['error'] = {
                        'predicted': [error_decoder(p) for p in error_pred],
                        'target': [error_decoder(t) for t in error_target]
                    }
                
                if 'phoneme_logits' in predictions and hasattr(batch, 'phoneme_tokens'):
                    phoneme_log_probs = predictions['phoneme_logits'][i:i+1].log_softmax(dim=-1)
                    input_length = torch.tensor([phoneme_log_probs.shape[1]])
                    phoneme_pred = decode_ctc_greedy(phoneme_log_probs, input_length)[0]
                    
                    phoneme_target = batch.phoneme_tokens.data[i][:batch.phoneme_tokens.lengths[i]].cpu().numpy().tolist()
                    
                    sample_result['predictions']['phoneme'] = {
                        'predicted': [phoneme_decoder(p) for p in phoneme_pred],
                        'target': [phoneme_decoder(t) for t in phoneme_target]
                    }
                
                results.append(sample_result)
                sample_count += 1
                
                if sample_count >= num_samples:
                    break
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Multi-task Model')
    parser.add_argument('hparams_file', type=str, help='Hyperparameters file')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--output_folder', type=str, default='./eval_results', help='Output folder')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    with open(args.hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
    
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
    
    from model import MultiTaskBrain, MultiTaskModel
    from data_prepare import dataio_prepare
    
    train_loader, val_loader, test_loader = dataio_prepare(hparams)
    
    model = MultiTaskModel(hparams)
    modules = {"model": model, "wav2vec2": model.wav2vec2}
    
    brain = MultiTaskBrain(
        modules=modules,
        hparams=hparams,
        run_opts={"device": args.device}
    )
    
    if args.model_checkpoint:
        logger.info(f"Loading checkpoint: {args.model_checkpoint}")
        brain.checkpointer.recover_if_possible()
    
    results = {}
    
    if hparams['task'] in ['error', 'both']:
        logger.info("Evaluating error detection...")
        error_results = evaluate_error_detection(
            brain, test_loader, hparams["error_decoder"], args.device
        )
        results['error_detection'] = error_results
        
        logger.info("=== Error Detection Results ===")
        logger.info(f"Sequence Accuracy: {error_results.get('sequence_accuracy', 0):.4f}")
        logger.info(f"Token Accuracy: {error_results.get('token_accuracy', 0):.4f}")
        logger.info(f"Weighted F1: {error_results.get('weighted_f1', 0):.4f}")
    
    if hparams['task'] in ['phoneme', 'both']:
        logger.info("Evaluating phoneme recognition...")
        phoneme_results = evaluate_phoneme_recognition(
            brain, test_loader, hparams["phoneme_decoder"], args.device
        )
        results['phoneme_recognition'] = phoneme_results
        
        logger.info("=== Phoneme Recognition Results ===")
        logger.info(f"PER: {phoneme_results.get('per', 0):.4f}")
        logger.info(f"Accuracy: {phoneme_results.get('accuracy', 0):.4f}")
    
    logger.info("Generating sample predictions...")
    sample_results = show_sample_predictions(
        brain, test_loader, hparams["phoneme_decoder"], hparams["error_decoder"], 
        num_samples=5, device=args.device
    )
    
    results_file = os.path.join(args.output_folder, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    samples_file = os.path.join(args.output_folder, 'sample_predictions.json')
    with open(samples_file, 'w') as f:
        json.dump(sample_results, f, indent=2)
    
    summary = {}
    if 'error_detection' in results:
        summary['error_token_accuracy'] = results['error_detection'].get('token_accuracy', 0)
        summary['error_sequence_accuracy'] = results['error_detection'].get('sequence_accuracy', 0)
    if 'phoneme_recognition' in results:
        summary['phoneme_error_rate'] = results['phoneme_recognition'].get('per', 0)
        summary['phoneme_accuracy'] = results['phoneme_recognition'].get('accuracy', 0)
    
    summary_file = os.path.join(args.output_folder, 'evaluation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {args.output_folder}")
    logger.info("=== SUMMARY ===")
    for key, value in summary.items():
        logger.info(f"{key}: {value:.4f}")
    
    return results


if __name__ == "__main__":
    main()