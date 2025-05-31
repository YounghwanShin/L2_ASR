import os
import json
import argparse
import logging
import numpy as np
from tqdm import tqdm
import sys

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from speechbrain.utils.edit_distance import wer_details_for_batch

from model import ErrorDetectionModel, PhonemeRecognitionModel
from data import EvaluationDataset

def remove_sil_tokens(sequences):
    cleaned_sequences = []
    for seq in sequences:
        if isinstance(seq[0], str):
            cleaned_seq = [token for token in seq if token != "sil"]
        else:
            cleaned_seq = [token for token in seq if token != "sil"]
        cleaned_sequences.append(cleaned_seq)
    return cleaned_sequences

def get_wav2vec2_output_lengths_official(model, input_lengths):
    actual_model = model.module if hasattr(model, 'module') else model
    
    if hasattr(actual_model, 'encoder'):
        wav2vec_model = actual_model.encoder.wav2vec2
    elif hasattr(actual_model, 'error_model'):
        wav2vec_model = actual_model.error_model.encoder.wav2vec2
    else:
        wav2vec_model = actual_model
    
    return wav2vec_model._get_feat_extract_output_lengths(input_lengths)

def decode_ctc(log_probs, input_lengths, blank_idx=0):
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

def decode_and_clean(log_probs, input_lengths, blank_idx=0, separator_idx=3):
    decoded_seqs = decode_ctc(log_probs, input_lengths, blank_idx)
    
    cleaned_seqs = []
    for seq in decoded_seqs:
        cleaned_seq = [token for token in seq if token != separator_idx]
        cleaned_seqs.append(cleaned_seq)
    
    return cleaned_seqs

def clean_targets(error_labels, label_lengths, separator_idx=3):
    targets = []
    for labels, length in zip(error_labels, label_lengths):
        target_seq = labels[:length].cpu().numpy().tolist()
        clean_target = [token for token in target_seq if token != separator_idx]
        targets.append(clean_target)
    return targets

def convert_ids_to_phonemes(sequences, id_to_phoneme):
    phoneme_sequences = []
    for seq in sequences:
        phoneme_seq = []
        for token_id in seq:
            phoneme = id_to_phoneme.get(str(token_id), f"UNK_{token_id}")
            phoneme_seq.append(phoneme)
        phoneme_sequences.append(phoneme_seq)
    return phoneme_sequences

def collate_fn(batch):
    (waveforms, error_labels, perceived_phoneme_ids, canonical_phoneme_ids,
     audio_lengths, error_label_lengths, perceived_lengths, canonical_lengths, wav_files) = zip(*batch)
    
    max_audio_len = max([waveform.shape[0] for waveform in waveforms])
    padded_waveforms = []
    
    for waveform in waveforms:
        audio_len = waveform.shape[0]
        padding = max_audio_len - audio_len
        padded_waveform = torch.nn.functional.pad(waveform, (0, padding))
        padded_waveforms.append(padded_waveform)
    
    max_error_len = max([labels.shape[0] for labels in error_labels])
    padded_error_labels = []
    
    for labels in error_labels:
        label_len = labels.shape[0]
        padding = max_error_len - label_len
        padded_labels = torch.nn.functional.pad(labels, (0, padding), value=0)
        padded_error_labels.append(padded_labels)
    
    max_perceived_len = max([ids.shape[0] for ids in perceived_phoneme_ids])
    padded_perceived_ids = []
    
    for ids in perceived_phoneme_ids:
        ids_len = ids.shape[0]
        padding = max_perceived_len - ids_len
        padded_ids = torch.nn.functional.pad(ids, (0, padding), value=0)
        padded_perceived_ids.append(padded_ids)
    
    max_canonical_len = max([ids.shape[0] for ids in canonical_phoneme_ids])
    padded_canonical_ids = []
    
    for ids in canonical_phoneme_ids:
        ids_len = ids.shape[0]
        padding = max_canonical_len - ids_len
        padded_ids = torch.nn.functional.pad(ids, (0, padding), value=0)
        padded_canonical_ids.append(padded_ids)
    
    padded_waveforms = torch.stack(padded_waveforms)
    padded_error_labels = torch.stack(padded_error_labels)
    padded_perceived_ids = torch.stack(padded_perceived_ids)
    padded_canonical_ids = torch.stack(padded_canonical_ids)
    
    audio_lengths = torch.tensor(audio_lengths)
    error_label_lengths = torch.tensor(error_label_lengths)
    perceived_lengths = torch.tensor(perceived_lengths)
    canonical_lengths = torch.tensor(canonical_lengths)
    
    return (
        padded_waveforms, padded_error_labels, padded_perceived_ids, padded_canonical_ids,
        audio_lengths, error_label_lengths, perceived_lengths, canonical_lengths, wav_files
    )

def evaluate_error_detection(model, dataloader, device, error_type_names=None):
    if error_type_names is None:
        error_type_names = {0: 'blank', 1: 'incorrect', 2: 'correct', 3: 'separator'}
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_ids = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Error Detection Evaluation')
        
        for (waveforms, error_labels, _, _, audio_lengths, error_label_lengths, 
             _, _, wav_files) in progress_bar:
            
            waveforms = waveforms.to(device)
            error_labels = error_labels.to(device)
            audio_lengths = audio_lengths.to(device)
            
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
            
            if hasattr(model, 'error_model') or (hasattr(model, 'module') and hasattr(model.module, 'error_model')):
                _, error_logits = model(waveforms, attention_mask)
            else:
                error_logits = model(waveforms, attention_mask)
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            input_lengths = torch.clamp(input_lengths, min=1, max=error_logits.size(1))
            
            log_probs = torch.log_softmax(error_logits, dim=-1)
            predictions = decode_and_clean(log_probs, input_lengths)
            targets = clean_targets(error_labels, error_label_lengths)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            all_ids.extend(wav_files)
    
    # Use speechbrain's wer_details_for_batch for exact compatibility
    wer_details = wer_details_for_batch(
        ids=all_ids,
        refs=all_targets,
        hyps=all_predictions,
        compute_alignments=True
    )
    
    total_sequences = len(wer_details)
    correct_sequences = sum(1 for detail in wer_details if detail['WER'] == 0.0)
    
    total_tokens = sum(detail['num_ref_tokens'] for detail in wer_details)
    total_errors = sum(detail['insertions'] + detail['deletions'] + detail['substitutions'] for detail in wer_details)
    total_insertions = sum(detail['insertions'] for detail in wer_details)
    total_deletions = sum(detail['deletions'] for detail in wer_details)
    total_substitutions = sum(detail['substitutions'] for detail in wer_details)
    
    sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
    token_accuracy = 1 - (total_errors / total_tokens) if total_tokens > 0 else 0
    avg_edit_distance = total_errors / total_sequences if total_sequences > 0 else 0
    
    flat_predictions = [token for pred in all_predictions for token in pred]
    flat_targets = [token for target in all_targets for token in target]
    
    weighted_f1 = 0
    macro_f1 = 0
    class_metrics = {}
    
    if len(flat_predictions) > 0 and len(flat_targets) > 0:
        try:
            min_len = min(len(flat_predictions), len(flat_targets))
            flat_predictions = flat_predictions[:min_len]
            flat_targets = flat_targets[:min_len]
            
            weighted_f1 = f1_score(flat_targets, flat_predictions, average='weighted', zero_division=0)
            macro_f1 = f1_score(flat_targets, flat_predictions, average='macro', zero_division=0)
            
            class_report = classification_report(flat_targets, flat_predictions, output_dict=True, zero_division=0)
            
            eval_error_types = {k: v for k, v in error_type_names.items() if k not in [0, 3]}
            for class_id, class_name in eval_error_types.items():
                if str(class_id) in class_report:
                    class_metrics[class_name] = {
                        'precision': float(class_report[str(class_id)]['precision']),
                        'recall': float(class_report[str(class_id)]['recall']),
                        'f1': float(class_report[str(class_id)]['f1-score']),
                        'support': int(class_report[str(class_id)]['support'])
                    }
        except Exception as e:
            print(f"Error calculating class metrics: {e}")
    
    return {
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

def evaluate_phoneme_recognition(model, dataloader, device, id_to_phoneme):
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_ids = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Phoneme Recognition Evaluation')
        
        for (waveforms, _, perceived_phoneme_ids, canonical_phoneme_ids, 
             audio_lengths, _, perceived_lengths, canonical_lengths, wav_files) in progress_bar:
            
            waveforms = waveforms.to(device)
            audio_lengths = audio_lengths.to(device)
            
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
            
            phoneme_logits, _ = model(waveforms, attention_mask)
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            input_lengths = torch.clamp(input_lengths, min=1, max=phoneme_logits.size(1))
            
            log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            batch_phoneme_preds = decode_ctc(log_probs, input_lengths)
            
            batch_targets = []
            for i, length in enumerate(perceived_lengths):
                target = perceived_phoneme_ids[i][:length].cpu().numpy().tolist()
                target = [int(p) for p in target]
                batch_targets.append(target)
            
            all_predictions.extend(batch_phoneme_preds)
            all_targets.extend(batch_targets)
            all_ids.extend(wav_files)
    
    # Convert to phoneme symbols for speechbrain compatibility
    pred_phonemes = convert_ids_to_phonemes(all_predictions, id_to_phoneme)
    target_phonemes = convert_ids_to_phonemes(all_targets, id_to_phoneme)
    
    # Remove 'sil' tokens just like speechbrain does
    pred_phonemes = remove_sil_tokens(pred_phonemes)
    target_phonemes = remove_sil_tokens(target_phonemes)
    
    # Use speechbrain's wer_details_for_batch for exact PER calculation
    per_details = wer_details_for_batch(
        ids=all_ids,
        refs=target_phonemes,
        hyps=pred_phonemes,
        compute_alignments=True
    )
    
    total_phonemes = sum(detail['num_ref_tokens'] for detail in per_details)
    total_errors = sum(detail['insertions'] + detail['deletions'] + detail['substitutions'] for detail in per_details)
    total_insertions = sum(detail['insertions'] for detail in per_details)
    total_deletions = sum(detail['deletions'] for detail in per_details)
    total_substitutions = sum(detail['substitutions'] for detail in per_details)
    
    per = total_errors / total_phonemes if total_phonemes > 0 else 0
    
    per_sample_metrics = []
    for detail in per_details:
        per_sample_metrics.append({
            'wav_file': detail['key'],
            'per': detail['WER'],
            'insertions': detail['insertions'],
            'deletions': detail['deletions'],
            'substitutions': detail['substitutions'],
            'true_phonemes': detail['ref_tokens'],
            'pred_phonemes': detail['hyp_tokens']
        })
    
    return {
        'per': float(per),
        'total_phonemes': int(total_phonemes),
        'total_errors': int(total_errors),
        'insertions': int(total_insertions),
        'deletions': int(total_deletions),
        'substitutions': int(total_substitutions),
        'per_sample': per_sample_metrics,
        'per_details': per_details
    }

# Keep backward compatibility
levenshtein_distance = lambda seq1, seq2: (0, 0, 0, 0)  # Placeholder
edit_distance = levenshtein_distance

def decode_and_remove_separators(log_probs, input_lengths, blank_idx=0, separator_idx=3):
    return decode_and_clean(log_probs, input_lengths, blank_idx, separator_idx)

def prepare_target_without_separators(error_labels, label_lengths, separator_idx=3):
    return clean_targets(error_labels, label_lengths, separator_idx)

def main():
    parser = argparse.ArgumentParser(description='L2 Phoneme Recognition and Error Detection Model Evaluation')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mode', type=str, choices=['error', 'phoneme', 'both'], default='both')
    
    parser.add_argument('--eval_data', type=str, required=True)
    parser.add_argument('--phoneme_map', type=str, required=True)
    
    parser.add_argument('--error_model_checkpoint', type=str)
    parser.add_argument('--phoneme_model_checkpoint', type=str)
    parser.add_argument('--pretrained_model', type=str, default='facebook/wav2vec2-large-xlsr-53')
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_phonemes', type=int, default=42)
    parser.add_argument('--num_error_types', type=int, default=4)
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_audio_length', type=int, default=None)
    
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--detailed', action='store_true')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading phoneme mapping from: {args.phoneme_map}")
    with open(args.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    error_type_names = {0: 'blank', 1: 'incorrect', 2: 'correct', 3: 'separator'}
    
    logger.info(f"Loading evaluation dataset: {args.eval_data}")
    eval_dataset = EvaluationDataset(args.eval_data, phoneme_to_id, max_length=args.max_audio_length)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    results = {}
    
    if args.mode in ['error', 'both']:
        if not args.error_model_checkpoint:
            logger.error("Error model checkpoint required for error detection evaluation.")
            if args.mode == 'error':
                sys.exit(1)
        else:
            logger.info("Initializing error detection model")
            error_model = ErrorDetectionModel(
                pretrained_model_name=args.pretrained_model,
                hidden_dim=args.hidden_dim,
                num_error_types=args.num_error_types
            )
            
            logger.info(f"Loading error detection model checkpoint: {args.error_model_checkpoint}")
            state_dict = torch.load(args.error_model_checkpoint, map_location=args.device)
            
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            error_model.load_state_dict(new_state_dict)
            error_model = error_model.to(args.device)
            
            logger.info("Evaluating error detection...")
            error_detection_results = evaluate_error_detection(error_model, eval_dataloader, args.device, error_type_names)
            
            logger.info("\n===== Error Detection Results =====")
            logger.info(f"Sequence Accuracy: {error_detection_results['sequence_accuracy']:.4f}")
            logger.info(f"Token Accuracy: {error_detection_results['token_accuracy']:.4f}")
            logger.info(f"Average Edit Distance: {error_detection_results['avg_edit_distance']:.4f}")
            logger.info(f"Weighted F1: {error_detection_results['weighted_f1']:.4f}")
            logger.info(f"Macro F1: {error_detection_results['macro_f1']:.4f}")
            
            logger.info("\nError Type Metrics:")
            for error_type, metrics in error_detection_results['class_metrics'].items():
                logger.info(f"  {error_type}:")
                logger.info(f"    Precision: {metrics['precision']:.4f}")
                logger.info(f"    Recall: {metrics['recall']:.4f}")
                logger.info(f"    F1 Score: {metrics['f1']:.4f}")
                logger.info(f"    Support: {metrics['support']}")
                
            results['error_detection'] = error_detection_results
    
    if args.mode in ['phoneme', 'both']:
        if not args.phoneme_model_checkpoint or (args.mode == 'both' and not args.error_model_checkpoint):
            logger.error("Phoneme model checkpoint required for phoneme recognition evaluation.")
            if args.mode == 'phoneme':
                sys.exit(1)
        else:
            logger.info("Initializing phoneme recognition model")
            phoneme_model = PhonemeRecognitionModel(
                pretrained_model_name=args.pretrained_model,
                error_model_checkpoint=args.error_model_checkpoint if args.mode == 'both' else None,
                hidden_dim=args.hidden_dim,
                num_phonemes=args.num_phonemes,
                num_error_types=args.num_error_types
            )
            
            logger.info(f"Loading phoneme recognition model checkpoint: {args.phoneme_model_checkpoint}")
            state_dict = torch.load(args.phoneme_model_checkpoint, map_location=args.device)
            
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            phoneme_model.load_state_dict(new_state_dict)
            phoneme_model = phoneme_model.to(args.device)
            
            logger.info("Evaluating phoneme recognition...")
            phoneme_recognition_results = evaluate_phoneme_recognition(phoneme_model, eval_dataloader, args.device, id_to_phoneme)
            
            logger.info("\n===== Phoneme Recognition Results =====")
            logger.info(f"Phoneme Error Rate (PER): {phoneme_recognition_results['per']:.4f}")
            logger.info(f"Total Phonemes: {phoneme_recognition_results['total_phonemes']}")
            logger.info(f"Total Errors: {phoneme_recognition_results['total_errors']}")
            logger.info(f"Insertions: {phoneme_recognition_results['insertions']}")
            logger.info(f"Deletions: {phoneme_recognition_results['deletions']}")
            logger.info(f"Substitutions: {phoneme_recognition_results['substitutions']}")
            
            results['phoneme_recognition'] = {
                'per': phoneme_recognition_results['per'],
                'total_phonemes': phoneme_recognition_results['total_phonemes'],
                'total_errors': phoneme_recognition_results['total_errors'],
                'insertions': phoneme_recognition_results['insertions'],
                'deletions': phoneme_recognition_results['deletions'],
                'substitutions': phoneme_recognition_results['substitutions']
            }
            
            if args.detailed:
                per_sample_results_path = os.path.join(args.output_dir, 'per_sample_results.json')
                with open(per_sample_results_path, 'w') as f:
                    json.dump(phoneme_recognition_results['per_sample'], f, indent=2)
                logger.info(f"Per-sample PER results saved to {per_sample_results_path}")
                
                detailed_results_path = os.path.join(args.output_dir, 'detailed_per_results.json')
                with open(detailed_results_path, 'w') as f:
                    json.dump(phoneme_recognition_results['per_details'], f, indent=2)
                logger.info(f"Detailed PER results saved to {detailed_results_path}")
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved to {results_path}")

if __name__ == "__main__":
    main()