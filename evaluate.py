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

from model import ErrorDetectionModel, PhonemeRecognitionModel
from data import EvaluationDataset

def levenshtein_distance(seq1, seq2):
    size_x, size_y = len(seq1) + 1, len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=np.int32)
    ops = np.zeros((size_x, size_y, 3), dtype=np.int32)
    
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    
    for x in range(1, size_x):
        ops[x, 0, 0] = 1
    for y in range(1, size_y):
        ops[0, y, 1] = 1
    
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = matrix[x-1, y-1]
                ops[x, y] = ops[x-1, y-1]
            else:
                delete = matrix[x-1, y] + 1
                insert = matrix[x, y-1] + 1
                subst = matrix[x-1, y-1] + 1
                
                min_val = min(delete, insert, subst)
                matrix[x, y] = min_val
                
                if min_val == delete:
                    ops[x, y] = ops[x-1, y].copy()
                    ops[x, y, 0] += 1
                elif min_val == insert:
                    ops[x, y] = ops[x, y-1].copy()
                    ops[x, y, 1] += 1
                else:
                    ops[x, y] = ops[x-1, y-1].copy()
                    ops[x, y, 2] += 1
    
    deletions, insertions, substitutions = ops[size_x-1, size_y-1]
    distance = int(matrix[size_x-1, size_y-1])
    
    return distance, int(insertions), int(deletions), int(substitutions)

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
        prev = -1
        actual_length = input_lengths[b].item()
        for t in range(min(greedy_preds.shape[1], actual_length)):
            pred = greedy_preds[b, t]
            if pred != blank_idx and pred != prev:
                seq.append(int(pred))
            prev = pred
        decoded_seqs.append(seq)
    
    return decoded_seqs

def decode_and_remove_separators(log_probs, input_lengths, blank_idx=0, separator_idx=3):
    greedy_preds = torch.argmax(log_probs, dim=-1).cpu().numpy()
    batch_size = greedy_preds.shape[0]
    decoded_seqs = []
    
    for b in range(batch_size):
        seq = []
        prev = -1
        actual_length = input_lengths[b].item()
        
        for t in range(min(greedy_preds.shape[1], actual_length)):
            pred = greedy_preds[b, t]
            if pred != blank_idx and pred != prev:
                if pred != separator_idx:
                    seq.append(int(pred))
            prev = pred
        
        decoded_seqs.append(seq)
    
    return decoded_seqs

def prepare_target_without_separators(error_labels, label_lengths, separator_idx=3):
    targets = []
    for labels, length in zip(error_labels, label_lengths):
        target_seq = labels[:length].cpu().numpy().tolist()
        clean_target = []
        for token in target_seq:
            if token != separator_idx:
                clean_target.append(token)
        targets.append(clean_target)
    return targets

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
    
    total_sequences = 0
    correct_sequences = 0
    total_edit_distance = 0
    total_tokens = 0
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    
    all_predictions = []
    all_targets = []
    
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
            predictions = decode_and_remove_separators(log_probs, input_lengths)
            targets = prepare_target_without_separators(error_labels, error_label_lengths)
            
            for pred, target in zip(predictions, targets):
                total_sequences += 1
                total_tokens += len(target)
                
                if pred == target:
                    correct_sequences += 1
                
                edit_dist, insertions, deletions, substitutions = levenshtein_distance(pred, target)
                total_edit_distance += edit_dist
                total_insertions += insertions
                total_deletions += deletions
                total_substitutions += substitutions
                
                all_predictions.extend(pred)
                all_targets.extend(target)
    
    sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
    token_accuracy = 1 - (total_edit_distance / total_tokens) if total_tokens > 0 else 0
    avg_edit_distance = total_edit_distance / total_sequences if total_sequences > 0 else 0
    
    weighted_f1 = 0
    macro_f1 = 0
    class_metrics = {}
    
    if len(all_predictions) > 0 and len(all_targets) > 0:
        try:
            min_len = min(len(all_predictions), len(all_targets))
            all_predictions = all_predictions[:min_len]
            all_targets = all_targets[:min_len]
            
            weighted_f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
            macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
            
            class_report = classification_report(all_targets, all_predictions, output_dict=True, zero_division=0)
            
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
        'total_substitutions': int(total_substitutions)
    }

def evaluate_phoneme_recognition(model, dataloader, device, id_to_phoneme):
    model.eval()
    
    total_phonemes = 0
    total_errors = 0
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    
    per_sample_metrics = []
    
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
            
            for i, (preds, true_phonemes, length, wav_file) in enumerate(
                zip(batch_phoneme_preds, perceived_phoneme_ids, perceived_lengths, wav_files)):
                
                true_phonemes = true_phonemes[:length].cpu().numpy().tolist()
                true_phonemes = [int(p) for p in true_phonemes]
                
                per, insertions, deletions, substitutions = levenshtein_distance(preds, true_phonemes)
                phoneme_count = len(true_phonemes)
                
                total_phonemes += phoneme_count
                total_errors += per
                total_insertions += insertions
                total_deletions += deletions
                total_substitutions += substitutions
                
                per_sample_metrics.append({
                    'wav_file': wav_file,
                    'per': float(per / phoneme_count) if phoneme_count > 0 else 0.0,
                    'insertions': insertions,
                    'deletions': deletions,
                    'substitutions': substitutions,
                    'true_phonemes': [id_to_phoneme.get(str(p), "UNK") for p in true_phonemes],
                    'pred_phonemes': [id_to_phoneme.get(str(p), "UNK") for p in preds]
                })
        
    per = total_errors / total_phonemes if total_phonemes > 0 else 0
    
    return {
        'per': float(per),
        'total_phonemes': int(total_phonemes),
        'total_errors': int(total_errors),
        'insertions': int(total_insertions),
        'deletions': int(total_deletions),
        'substitutions': int(total_substitutions),
        'per_sample': per_sample_metrics
    }

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
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved to {results_path}")

if __name__ == "__main__":
    main()