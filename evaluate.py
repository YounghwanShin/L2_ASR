import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from speechbrain.utils.edit_distance import wer_details_for_batch

def get_wav2vec2_output_lengths_official(model, input_lengths):
    actual_model = model.module if hasattr(model, 'module') else model
    wav2vec_model = actual_model.encoder.wav2vec2
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

def clean_targets(error_labels, label_lengths):
    return [labels[:length].cpu().numpy().tolist() for labels, length in zip(error_labels, label_lengths)]

def convert_ids_to_phonemes(sequences, id_to_phoneme):
    return [[id_to_phoneme.get(str(token_id), f"UNK_{token_id}") for token_id in seq] for seq in sequences]

def remove_sil_tokens(sequences):
    return [[token for token in seq if token != "sil"] for seq in sequences]

def evaluate_error_detection(model, dataloader, device, error_type_names=None):
    if error_type_names is None:
        error_type_names = {0: 'blank', 1: 'incorrect', 2: 'correct'}
    
    model.eval()
    all_predictions, all_targets, all_ids = [], [], []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Error Detection Evaluation')
        
        for batch_data in progress_bar:
            (waveforms, error_labels, _, _, audio_lengths, error_label_lengths, 
             _, _, wav_files) = batch_data
            
            waveforms = waveforms.to(device)
            error_labels = error_labels.to(device)
            audio_lengths = audio_lengths.to(device)
            
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
            
            outputs = model(waveforms, attention_mask, task='error')
            error_logits = outputs['error_logits']
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            input_lengths = torch.clamp(input_lengths, min=1, max=error_logits.size(1))
            
            log_probs = torch.log_softmax(error_logits, dim=-1)
            predictions = decode_ctc(log_probs, input_lengths)
            targets = clean_targets(error_labels, error_label_lengths)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            all_ids.extend(wav_files)
    
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
            
            eval_error_types = {k: v for k, v in error_type_names.items() if k != 0}
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
    all_predictions, all_targets, all_ids = [], [], []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Phoneme Recognition Evaluation')
        
        for batch_data in progress_bar:
            (waveforms, _, perceived_phoneme_ids, canonical_phoneme_ids, 
             audio_lengths, _, perceived_lengths, canonical_lengths, wav_files) = batch_data
            
            waveforms = waveforms.to(device)
            audio_lengths = audio_lengths.to(device)
            
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
            
            outputs = model(waveforms, attention_mask, task='phoneme')
            phoneme_logits = outputs['phoneme_logits']
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            input_lengths = torch.clamp(input_lengths, min=1, max=phoneme_logits.size(1))
            
            log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            batch_phoneme_preds = decode_ctc(log_probs, input_lengths)
            
            batch_targets = [
                perceived_phoneme_ids[i][:length].cpu().numpy().tolist()
                for i, length in enumerate(perceived_lengths)
            ]
            
            all_predictions.extend(batch_phoneme_preds)
            all_targets.extend(batch_targets)
            all_ids.extend(wav_files)
    
    pred_phonemes = convert_ids_to_phonemes(all_predictions, id_to_phoneme)
    target_phonemes = convert_ids_to_phonemes(all_targets, id_to_phoneme)
    
    pred_phonemes = remove_sil_tokens(pred_phonemes)
    target_phonemes = remove_sil_tokens(target_phonemes)
    
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
    
    per_sample_metrics = [
        {
            'wav_file': detail['key'],
            'per': detail['WER'],
            'insertions': detail['insertions'],
            'deletions': detail['deletions'],
            'substitutions': detail['substitutions'],
            'true_phonemes': detail['ref_tokens'],
            'pred_phonemes': detail['hyp_tokens']
        }
        for detail in per_details
    ]
    
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

def show_multitask_samples(model, dataloader, device, error_type_names, id_to_phoneme, num_samples=3):
    model.eval()
    
    with torch.no_grad():
        sample_count = 0
        for batch_data in dataloader:
            if sample_count >= num_samples:
                break
                
            (waveforms, error_labels, perceived_phoneme_ids, _, 
             audio_lengths, error_label_lengths, perceived_lengths, _, wav_files) = batch_data
            
            waveforms = waveforms.to(device)
            audio_lengths = audio_lengths.to(device)
            
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
            
            outputs = model(waveforms, attention_mask, task='both')
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            
            error_predictions = []
            phoneme_predictions = []
            
            if 'error_logits' in outputs:
                error_log_probs = torch.log_softmax(outputs['error_logits'], dim=-1)
                error_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['error_logits'].size(1))
                error_predictions = decode_ctc(error_log_probs, error_input_lengths)
                
            if 'phoneme_logits' in outputs:
                phoneme_log_probs = torch.log_softmax(outputs['phoneme_logits'], dim=-1)
                phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=outputs['phoneme_logits'].size(1))
                phoneme_predictions = decode_ctc(phoneme_log_probs, phoneme_input_lengths)
            
            for i in range(min(len(wav_files), num_samples - sample_count)):
                print(f"\n--- Multi-task Sample {sample_count + 1} ---")
                print(f"File: {wav_files[i]}")
                
                if error_predictions and error_label_lengths[i] > 0:
                    error_target = error_labels[i][:error_label_lengths[i]].cpu().numpy().tolist()
                    error_pred = error_predictions[i] if i < len(error_predictions) else []
                    
                    error_target_symbols = [error_type_names.get(t, str(t)) for t in error_target]
                    error_pred_symbols = [error_type_names.get(p, str(p)) for p in error_pred]
                    
                    print(f"Error Actual:    {' '.join(error_target_symbols)}")
                    print(f"Error Predicted: {' '.join(error_pred_symbols)}")
                    
                if phoneme_predictions and perceived_lengths[i] > 0:
                    phoneme_target = perceived_phoneme_ids[i][:perceived_lengths[i]].cpu().numpy().tolist()
                    phoneme_pred = phoneme_predictions[i] if i < len(phoneme_predictions) else []
                    
                    phoneme_target_symbols = [id_to_phoneme.get(str(t), f"UNK({t})") for t in phoneme_target]
                    phoneme_pred_symbols = [id_to_phoneme.get(str(p), f"UNK({p})") for p in phoneme_pred]
                    
                    print(f"Phoneme Actual:    {' '.join(phoneme_target_symbols)}")
                    print(f"Phoneme Predicted: {' '.join(phoneme_pred_symbols)}")
                
                sample_count += 1
                if sample_count >= num_samples:
                    break
    
    model.train()