import os
import logging
import torch
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score
from speechbrain.utils.edit_distance import wer_details_for_batch

EDIT_SYMBOLS = {
    "eq": "=",
    "ins": "I",
    "del": "D",
    "sub": "S",
}

def make_attn_mask(wavs, wav_lens):
    abs_lens = (wav_lens * wavs.shape[1]).long()
    attn_mask = wavs.new(wavs.shape).zero_().long()
    for i in range(len(abs_lens)):
        attn_mask[i, :abs_lens[i]] = 1
    return attn_mask

def get_multitask_model_class(model_type):
    if model_type == 'simple':
        from models.multitask_model import SimpleMultiTaskModel
        from models.loss_functions import MultiTaskLoss
        return SimpleMultiTaskModel, MultiTaskLoss
    elif model_type == 'transformer':
        from models.multitask_model_transformer import TransformerMultiTaskModel
        from models.loss_functions import MultiTaskLoss
        return TransformerMultiTaskModel, MultiTaskLoss
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: simple, transformer")

def get_phoneme_model_class(model_type):
    if model_type == 'simple':
        from models.phoneme_model import SimplePhonemeModel
        from models.loss_functions import PhonemeLoss
        return SimplePhonemeModel, PhonemeLoss
    elif model_type == 'transformer':
        from models.phoneme_model_transformer import TransformerPhonemeModel
        from models.loss_functions import PhonemeLoss
        return TransformerPhonemeModel, PhonemeLoss
    else:
        raise ValueError(f"Unknown phoneme model type: {model_type}. Available: simple, transformer")

def detect_model_type_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict
    
    state_dict = remove_module_prefix(state_dict)
    keys = list(state_dict.keys())
    
    if any('transformer_encoder' in key for key in keys):
        return 'transformer'
    elif any('shared_encoder' in key for key in keys):
        return 'simple'
    else:
        return 'simple'

def detect_phoneme_model_type_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict
    
    state_dict = remove_module_prefix(state_dict)
    keys = list(state_dict.keys())
    
    if any('transformer_encoder' in key for key in keys):
        return 'transformer'
    elif any('shared_encoder' in key for key in keys):
        return 'simple'

def setup_experiment_dirs(config, resume=False):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    config_path = os.path.join(config.experiment_dir, 'config.json')
    if not resume:
        config.save_config(config_path)
    
    log_file = os.path.join(config.log_dir, 'training.log')
    file_mode = 'a' if resume else 'w'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode=file_mode),
            logging.StreamHandler()
        ]
    )

def enable_wav2vec2_specaug(model, enable=True):
    actual_model = model.module if hasattr(model, 'module') else model
    if hasattr(actual_model.encoder.wav2vec2, 'config'):
        actual_model.encoder.wav2vec2.config.apply_spec_augment = enable

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

def calculate_soft_length(outputs, config):
    probs = torch.softmax(outputs, dim=-1)
    non_blank_probs = 1.0 - probs[:, :, 0]
    non_blank_probs = torch.sigmoid(10 * (non_blank_probs - 0.5))
    
    phoneme_probs = probs[:, :, 1:]
    soft_preds = torch.matmul(
        phoneme_probs, torch.arange(1, phoneme_probs.size(-1) + 1, device=phoneme_probs.device, dtype=phoneme_probs.dtype)
    )
    
    preds_shift = torch.roll(soft_preds, shifts=1, dims=1)

    diff = torch.abs(soft_preds - preds_shift) / 42.0
    change_probs = torch.sigmoid(config.sigmoid_k * (diff - config.sigmoid_threshold))
    change_probs = torch.cat([torch.ones(change_probs.size(0), 1, device=change_probs.device, dtype=change_probs.dtype), change_probs[:, 1:]], dim=1)

    soft_length = (non_blank_probs * change_probs).sum(dim=1)
    return soft_length

def show_sample_predictions(task_mode, model, eval_dataloader, device, id_to_phoneme, logger, error_type_names=None, num_samples=3):
    model.eval()
    enable_wav2vec2_specaug(model, False)
    samples_shown = 0
    
    with torch.no_grad():
        for batch_data in eval_dataloader:
            if samples_shown >= num_samples:
                break
                
            if task_mode == 'multi_eval':
                (waveforms, error_labels, perceived_phoneme_ids, canonical_phoneme_ids, 
                 audio_lengths, error_label_lengths, perceived_lengths, canonical_lengths, wav_files, spk_ids) = batch_data
            elif task_mode == 'phoneme_eval':
                (waveforms, perceived_phoneme_ids, canonical_phoneme_ids, 
                 audio_lengths, perceived_lengths, canonical_lengths, wav_files, spk_ids) = batch_data
            
            waveforms = waveforms.to(device)
            audio_lengths = audio_lengths.to(device)
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
            attention_mask = make_attn_mask(waveforms, wav_lens_norm)
            
            if task_mode == 'multi_eval':
                outputs = model(waveforms, attention_mask, task_mode=task_mode)
            
                error_logits = outputs['error_logits']
                error_input_lengths = torch.clamp(input_lengths, min=1, max=error_logits.size(1))
                error_log_probs = torch.log_softmax(error_logits, dim=-1)
                error_predictions = decode_ctc(error_log_probs, error_input_lengths)

                phoneme_logits = outputs['phoneme_logits']
                phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=phoneme_logits.size(1))
                phoneme_log_probs = torch.log_softmax(phoneme_logits, dim=-1)
                phoneme_predictions = decode_ctc(phoneme_log_probs, phoneme_input_lengths)
            
                for i in range(min(waveforms.shape[0], num_samples - samples_shown)):
                    logger.info(f"\n--- Multi-task Sample {samples_shown + 1} ---")
                    logger.info(f"File: {wav_files[i]}")

                    error_actual = [error_type_names.get(int(label), str(label)) 
                                  for label in error_labels[i][:error_label_lengths[i]]]
                    error_pred = [error_type_names.get(int(pred), str(pred)) 
                                for pred in error_predictions[i]]

                    phoneme_actual = [id_to_phoneme.get(str(int(pid)), f"UNK_{pid}") 
                                    for pid in perceived_phoneme_ids[i][:perceived_lengths[i]]]
                    phoneme_pred = [id_to_phoneme.get(str(int(pid)), f"UNK_{pid}") 
                                  for pid in phoneme_predictions[i]]

                    logger.info(f"Error Actual:    {' '.join(error_actual)}")
                    logger.info(f"Error Predicted: {' '.join(error_pred)}")
                    logger.info(f"Phoneme Actual:    {' '.join(phoneme_actual)}")
                    logger.info(f"Phoneme Predicted: {' '.join(phoneme_pred)}")

                    samples_shown += 1
                    if samples_shown >= num_samples:
                        break
            
            elif task_mode == 'phoneme_eval':
                outputs = model(waveforms, attention_mask)
                
                phoneme_logits = outputs['phoneme_logits']
                phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=phoneme_logits.size(1))
                phoneme_log_probs = torch.log_softmax(phoneme_logits, dim=-1)
                phoneme_predictions = decode_ctc(phoneme_log_probs, phoneme_input_lengths)

                for i in range(min(waveforms.shape[0], num_samples - samples_shown)):
                    logger.info(f"\n--- Phoneme Sample {samples_shown + 1} ---")
                    logger.info(f"File: {wav_files[i]}")

                    phoneme_actual = [id_to_phoneme.get(str(int(pid)), f"UNK_{pid}") 
                                    for pid in perceived_phoneme_ids[i][:perceived_lengths[i]]]
                    phoneme_pred = [id_to_phoneme.get(str(int(pid)), f"UNK_{pid}") 
                                  for pid in phoneme_predictions[i]]

                    logger.info(f"Phoneme Actual:    {' '.join(phoneme_actual)}")
                    logger.info(f"Phoneme Predicted: {' '.join(phoneme_pred)}")
                
                    samples_shown += 1
                    if samples_shown >= num_samples:
                        break
            
            if samples_shown >= num_samples:
                break

def get_wav2vec2_output_lengths_official(model, input_lengths):
    actual_model = model.module if hasattr(model, 'module') else model
    wav2vec_model = actual_model.encoder.wav2vec2
    return wav2vec_model._get_feat_extract_output_lengths(input_lengths)

def convert_ids_to_phonemes(sequences, id_to_phoneme):
    return [[id_to_phoneme.get(str(token_id), f"UNK_{token_id}") for token_id in seq] for seq in sequences]

def remove_sil_tokens(sequences):
    return [[token for token in seq if token != "sil"] for seq in sequences]

def calculate_mispronunciation_metrics(all_predictions, all_canonical, all_perceived, id_to_phoneme):
    pred_phonemes = convert_ids_to_phonemes(all_predictions, id_to_phoneme)
    canonical_phonemes = convert_ids_to_phonemes(all_canonical, id_to_phoneme)
    perceived_phonemes = convert_ids_to_phonemes(all_perceived, id_to_phoneme)
    
    pred_phonemes = remove_sil_tokens(pred_phonemes)
    canonical_phonemes = remove_sil_tokens(canonical_phonemes)
    perceived_phonemes = remove_sil_tokens(perceived_phonemes)
    
    total_ta, total_fr, total_fa, total_tr = 0, 0, 0, 0
    
    for pred, canonical, perceived in zip(pred_phonemes, canonical_phonemes, perceived_phonemes):
        if len(canonical) == 0 or len(perceived) == 0 or len(pred) == 0:
            continue
            
        max_len = max(len(canonical), len(perceived), len(pred))
        
        canonical_padded = canonical + ['<pad>'] * (max_len - len(canonical))
        perceived_padded = perceived + ['<pad>'] * (max_len - len(perceived))
        pred_padded = pred + ['<pad>'] * (max_len - len(pred))
        
        for c_phone, p_phone, pred_phone in zip(canonical_padded, perceived_padded, pred_padded):
            if c_phone == '<pad>' or p_phone == '<pad>' or pred_phone == '<pad>':
                continue
                
            is_correct = (c_phone == p_phone)
            pred_correct = (c_phone == pred_phone)
            
            if is_correct and pred_correct:
                total_ta += 1
            elif is_correct and not pred_correct:
                total_fr += 1
            elif not is_correct and pred_correct:
                total_fa += 1
            elif not is_correct and not pred_correct:
                total_tr += 1
    
    if (total_tr + total_fa) > 0:
        recall = total_tr / (total_tr + total_fa)
    else:
        recall = 0.0
        
    if (total_tr + total_fr) > 0:
        precision = total_tr / (total_tr + total_fr)
    else:
        precision = 0.0
        
    if (precision + recall) > 0:
        f1_score = 2.0 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'ta': total_ta,
        'fr': total_fr,
        'fa': total_fa,
        'tr': total_tr
    }

def clean_targets(error_labels, label_lengths):
    return [labels[:length].cpu().numpy().tolist() for labels, length in zip(error_labels, label_lengths)]

def evaluate_error_detection(model, dataloader, device, task_mode='', error_type_names=None):
    if error_type_names is None:
        error_type_names = {0: 'blank', 1: 'incorrect', 2: 'correct'}
    
    model.eval()
    all_predictions, all_targets, all_ids, all_spk_ids = [], [], [], []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Error Detection Evaluation', dynamic_ncols=True)
        
        for batch_data in progress_bar:
            (waveforms, error_labels, _, _, audio_lengths, error_label_lengths, 
             _, _, wav_files, spk_ids) = batch_data
            
            waveforms = waveforms.to(device)
            error_labels = error_labels.to(device)
            audio_lengths = audio_lengths.to(device)
            
            wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
            attention_mask = make_attn_mask(waveforms, wav_lens_norm)
            
            outputs = model(waveforms, attention_mask, task_mode=task_mode)
            error_logits = outputs['error_logits']
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            input_lengths = torch.clamp(input_lengths, min=1, max=error_logits.size(1))
            
            log_probs = torch.log_softmax(error_logits, dim=-1)
            predictions = decode_ctc(log_probs, input_lengths)
            targets = clean_targets(error_labels, error_label_lengths)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            all_ids.extend(wav_files)
            all_spk_ids.extend(spk_ids)
    
    results = _calculate_error_metrics(all_predictions, all_targets, all_ids, error_type_names)
    
    by_country_results = {}
    country_data = defaultdict(lambda: {'predictions': [], 'targets': [], 'ids': []})
    
    for pred, target, id_val, spk_id in zip(all_predictions, all_targets, all_ids, all_spk_ids):
        country_data[spk_id]['predictions'].append(pred)
        country_data[spk_id]['targets'].append(target)
        country_data[spk_id]['ids'].append(id_val)
    
    for country, data in country_data.items():
        by_country_results[country] = _calculate_error_metrics(
            data['predictions'], data['targets'], data['ids'], error_type_names
        )
    
    results['by_country'] = by_country_results
    return results

def _calculate_error_metrics(all_predictions, all_targets, all_ids, error_type_names):
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
        'total_tokens': int(total_tokens)
    }

def evaluate_phoneme_recognition(model, dataloader, device, task_mode=None, id_to_phoneme=None):
    model.eval()
    all_predictions, all_targets, all_canonical, all_perceived, all_ids, all_spk_ids = [], [], [], [], [], []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Phoneme Recognition Evaluation', dynamic_ncols=True)
        
        for batch_data in progress_bar:
            if len(batch_data) == 8:
                (waveforms, perceived_phoneme_ids, canonical_phoneme_ids, 
                 audio_lengths, perceived_lengths, canonical_lengths, wav_files, spk_ids) = batch_data
            elif len(batch_data) == 10:
                (waveforms, _, perceived_phoneme_ids, canonical_phoneme_ids, 
                 audio_lengths, _, perceived_lengths, canonical_lengths, wav_files, spk_ids) = batch_data
            else:
                raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
            
            waveforms = waveforms.to(device)
            audio_lengths = audio_lengths.to(device)
            
            wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
            attention_mask = make_attn_mask(waveforms, wav_lens_norm)
            
            outputs = model(waveforms, attention_mask, task_mode=task_mode)
            if 'phoneme_logits' in outputs:
                phoneme_logits = outputs['phoneme_logits']
            else:
                raise ValueError("Model output does not contain 'phoneme_logits'")
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            input_lengths = torch.clamp(input_lengths, min=1, max=phoneme_logits.size(1))
            
            log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            batch_phoneme_preds = decode_ctc(log_probs, input_lengths)
            
            batch_targets = [
                perceived_phoneme_ids[i][:length].cpu().numpy().tolist()
                for i, length in enumerate(perceived_lengths)
            ]
            
            batch_canonical = [
                canonical_phoneme_ids[i][:length].cpu().numpy().tolist()
                for i, length in enumerate(canonical_lengths)
            ]
            
            batch_perceived = [
                perceived_phoneme_ids[i][:length].cpu().numpy().tolist()
                for i, length in enumerate(perceived_lengths)
            ]
            
            all_predictions.extend(batch_phoneme_preds)
            all_targets.extend(batch_targets)
            all_canonical.extend(batch_canonical)
            all_perceived.extend(batch_perceived)
            all_ids.extend(wav_files)
            all_spk_ids.extend(spk_ids)
    
    results = _calculate_phoneme_metrics(all_predictions, all_targets, all_canonical, all_perceived, all_ids, id_to_phoneme)
    
    by_country_results = {}
    country_data = defaultdict(lambda: {'predictions': [], 'targets': [], 'canonical': [], 'perceived': [], 'ids': []})
    
    for pred, target, canonical, perceived, id_val, spk_id in zip(all_predictions, all_targets, all_canonical, all_perceived, all_ids, all_spk_ids):
        country_data[spk_id]['predictions'].append(pred)
        country_data[spk_id]['targets'].append(target)
        country_data[spk_id]['canonical'].append(canonical)
        country_data[spk_id]['perceived'].append(perceived)
        country_data[spk_id]['ids'].append(id_val)
    
    for country, data in country_data.items():
        by_country_results[country] = _calculate_phoneme_metrics(
            data['predictions'], data['targets'], data['canonical'], 
            data['perceived'], data['ids'], id_to_phoneme
        )
    
    results['by_country'] = by_country_results
    return results

def _calculate_phoneme_metrics(all_predictions, all_targets, all_canonical, all_perceived, all_ids, id_to_phoneme):
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
    
    misp_metrics = {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'ta': 0, 'fr': 0, 'fa': 0, 'tr': 0}
    if all_canonical and all_perceived:
        misp_metrics = calculate_mispronunciation_metrics(all_predictions, all_canonical, all_perceived, id_to_phoneme)
    
    return {
        'per': float(per),
        'mispronunciation_precision': float(misp_metrics['precision']),
        'mispronunciation_recall': float(misp_metrics['recall']),
        'mispronunciation_f1': float(misp_metrics['f1_score']),
        'total_phonemes': int(total_phonemes),
        'total_errors': int(total_errors),
        'insertions': int(total_insertions),
        'deletions': int(total_deletions),
        'substitutions': int(total_substitutions),
        'confusion_matrix': {
            'true_acceptance': int(misp_metrics['ta']),
            'false_rejection': int(misp_metrics['fr']),
            'false_acceptance': int(misp_metrics['fa']),
            'true_rejection': int(misp_metrics['tr'])
        }
    }

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
