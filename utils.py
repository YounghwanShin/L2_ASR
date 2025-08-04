import os
import logging
import torch
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

def get_model_class(model_type):
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
