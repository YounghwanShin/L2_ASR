import torch
import numpy as np
import logging
from tqdm import tqdm
from speechbrain.utils.edit_distance import wer_details_for_batch
from collections import defaultdict

logger = logging.getLogger(__name__)

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

def evaluate_phoneme_recognition(model, dataloader, device, id_to_phoneme):
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
            
            outputs = model(waveforms, attention_mask)
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