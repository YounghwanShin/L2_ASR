import torch
import numpy as np
import logging
from tqdm import tqdm
from speechbrain.utils.edit_distance import wer_details_for_batch

logger = logging.getLogger(__name__)

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

def evaluate_phoneme_recognition(model, dataloader, device, id_to_phoneme):
    model.eval()
    all_predictions, all_targets, all_ids = [], [], []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Phoneme Recognition Evaluation', dynamic_ncols=True)
        
        for batch_data in progress_bar:
            if len(batch_data) == 7:
                (waveforms, perceived_phoneme_ids, canonical_phoneme_ids, 
                 audio_lengths, perceived_lengths, canonical_lengths, wav_files) = batch_data
            elif len(batch_data) == 9:
                (waveforms, _, perceived_phoneme_ids, canonical_phoneme_ids, 
                 audio_lengths, _, perceived_lengths, canonical_lengths, wav_files) = batch_data
            else:
                raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
            
            waveforms = waveforms.to(device)
            audio_lengths = audio_lengths.to(device)
            
            attention_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attention_mask = (attention_mask < audio_lengths.unsqueeze(1)).float()
            
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
    
    return {
        'per': float(per),
        'total_phonemes': int(total_phonemes),
        'total_errors': int(total_errors),
        'insertions': int(total_insertions),
        'deletions': int(total_deletions),
        'substitutions': int(total_substitutions)
    }