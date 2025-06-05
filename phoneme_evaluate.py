import torch
import numpy as np
import logging
from tqdm import tqdm
from speechbrain.utils.edit_distance import wer_details_for_batch

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

def rm_parallel_sil_batch(canos, percs):
    canos_out, percs_out = [], []
    assert len(canos) == len(percs)
    for cano, perc in zip(canos, percs):
        cano, perc = rm_parallel_sil(cano, perc)
        canos_out.append(cano)
        percs_out.append(perc)
    return canos_out, percs_out

def rm_parallel_sil(canos, percs):
    canos_out, percs_out = [], []
    assert len(canos) == len(percs)
    for cano, perc in zip(canos, percs):
        if (cano==perc and cano=="sil"):
            continue
        canos_out.append(cano)
        percs_out.append(perc)
    return canos_out, percs_out

def extract_alignment(a, b, gap_token="sil"):
    alignment = []
    idx_a, idx_b = 0, 0
    for str_a, str_b in zip(a, b):
        if str_a == gap_token and str_b != gap_token:
            alignment.append((EDIT_SYMBOLS["ins"], None, idx_b))
            idx_b += 1
        elif str_a != gap_token and str_b == gap_token:
            alignment.append((EDIT_SYMBOLS["del"], idx_a, None))
            idx_a += 1
        elif str_a != gap_token and str_b != gap_token and str_a != str_b:
            alignment.append((EDIT_SYMBOLS["sub"], idx_a, idx_b))
            idx_a += 1
            idx_b += 1
        else:
            alignment.append((EDIT_SYMBOLS["eq"], idx_a, idx_b))
            idx_a += 1
            idx_b += 1
    return alignment

def mpd_stats(align_c2p, align_c2h, c, p, h):
    ta, fr, fa, tr, cor_diag, err_diag = 0, 0, 0, 0, 0, 0
    
    if not align_c2p or not align_c2h:
        return ta, fr, fa, tr, cor_diag, err_diag
    
    max_c2p = max((x[1] for x in align_c2p if x[1] is not None), default=-1)
    max_c2h = max((x[1] for x in align_c2h if x[1] is not None), default=-1)
    
    if max_c2p != max_c2h:
        return ta, fr, fa, tr, cor_diag, err_diag

    i, j = 0, 0
    while i < len(align_c2p) and j < len(align_c2h):
        if align_c2p[i][1] is not None and \
           align_c2h[j][1] is not None and \
           align_c2p[i][1] == align_c2h[j][1]:
            assert align_c2p[i][0] != EDIT_SYMBOLS["ins"]
            assert align_c2h[j][0] != EDIT_SYMBOLS["ins"]
            if align_c2p[i][0] == EDIT_SYMBOLS["eq"]:
                if align_c2h[j][0] == EDIT_SYMBOLS["eq"]:
                    ta += 1
                else:
                    fr += 1
            elif align_c2p[i][0] != EDIT_SYMBOLS["eq"]:
                if align_c2h[j][0] == EDIT_SYMBOLS["eq"]:
                    fa += 1
                else:
                    tr += 1
                    if align_c2p[i][0] != align_c2h[j][0]:
                        err_diag += 1
                    elif align_c2p[i][0] == EDIT_SYMBOLS["del"] and align_c2h[j][0] == EDIT_SYMBOLS["del"]:
                        cor_diag += 1
                    elif align_c2p[i][0] == EDIT_SYMBOLS["sub"] and align_c2h[j][0] == EDIT_SYMBOLS["sub"]:
                        if i < len(p) and j < len(h) and p[align_c2p[i][2]] == h[align_c2h[j][2]]:
                            cor_diag += 1
                        else:
                            err_diag += 1
            i += 1
            j += 1
        elif align_c2p[i][1] is None and \
             align_c2h[j][1] is not None:
            fa += 1
            i += 1
        elif align_c2p[i][1] is not None and  \
             align_c2h[j][1] is None:
            fr += 1
            j += 1
        elif align_c2p[i][1] is None and align_c2h[j][1] is None:
            tr += 1
            if (align_c2p[i][2] < len(p) and align_c2h[j][2] < len(h) and 
                p[align_c2p[i][2]] == h[align_c2h[j][2]]):
                cor_diag += 1
            else:
                err_diag += 1
            i += 1
            j += 1
    
    if i == len(align_c2p) and j != len(align_c2h):
        fr += len(align_c2h[j:])
    if i != len(align_c2p) and j == len(align_c2h):
        fa += len(align_c2p[i:])

    return ta, fr, fa, tr, cor_diag, err_diag

def calculate_mpd_f1(all_predictions, all_targets, all_canonical, all_perceived, all_ids, id_to_phoneme):
    pred_phonemes = convert_ids_to_phonemes(all_predictions, id_to_phoneme)
    target_phonemes = convert_ids_to_phonemes(all_targets, id_to_phoneme)
    canonical_phonemes = convert_ids_to_phonemes(all_canonical, id_to_phoneme)
    perceived_phonemes = convert_ids_to_phonemes(all_perceived, id_to_phoneme)
    
    pred_phonemes = remove_sil_tokens(pred_phonemes)
    target_phonemes = remove_sil_tokens(target_phonemes)
    canonical_phonemes = remove_sil_tokens(canonical_phonemes)
    perceived_phonemes = remove_sil_tokens(perceived_phonemes)
    
    canonical_phonemes, perceived_phonemes = rm_parallel_sil_batch(canonical_phonemes, perceived_phonemes)
    
    total_ta, total_fr, total_fa, total_tr = 0, 0, 0, 0
    
    for i, (pred, canonical, perceived, id_name) in enumerate(zip(pred_phonemes, canonical_phonemes, perceived_phonemes, all_ids)):
        if len(canonical) == 0 or len(perceived) == 0:
            continue
            
        alignment_c2p = extract_alignment(canonical, perceived)
        
        wer_details = wer_details_for_batch(
            ids=[id_name],
            refs=[canonical],
            hyps=[pred],
            compute_alignments=True
        )
        
        if wer_details:
            alignment_c2h = wer_details[0]['alignment']
            ta, fr, fa, tr, cor_diag, err_diag = mpd_stats(alignment_c2p, alignment_c2h, canonical, perceived, pred)
            
            total_ta += ta
            total_fr += fr
            total_fa += fa
            total_tr += tr
    
    if (total_fr + total_tr) > 0 and (total_fa + total_tr) > 0:
        precision = total_tr / (total_fr + total_tr)
        recall = total_tr / (total_fa + total_tr)
        if (precision + recall) > 0:
            mpd_f1 = 2.0 * precision * recall / (precision + recall)
        else:
            mpd_f1 = 0.0
    else:
        mpd_f1 = 0.0
    
    return mpd_f1

def evaluate_phoneme_recognition(model, dataloader, device, id_to_phoneme):
    model.eval()
    all_predictions, all_targets, all_canonical, all_perceived, all_ids = [], [], [], [], []
    
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
    
    mpd_f1 = 0.0
    if all_canonical and all_perceived:
        mpd_f1 = calculate_mpd_f1(all_predictions, all_targets, all_canonical, all_perceived, all_ids, id_to_phoneme)
    
    return {
        'per': float(per),
        'mpd_f1': float(mpd_f1),
        'total_phonemes': int(total_phonemes),
        'total_errors': int(total_errors),
        'insertions': int(total_insertions),
        'deletions': int(total_deletions),
        'substitutions': int(total_substitutions)
    }