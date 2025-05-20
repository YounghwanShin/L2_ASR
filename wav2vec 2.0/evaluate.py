import numpy as np
import torch
from tqdm import tqdm

def levenshtein_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=np.int32)

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x, y - 1] + 1,
                    matrix[x - 1, y - 1] + 1
                )

    return int(matrix[size_x - 1, size_y - 1])

def decode_ctc(log_probs, input_lengths, blank_idx=0):
    preds = torch.argmax(log_probs, dim=-1).cpu().numpy()
    batch_size = preds.shape[0]
    decoded = []

    for b in range(batch_size):
        seq = []
        prev = -1
        for t in range(min(preds.shape[1], input_lengths[b].item())):
            p = preds[b, t]
            if p != blank_idx and p != prev:
                seq.append(int(p))
            prev = p
        decoded.append(seq)
    return decoded

def evaluate_phoneme_recognition(model, dataloader, device):
    model.eval()
    total_phonemes = 0
    total_errors = 0

    with torch.no_grad():
        for (waveforms, _, perceived_phoneme_ids, _, 
             audio_lengths, _, perceived_lengths, _, _) in tqdm(dataloader, desc='Eval'):

            waveforms = waveforms.to(device)
            audio_lengths = audio_lengths.to(device)

            attn_mask = torch.arange(waveforms.shape[1]).expand(waveforms.shape[0], -1).to(device)
            attn_mask = (attn_mask < audio_lengths.unsqueeze(1)).float()

            phoneme_logits, _ = model(waveforms, attn_mask)

            input_len = waveforms.size(1)
            output_len = phoneme_logits.size(1)
            input_lengths = torch.floor((audio_lengths.float() / input_len) * output_len).long()
            input_lengths = torch.clamp(input_lengths, min=1, max=output_len)

            log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            pred_seqs = decode_ctc(log_probs, input_lengths)

            for preds, true_ids, length in zip(pred_seqs, perceived_phoneme_ids, perceived_lengths):
                true = true_ids[:length].cpu().numpy().tolist()
                errors = levenshtein_distance(preds, true)
                total_errors += errors
                total_phonemes += len(true)

    per = total_errors / total_phonemes if total_phonemes > 0 else 0.0
    return {
        'per': float(per),
        'total_phonemes': int(total_phonemes),
        'total_errors': int(total_errors)
    }
