import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from wav2vec import NaiveWav2Vec2PhonemeModel
from data import EvaluationDataset, phoneme_collate_fn


# CTC 디코딩 함수: blank 토큰과 반복 제거
def decode_ctc(logits, input_lengths, blank=0):
    #pred = torch.argmax(logits, dim=-1).permute(1, 0)  # [B, T]
    pred = torch.argmax(logits, dim=-1)
    decoded = []
    for i, seq in enumerate(pred):
        prev = blank
        result = []
        for j in range(min(input_lengths[i], len(seq))):
            token = seq[j].item()
            if token != prev and token != blank:
                result.append(token)
            prev = token
        decoded.append(result)
    return decoded

# 레벤슈타인 거리 함수 
def levenshtein(seq1, seq2):
    import numpy as np
    dp = np.zeros((len(seq1) + 1, len(seq2) + 1), dtype=int)
    for i in range(len(seq1) + 1):
        dp[i][0] = i
    for j in range(len(seq2) + 1):
        dp[0][j] = j
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[-1][-1]

# 경로
test_json = '/home/ellt/Workspace/wav2vec/wav2vec 2.0/split_data/test.json' 
phoneme_map_path = '/home/ellt/Workspace/wav2vec/wav2vec 2.0/split_data/phoneme_to_id.json'
model_path = '/home/ellt/Workspace/wav2vec/wav2vec 2.0/checkpoints/best_model.pth'
output_path = '/home/ellt/Workspace/wav2vec/wav2vec 2.0/results/evaluate_results.json'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(phoneme_map_path, 'r') as f:
    phoneme_to_id = json.load(f)
id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}

test_dataset = EvaluationDataset(test_json, phoneme_to_id)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=phoneme_collate_fn)

model = NaiveWav2Vec2PhonemeModel(num_phonemes=len(phoneme_to_id))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
    
model.eval()

total_phonemes, total_errors = 0, 0
results = []

with torch.no_grad():
    for waveforms, _, perceived_ids, _, audio_lengths, _, perceived_lengths, _, wav_files in tqdm(test_loader, desc='Evaluating'):
        waveforms = waveforms.to(device)
        attention_mask = (
        torch.arange(waveforms.shape[1], device=device)
        .unsqueeze(0) < audio_lengths.to(device).unsqueeze(1)
        ).float()

        logits = model(waveforms, attention_mask)
        log_probs = F.log_softmax(logits, dim=2)
        #print("logits shape:", logits.shape)
        #수정함 1로
        #input_lengths = torch.full((waveforms.size(0),), logits.size(1), dtype=torch.long).to(device)
        input_lengths = torch.full(
          size=(log_probs.size(0),),  # batch size
          fill_value=log_probs.size(1),  # time steps
          dtype=torch.long
          ).to(device)

        preds = decode_ctc(log_probs, input_lengths)

        for pred, true, length, fname in zip(preds, perceived_ids, perceived_lengths, wav_files):
            true = true[:length].tolist()
            dist = levenshtein(pred, true)
            total_errors += dist
            total_phonemes += len(true)

            results.append({
                "file": fname,
                "PER": float(dist) / len(true) if len(true) > 0 else 0.0,
                "true": [id_to_phoneme.get(int(i), "UNK") for i in true],
                "pred": [id_to_phoneme.get(int(i), "UNK") for i in pred]
            })

            #results.append({
             #   "file": fname,
              #  "PER": dist / len(true) if len(true) > 0 else 0.0,
               # "true": [id_to_phoneme.get(i, "UNK") for i in true],
                #"pred": [id_to_phoneme.get(i, "UNK") for i in pred]
            #})

final_result = {
    "PER": total_errors / total_phonemes,
    "total_phonemes": int(total_phonemes),
    "total_errors": int(total_errors),
    "samples": results
}


with open(output_path, 'w') as f:
    json.dump(final_result, f, indent=2)

print(f" 평가 완료. 결과가 {output_path}에 저장되었습니다.")
