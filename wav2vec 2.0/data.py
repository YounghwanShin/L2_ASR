from torch.utils.data import Dataset
import json
import torchaudio
import torch

class PhonemeRecognitionDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, max_length=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.phoneme_to_id = phoneme_to_id
        self.items = list(self.data.items())
        self.max_length = max_length

    def __getitem__(self, idx):
        _, item = self.items[idx]
        wav_path = item["wav"]
        phoneme_str = item["perceived_train_target"]

        waveform, _ = torchaudio.load(wav_path)
        phoneme_ids = [self.phoneme_to_id[p] for p in phoneme_str.split()]
        phoneme_tensor = torch.tensor(phoneme_ids, dtype=torch.long)
        
        label_length = int(phoneme_tensor.shape[0])  # 확실히 int로 변환
        return waveform.squeeze(0), phoneme_tensor, label_length, wav_path

    def __len__(self):  
        return len(self.items)

class EvaluationDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.items = list(self.data.items())
        self.phoneme_to_id = phoneme_to_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, item = self.items[idx]
        waveform, _ = torchaudio.load(item["wav"])

        # 음소 시퀀스
        phonemes = item.get("perceived_train_target", "").split()
        phoneme_ids = [self.phoneme_to_id[p] for p in phonemes if p in self.phoneme_to_id]
        phoneme_tensor = torch.tensor(phoneme_ids, dtype=torch.long)
        phoneme_length = torch.tensor(len(phoneme_ids), dtype=torch.long)
        audio_length = torch.tensor(waveform.shape[1], dtype=torch.long)

        return (
            waveform.squeeze(0),         # [T]
            torch.tensor([]),            # dummy error label (unused)
            phoneme_tensor,              # 정답 음소 시퀀스
            torch.tensor([]),            # dummy canonical (unused)
            audio_length,                # 오디오 길이
            torch.tensor([]),            # dummy error_len (unused)
            phoneme_length,              # 정답 음소 길이
            torch.tensor([]),            # dummy canonical_len
            wav_path                     # 파일 이름
        )

def phoneme_collate_fn(batch):
    waveforms, _, perceived_ids, _, audio_lengths, _, perceived_lengths, _, wav_files = zip(*batch)

    # waveform padding
    max_audio_len = max([w.shape[0] for w in waveforms])
    padded_waveforms = torch.stack([
        torch.nn.functional.pad(w, (0, max_audio_len - w.shape[0])) for w in waveforms
    ])

    # label padding
    max_label_len = max([p.shape[0] for p in perceived_ids])
    padded_labels = torch.stack([
        torch.nn.functional.pad(p, (0, max_label_len - p.shape[0]), value=0) for p in perceived_ids
    ])

    return (
        padded_waveforms,          # [B, T]
        padded_labels,             # [B, L]
        perceived_ids,             # list of raw label tensors
        None,                      # canonical (unused)
        torch.tensor(audio_lengths),
        None,                      # error_len (unused)
        torch.tensor(perceived_lengths),
        None,                      # canonical_len (unused)
        wav_files
    )
