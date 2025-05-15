from torch.utils.data import Dataset
import json
import torchaudio
import torch

class PhonemeRecognitionDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.phoneme_to_id = phoneme_to_id
        self.items = list(self.data.items())

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
