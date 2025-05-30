import torch
import json
import torchaudio
from torch.utils.data import Dataset

class ErrorLabelDataset(Dataset):
    def __init__(self, json_path, max_length=None, sampling_rate=16000):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.error_mapping = {'C': 2, 'I': 1}
        self.blank_token = 0
        self.separator_token = 3
        
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        item = self.data[wav_file]
        
        waveform, sample_rate = torchaudio.load(wav_file)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        
        if self.max_length and waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        
        error_labels = item.get('error_labels', '').split()
        modified_labels = []
        
        for i, label in enumerate(error_labels):
            modified_labels.append(self.error_mapping[label])
            if i < len(error_labels) - 1:
                modified_labels.append(self.separator_token)
        
        return (
            waveform.squeeze(0),
            torch.tensor(modified_labels, dtype=torch.long),
            torch.tensor(len(modified_labels), dtype=torch.long),
            wav_file
        )

class PhonemeRecognitionDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, max_length=None, sampling_rate=16000):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        item = self.data[wav_file]
        
        waveform, sample_rate = torchaudio.load(wav_file)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        
        if self.max_length and waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        
        phoneme_target = item.get('perceived_train_target', '')
        phoneme_labels = [
            self.phoneme_to_id[p] for p in phoneme_target.split() 
            if p in self.phoneme_to_id
        ]
        
        return (
            waveform.squeeze(0),
            torch.tensor(phoneme_labels, dtype=torch.long),
            torch.tensor(len(phoneme_labels), dtype=torch.long),
            wav_file
        )

class EvaluationDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, max_length=None, sampling_rate=16000):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.error_mapping = {'C': 2, 'I': 1}
        self.blank_token = 0
        self.separator_token = 3
        
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        item = self.data[wav_file]
        
        waveform, sample_rate = torchaudio.load(wav_file)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        
        if self.max_length and waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        
        error_labels = item.get('error_labels', '').split()
        modified_error_labels = []
        for i, label in enumerate(error_labels):
            modified_error_labels.append(self.error_mapping.get(label, 0))
            if i < len(error_labels) - 1:
                modified_error_labels.append(self.separator_token)
        
        perceived_phonemes = item.get('perceived_train_target', '').split()
        perceived_ids = [
            self.phoneme_to_id[p] for p in perceived_phonemes 
            if p in self.phoneme_to_id
        ]
        
        canonical_phonemes = item.get('canonical_aligned', '').split()
        canonical_ids = [
            self.phoneme_to_id[p] for p in canonical_phonemes 
            if p in self.phoneme_to_id
        ]
        
        return (
            waveform.squeeze(0),
            torch.tensor(modified_error_labels, dtype=torch.long),
            torch.tensor(perceived_ids, dtype=torch.long),
            torch.tensor(canonical_ids, dtype=torch.long),
            torch.tensor(waveform.shape[1], dtype=torch.long),
            torch.tensor(len(modified_error_labels), dtype=torch.long),
            torch.tensor(len(perceived_ids), dtype=torch.long),
            torch.tensor(len(canonical_ids), dtype=torch.long),
            wav_file
        )