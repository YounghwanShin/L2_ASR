import torch
import json
import torchaudio
from torch.utils.data import Dataset

class PhonemeDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, max_length=None, sampling_rate=16000):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        
        self.valid_files = []
        
        for wav_file in self.wav_files:
            item = self.data[wav_file]
            has_phoneme_labels = 'perceived_train_target' in item and item['perceived_train_target'].strip()
            
            if has_phoneme_labels:
                self.valid_files.append(wav_file)
        
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        wav_file = self.valid_files[idx]
        item = self.data[wav_file]
        
        waveform, sample_rate = torchaudio.load(wav_file)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        
        if self.max_length and waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
            
        waveform = waveform.squeeze(0)
        
        result = {
            'waveform': waveform,
            'audio_length': torch.tensor(waveform.shape[0], dtype=torch.long),
            'wav_file': wav_file
        }
        
        phoneme_target = item['perceived_train_target']
        if phoneme_target:
            phoneme_labels = []
            for p in phoneme_target.split():
                if p in self.phoneme_to_id:
                    phoneme_labels.append(self.phoneme_to_id[p])
            
            result['phoneme_labels'] = torch.tensor(phoneme_labels, dtype=torch.long)
            result['phoneme_length'] = torch.tensor(len(phoneme_labels), dtype=torch.long)
        else:
            result['phoneme_labels'] = None
            result['phoneme_length'] = None
            
        return result

class PhonemeEvaluationDataset(Dataset):
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
        
        waveform = waveform.squeeze(0)
        
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
            waveform,
            torch.tensor(perceived_ids, dtype=torch.long),
            torch.tensor(canonical_ids, dtype=torch.long),
            torch.tensor(waveform.shape[0], dtype=torch.long),
            torch.tensor(len(perceived_ids), dtype=torch.long),
            torch.tensor(len(canonical_ids), dtype=torch.long),
            wav_file
        )

def phoneme_collate_fn(batch):
    valid_samples = [item for item in batch if item['phoneme_labels'] is not None]
    
    if not valid_samples:
        return None
    
    waveforms = [sample['waveform'] for sample in valid_samples]
    max_len = max(w.shape[0] for w in waveforms)
    padded_waveforms = torch.stack([
        torch.nn.functional.pad(w, (0, max_len - w.shape[0]))
        for w in waveforms
    ])
    
    audio_lengths = torch.tensor([sample['audio_length'] for sample in valid_samples])
    
    phoneme_labels = [sample['phoneme_labels'] for sample in valid_samples]
    max_label_len = max(l.shape[0] for l in phoneme_labels)
    padded_phoneme_labels = torch.stack([
        torch.nn.functional.pad(l, (0, max_label_len - l.shape[0]), value=0)
        for l in phoneme_labels
    ])
    phoneme_lengths = torch.tensor([sample['phoneme_length'] for sample in valid_samples])
    
    return {
        'waveforms': padded_waveforms,
        'audio_lengths': audio_lengths,
        'phoneme_labels': padded_phoneme_labels,
        'phoneme_lengths': phoneme_lengths,
        'wav_files': [sample['wav_file'] for sample in valid_samples]
    }

def phoneme_evaluation_collate_fn(batch):
    (waveforms, perceived_phoneme_ids, canonical_phoneme_ids,
     audio_lengths, perceived_lengths, canonical_lengths, wav_files) = zip(*batch)
    
    def pad_tensors(tensors, pad_value=0):
        max_len = max(tensor.shape[0] for tensor in tensors)
        return torch.stack([
            torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[0]), value=pad_value)
            for tensor in tensors
        ])
    
    max_audio_len = max(waveform.shape[0] for waveform in waveforms)
    padded_waveforms = torch.stack([
        torch.nn.functional.pad(waveform, (0, max_audio_len - waveform.shape[0]))
        for waveform in waveforms
    ])
    
    return (
        padded_waveforms,
        pad_tensors(perceived_phoneme_ids),
        pad_tensors(canonical_phoneme_ids),
        torch.tensor(audio_lengths),
        torch.tensor(perceived_lengths),
        torch.tensor(canonical_lengths),
        wav_files
    )