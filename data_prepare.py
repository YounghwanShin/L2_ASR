import torch
import json
import torchaudio
from torch.utils.data import Dataset
import random

class MultiTaskDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, task_mode='both', error_task_ratio=0.5, 
                 max_length=None, sampling_rate=16000):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.task_mode = task_mode
        self.error_task_ratio = error_task_ratio
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.error_mapping = {'C': 2, 'I': 1}
        
        self.valid_files = []
        
        for wav_file in self.wav_files:
            item = self.data[wav_file]
            has_error_labels = 'error_labels' in item and item['error_labels'].strip()
            has_phoneme_labels = 'perceived_train_target' in item and item['perceived_train_target'].strip()
            
            if task_mode == 'both' and (has_error_labels or has_phoneme_labels):
                self.valid_files.append(wav_file)
            elif task_mode == 'error' and has_error_labels:
                self.valid_files.append(wav_file)
            elif task_mode == 'phoneme' and has_phoneme_labels:
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
        
        if 'error_labels' in item and item['error_labels'].strip():
            error_labels = item['error_labels'].split()
            modified_labels = []
            for label in error_labels:
                if label in self.error_mapping:
                    modified_labels.append(self.error_mapping[label])
                else:
                    modified_labels.append(0)
            
            result['error_labels'] = torch.tensor(modified_labels, dtype=torch.long)
            result['error_length'] = torch.tensor(len(modified_labels), dtype=torch.long)
        else:
            result['error_labels'] = None
            result['error_length'] = None
        
        if 'perceived_train_target' in item and item['perceived_train_target'].strip():
            phoneme_target = item['perceived_train_target']
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

class EvaluationDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, max_length=None, sampling_rate=16000):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.error_mapping = {'C': 2, 'I': 1}
        
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
        
        error_labels = item.get('error_labels', '').split()
        error_ids = []
        for label in error_labels:
            if label in self.error_mapping:
                error_ids.append(self.error_mapping[label])
            else:
                error_ids.append(0)
        
        return (
            waveform,
            torch.tensor(error_ids, dtype=torch.long),
            torch.tensor(perceived_ids, dtype=torch.long),
            torch.tensor(canonical_ids, dtype=torch.long),
            torch.tensor(waveform.shape[0], dtype=torch.long),
            torch.tensor(len(error_ids), dtype=torch.long),
            torch.tensor(len(perceived_ids), dtype=torch.long),
            torch.tensor(len(canonical_ids), dtype=torch.long),
            wav_file
        )

def simultaneous_multitask_collate_fn(batch):
    valid_samples = [item for item in batch if item['error_labels'] is not None or item['phoneme_labels'] is not None]
    
    if not valid_samples:
        return None
    
    waveforms = [sample['waveform'] for sample in valid_samples]
    max_len = max(w.shape[0] for w in waveforms)
    padded_waveforms = torch.stack([
        torch.nn.functional.pad(w, (0, max_len - w.shape[0]))
        for w in waveforms
    ])
    
    audio_lengths = torch.tensor([sample['audio_length'] for sample in valid_samples])
    
    error_labels_list = []
    error_lengths_list = []
    phoneme_labels_list = []
    phoneme_lengths_list = []
    
    for sample in valid_samples:
        if sample['error_labels'] is not None:
            error_labels_list.append(sample['error_labels'])
            error_lengths_list.append(sample['error_length'])
        else:
            error_labels_list.append(None)
            error_lengths_list.append(None)
        
        if sample['phoneme_labels'] is not None:
            phoneme_labels_list.append(sample['phoneme_labels'])
            phoneme_lengths_list.append(sample['phoneme_length'])
        else:
            phoneme_labels_list.append(None)
            phoneme_lengths_list.append(None)
    
    return {
        'waveforms': padded_waveforms,
        'audio_lengths': audio_lengths,
        'error_labels': error_labels_list,
        'error_lengths': error_lengths_list,
        'phoneme_labels': phoneme_labels_list,
        'phoneme_lengths': phoneme_lengths_list,
        'wav_files': [sample['wav_file'] for sample in valid_samples]
    }

def evaluation_collate_fn(batch):
    (waveforms, error_ids, perceived_phoneme_ids, canonical_phoneme_ids,
     audio_lengths, error_lengths, perceived_lengths, canonical_lengths, wav_files) = zip(*batch)
    
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
        pad_tensors(error_ids),
        pad_tensors(perceived_phoneme_ids),
        pad_tensors(canonical_phoneme_ids),
        torch.tensor(audio_lengths),
        torch.tensor(error_lengths),
        torch.tensor(perceived_lengths),
        torch.tensor(canonical_lengths),
        wav_files
    )