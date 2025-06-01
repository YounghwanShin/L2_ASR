import torch
import json
import torchaudio
from torch.utils.data import Dataset
import random

class MultiTaskDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, max_length=None, sampling_rate=16000, 
                 task_mode='both', error_task_ratio=0.5, oversample_error=True):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.error_mapping = {'C': 2, 'I': 1}
        self.task_mode = task_mode
        self.error_task_ratio = error_task_ratio
        
        self.valid_files = []
        self.error_files = []
        self.phoneme_files = []
        self.both_files = []
        
        for wav_file in self.wav_files:
            item = self.data[wav_file]
            has_error_labels = 'error_labels' in item and item['error_labels'].strip()
            has_phoneme_labels = 'perceived_train_target' in item and item['perceived_train_target'].strip()
            
            if has_error_labels and has_phoneme_labels:
                self.both_files.append(wav_file)
                self.valid_files.append(wav_file)
            elif has_error_labels:
                self.error_files.append(wav_file)
                if task_mode in ['error', 'both']:
                    self.valid_files.append(wav_file)
            elif has_phoneme_labels:
                self.phoneme_files.append(wav_file)
                if task_mode in ['phoneme', 'both']:
                    self.valid_files.append(wav_file)
        
        if oversample_error and task_mode == 'both':
            error_only_files = [f for f in self.error_files if 'I' in self.data[f].get('error_labels', '')]
            self.valid_files.extend(error_only_files * 2)
        
        print(f"Dataset loaded: {len(self.valid_files)} valid files")
        print(f"  - Error only: {len(self.error_files)}")
        print(f"  - Phoneme only: {len(self.phoneme_files)}")
        print(f"  - Both tasks: {len(self.both_files)}")
        
        if task_mode == 'both':
            self._create_task_schedule()
        
    def _create_task_schedule(self):
        self.task_schedule = []
        
        for wav_file in self.valid_files:
            item = self.data[wav_file]
            has_error = 'error_labels' in item and item['error_labels'].strip()
            has_phoneme = 'perceived_train_target' in item and item['perceived_train_target'].strip()
            
            if has_error and has_phoneme:
                if random.random() < 0.3:
                    self.task_schedule.append(('joint', wav_file))
                elif random.random() < self.error_task_ratio:
                    self.task_schedule.append(('error', wav_file))
                else:
                    self.task_schedule.append(('phoneme', wav_file))
            elif has_error:
                self.task_schedule.append(('error', wav_file))
            elif has_phoneme:
                self.task_schedule.append(('phoneme', wav_file))
                
        random.shuffle(self.task_schedule)
        
    def __len__(self):
        if self.task_mode == 'both':
            return len(self.task_schedule)
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        if self.task_mode == 'both':
            task_type, wav_file = self.task_schedule[idx]
        else:
            task_type = self.task_mode
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
            'wav_file': wav_file,
            'task': task_type
        }
        
        if task_type in ['error', 'joint'] and 'error_labels' in item:
            error_labels = item['error_labels'].split()
            if error_labels:
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
        else:
            result['error_labels'] = None
            result['error_length'] = None
            
        if task_type in ['phoneme', 'joint'] and 'perceived_train_target' in item:
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
        
        error_labels = item.get('error_labels', '').split()
        modified_error_labels = [self.error_mapping.get(label, 0) for label in error_labels]
        
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
            torch.tensor(modified_error_labels, dtype=torch.long),
            torch.tensor(perceived_ids, dtype=torch.long),
            torch.tensor(canonical_ids, dtype=torch.long),
            torch.tensor(waveform.shape[0], dtype=torch.long),
            torch.tensor(len(modified_error_labels), dtype=torch.long),
            torch.tensor(len(perceived_ids), dtype=torch.long),
            torch.tensor(len(canonical_ids), dtype=torch.long),
            wav_file
        )

def multitask_collate_fn(batch):
    error_samples = [item for item in batch if item['task'] in ['error', 'joint'] and item['error_labels'] is not None]
    phoneme_samples = [item for item in batch if item['task'] in ['phoneme', 'joint'] and item['phoneme_labels'] is not None]
    
    def pad_waveforms(samples):
        if not samples:
            return None, None
        waveforms = [sample['waveform'] for sample in samples]
        max_len = max(w.shape[0] for w in waveforms)
        padded = torch.stack([
            torch.nn.functional.pad(w, (0, max_len - w.shape[0]))
            for w in waveforms
        ])
        audio_lengths = torch.tensor([sample['audio_length'] for sample in samples])
        return padded, audio_lengths
    
    def pad_labels(samples, label_key, length_key):
        if not samples:
            return None, None
        labels = [sample[label_key] for sample in samples]
        max_len = max(l.shape[0] for l in labels)
        padded = torch.stack([
            torch.nn.functional.pad(l, (0, max_len - l.shape[0]), value=0)
            for l in labels
        ])
        lengths = torch.tensor([sample[length_key] for sample in samples])
        return padded, lengths
    
    result = {}
    
    if error_samples:
        error_waveforms, error_audio_lengths = pad_waveforms(error_samples)
        error_labels, error_label_lengths = pad_labels(error_samples, 'error_labels', 'error_length')
        
        result['error'] = {
            'waveforms': error_waveforms,
            'audio_lengths': error_audio_lengths,
            'labels': error_labels,
            'label_lengths': error_label_lengths,
            'wav_files': [sample['wav_file'] for sample in error_samples]
        }
    
    if phoneme_samples:
        phoneme_waveforms, phoneme_audio_lengths = pad_waveforms(phoneme_samples)
        phoneme_labels, phoneme_label_lengths = pad_labels(phoneme_samples, 'phoneme_labels', 'phoneme_length')
        
        result['phoneme'] = {
            'waveforms': phoneme_waveforms,
            'audio_lengths': phoneme_audio_lengths,
            'labels': phoneme_labels,
            'label_lengths': phoneme_label_lengths,
            'wav_files': [sample['wav_file'] for sample in phoneme_samples]
        }
    
    return result

def evaluation_collate_fn(batch):
    (waveforms, error_labels, perceived_phoneme_ids, canonical_phoneme_ids,
     audio_lengths, error_label_lengths, perceived_lengths, canonical_lengths, wav_files) = zip(*batch)
    
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
        pad_tensors(error_labels),
        pad_tensors(perceived_phoneme_ids),
        pad_tensors(canonical_phoneme_ids),
        torch.tensor(audio_lengths),
        torch.tensor(error_label_lengths),
        torch.tensor(perceived_lengths),
        torch.tensor(canonical_lengths),
        wav_files
    )