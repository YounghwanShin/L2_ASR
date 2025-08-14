import torch
import json
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

class UnifiedDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, training_mode, max_length=None, sampling_rate=16000, device='cuda'):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.training_mode = training_mode
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.device = device
        self.error_mapping = {'C': 2, 'I': 1}
        
        self.valid_files = []
        for wav_file in self.wav_files:
            item = self.data[wav_file]
            has_error_labels = 'error_labels' in item and item['error_labels'].strip()
            has_phoneme_labels = 'perceived_train_target' in item and item['perceived_train_target'].strip()
            
            if training_mode == 'phoneme_only' and has_phoneme_labels:
                self.valid_files.append(wav_file)
            elif training_mode in ['phoneme_error', 'phoneme_error_length']:
                if has_phoneme_labels or has_error_labels:
                    self.valid_files.append(wav_file)
            else:
                self.valid_files.append(wav_file)
        
        if self.max_length:
            self._filter_by_length()
    
    def _filter_by_length(self):
        filtered_files = []
        excluded_count = 0
        
        if torch.cuda.is_available():
            resampler_16k = torchaudio.transforms.Resample(
                orig_freq=None, new_freq=self.sampling_rate
            ).to(self.device)
        
        for wav_file in tqdm(self.wav_files[:len(self.valid_files)], desc="Processing audio files"):
            try:
                waveform, sample_rate = torchaudio.load(wav_file)
                
                if torch.cuda.is_available():
                    waveform = waveform.to(self.device)
                
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                if sample_rate != self.sampling_rate:
                    if torch.cuda.is_available():
                        resampler_16k.orig_freq = sample_rate
                        waveform = resampler_16k(waveform)
                    else:
                        resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
                        waveform = resampler(waveform)
                
                if waveform.shape[1] <= self.max_length:
                    filtered_files.append(wav_file)
                else:
                    excluded_count += 1
                    print(f"Excluding long file: {wav_file} ({waveform.shape[1]} samples, {waveform.shape[1]/self.sampling_rate:.1f}s)")
            
            except Exception as e:
                print(f"Error loading {wav_file}: {e}")
                excluded_count += 1
        
        print(f"Length filtering: {len(self.valid_files)} → {len(filtered_files)} files ({excluded_count} excluded)")
        self.valid_files = filtered_files
    
    def __len__(self):
        return len(self.valid_files)
    
    def load_waveform(self, wav_file):
        waveform, sample_rate = torchaudio.load(wav_file)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)

    def __getitem__(self, idx):
        wav_file = self.valid_files[idx]
        item = self.data[wav_file]
        
        waveform = self.load_waveform(wav_file)
        result = {
            'waveform': waveform,
            'audio_lengths': torch.tensor(waveform.shape[0], dtype=torch.long),
            'wav_file': wav_file
        }

        perceived = item.get('perceived_train_target', '').split()
        canonical = item.get('canonical_aligned', '').split()
        
        result['phoneme_labels'] = torch.tensor(
            [self.phoneme_to_id[p] for p in perceived if p in self.phoneme_to_id],
            dtype=torch.long
        ) if perceived else None
        
        result['phoneme_length'] = torch.tensor(
            len(result['phoneme_labels']) if result['phoneme_labels'] is not None else 0
        )
        
        result['canonical_labels'] = torch.tensor(
            [self.phoneme_to_id[p] for p in canonical if p in self.phoneme_to_id],
            dtype=torch.long
        ) if canonical else None

        result['canonical_length'] = torch.tensor(
            len(result['canonical_labels']) if result['canonical_labels'] is not None else 0
        )

        if self.training_mode in ['phoneme_error', 'phoneme_error_length']:
            errors = item.get('error_labels', '').split()
            error_ids = [self.error_mapping.get(e, 0) for e in errors]
            
            result['error_labels'] = torch.tensor(error_ids, dtype=torch.long) if errors else None
            result['error_length'] = torch.tensor(len(error_ids) if error_ids else 0)
        
        result['spk_id'] = item.get('spk_id', 'UNKNOWN')
        
        return result

def collate_fn(batch, training_mode='phoneme_only'):
    valid_samples = [item for item in batch if item['phoneme_labels'] is not None]
    
    if not valid_samples:
        return None

    waveforms = [sample['waveform'] for sample in valid_samples]
    max_len = max(waveform.shape[0] for waveform in waveforms)
    padded_waveforms = torch.stack([
        torch.nn.functional.pad(waveform, (0, max_len - waveform.shape[0]))
        for waveform in waveforms
    ])

    result = {
        'waveforms': padded_waveforms,
        'audio_lengths': torch.tensor([sample['audio_lengths'] for sample in valid_samples]),
        'wav_files': [sample['wav_file'] for sample in valid_samples],
        'spk_ids': [sample['spk_id'] for sample in valid_samples]
    }

    phoneme_labels = [sample['phoneme_labels'] for sample in valid_samples]
    max_phoneme_len = max(l.shape[0] for l in phoneme_labels)
    result['phoneme_labels'] = torch.stack([
        torch.nn.functional.pad(l, (0, max_phoneme_len - l.shape[0]), value=0)
        for l in phoneme_labels
    ])
    result['phoneme_lengths'] = torch.tensor([sample['phoneme_length'] for sample in valid_samples])

    canonical_labels = [sample['canonical_labels'] for sample in valid_samples if sample['canonical_labels'] is not None]
    if canonical_labels:
        max_canonical_len = max(l.shape[0] for l in canonical_labels)
        result['canonical_labels'] = torch.stack([
            torch.nn.functional.pad(l, (0, max_canonical_len - l.shape[0]), value=0)
            for l in canonical_labels
        ])
        result['canonical_lengths'] = torch.tensor([sample['canonical_length'] for sample in valid_samples])

    if training_mode in ['phoneme_error', 'phoneme_error_length']:
        error_labels = [sample.get('error_labels') for sample in valid_samples]
        valid_error_labels = [l for l in error_labels if l is not None]
        
        if valid_error_labels:
            max_error_len = max(l.shape[0] for l in valid_error_labels)
            padded_error_labels = []
            error_lengths = []
            
            for sample in valid_samples:
                if sample.get('error_labels') is not None:
                    padded_error_labels.append(
                        torch.nn.functional.pad(sample['error_labels'], (0, max_error_len - sample['error_labels'].shape[0]), value=0)
                    )
                    error_lengths.append(sample['error_length'])
                else:
                    padded_error_labels.append(torch.zeros(max_error_len, dtype=torch.long))
                    error_lengths.append(torch.tensor(0))
            
            result['error_labels'] = torch.stack(padded_error_labels)
            result['error_lengths'] = torch.tensor(error_lengths)

    return result