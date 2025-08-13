import torch
import json
import torchaudio
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, task_mode, error_task_ratio=None, max_length=None, sampling_rate=16000):
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
            
            if task_mode == 'multi_train' and (has_error_labels or has_phoneme_labels):
                self.valid_files.append(wav_file)
            elif task_mode == 'error_train' and has_error_labels:
                self.valid_files.append(wav_file)
            elif task_mode == 'phoneme_train' and has_phoneme_labels:
                self.valid_files.append(wav_file)
            elif task_mode.endswith('eval'):
                self.valid_files.append(wav_file)
        
        if self.max_length:
            filtered_files = []
            excluded_count = 0
            
            for wav_file in self.valid_files:
                try:
                    waveform, sample_rate = torchaudio.load(wav_file)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    if sample_rate != self.sampling_rate:
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

        if self.task_mode in ['multi_train', 'multi_eval']:
            errors = item.get('error_labels', '').split()
            error_ids = [self.error_mapping.get(e, 0) for e in errors]
            
            result['error_labels'] = torch.tensor(error_ids, dtype=torch.long) if errors else None
            result['error_length'] = torch.tensor(len(error_ids) if error_ids else 0)
        
        if self.task_mode in ['multi_train', 'multi_eval', 'phoneme_train', 'phoneme_eval']:
            perceived = item.get('perceived_train_target', '').split()
            canonical = item.get('canonical_aligned', '').split()
            
            result['phoneme_labels'] = torch.tensor(
                [self.phoneme_to_id[p] for p in perceived if p in self.phoneme_to_id],
                dtype=torch.long
            ) if perceived else None
            
            result['phoneme_length'] = torch.tensor(
                len(result['phoneme_labels']) if result['phoneme_labels'] is not None else 0
            )
            
            if self.task_mode in ['multi_eval', 'phoneme_eval']:
                result['canonical_labels'] = torch.tensor(
                    [self.phoneme_to_id[p] for p in canonical if p in self.phoneme_to_id],
                    dtype=torch.long
                ) if canonical else None

                result['canonical_length'] = torch.tensor(
                    len(result['canonical_labels']) if result['canonical_labels'] is not None else 0
                )
        
        if self.task_mode in ['multi_eval', 'phoneme_eval']:
            result['spk_id'] = item.get('spk_id', 'UNKNOWN')
        
        if self.task_mode in ['multi_eval', 'phoneme_eval']:
            if self.task_mode == 'multi_eval':
                return (
                    result['waveform'],
                    result.get('error_labels', None),
                    result.get('phoneme_labels', None),
                    result.get('canonical_labels', None),
                    result['audio_lengths'],
                    result.get('error_length', None),
                    result.get('phoneme_length', None),
                    result.get('canonical_length', None),
                    result['wav_file'],
                    result.get('spk_id', None)
                )
            elif self.task_mode == 'phoneme_eval':
                return (
                    result['waveform'],
                    result.get('phoneme_labels', None),
                    result.get('canonical_labels', None),
                    result['audio_lengths'],
                    result.get('phoneme_length', None),
                    result.get('canonical_length', None),
                    result['wav_file'],
                    result.get('spk_id', None)
                )

        return result

def collate_fn(batch, task_mode):
    if task_mode in ['multi_train', 'phoneme_train']:
        valid_samples = [item for item in batch if 
                         (item['phoneme_labels'] is not None if task_mode in ['multi_train', 'phoneme_train'] else True)
                         or (item['error_labels'] is not None if task_mode in ['multi_train'] else True)]
    
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
            'wav_files': [sample['wav_file'] for sample in valid_samples]
        }

        if task_mode == 'multi_train':
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
            
            result['error_labels'] = error_labels_list
            result['error_lengths'] = error_lengths_list
            result['phoneme_labels'] = phoneme_labels_list
            result['phoneme_lengths'] = phoneme_lengths_list

            return result

        elif task_mode == 'phoneme_train':
            phoneme_labels = [sample['phoneme_labels'] for sample in valid_samples]
            max_label_len = max(l.shape[0] for l in phoneme_labels)
            padded_phoneme_labels = torch.stack([
                torch.nn.functional.pad(l, (0, max_label_len - l.shape[0]), value=0)
                for l in phoneme_labels
            ])
            phoneme_lengths = torch.tensor([sample['phoneme_length'] for sample in valid_samples])

            result['phoneme_labels'] = padded_phoneme_labels
            result['phoneme_lengths'] = phoneme_lengths

            return result
    
    if task_mode in ['multi_eval', 'phoneme_eval']:
        def pad_tensors(tensors, pad_value=0):
            max_len = max(tensor.shape[0] for tensor in tensors)
            return torch.stack([
                torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[0]), value=pad_value)
                for tensor in tensors
            ])

        if task_mode == 'multi_eval':
            (waveforms, error_ids, perceived_phoneme_ids, canonical_phoneme_ids,
             audio_lengths, error_lengths, perceived_lengths, canonical_lengths, wav_files, spk_ids) = zip(*batch)
        
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
                wav_files,
                spk_ids
            )

        elif task_mode == 'phoneme_eval':
            (waveforms, perceived_phoneme_ids, canonical_phoneme_ids,
             audio_lengths, perceived_lengths, canonical_lengths, wav_files, spk_ids) = zip(*batch)

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
                wav_files,
                spk_ids
            )