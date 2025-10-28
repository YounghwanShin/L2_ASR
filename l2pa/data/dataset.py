"""Dataset module for pronunciation assessment model.

This module implements the dataset class for loading and preprocessing
audio data with canonical phoneme, perceived phoneme, and error labels.
"""

import torch
import json
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, List, Optional


class PronunciationDataset(Dataset):
    """Dataset class for pronunciation assessment.
    
    Loads audio files with corresponding labels for multiple tasks.
    Supports three training modes: phoneme_only, phoneme_error, and multitask.
    
    Attributes:
        data: Dictionary mapping audio paths to label dictionaries.
        file_paths: List of audio file paths.
        phoneme_to_id: Mapping from phoneme strings to integer IDs.
        training_mode: Current training mode.
        sampling_rate: Target sampling rate for audio.
        max_length: Maximum audio length in samples.
        device: Device for data loading.
        error_mapping: Mapping from error types to integer IDs.
        valid_files: List of valid files after filtering.
    """
    
    def __init__(self, 
                 json_path: str, 
                 phoneme_to_id: Dict[str, int], 
                 training_mode: str,
                 max_length: Optional[int] = None, 
                 sampling_rate: int = 16000, 
                 device: str = 'cuda'):
        """Initializes the dataset.
        
        Args:
            json_path: Path to data JSON file.
            phoneme_to_id: Phoneme to ID mapping dictionary.
            training_mode: Training mode selection.
            max_length: Maximum audio length in samples.
            sampling_rate: Target sampling rate.
            device: Device for data loading.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.file_paths = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.training_mode = training_mode
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.device = device
        
        # Error label mapping
        self.error_mapping = {
            'D': 1,
            'I': 2,
            'S': 3,
            'C': 4
        }

        self.valid_files = []
        self._filter_valid_files()

        if self.max_length:
            self._filter_by_length()

    def _filter_valid_files(self):
        """Filters valid files based on training mode and label availability."""
        for file_path in self.file_paths:
            item = self.data[file_path]
            has_error_labels = 'error_labels' in item and item['error_labels'] and item['error_labels'].strip()
            has_canonical_labels = 'canonical_train_target' in item and item['canonical_train_target'] and item['canonical_train_target'].strip()
            has_perceived_labels = 'perceived_train_target' in item and item['perceived_train_target'] and item['perceived_train_target'].strip()

            if self.training_mode == 'phoneme_only' and has_perceived_labels:
                self.valid_files.append(file_path)
            elif self.training_mode == 'phoneme_error':
                if has_perceived_labels or has_error_labels:
                    self.valid_files.append(file_path)
            elif self.training_mode == 'multitask':
                if has_canonical_labels and has_perceived_labels and has_error_labels:
                    self.valid_files.append(file_path)
            else:
                self.valid_files.append(file_path)

    def _filter_by_length(self):
        """Filters files by audio length.
        
        Excludes files longer than max_length to prevent memory issues.
        """
        filtered_files = []
        excluded_count = 0
        resamplers_cache = {}

        for file_path in tqdm(self.valid_files, desc="Filtering by length"):
            try:
                waveform, sample_rate = torchaudio.load(file_path)

                if torch.cuda.is_available():
                    waveform = waveform.to(self.device)

                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Resample if necessary
                if sample_rate != self.sampling_rate:
                    if torch.cuda.is_available():
                        if sample_rate not in resamplers_cache:
                            resamplers_cache[sample_rate] = torchaudio.transforms.Resample(
                                orig_freq=sample_rate, new_freq=self.sampling_rate
                            ).to(self.device)
                        waveform = resamplers_cache[sample_rate](waveform)
                    else:
                        resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
                        waveform = resampler(waveform)

                if waveform.shape[1] <= self.max_length:
                    filtered_files.append(file_path)
                else:
                    excluded_count += 1
                    print(f"Excluding long file: {file_path} ({waveform.shape[1]} samples)")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                excluded_count += 1

        print(f"Length filtering: {len(self.valid_files)} → {len(filtered_files)} files ({excluded_count} excluded)")
        self.valid_files = filtered_files

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.valid_files)

    def load_waveform(self, file_path: str) -> torch.Tensor:
        """Loads and preprocesses an audio file.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Preprocessed audio waveform tensor.
        """
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0)

    def __getitem__(self, idx: int) -> Dict:
        """Returns a single dataset item.
        
        Args:
            idx: Index of the item.
            
        Returns:
            Dictionary containing waveform, labels, and metadata.
        """
        file_path = self.valid_files[idx]
        item = self.data[file_path]

        waveform = self.load_waveform(file_path)
        result = {
            'waveform': waveform,
            'audio_lengths': torch.tensor(waveform.shape[0], dtype=torch.long),
            'file_path': file_path
        }

        # Process canonical labels
        canonical = item.get('canonical_train_target', '')
        if canonical and canonical.strip():
            canonical_tokens = canonical.split()
            result['canonical_labels'] = torch.tensor(
                [self.phoneme_to_id.get(p, 0) for p in canonical_tokens],
                dtype=torch.long
            )
        else:
            result['canonical_labels'] = torch.tensor([], dtype=torch.long)
        result['canonical_length'] = torch.tensor(len(result['canonical_labels']))

        # Process perceived labels
        perceived = item.get('perceived_train_target', '')
        if perceived and perceived.strip():
            perceived_tokens = perceived.split()
            result['perceived_labels'] = torch.tensor(
                [self.phoneme_to_id.get(p, 0) for p in perceived_tokens],
                dtype=torch.long
            )
        else:
            result['perceived_labels'] = torch.tensor([], dtype=torch.long)
        result['perceived_length'] = torch.tensor(len(result['perceived_labels']))

        # Backward compatibility
        result['phoneme_labels'] = result['perceived_labels']
        result['phoneme_length'] = result['perceived_length']

        # Process error labels
        if self.training_mode in ['phoneme_error', 'multitask']:
            errors = item.get('error_labels', '')
            if errors and errors.strip():
                error_tokens = errors.split()
                error_ids = [self.error_mapping.get(e, 0) for e in error_tokens]
                result['error_labels'] = torch.tensor(error_ids, dtype=torch.long)
            else:
                result['error_labels'] = torch.tensor([], dtype=torch.long)
            result['error_length'] = torch.tensor(len(result['error_labels']))

        result['speaker_id'] = item.get('spk_id', 'UNKNOWN')

        return result


def collate_batch(batch: List[Dict], training_mode: str = 'phoneme_only') -> Optional[Dict]:
    """Collates batch data with proper padding.
    
    Args:
        batch: List of dataset items.
        training_mode: Current training mode.
        
    Returns:
        Collated batch dictionary with padded tensors, or None if batch is empty.
    """
    valid_samples = [item for item in batch if item['perceived_labels'] is not None and len(item['perceived_labels']) > 0]

    if not valid_samples:
        return None

    # Pad audio waveforms
    waveforms = [sample['waveform'] for sample in valid_samples]
    max_len = max(waveform.shape[0] for waveform in waveforms)
    padded_waveforms = torch.stack([
        torch.nn.functional.pad(waveform, (0, max_len - waveform.shape[0]))
        for waveform in waveforms
    ])

    result = {
        'waveforms': padded_waveforms,
        'audio_lengths': torch.tensor([sample['audio_lengths'] for sample in valid_samples]),
        'file_paths': [sample['file_path'] for sample in valid_samples],
        'speaker_ids': [sample['speaker_id'] for sample in valid_samples]
    }

    # Pad canonical labels
    if training_mode == 'multitask':
        canonical_labels = [sample['canonical_labels'] for sample in valid_samples]
        if canonical_labels and all(len(l) > 0 for l in canonical_labels):
            max_canonical_len = max(l.shape[0] for l in canonical_labels)
            result['canonical_labels'] = torch.stack([
                torch.nn.functional.pad(l, (0, max_canonical_len - l.shape[0]), value=0)
                for l in canonical_labels
            ])
            result['canonical_lengths'] = torch.tensor([sample['canonical_length'] for sample in valid_samples])

    # Pad perceived labels
    perceived_labels = [sample['perceived_labels'] for sample in valid_samples]
    if perceived_labels and all(len(l) > 0 for l in perceived_labels):
        max_perceived_len = max(l.shape[0] for l in perceived_labels)
        result['perceived_labels'] = torch.stack([
            torch.nn.functional.pad(l, (0, max_perceived_len - l.shape[0]), value=0)
            for l in perceived_labels
        ])
    else:
        result['perceived_labels'] = torch.zeros((len(valid_samples), 1), dtype=torch.long)
    result['perceived_lengths'] = torch.tensor([sample['perceived_length'] for sample in valid_samples])

    # Backward compatibility
    result['phoneme_labels'] = result['perceived_labels']
    result['phoneme_lengths'] = result['perceived_lengths']

    # Pad error labels
    if training_mode in ['phoneme_error', 'multitask']:
        error_labels = [sample.get('error_labels', torch.tensor([], dtype=torch.long)) for sample in valid_samples]
        valid_error_labels = [l for l in error_labels if len(l) > 0]

        if valid_error_labels:
            max_error_len = max(l.shape[0] for l in valid_error_labels)
            padded_error_labels = []
            error_lengths = []

            for sample in valid_samples:
                error_data = sample.get('error_labels', torch.tensor([], dtype=torch.long))
                if len(error_data) > 0:
                    padded_error_labels.append(
                        torch.nn.functional.pad(error_data, (0, max_error_len - error_data.shape[0]), value=0)
                    )
                    error_lengths.append(sample['error_length'])
                else:
                    padded_error_labels.append(torch.zeros(max_error_len, dtype=torch.long))
                    error_lengths.append(torch.tensor(0))

            result['error_labels'] = torch.stack(padded_error_labels)
            result['error_lengths'] = torch.tensor(error_lengths)

    return result
