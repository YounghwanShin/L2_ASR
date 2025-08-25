import os
import sys
import torch
import numpy as np
import librosa
import json
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

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
            has_error_labels = 'error_labels' in item and item['error_labels'] and item['error_labels'].strip()
            has_phoneme_labels = 'perceived_train_target' in item and item['perceived_train_target'] and item['perceived_train_target'].strip()

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
        resamplers_cache = {}

        for wav_file in tqdm(self.valid_files, desc="Processing audio files"):
            try:
                waveform, sample_rate = torchaudio.load(wav_file)

                if torch.cuda.is_available():
                    waveform = waveform.to(self.device)

                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

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

        perceived = item.get('perceived_train_target', '')
        canonical = item.get('canonical_aligned', '')

        if perceived and perceived.strip():
            perceived_tokens = perceived.split()
            result['phoneme_labels'] = torch.tensor(
                [self.phoneme_to_id.get(p, 0) for p in perceived_tokens],
                dtype=torch.long
            )
        else:
            result['phoneme_labels'] = torch.tensor([], dtype=torch.long)

        result['phoneme_length'] = torch.tensor(len(result['phoneme_labels']))

        if canonical and canonical.strip():
            canonical_tokens = canonical.split()
            result['canonical_labels'] = torch.tensor(
                [self.phoneme_to_id.get(p, 0) for p in canonical_tokens],
                dtype=torch.long
            )
        else:
            result['canonical_labels'] = torch.tensor([], dtype=torch.long)

        result['canonical_length'] = torch.tensor(len(result['canonical_labels']))

        if self.training_mode in ['phoneme_error', 'phoneme_error_length']:
            errors = item.get('error_labels', '')
            if errors and errors.strip():
                error_tokens = errors.split()
                error_ids = [self.error_mapping.get(e, 0) for e in error_tokens]
                result['error_labels'] = torch.tensor(error_ids, dtype=torch.long)
            else:
                result['error_labels'] = torch.tensor([], dtype=torch.long)

            result['error_length'] = torch.tensor(len(result['error_labels']))

        result['spk_id'] = item.get('spk_id', 'UNKNOWN')

        return result

def collate_fn(batch, training_mode='phoneme_only'):
    valid_samples = [item for item in batch if item['phoneme_labels'] is not None and len(item['phoneme_labels']) > 0]

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
    if phoneme_labels and all(len(l) > 0 for l in phoneme_labels):
        max_phoneme_len = max(l.shape[0] for l in phoneme_labels)
        result['phoneme_labels'] = torch.stack([
            torch.nn.functional.pad(l, (0, max_phoneme_len - l.shape[0]), value=0)
            for l in phoneme_labels
        ])
    else:
        result['phoneme_labels'] = torch.zeros((len(valid_samples), 1), dtype=torch.long)

    result['phoneme_lengths'] = torch.tensor([sample['phoneme_length'] for sample in valid_samples])

    canonical_labels = [sample.get('canonical_labels', torch.tensor([], dtype=torch.long)) for sample in valid_samples]
    valid_canonical = [l for l in canonical_labels if len(l) > 0]

    if valid_canonical:
        max_canonical_len = max(l.shape[0] for l in valid_canonical)
        padded_canonical = []
        canonical_lengths = []

        for sample in valid_samples:
            canonical = sample.get('canonical_labels', torch.tensor([], dtype=torch.long))
            if len(canonical) > 0:
                padded_canonical.append(
                    torch.nn.functional.pad(canonical, (0, max_canonical_len - canonical.shape[0]), value=0)
                )
                canonical_lengths.append(sample['canonical_length'])
            else:
                padded_canonical.append(torch.zeros(max_canonical_len, dtype=torch.long))
                canonical_lengths.append(torch.tensor(0))

        result['canonical_labels'] = torch.stack(padded_canonical)
        result['canonical_lengths'] = torch.tensor(canonical_lengths)

    if training_mode in ['phoneme_error', 'phoneme_error_length']:
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

def load_audiofile(file_path, config):
    audio_signal, sample_rate = librosa.load(file_path, duration=10, offset=0.5, sr=config.sampling_rate)
    signal = np.zeros(int(config.max_length))
    signal[:len(audio_signal)] = audio_signal
    return signal

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return list(data.keys())

def calculate_mfcc(audio, sample_rate, n_mfcc=13):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        win_length=512,
        window='hamming',
        hop_length=256,
        n_mels=128,
        fmax=sample_rate/2
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=n_mfcc)
    return mfcc

def process_data(file_list, config, batch_size=500):
    all_mel = []

    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i:i+batch_size]
        batch_audio = [load_audiofile(f, config) for f in batch_files]
        batch_audio = np.stack(batch_audio, 0)

        batch_mel = []
        for j in range(batch_audio.shape[0]):
            mel = calculate_mfcc(batch_audio[j, :], config.sampling_rate)
            batch_mel.append(mel)
            print(f"\rProcessed {i+j+1}/{len(file_list)} files", end='')
        batch_mel = np.stack(batch_mel, axis=0)
        all_mel.append(batch_mel)
    print('')
    return np.concatenate(all_mel, axis=0)

def scale_and_save(all_mel, save_path):
    b,h,w = all_mel.shape
    all_mel_flat = all_mel.reshape(b, -1)

    scaler = StandardScaler()
    all_mel_flat = scaler.fit_transform(all_mel_flat)

    all_mel = all_mel_flat.reshape(b,h,w)
    np.save(save_path, all_mel)
    print(f"{save_path} saved. shape: {all_mel.shape}")

def main():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import Config
    config = Config()

    train_files = load_data(config.train_data)
    val_files = load_data(config.val_data)
    eval_files = load_data(config.eval_data)

    mel_train = process_data(train_files, config)
    mel_val   = process_data(val_files, config)
    mel_eval  = process_data(eval_files, config)

    scale_and_save(mel_train, config.train_mfcc_data)
    scale_and_save(mel_val, config.val_mfcc_data)
    scale_and_save(mel_eval, config.eval_mfcc_data)

if __name__ == "__main__":
    main()
