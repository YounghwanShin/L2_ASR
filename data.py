import torch
import torchaudio
import json
from torch.utils.data import Dataset

class PhonemeRecognitionDataset(Dataset):
    """음소 인식을 위한 데이터셋"""
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
        
        # 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 리샘플링
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        
        # 길이 제한
        if self.max_length is not None and waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        
        # 음소 레이블 변환
        phoneme_target = item.get('perceived_train_target', '')
        phoneme_labels = []
        for phoneme in phoneme_target.split():
            if phoneme in self.phoneme_to_id:
                phoneme_labels.append(self.phoneme_to_id[phoneme])
        
        phoneme_labels = torch.tensor(phoneme_labels, dtype=torch.long)
        label_length = torch.tensor(len(phoneme_labels), dtype=torch.long)
        
        return waveform.squeeze(0), phoneme_labels, label_length, wav_file

class PhonemeEvaluationDataset(Dataset):
    """음소 인식 평가를 위한 데이터셋"""
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
        
        # 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 리샘플링
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        
        # 길이 제한
        if self.max_length is not None and waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        
        # 음소 레이블 변환
        perceived_phonemes = item.get('perceived_train_target', '').split()
        perceived_phoneme_ids = []
        for phoneme in perceived_phonemes:
            if phoneme in self.phoneme_to_id:
                perceived_phoneme_ids.append(self.phoneme_to_id[phoneme])
        
        perceived_phoneme_ids = torch.tensor(perceived_phoneme_ids, dtype=torch.long)
        
        # 정규 발음 음소 레이블 변환 (참고용)
        canonical_phonemes = item.get('canonical_aligned', '').split()
        canonical_phoneme_ids = []
        for phoneme in canonical_phonemes:
            if phoneme in self.phoneme_to_id:
                canonical_phoneme_ids.append(self.phoneme_to_id[phoneme])
        
        canonical_phoneme_ids = torch.tensor(canonical_phoneme_ids, dtype=torch.long)
        
        # 음성 길이와 레이블 길이
        audio_length = torch.tensor(waveform.shape[1], dtype=torch.long)
        perceived_length = torch.tensor(len(perceived_phoneme_ids), dtype=torch.long)
        canonical_length = torch.tensor(len(canonical_phoneme_ids), dtype=torch.long)
        
        return (
            waveform.squeeze(0), 
            perceived_phoneme_ids, 
            canonical_phoneme_ids,
            audio_length,
            perceived_length,
            canonical_length,
            wav_file
        )