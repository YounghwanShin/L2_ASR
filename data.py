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
        
        # C: 정확함(4), D: 삭제(1), A: 추가/삽입(3), S: 대체(2), 0: blank, 5: separator
        self.error_type_mapping = {'C': 4, 'D': 1, 'A': 3, 'S': 2}
        self.separator_token = 5  # 구분자 토큰
        
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
        
        # 오류 라벨 변환 - 구분자 토큰을 사용하여 CTC 중복 제거 문제 해결
        error_labels_str = item.get('error_labels', '')
        error_labels_list = error_labels_str.split()
        
        # 각 라벨 사이에 구분자 토큰 삽입
        modified_error_labels = []
        for i, label in enumerate(error_labels_list):
            error_code = self.error_type_mapping[label]
            modified_error_labels.append(error_code)
            
            # 마지막 라벨이 아니면 구분자 토큰 추가
            if i < len(error_labels_list) - 1:
                modified_error_labels.append(self.separator_token)
        
        error_labels = torch.tensor(modified_error_labels, dtype=torch.long)
        label_length = torch.tensor(len(error_labels), dtype=torch.long)
        
        return waveform.squeeze(0), error_labels, label_length, wav_file

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
        
        # 음소 라벨 변환
        phoneme_target = item.get('perceived_train_target', '')
        phoneme_labels = []
        for phoneme in phoneme_target.split():
            if phoneme in self.phoneme_to_id:
                phoneme_labels.append(self.phoneme_to_id[phoneme])
        
        phoneme_labels = torch.tensor(phoneme_labels, dtype=torch.long)
        label_length = torch.tensor(len(phoneme_labels), dtype=torch.long)
        
        return waveform.squeeze(0), phoneme_labels, label_length, wav_file

class EvaluationDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, max_length=None, sampling_rate=16000):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        
        # 오류 유형 매핑: C (정확함), D (삭제), A/I (추가/삽입), S (대체)
        self.error_type_mapping = {'C': 4, 'D': 1, 'A': 3, 'I': 3, 'S': 2}
        self.separator_token = 5  # 구분자 토큰
        
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
        
        # 오류 라벨 변환 - 구분자 토큰을 사용하여 CTC 중복 제거 문제 해결
        error_labels_str = item.get('error_labels', '')
        error_labels_list = error_labels_str.split()
        
        # 각 라벨 사이에 구분자 토큰 삽입
        modified_error_labels = []
        for i, label in enumerate(error_labels_list):
            error_code = self.error_type_mapping.get(label, 0)
            modified_error_labels.append(error_code)
            
            # 마지막 라벨이 아니면 구분자 토큰 추가
            if i < len(error_labels_list) - 1:
                modified_error_labels.append(self.separator_token)
        
        error_labels = torch.tensor(modified_error_labels, dtype=torch.long)
        
        # 인식된 음소 레이블 변환
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
        error_label_length = torch.tensor(len(error_labels), dtype=torch.long)
        perceived_length = torch.tensor(len(perceived_phoneme_ids), dtype=torch.long)
        canonical_length = torch.tensor(len(canonical_phoneme_ids), dtype=torch.long)
        
        return (
            waveform.squeeze(0), 
            error_labels, 
            perceived_phoneme_ids, 
            canonical_phoneme_ids,
            audio_length,
            error_label_length,
            perceived_length,
            canonical_length,
            wav_file
        )