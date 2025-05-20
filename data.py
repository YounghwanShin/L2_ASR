import torch
import torchaudio
import json
from torch.utils.data import Dataset
from transformers import BertTokenizer

class PhonemeRecognitionDataset(Dataset):
    """음소 인식을 위한 데이터셋 - 텍스트 데이터 포함"""
    def __init__(self, json_path, phoneme_to_id, text_model_name="bert-base-uncased", max_length=None, sampling_rate=16000, max_text_length=128):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.max_text_length = max_text_length
        
        # BERT 토크나이저 초기화
        self.tokenizer = BertTokenizer.from_pretrained(text_model_name)
        
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
        
        # 텍스트 데이터 가져오기 및 토큰화
        text = item.get('wrd', '')
        text_encoding = self.tokenizer(
            text, 
            padding='max_length',
            max_length=self.max_text_length,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = text_encoding['input_ids'].squeeze(0)
        attention_mask = text_encoding['attention_mask'].squeeze(0)
        
        # 음소 레이블 변환
        phoneme_target = item.get('perceived_train_target', '')
        phoneme_labels = []
        for phoneme in phoneme_target.split():
            if phoneme in self.phoneme_to_id:
                phoneme_labels.append(self.phoneme_to_id[phoneme])
        
        phoneme_labels = torch.tensor(phoneme_labels, dtype=torch.long)
        label_length = torch.tensor(len(phoneme_labels), dtype=torch.long)
        
        return waveform.squeeze(0), input_ids, attention_mask, phoneme_labels, label_length, wav_file

class PhonemeEvaluationDataset(Dataset):
    """음소 인식 평가를 위한 데이터셋 - 텍스트 데이터 포함"""
    def __init__(self, json_path, phoneme_to_id, text_model_name="bert-base-uncased", max_length=None, sampling_rate=16000, max_text_length=128):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.max_text_length = max_text_length
        
        # BERT 토크나이저 초기화
        self.tokenizer = BertTokenizer.from_pretrained(text_model_name)
        
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
        
        # 텍스트 데이터 가져오기 및 토큰화
        text = item.get('wrd', '')
        text_encoding = self.tokenizer(
            text, 
            padding='max_length',
            max_length=self.max_text_length,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = text_encoding['input_ids'].squeeze(0)
        attention_mask = text_encoding['attention_mask'].squeeze(0)
        
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
        audio_length = torch.tensor(waveform.shape[0], dtype=torch.long)
        perceived_length = torch.tensor(len(perceived_phoneme_ids), dtype=torch.long)
        canonical_length = torch.tensor(len(canonical_phoneme_ids), dtype=torch.long)
        
        return (
            waveform, 
            input_ids,
            attention_mask,
            perceived_phoneme_ids, 
            canonical_phoneme_ids,
            audio_length,
            perceived_length,
            canonical_length,
            wav_file
        )

def phoneme_recognition_collate_fn(batch):
    """음소 인식용 배치 콜레이션 함수"""
    waveforms, input_ids, attention_masks, phoneme_labels, label_lengths, wav_files = zip(*batch)
    
    # 가변 길이 오디오 패딩
    max_audio_len = max([waveform.shape[0] for waveform in waveforms])
    padded_waveforms = []
    
    for waveform in waveforms:
        audio_len = waveform.shape[0]
        padding = max_audio_len - audio_len
        padded_waveform = torch.nn.functional.pad(waveform, (0, padding))
        padded_waveforms.append(padded_waveform)
    
    # 텍스트 입력은 이미 패딩되어 있으므로 스택만 수행
    
    # 음소 레이블 패딩
    max_phoneme_len = max([labels.shape[0] for labels in phoneme_labels])
    padded_phoneme_labels = []
    
    for labels in phoneme_labels:
        label_len = labels.shape[0]
        padding = max_phoneme_len - label_len
        padded_labels = torch.nn.functional.pad(labels, (0, padding), value=0)
        padded_phoneme_labels.append(padded_labels)
    
    # 텐서로 변환
    padded_waveforms = torch.stack(padded_waveforms)
    padded_input_ids = torch.stack(input_ids)
    padded_attention_masks = torch.stack(attention_masks)
    padded_phoneme_labels = torch.stack(padded_phoneme_labels)
    label_lengths = torch.tensor(label_lengths)
    
    return (
        padded_waveforms, 
        padded_input_ids, 
        padded_attention_masks, 
        padded_phoneme_labels, 
        label_lengths, 
        wav_files
    )

def phoneme_evaluation_collate_fn(batch):
    """음소 인식 평가용 콜레이션 함수"""
    (
        waveforms, 
        input_ids,
        attention_masks,
        perceived_phoneme_ids, 
        canonical_phoneme_ids,
        audio_lengths,
        perceived_lengths,
        canonical_lengths,
        wav_files
    ) = zip(*batch)
    
    # 가변 길이 오디오 패딩
    max_audio_len = max([waveform.shape[0] for waveform in waveforms])
    padded_waveforms = []
    
    for waveform in waveforms:
        audio_len = waveform.shape[0]
        padding = max_audio_len - audio_len
        padded_waveform = torch.nn.functional.pad(waveform, (0, padding))
        padded_waveforms.append(padded_waveform)
    
    # 텍스트 입력은 이미 패딩되어 있으므로 스택만 수행
    
    # 인식된 음소 레이블 패딩
    max_perceived_len = max([ids.shape[0] for ids in perceived_phoneme_ids])
    padded_perceived_ids = []
    
    for ids in perceived_phoneme_ids:
        ids_len = ids.shape[0]
        padding = max_perceived_len - ids_len
        padded_ids = torch.nn.functional.pad(ids, (0, padding), value=0)
        padded_perceived_ids.append(padded_ids)
    
    # 정규 발음 음소 레이블 패딩
    max_canonical_len = max([ids.shape[0] for ids in canonical_phoneme_ids])
    padded_canonical_ids = []
    
    for ids in canonical_phoneme_ids:
        ids_len = ids.shape[0]
        padding = max_canonical_len - ids_len
        padded_ids = torch.nn.functional.pad(ids, (0, padding), value=0)
        padded_canonical_ids.append(padded_ids)
    
    # 텐서로 변환
    padded_waveforms = torch.stack(padded_waveforms)
    padded_input_ids = torch.stack(input_ids)
    padded_attention_masks = torch.stack(attention_masks)
    padded_perceived_ids = torch.stack(padded_perceived_ids)
    padded_canonical_ids = torch.stack(padded_canonical_ids)
    
    audio_lengths = torch.tensor(audio_lengths)
    perceived_lengths = torch.tensor(perceived_lengths)
    canonical_lengths = torch.tensor(canonical_lengths)
    
    return (
        padded_waveforms,
        padded_input_ids,
        padded_attention_masks, 
        padded_perceived_ids, 
        padded_canonical_ids,
        audio_lengths,
        perceived_lengths,
        canonical_lengths,
        wav_files
    )