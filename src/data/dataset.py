import torch
import json
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, List, Optional


class UnifiedDataset(Dataset):
    """통합 음성 데이터셋 클래스"""
    
    def __init__(self, 
                 json_path: str, 
                 phoneme_to_id: Dict[str, int], 
                 training_mode: str,
                 max_length: Optional[int] = None, 
                 sampling_rate: int = 16000, 
                 device: str = 'cuda'):
        """
        Args:
            json_path: 데이터 JSON 파일 경로
            phoneme_to_id: 음소-ID 매핑 딕셔너리
            training_mode: 훈련 모드 ('phoneme_only', 'phoneme_error', 'phoneme_error_length')
            max_length: 최대 오디오 길이 (샘플 단위)
            sampling_rate: 샘플링 레이트
            device: 디바이스 ('cuda' or 'cpu')
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.wav_files = list(self.data.keys())
        self.phoneme_to_id = phoneme_to_id
        self.training_mode = training_mode
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.device = device
        
        # 새로운 에러 라벨 매핑
        self.error_mapping = {
            'D': 1,  # Deletion
            'I': 2,  # Insertion
            'S': 3,  # Substitution
            'C': 4   # Correct
        }

        self.valid_files = []
        self._filter_valid_files()

        if self.max_length:
            self._filter_by_length()

    def _filter_valid_files(self):
        """유효한 파일들을 필터링합니다."""
        for wav_file in self.wav_files:
            item = self.data[wav_file]
            has_error_labels = 'error_labels' in item and item['error_labels'] and item['error_labels'].strip()
            has_phoneme_labels = 'perceived_train_target' in item and item['perceived_train_target'] and item['perceived_train_target'].strip()

            if self.training_mode == 'phoneme_only' and has_phoneme_labels:
                self.valid_files.append(wav_file)
            elif self.training_mode in ['phoneme_error', 'phoneme_error_length']:
                if has_phoneme_labels or has_error_labels:
                    self.valid_files.append(wav_file)
            else:
                self.valid_files.append(wav_file)

    def _filter_by_length(self):
        """오디오 길이로 파일들을 필터링합니다."""
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

    def load_waveform(self, wav_file: str) -> torch.Tensor:
        """오디오 파일을 로드하고 전처리합니다."""
        waveform, sample_rate = torchaudio.load(wav_file)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0)

    def __getitem__(self, idx: int) -> Dict:
        """데이터셋의 단일 아이템을 반환합니다."""
        wav_file = self.valid_files[idx]
        item = self.data[wav_file]

        waveform = self.load_waveform(wav_file)
        result = {
            'waveform': waveform,
            'audio_lengths': torch.tensor(waveform.shape[0], dtype=torch.long),
            'wav_file': wav_file
        }

        # 음소 라벨 처리
        perceived = item.get('perceived_train_target', '')
        canonical = item.get('canonical_train_target', '')

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

        # 에러 라벨 처리
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


def collate_fn(batch: List[Dict], training_mode: str = 'phoneme_only') -> Optional[Dict]:
    """배치 데이터를 콜레이션합니다."""
    valid_samples = [item for item in batch if item['phoneme_labels'] is not None and len(item['phoneme_labels']) > 0]

    if not valid_samples:
        return None

    # 오디오 데이터 패딩
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

    # 음소 라벨 패딩
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

    # 정규 음소 라벨 패딩
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

    # 에러 라벨 패딩
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