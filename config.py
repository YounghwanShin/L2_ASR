import os
import json
from datetime import datetime
from dataclasses import dataclass
from pytz import timezone

@dataclass
class Config:
    # 모델 관련 설정
    pretrained_model = "facebook/wav2vec2-large-xlsr-53"
    sampling_rate = 16000
    max_length = 160000

    num_phonemes = 42
    num_error_types = 3

    # 훈련 모드: 'phoneme_only', 'phoneme_error', 'phoneme_error_length'
    training_mode = 'phoneme_error_length'

    # 모델 아키텍처: 'simple' 또는 'transformer'
    model_type = 'transformer'

    # 훈련 하이퍼파라미터
    batch_size = 16
    eval_batch_size = 16
    num_epochs = 50
    gradient_accumulation = 2

    # 학습률 설정 (차별적 학습률)
    main_lr = 3e-4
    wav2vec_lr = 1e-5

    # 손실 가중치 (error + phoneme = 1.0 for multitask, length separate)
    error_weight = 0.4
    phoneme_weight = 0.6
    length_weight = 1.0

    # Focal loss 파라미터
    focal_alpha = 0.25
    focal_gamma = 2.0

    # 길이 손실 관련 설정
    length_loss_type = 'smooth_l1'  # 'mse', 'mae', 'smooth_l1'
    length_beta = 1.0  # SmoothL1Loss의 beta 파라미터

    # 체크포인트 저장 옵션
    save_best_error = True
    save_best_phoneme = True
    save_best_loss = True

    # 기타 설정
    wav2vec2_specaug = True
    seed = 42

    # 디렉토리 및 파일 경로
    base_experiment_dir = "../shared/experiments"
    experiment_name = None

    train_data = "../shared/data/train_labels.json"
    val_data = "../shared/data/val_labels.json"
    eval_data = "../shared/data/test_labels.json"
    phoneme_map = "../shared/data/phoneme_map.json"

    device = "cuda"

    # 모델 아키텍처별 설정
    model_configs = {
        'simple': {
            'hidden_dim': 1024,
            'dropout': 0.1
        },
        'transformer': {
            'hidden_dim': 1024,
            'num_layers': 2,
            'num_heads': 8,
            'dropout': 0.1
        }
    }

    def __post_init__(self):
        # 훈련 모드에 따른 가중치 검증
        self._validate_weights()

        # 실험명 자동 생성 (모델 타입이 변경되었거나 없는 경우)
        if self.experiment_name is None or not hasattr(self, '_last_model_type') or self._last_model_type != self.model_type:
            current_date = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d%H%M%S')

            # 훈련 모드와 모델 타입에 따른 실험명 생성
            if self.training_mode == 'phoneme_only':
                model_prefix = f'phoneme_{self.model_type}'
                self.experiment_name = f"{model_prefix}_{current_date}"
            elif self.training_mode == 'phoneme_error':
                model_prefix = f'phoneme_error_{self.model_type}'
                error_ratio = str(int(self.error_weight * 10)).zfill(2)
                phoneme_ratio = str(int(self.phoneme_weight * 10)).zfill(2)
                self.experiment_name = f"{model_prefix}{error_ratio}{phoneme_ratio}_{current_date}"
            elif self.training_mode == 'phoneme_error_length':
                model_prefix = f'phoneme_error_length_{self.model_type}'
                error_ratio = str(int(self.error_weight * 10)).zfill(2)
                phoneme_ratio = str(int(self.phoneme_weight * 10)).zfill(2)
                length_ratio = str(int(self.length_weight * 100)).zfill(2)
                # 길이 손실 타입도 실험명에 포함
                length_type_short = {'mse': 'ms', 'mae': 'ma', 'smooth_l1': 'sl'}[self.length_loss_type]
                self.experiment_name = f"{model_prefix}{error_ratio}{phoneme_ratio}l{length_ratio}{length_type_short}_{current_date}"

            self._last_model_type = self.model_type

        # 디렉토리 경로 설정
        self.experiment_dir = os.path.join(self.base_experiment_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')
        self.result_dir = os.path.join(self.experiment_dir, 'results')
        self.output_dir = self.checkpoint_dir

    def _validate_weights(self):
        """훈련 모드에 따른 가중치 검증 및 정규화"""
        if self.training_mode == 'phoneme_only':
            # Phoneme만 사용하는 경우 가중치 검사 불필요
            pass
        elif self.training_mode in ['phoneme_error', 'phoneme_error_length']:
            # phoneme_weight + error_weight = 1.0이어야 함
            total = self.phoneme_weight + self.error_weight
            if abs(total - 1.0) > 1e-6:
                print(f"경고: phoneme_weight ({self.phoneme_weight}) + error_weight ({self.error_weight}) = {total} != 1.0")
                print("가중치 자동 정규화 중...")
                self.phoneme_weight = self.phoneme_weight / total
                self.error_weight = self.error_weight / total
                print(f"정규화된 가중치: phoneme_weight={self.phoneme_weight:.3f}, error_weight={self.error_weight:.3f}")
                if self.training_mode == 'phoneme_error_length':
                    print(f"길이 가중치 (별도 페널티): {self.length_weight:.3f}")
                    print(f"길이 손실 타입: {self.length_loss_type}")

    def get_model_config(self):
        """모델 설정 반환 및 use_transformer 설정"""
        config = self.model_configs.get(self.model_type, self.model_configs['simple']).copy()
        config['use_transformer'] = (self.model_type == 'transformer')
        return config

    def has_error_component(self):
        """현재 훈련 모드가 에러 감지를 포함하는지 확인"""
        return self.training_mode in ['phoneme_error', 'phoneme_error_length']

    def has_length_component(self):
        """현재 훈련 모드가 길이 손실을 포함하는지 확인"""
        return self.training_mode == 'phoneme_error_length'

    def get_length_loss_config(self):
        """길이 손실 관련 설정 반환"""
        return {
            'loss_type': self.length_loss_type,
            'beta': self.length_beta,
            'weight': self.length_weight
        }

    def save_config(self, path):
        """설정을 JSON 파일로 저장"""
        config_dict = {
            attr: getattr(self, attr) for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)