import os
import json
from datetime import datetime
from dataclasses import dataclass
from pytz import timezone

@dataclass
class Config:
    """모델 훈련 및 평가를 위한 설정 클래스"""
    
    # 모델 설정
    pretrained_model = "facebook/wav2vec2-large-xlsr-53"
    sampling_rate = 16000
    max_length = 140000

    # 출력 차원 설정
    num_phonemes = 42
    num_error_types = 5  # blank(0), D(1), I(2), S(3), C(4)

    # 훈련 모드: 'phoneme_only', 'phoneme_error', 'phoneme_error_length'
    training_mode = 'phoneme_error'

    # 모델 아키텍처: 'simple' or 'transformer'
    model_type = 'transformer'

    # 훈련 하이퍼파라미터
    batch_size = 16
    eval_batch_size = 16
    num_epochs = 100
    gradient_accumulation = 2

    # 학습률 설정
    main_lr = 3e-4
    wav2vec_lr = 1e-5

    # 손실 함수 가중치 (error + phoneme = 1.0 for multitask, length separate)
    error_weight = 0.4
    phoneme_weight = 0.6
    length_weight = 1.0

    # Focal Loss 파라미터
    focal_alpha = 0.25
    focal_gamma = 2.0

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

    train_data = "data/train_labels.json"
    val_data = "data/val_labels.json"
    eval_data = "data/test_labels.json"
    phoneme_map = "data/phoneme_map.json"

    device = "cuda"

    # 모델 설정
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
        """초기화 후 처리"""
        self._validate_weights()

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
                self.experiment_name = f"{model_prefix}{error_ratio}{phoneme_ratio}l{length_ratio}_{current_date}"

            self._last_model_type = self.model_type

        # 디렉토리 경로 설정
        self.experiment_dir = os.path.join(self.base_experiment_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.length_logs_dir = os.path.join(self.experiment_dir, 'length_logs')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')
        self.result_dir = os.path.join(self.experiment_dir, 'results')
        self.output_dir = self.checkpoint_dir

    def _validate_weights(self):
        """가중치 유효성 검사 및 정규화"""
        if self.training_mode == 'phoneme_only':
            pass
        elif self.training_mode in ['phoneme_error', 'phoneme_error_length']:
            # phoneme_weight + error_weight should equal 1.0
            total = self.phoneme_weight + self.error_weight
            if abs(total - 1.0) > 1e-6:
                print(f"Warning: phoneme_weight ({self.phoneme_weight}) + error_weight ({self.error_weight}) = {total} != 1.0")
                print("Auto-normalizing weights...")
                self.phoneme_weight = self.phoneme_weight / total
                self.error_weight = self.error_weight / total
                print(f"Normalized weights: phoneme_weight={self.phoneme_weight:.3f}, error_weight={self.error_weight:.3f}")
                if self.training_mode == 'phoneme_error_length':
                    print(f"Length weight (separate penalty): {self.length_weight:.3f}")

    def get_model_config(self):
        """모델 설정을 반환하며 use_transformer 플래그를 설정합니다."""
        config = self.model_configs.get(self.model_type, self.model_configs['simple']).copy()
        config['use_transformer'] = (self.model_type == 'transformer')
        return config

    def has_error_component(self):
        """현재 훈련 모드에 에러 탐지가 포함되어 있는지 확인합니다."""
        return self.training_mode in ['phoneme_error', 'phoneme_error_length']

    def has_length_component(self):
        """현재 훈련 모드에 길이 손실이 포함되어 있는지 확인합니다."""
        return self.training_mode == 'phoneme_error_length'

    def save_config(self, path):
        """설정을 JSON 파일로 저장합니다."""
        config_dict = {
            attr: getattr(self, attr) for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def get_error_mapping(self):
        """에러 라벨 매핑을 반환합니다."""
        return {
            'blank': 0,
            'D': 1,  # Deletion
            'I': 2,  # Insertion
            'S': 3,  # Substitution
            'C': 4   # Correct
        }

    def get_error_type_names(self):
        """에러 타입 이름 매핑을 반환합니다."""
        return {
            0: 'blank',
            1: 'deletion',
            2: 'insertion',
            3: 'substitution',
            4: 'correct'
        }