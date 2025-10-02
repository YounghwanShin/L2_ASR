import os
import json
import random
import logging
import torch
import numpy as np
from datetime import datetime
import pytz


def seed_everything(seed: int):
    """모든 랜덤 시드를 설정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_experiment_dirs(config, resume: bool = False):
    """실험 디렉토리를 설정합니다."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)

    config_path = os.path.join(config.experiment_dir, 'config.json')
    if not resume:
        config.save_config(config_path)

    log_file = os.path.join(config.log_dir, 'training.log')
    file_mode = 'a' if resume else 'w'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode=file_mode),
            logging.StreamHandler()
        ]
    )


def save_checkpoint(model, wav2vec_opt, main_opt, epoch, val_loss, train_loss, metrics, path):
    """모델 체크포인트를 저장합니다."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'wav2vec_optimizer_state_dict': wav2vec_opt.state_dict(),
        'main_optimizer_state_dict': main_opt.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'metrics': metrics,
        'saved_time': datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(checkpoint_path, model, wav2vec_optimizer, main_optimizer, device):
    """체크포인트를 로드합니다."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    wav2vec_optimizer.load_state_dict(checkpoint['wav2vec_optimizer_state_dict'])
    main_optimizer.load_state_dict(checkpoint['main_optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_metrics = checkpoint.get('metrics', {})
    
    logger.info(f"Resumed from epoch {checkpoint['epoch']}")
    if 'saved_time' in checkpoint:
        logger.info(f"Checkpoint saved at: {checkpoint['saved_time']}")
    logger.info(f"Previous metrics: {best_metrics}")
    
    return start_epoch, best_metrics


def get_model_class(model_type: str):
    """모델 타입에 따른 모델 클래스와 손실 함수를 반환합니다."""
    from ..models.unified_model import UnifiedModel
    from ..models.losses import UnifiedLoss
    return UnifiedModel, UnifiedLoss


def detect_model_type_from_checkpoint(checkpoint_path: str) -> str:
    """체크포인트에서 모델 타입을 자동 감지합니다."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    state_dict = remove_module_prefix(state_dict)
    keys = list(state_dict.keys())

    if any('feature_encoder.transformer' in key for key in keys):
        return 'transformer'
    else:
        return 'simple'


def remove_module_prefix(state_dict):
    """DataParallel에서 사용되는 'module.' 접두사를 제거합니다."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict