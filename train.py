import os
import json
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from src.data.dataset import UnifiedDataset, collate_fn
from src.models.unified_model import UnifiedModel
from src.models.losses import UnifiedLoss
from src.training.trainer import UnifiedTrainer
from src.training.utils import (
    seed_everything, setup_experiment_dirs, save_checkpoint, 
    load_checkpoint, get_model_class, detect_model_type_from_checkpoint
)
from src.evaluation.evaluator import UnifiedEvaluator

logger = logging.getLogger(__name__)


def main():
    """메인 훈련 함수"""
    parser = argparse.ArgumentParser(description='L2 발음 평가 모델 훈련')
    parser.add_argument('--training_mode', type=str, 
                       choices=['phoneme_only', 'phoneme_error', 'phoneme_error_length'], 
                       help='훈련 모드')
    parser.add_argument('--model_type', type=str, 
                       choices=['simple', 'transformer'], 
                       help='모델 아키텍처')
    parser.add_argument('--config', type=str, 
                       help='설정값 오버라이드 (key=value 형식)')
    parser.add_argument('--train_data', type=str, 
                       help='훈련 데이터 경로 오버라이드')
    parser.add_argument('--val_data', type=str, 
                       help='검증 데이터 경로 오버라이드')
    parser.add_argument('--eval_data', type=str, 
                       help='평가 데이터 경로 오버라이드')
    parser.add_argument('--phoneme_map', type=str, 
                       help='음소 맵 경로 오버라이드')
    parser.add_argument('--output_dir', type=str, 
                       help='출력 디렉토리 오버라이드')
    parser.add_argument('--resume', type=str, 
                       help='체크포인트에서 훈련 재개')
    parser.add_argument('--experiment_name', type=str, 
                       help='실험명 오버라이드')
    
    args = parser.parse_args()

    # 설정 로드 및 오버라이드
    config = Config()
    
    if args.training_mode:
        config.training_mode = args.training_mode
    if args.model_type:
        config.model_type = args.model_type
    if args.train_data:
        config.train_data = args.train_data
    if args.val_data:
        config.val_data = args.val_data
    if args.eval_data:
        config.eval_data = args.eval_data
    if args.phoneme_map:
        config.phoneme_map = args.phoneme_map
    if args.output_dir:
        config.output_dir = args.output_dir

    # 체크포인트에서 모델 타입 자동 감지
    if args.resume:
        detected_model_type = detect_model_type_from_checkpoint(args.resume)
        config.model_type = detected_model_type
        logger.info(f"체크포인트에서 모델 타입 자동 감지: {detected_model_type}")

    if args.experiment_name:
        config.experiment_name = args.experiment_name
    elif args.resume:
        resume_exp_dir = os.path.dirname(os.path.dirname(args.resume))
        config.experiment_name = os.path.basename(resume_exp_dir)

    # 개별 설정 오버라이드
    if args.config:
        for override in args.config.split(','):
            key, value = override.split('=')
            if hasattr(config, key):
                attr_type = type(getattr(config, key))
                if attr_type == bool:
                    setattr(config, key, value.lower() == 'true')
                else:
                    setattr(config, key, attr_type(value))

    config.__post_init__()

    # 재현 가능성을 위한 시드 설정
    seed_everything(config.seed)
    
    # 실험 디렉토리 설정
    setup_experiment_dirs(config, resume=bool(args.resume))

    # 데이터 로드
    with open(config.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    error_type_names = config.get_error_type_names()

    # 모델 생성
    model = UnifiedModel(
        pretrained_model_name=config.pretrained_model,
        num_phonemes=config.num_phonemes,
        num_error_types=config.num_error_types,
        **config.get_model_config()
    )

    # 멀티 GPU 지원
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(config.device)

    # 손실 함수 생성
    criterion = UnifiedLoss(
        training_mode=config.training_mode,
        error_weight=config.error_weight,
        phoneme_weight=config.phoneme_weight,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma
    )

    # 훈련자 생성
    trainer = UnifiedTrainer(model, config, config.device, logger)
    wav2vec_optimizer, main_optimizer = trainer.get_optimizers()

    # 데이터셋 및 데이터로더 생성
    train_dataset = UnifiedDataset(
        config.train_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=config.device
    )
    val_dataset = UnifiedDataset(
        config.val_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=config.device
    )
    eval_dataset = UnifiedDataset(
        config.eval_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=config.device
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, training_mode=config.training_mode)
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, training_mode=config.training_mode)
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.eval_batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, training_mode=config.training_mode)
    )

    # 평가자 생성
    evaluator = UnifiedEvaluator(config.device)

    # 훈련 상태 초기화
    best_val_loss = float('inf')
    best_error_accuracy = 0.0
    best_phoneme_accuracy = 0.0
    start_epoch = 1

    # 체크포인트에서 재개
    if args.resume:
        start_epoch, resume_metrics = load_checkpoint(
            args.resume, model, wav2vec_optimizer, main_optimizer, config.device
        )
        if 'error_accuracy' in resume_metrics:
            best_error_accuracy = resume_metrics['error_accuracy']
        if 'phoneme_accuracy' in resume_metrics:
            best_phoneme_accuracy = resume_metrics['phoneme_accuracy']
        checkpoint = torch.load(args.resume, map_location=config.device)
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        
        logger.info("=" * 50)
        logger.info("훈련 재개")
        logger.info("=" * 50)
        logger.info(f"훈련 모드: {config.training_mode}")
        logger.info(f"에포크 {start_epoch}부터 재개")
        logger.info(f"현재 최고 에러 정확도: {best_error_accuracy:.4f}")
        logger.info(f"현재 최고 음소 정확도: {best_phoneme_accuracy:.4f}")
        logger.info(f"현재 최고 검증 손실: {best_val_loss:.4f}")
        logger.info("=" * 50)
    else:
        logger.info(f"훈련 모드: {config.training_mode}로 훈련 시작")
        logger.info(f"모델 타입: {config.model_type}")
        logger.info(f"실험명: {config.experiment_name}")
        logger.info(f"{config.num_epochs} 에포크 동안 훈련")
        logger.info(f"SpecAugment 활성화: {config.wav2vec2_specaug}")
        logger.info(f"길이 손실 활성화: {config.has_length_component()}")
        logger.info(f"에러 탐지 활성화: {config.has_error_component()}")
        logger.info(f"기본 파라미터로 Focal Loss 사용")
        if config.has_length_component():
            logger.info(f"길이 패널티는 음소와 에러 디코딩 길이의 평균 사용")

    # 훈련 루프
    for epoch in range(start_epoch, config.num_epochs + 1):
        # 훈련
        train_loss = trainer.train_epoch(train_dataloader, criterion, epoch)
        
        # 검증
        val_loss = trainer.validate_epoch(val_dataloader, criterion)
        
        logger.info(f"에포크 {epoch}: 훈련 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")

        # 샘플 예측 출력
        logger.info(f"에포크 {epoch} - 샘플 예측")
        logger.info("=" * 50)
        evaluator.show_sample_predictions(
            model=model,
            eval_dataloader=eval_dataloader,
            id_to_phoneme=id_to_phoneme,
            logger=logger,
            training_mode=config.training_mode,
            error_type_names=error_type_names
        )

        # 음소 인식 평가
        logger.info(f"에포크 {epoch}: 음소 인식 평가 중...")
        phoneme_recognition_results = evaluator.evaluate_phoneme_recognition(
            model=model,
            dataloader=eval_dataloader,
            training_mode=config.training_mode,
            id_to_phoneme=id_to_phoneme
        )
        logger.info(f"음소 에러율 (PER): {phoneme_recognition_results['per']:.4f}")
        logger.info(f"음소 정확도: {1.0 - phoneme_recognition_results['per']:.4f}")

        current_phoneme_accuracy = 1.0 - phoneme_recognition_results['per']
        current_error_accuracy = 0.0

        # 에러 탐지 평가
        if config.has_error_component():
            logger.info(f"에포크 {epoch}: 에러 탐지 평가 중...")
            error_detection_results = evaluator.evaluate_error_detection(
                model=model,
                dataloader=eval_dataloader,
                training_mode=config.training_mode,
                error_type_names=error_type_names
            )
            logger.info(f"에러 토큰 정확도: {error_detection_results['token_accuracy']:.4f}")
            logger.info(f"에러 가중 F1: {error_detection_results['weighted_f1']:.4f}")
            for error_type, metrics in error_detection_results['class_metrics'].items():
                if error_type != 'blank':
                    logger.info(f"  {error_type}: 정밀도={metrics['precision']:.4f}, 재현율={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
            current_error_accuracy = error_detection_results['token_accuracy']

        # 체크포인트 저장
        if config.save_best_error and current_error_accuracy > best_error_accuracy:
            best_error_accuracy = current_error_accuracy
            metrics = {
                'error_accuracy': best_error_accuracy,
                'phoneme_accuracy': best_phoneme_accuracy,
                'per': phoneme_recognition_results['per']
            }
            model_path = os.path.join(config.output_dir, 'best_error.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"새로운 최고 에러 정확도: {best_error_accuracy:.4f}")

        if config.save_best_phoneme and current_phoneme_accuracy > best_phoneme_accuracy:
            best_phoneme_accuracy = current_phoneme_accuracy
            metrics = {
                'error_accuracy': best_error_accuracy,
                'phoneme_accuracy': best_phoneme_accuracy,
                'per': phoneme_recognition_results['per']
            }
            model_path = os.path.join(config.output_dir, 'best_phoneme.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"새로운 최고 음소 정확도: {best_phoneme_accuracy:.4f} (PER: {phoneme_recognition_results['per']:.4f})")

        if config.save_best_loss and val_loss < best_val_loss:
            best_val_loss = val_loss
            metrics = {
                'error_accuracy': best_error_accuracy,
                'phoneme_accuracy': best_phoneme_accuracy,
                'per': phoneme_recognition_results['per']
            }
            model_path = os.path.join(config.output_dir, 'best_loss.pth')
            save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                          epoch, val_loss, train_loss, metrics, model_path)
            logger.info(f"새로운 최고 검증 손실: {best_val_loss:.4f}")

        # 최신 체크포인트 저장
        latest_metrics = {
            'error_accuracy': best_error_accuracy,
            'phoneme_accuracy': best_phoneme_accuracy,
            'per': phoneme_recognition_results['per']
        }
        latest_path = os.path.join(config.output_dir, 'latest.pth')
        save_checkpoint(model, wav2vec_optimizer, main_optimizer,
                      epoch, val_loss, train_loss, latest_metrics, latest_path)

        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 최종 메트릭 저장
    final_metrics = {
        'best_error_accuracy': best_error_accuracy,
        'best_phoneme_accuracy': best_phoneme_accuracy,
        'best_val_loss': best_val_loss,
        'completed_epochs': config.num_epochs,
        'training_mode': config.training_mode,
        'model_type': config.model_type,
        'experiment_name': config.experiment_name
    }
    metrics_path = os.path.join(config.result_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("훈련 완료!")
    logger.info(f"훈련 모드: {config.training_mode}")
    logger.info(f"최고 에러 정확도: {best_error_accuracy:.4f}")
    logger.info(f"최고 음소 정확도: {best_phoneme_accuracy:.4f}")
    logger.info(f"최종 메트릭 저장 위치: {metrics_path}")


if __name__ == "__main__":
    main()