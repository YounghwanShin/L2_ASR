import os
import json
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import pytz

from config import Config
from src.utils import (
    get_model_class,
    detect_model_type_from_checkpoint,
    evaluate_error_detection,
    evaluate_phoneme_recognition,
    remove_module_prefix,
)
from src.data_prepare import UnifiedDataset, collate_fn

def evaluate_length_prediction(model, dataloader, device, training_mode='phoneme_error_length'):
    """길이 예측 성능 평가"""
    if training_mode != 'phoneme_error_length':
        return None
    
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if batch_data is None:
                continue
                
            waveforms = batch_data['waveforms'].to(device)
            audio_lengths = batch_data['audio_lengths'].to(device)
            phoneme_lengths = batch_data['phoneme_lengths'].to(device)
            
            # Attention mask 계산
            wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
            attention_mask = (torch.arange(waveforms.shape[1], device=device)[None, :] < 
                            (wav_lens_norm * waveforms.shape[1])[:, None]).long()
            
            # 모델 순전파
            outputs = model(waveforms, attention_mask=attention_mask, training_mode=training_mode)
            
            if 'length_prediction' in outputs:
                predicted_lengths = outputs['length_prediction']
                target_lengths = phoneme_lengths.float()
                
                # 메트릭 계산
                mae = torch.abs(predicted_lengths - target_lengths).mean()
                mse = torch.pow(predicted_lengths - target_lengths, 2).mean()
                
                total_mae += mae.item()
                total_mse += mse.item()
                total_samples += 1
                
                # 예측값과 타겟값 저장
                predictions.extend(predicted_lengths.cpu().numpy().tolist())
                targets.extend(target_lengths.cpu().numpy().tolist())
    
    if total_samples == 0:
        return None
    
    return {
        'mae': total_mae / total_samples,
        'mse': total_mse / total_samples,
        'rmse': (total_mse / total_samples) ** 0.5,
        'predictions': predictions,
        'targets': targets,
        'total_samples': total_samples
    }

def main():
    parser = argparse.ArgumentParser(description='통합 L2 발음 모델 평가')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--training_mode', type=str, choices=['phoneme_only', 'phoneme_error', 'phoneme_error_length'], help='모델에 사용된 훈련 모드')
    parser.add_argument('--eval_data', type=str, help='평가 데이터 경로 오버라이드')
    parser.add_argument('--phoneme_map', type=str, help='음소 맵 경로 오버라이드')
    parser.add_argument('--model_type', type=str, help='모델 타입 강제 지정 (simple/transformer)')
    parser.add_argument('--batch_size', type=int, default=8, help='평가 배치 크기')
    parser.add_argument('--save_predictions', action='store_true', help='상세한 예측 결과 저장')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 설정 로드
    config = Config()

    if args.training_mode:
        config.training_mode = args.training_mode
    if args.eval_data:
        config.eval_data = args.eval_data
    if args.phoneme_map:
        config.phoneme_map = args.phoneme_map
    if args.batch_size:
        config.eval_batch_size = args.batch_size

    # 모델 타입 결정
    model_type = args.model_type
    if model_type is None:
        model_type = detect_model_type_from_checkpoint(args.model_checkpoint)
        logger.info(f"체크포인트에서 모델 타입 자동 감지: {model_type}")
    else:
        detected_type = detect_model_type_from_checkpoint(args.model_checkpoint)
        if model_type != detected_type:
            logger.warning(f"지정된 모델 타입 '{model_type}'이 체크포인트 '{detected_type}'와 일치하지 않습니다. 체크포인트 타입을 사용합니다.")
            model_type = detected_type

    config.model_type = model_type

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용 디바이스: {device}")
    logger.info(f"훈련 모드: {config.training_mode}")
    logger.info(f"모델 타입: {model_type}")
    logger.info(f"체크포인트: {args.model_checkpoint}")

    # 음소 맵 로드
    with open(config.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    error_type_names = {0: 'blank', 1: 'incorrect', 2: 'correct'}

    # 모델 초기화
    model_class, loss_class = get_model_class(model_type)
    model_config = config.model_configs[model_type]

    model = model_class(
        pretrained_model_name=config.pretrained_model,
        num_phonemes=config.num_phonemes,
        num_error_types=config.num_error_types,
        **model_config
    )

    # 체크포인트 로드
    checkpoint = torch.load(args.model_checkpoint, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        logger.info(f"에폭 {checkpoint.get('epoch', 'unknown')}에서 로드된 모델")
        if 'metrics' in checkpoint:
            logger.info(f"훈련 메트릭: {checkpoint['metrics']}")
    else:
        state_dict = checkpoint

    state_dict = remove_module_prefix(state_dict)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.error(f"체크포인트 로드 실패. 오류: {str(e)}")
        logger.error(f"모델 아키텍처: {model_type}")
        logger.error("이는 체크포인트가 다른 모델 아키텍처로 저장되었을 때 발생합니다.")
        logger.error("--model_type 없이 실행하여 자동 감지하거나 체크포인트 경로를 확인하세요.")
        raise

    model = model.to(device)
    model.eval()

    # 평가 데이터셋 준비
    eval_dataset = UnifiedDataset(
        config.eval_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=device
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.eval_batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, training_mode=config.training_mode)
    )

    logger.info("평가 시작...")

    # 음소 인식 평가
    logger.info("음소 인식 평가 중...")
    phoneme_recognition_results = evaluate_phoneme_recognition(
        model=model,
        dataloader=eval_dataloader,
        device=device,
        training_mode=config.training_mode,
        id_to_phoneme=id_to_phoneme
    )

    # 에러 감지 평가 (필요한 경우)
    error_detection_results = None
    if config.has_error_component():
        logger.info("에러 감지 평가 중...")
        error_detection_results = evaluate_error_detection(
            model=model,
            dataloader=eval_dataloader,
            device=device,
            training_mode=config.training_mode,
            error_type_names=error_type_names
        )

    # 길이 예측 평가 (필요한 경우)
    length_prediction_results = None
    if config.has_length_component():
        logger.info("길이 예측 평가 중...")
        length_prediction_results = evaluate_length_prediction(
            model=model,
            dataloader=eval_dataloader,
            device=device,
            training_mode=config.training_mode
        )

    # 결과 출력
    logger.info("\n" + "="*80)
    logger.info("완전한 평가 결과")
    logger.info("="*80)

    logger.info("\n--- 음소 인식 결과 ---")
    logger.info(f"음소 에러율 (PER): {phoneme_recognition_results['per']:.4f}")
    logger.info(f"음소 정확도: {1.0 - phoneme_recognition_results['per']:.4f}")
    logger.info(f"총 음소 수: {phoneme_recognition_results['total_phonemes']}")
    logger.info(f"총 에러 수: {phoneme_recognition_results['total_errors']}")
    logger.info(f"삽입 에러: {phoneme_recognition_results['insertions']}")
    logger.info(f"삭제 에러: {phoneme_recognition_results['deletions']}")
    logger.info(f"치환 에러: {phoneme_recognition_results['substitutions']}")

    logger.info("\n--- 잘못된 발음 감지 메트릭 ---")
    logger.info(f"정밀도: {phoneme_recognition_results['mispronunciation_precision']:.4f}")
    logger.info(f"재현율: {phoneme_recognition_results['mispronunciation_recall']:.4f}")
    logger.info(f"F1-점수: {phoneme_recognition_results['mispronunciation_f1']:.4f}")

    logger.info("\n--- 혼동 행렬 ---")
    cm = phoneme_recognition_results['confusion_matrix']
    logger.info(f"올바른 수용 (TA): {cm['true_acceptance']}")
    logger.info(f"잘못된 거부 (FR): {cm['false_rejection']}")
    logger.info(f"잘못된 수용 (FA): {cm['false_acceptance']}")
    logger.info(f"올바른 거부 (TR): {cm['true_rejection']}")

    if error_detection_results:
        logger.info("\n--- 에러 감지 결과 ---")
        logger.info(f"시퀀스 정확도: {error_detection_results['sequence_accuracy']:.4f}")
        logger.info(f"토큰 정확도: {error_detection_results['token_accuracy']:.4f}")
        logger.info(f"평균 편집 거리: {error_detection_results['avg_edit_distance']:.4f}")
        logger.info(f"가중 F1: {error_detection_results['weighted_f1']:.4f}")
        logger.info(f"매크로 F1: {error_detection_results['macro_f1']:.4f}")
        logger.info(f"총 시퀀스 수: {error_detection_results['total_sequences']}")
        logger.info(f"총 토큰 수: {error_detection_results['total_tokens']}")

        logger.info("\n--- 에러 타입별 메트릭 ---")
        for error_type, metrics in error_detection_results['class_metrics'].items():
            logger.info(f"{error_type.upper()}:")
            logger.info(f"  정밀도: {metrics['precision']:.4f}")
            logger.info(f"  재현율: {metrics['recall']:.4f}")
            logger.info(f"  F1-점수: {metrics['f1']:.4f}")
            logger.info(f"  지원: {metrics['support']}")

    if length_prediction_results:
        logger.info("\n--- 길이 예측 결과 ---")
        logger.info(f"평균 절대 오차 (MAE): {length_prediction_results['mae']:.4f}")
        logger.info(f"평균 제곱 오차 (MSE): {length_prediction_results['mse']:.4f}")
        logger.info(f"제곱근 평균 제곱 오차 (RMSE): {length_prediction_results['rmse']:.4f}")
        logger.info(f"총 샘플 수: {length_prediction_results['total_samples']}")

    # 실험 디렉토리명 추출
    experiment_dir_name = os.path.basename(os.path.dirname(os.path.dirname(args.model_checkpoint)))

    # 설정 정보
    config_info = {
        'training_mode': config.training_mode,
        'model_type': model_type,
        'checkpoint_path': args.model_checkpoint,
        'experiment_name': experiment_dir_name,
        'evaluation_date': datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S'),
        'model_config': config.model_configs.get(model_type, {})
    }

    # 국가별 결과 출력
    logger.info("\n--- 국가별 결과 ---")
    for country in sorted(phoneme_recognition_results.get('by_country', {}).keys()):
        logger.info(f"\n{country}:")
        phoneme_country = phoneme_recognition_results['by_country'][country]
        logger.info(f"  음소 정확도: {1.0 - phoneme_country['per']:.4f}")
        logger.info(f"  잘못된 발음 F1: {phoneme_country['mispronunciation_f1']:.4f}")

        if error_detection_results and country in error_detection_results.get('by_country', {}):
            error_country = error_detection_results['by_country'][country]
            logger.info(f"  에러 토큰 정확도: {error_country['token_accuracy']:.4f}")
            logger.info(f"  에러 가중 F1: {error_country['weighted_f1']:.4f}")

    # 평가 결과 구성
    evaluation_results = {
        'phoneme_recognition': {
            'per': phoneme_recognition_results['per'],
            'accuracy': 1.0 - phoneme_recognition_results['per'],
            'mispronunciation_precision': phoneme_recognition_results['mispronunciation_precision'],
            'mispronunciation_recall': phoneme_recognition_results['mispronunciation_recall'],
            'mispronunciation_f1': phoneme_recognition_results['mispronunciation_f1'],
            'total_phonemes': phoneme_recognition_results['total_phonemes'],
            'total_errors': phoneme_recognition_results['total_errors'],
            'insertions': phoneme_recognition_results['insertions'],
            'deletions': phoneme_recognition_results['deletions'],
            'substitutions': phoneme_recognition_results['substitutions'],
            'confusion_matrix': phoneme_recognition_results['confusion_matrix'],
            'by_country': phoneme_recognition_results.get('by_country', {})
        }
    }

    if error_detection_results:
        evaluation_results['error_detection'] = {
            'sequence_accuracy': error_detection_results['sequence_accuracy'],
            'token_accuracy': error_detection_results['token_accuracy'],
            'avg_edit_distance': error_detection_results['avg_edit_distance'],
            'weighted_f1': error_detection_results['weighted_f1'],
            'macro_f1': error_detection_results['macro_f1'],
            'total_sequences': error_detection_results['total_sequences'],
            'total_tokens': error_detection_results['total_tokens'],
            'class_metrics': error_detection_results['class_metrics'],
            'by_country': error_detection_results.get('by_country', {})
        }

    if length_prediction_results:
        evaluation_results['length_prediction'] = {
            'mae': length_prediction_results['mae'],
            'mse': length_prediction_results['mse'],
            'rmse': length_prediction_results['rmse'],
            'total_samples': length_prediction_results['total_samples']
        }

    # 최종 결과 구성
    final_results = {
        'config': config_info,
        'evaluation_results': evaluation_results
    }

    # 요약 출력
    logger.info("\n--- 요약 ---")
    logger.info(f"훈련 모드: {config.training_mode}")
    logger.info(f"전체 음소 인식 성능: {1.0 - phoneme_recognition_results['per']:.4f} (정확도)")
    logger.info(f"잘못된 발음 감지 F1: {phoneme_recognition_results['mispronunciation_f1']:.4f}")
    if error_detection_results:
        logger.info(f"전체 에러 감지 성능: {error_detection_results['weighted_f1']:.4f} (가중 F1)")
    if length_prediction_results:
        logger.info(f"길이 예측 성능: {length_prediction_results['mae']:.4f} (MAE)")

    # 결과 저장
    evaluation_results_dir = 'evaluation_results'
    os.makedirs(evaluation_results_dir, exist_ok=True)

    results_filename = f"{experiment_dir_name}_eval_results.json"
    results_path = os.path.join(evaluation_results_dir, results_filename)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n완전한 평가 결과 저장 위치: {results_path}")
    logger.info(f"결과에 포함된 내용: 설정 및 국가별 분석을 포함한 전체 평가 메트릭")

if __name__ == "__main__":
    main()