import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
import random
import numpy as np
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import LinearLR

from model import PhonemeRecognitionModel
from data import PhonemeRecognitionDataset, PhonemeEvaluationDataset, phoneme_recognition_collate_fn, phoneme_evaluation_collate_fn
from evaluate import evaluate_phoneme_recognition

def seed_everything(seed):
    """재현성을 위한 랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_phoneme_recognition(model, dataloader, criterion, optimizer, device, epoch, max_grad_norm=1.0):
    """음소 인식 모델 학습"""
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f'에폭 {epoch} [음소 인식]')
    
    for batch_idx, (waveforms, input_ids, attention_masks, phoneme_labels, label_lengths, _) in enumerate(progress_bar):
        waveforms = waveforms.to(device)
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        phoneme_labels = phoneme_labels.to(device)
        label_lengths = label_lengths.to(device)
        
        # wav2vec용 어텐션 마스크 생성
        batch_size, audio_len = waveforms.shape
        audio_attention_mask = torch.ones((batch_size, audio_len), device=device)
        
        # 순전파
        phoneme_logits = model(
            waveforms, 
            input_ids, 
            audio_attention_mask=audio_attention_mask,
            text_attention_mask=attention_masks
        )
        
        # CTC 손실 계산
        log_probs = torch.log_softmax(phoneme_logits, dim=-1)
        
        # 출력 길이 계산
        output_lengths = torch.full(
            size=(batch_size,), 
            fill_value=phoneme_logits.size(1), 
            dtype=torch.long, 
            device=device
        )
        
        loss = criterion(log_probs.transpose(0, 1), phoneme_labels, output_lengths, label_lengths)
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        running_loss += loss.item()
        
        progress_bar.set_postfix({
            '손실': running_loss / (batch_idx + 1)
        })
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def validate_phoneme_recognition(model, dataloader, criterion, device):
    """음소 인식 모델 검증"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='검증 [음소 인식]')
        
        for batch_idx, (waveforms, input_ids, attention_masks, phoneme_labels, label_lengths, _) in enumerate(progress_bar):
            waveforms = waveforms.to(device)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            phoneme_labels = phoneme_labels.to(device)
            label_lengths = label_lengths.to(device)
            
            # wav2vec용 어텐션 마스크 생성
            batch_size, audio_len = waveforms.shape
            audio_attention_mask = torch.ones((batch_size, audio_len), device=device)
            
            # 순전파
            phoneme_logits = model(
                waveforms, 
                input_ids, 
                audio_attention_mask=audio_attention_mask,
                text_attention_mask=attention_masks
            )
            
            # CTC 손실 계산
            log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            
            # 출력 길이 계산
            output_lengths = torch.full(
                size=(batch_size,), 
                fill_value=phoneme_logits.size(1), 
                dtype=torch.long, 
                device=device
            )
            
            loss = criterion(log_probs.transpose(0, 1), phoneme_labels, output_lengths, label_lengths)
            
            running_loss += loss.item()
            
            progress_bar.set_postfix({
                '검증_손실': running_loss / (batch_idx + 1)
            })
    
    val_loss = running_loss / len(dataloader)
    return val_loss

def main():
    parser = argparse.ArgumentParser(description='문맥 인식 정렬을 활용한 L2 음소 인식 모델 학습')
    
    # 기본 설정
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='사용할 장치')
    
    # 데이터 설정
    parser.add_argument('--train_data', type=str, required=True, help='학습 데이터 JSON 파일')
    parser.add_argument('--val_data', type=str, required=True, help='검증 데이터 JSON 파일')
    parser.add_argument('--eval_data', type=str, help='평가 데이터 JSON 파일')
    parser.add_argument('--phoneme_map', type=str, default='data/phoneme_to_id.json', 
                        help='음소-ID 매핑')
    
    # 모델 설정
    parser.add_argument('--pretrained_audio_model', type=str, default='facebook/wav2vec2-base-960h', 
                        help='사전학습된 wav2vec2 모델')
    parser.add_argument('--pretrained_text_model', type=str, default='bert-base-uncased', 
                        help='사전학습된 BERT 모델')
    parser.add_argument('--hidden_dim', type=int, default=768, help='은닉층 차원')
    parser.add_argument('--num_phonemes', type=int, default=42, help='음소 수')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='어텐션 헤드 수')
    parser.add_argument('--dropout', type=float, default=0.1, help='드롭아웃 비율')
    
    # 학습 설정
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='학습률')
    parser.add_argument('--num_epochs', type=int, default=10, help='에폭 수')
    parser.add_argument('--max_audio_length', type=int, default=None, help='최대 오디오 길이(샘플 단위)')
    parser.add_argument('--max_text_length', type=int, default=128, help='최대 텍스트 길이(토큰 단위)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='그라디언트 클리핑을 위한 최대 노름값')
    
    # 학습률 스케줄러 설정
    parser.add_argument('--use_scheduler', action='store_true', help='학습률 스케줄러 사용 여부')
    parser.add_argument('--end_factor', type=float, default=0.1, help='선형 스케줄러 최종 비율')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='웜업 스텝 수')
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='models', help='모델 체크포인트 출력 디렉토리')
    parser.add_argument('--result_dir', type=str, default='results', help='결과 출력 디렉토리')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='로드할 모델 체크포인트 경로')
    parser.add_argument('--save_steps', type=int, default=1000, help='체크포인트 저장 간격 (스텝 단위)')
    
    args = parser.parse_args()
    
    # 재현성을 위한 시드 설정
    seed_everything(args.seed)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 로깅 설정
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.result_dir, 'train_phoneme_recognition.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 하이퍼파라미터 저장
    with open(os.path.join(args.result_dir, 'hyperparams_phoneme_recognition.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # 음소 매핑 로드
    if os.path.exists(args.phoneme_map):
        with open(args.phoneme_map, 'r') as f:
            phoneme_to_id = json.load(f)
    else:
        logger.error(f"음소-ID 매핑 파일({args.phoneme_map})이 필요합니다.")
        import sys
        sys.exit(1)
    
    # ID를 음소로 변환하는 역매핑 생성
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    
    # 모델 초기화
    logger.info("문맥 인식 정렬을 활용한 음소 인식 모델 초기화")
    model = PhonemeRecognitionModel(
        pretrained_audio_model=args.pretrained_audio_model,
        pretrained_text_model=args.pretrained_text_model,
        hidden_dim=args.hidden_dim,
        num_phonemes=args.num_phonemes,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout
    )
    
    # 체크포인트 로드 (있는 경우)
    if args.model_checkpoint:
        logger.info(f"{args.model_checkpoint}에서 체크포인트 로드 중")
        state_dict = torch.load(args.model_checkpoint, map_location=args.device)
        
        # "module." 접두사 제거
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 'module.' 접두사 제거
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
    
    # 다중 GPU 사용
    if torch.cuda.device_count() > 1:
        logger.info(f"{torch.cuda.device_count()}개의 GPU가 감지되었습니다. DataParallel 사용")
        model = nn.DataParallel(model)
    
    model = model.to(args.device)
    
    # 데이터셋 및 데이터로더 초기화
    logger.info(f"학습 데이터 로드 중: {args.train_data}")
    train_dataset = PhonemeRecognitionDataset(
        args.train_data, 
        phoneme_to_id, 
        text_model_name=args.pretrained_text_model,
        max_length=args.max_audio_length,
        max_text_length=args.max_text_length
    )
    
    logger.info(f"검증 데이터 로드 중: {args.val_data}")
    val_dataset = PhonemeRecognitionDataset(
        args.val_data, 
        phoneme_to_id, 
        text_model_name=args.pretrained_text_model,
        max_length=args.max_audio_length,
        max_text_length=args.max_text_length
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=phoneme_recognition_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=phoneme_recognition_collate_fn
    )
    
    # 평가 데이터셋 설정 (제공된 경우)
    eval_dataloader = None
    if args.eval_data:
        logger.info(f"평가 데이터 로드 중: {args.eval_data}")
        eval_dataset = PhonemeEvaluationDataset(
            args.eval_data, 
            phoneme_to_id, 
            text_model_name=args.pretrained_text_model,
            max_length=args.max_audio_length,
            max_text_length=args.max_text_length
        )
        
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, 
            collate_fn=phoneme_evaluation_collate_fn
        )
    
    # CTC 손실 함수
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # 옵티마이저 - 파라미터 그룹별 차별화된 학습률 적용
    # BERT, Wav2Vec, 정렬 모듈, 기타 파라미터 구분
    if isinstance(model, nn.DataParallel):
        bert_params = list(model.module.text_encoder.parameters())
        wav2vec_params = list(model.module.learnable_wav2vec.parameters()) 
        alignment_params = list(model.module.alignment_module.parameters())
        other_params = [p for n, p in model.named_parameters() if 
                       not any(n.startswith(prefix) for prefix in 
                              ['module.text_encoder.', 'module.learnable_wav2vec.', 'module.alignment_module.'])]
    else:
        bert_params = list(model.text_encoder.parameters())
        wav2vec_params = list(model.learnable_wav2vec.parameters())
        alignment_params = list(model.alignment_module.parameters())
        other_params = [p for n, p in model.named_parameters() if 
                       not any(n.startswith(prefix) for prefix in 
                              ['text_encoder.', 'learnable_wav2vec.', 'alignment_module.'])]
    
    param_groups = [
        {'params': bert_params, 'lr': args.learning_rate * 0.1},  # BERT: 0.1x 학습률
        {'params': wav2vec_params, 'lr': args.learning_rate * 0.2},  # Wav2Vec: 0.2x 학습률
        {'params': alignment_params, 'lr': args.learning_rate * 1.5},  # 정렬 모듈: 1.5x 학습률 (더 적극적으로 학습)
        {'params': other_params, 'lr': args.learning_rate}  # 나머지: 기본 학습률
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    
    # Linear 학습률 스케줄러
    scheduler = None
    if args.use_scheduler:
        # 웜업과 선형 감소 스케줄러
        total_steps = len(train_dataloader) * args.num_epochs
        
        # 커스텀 람다 스케줄러 (웜업 + 선형 감소)
        def lr_lambda(current_step):
            if current_step < args.warmup_steps:
                return float(current_step) / float(max(1, args.warmup_steps))
            return max(
                args.end_factor,
                float(total_steps - current_step) / float(max(1, total_steps - args.warmup_steps))
            )
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        logger.info(f"웜업 + 선형 감소 학습률 스케줄러 초기화됨 (웜업 스텝: {args.warmup_steps}, 종료 비율: {args.end_factor})")
    
    best_val_loss = float('inf')
    best_per = float('inf') if eval_dataloader else None
    global_step = 0
    
    logger.info("학습 시작")
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"에폭 {epoch}/{args.num_epochs} 시작")
        
        # 학습
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f'에폭 {epoch} [음소 인식]')
        
        for batch_idx, (waveforms, input_ids, attention_masks, phoneme_labels, label_lengths, _) in enumerate(progress_bar):
            waveforms = waveforms.to(args.device)
            input_ids = input_ids.to(args.device)
            attention_masks = attention_masks.to(args.device)
            phoneme_labels = phoneme_labels.to(args.device)
            label_lengths = label_lengths.to(args.device)
            
            # wav2vec용 어텐션 마스크 생성
            batch_size, audio_len = waveforms.shape
            audio_attention_mask = torch.ones((batch_size, audio_len), device=args.device)
            
            # 순전파
            phoneme_logits = model(
                waveforms, 
                input_ids, 
                audio_attention_mask=audio_attention_mask,
                text_attention_mask=attention_masks
            )
            
            # CTC 손실 계산
            log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            
            # 출력 길이 계산
            output_lengths = torch.full(
                size=(batch_size,), 
                fill_value=phoneme_logits.size(1), 
                dtype=torch.long, 
                device=args.device
            )
            
            loss = criterion(log_probs.transpose(0, 1), phoneme_labels, output_lengths, label_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 현재 학습률 로깅
            lr = optimizer.param_groups[0]['lr']
            
            progress_bar.set_postfix({
                '손실': epoch_loss / (batch_idx + 1),
                'LR': f"{lr:.2e}"
            })
            
            # 중간 체크포인트 저장
            if global_step % args.save_steps == 0:
                checkpoint_path = os.path.join(args.output_dir, f'step_{global_step}_model.pth')
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), checkpoint_path)
                else:
                    torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"스텝 {global_step} 체크포인트 저장: {checkpoint_path}")
        
        # 에폭 평균 손실
        train_loss = epoch_loss / len(train_dataloader)
        
        # 검증
        val_loss = validate_phoneme_recognition(
            model, val_dataloader, criterion, args.device
        )
        
        logger.info(f"에폭 {epoch}: 학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")
        
        # 매 에포크마다 평가
        epoch_results = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rates': [pg['lr'] for pg in optimizer.param_groups]
        }
        
        # 평가 데이터로 모델 평가
        if eval_dataloader:
            logger.info(f"에폭 {epoch}: 음소 인식 평가 중...")
            eval_results = evaluate_phoneme_recognition(
                model, eval_dataloader, args.device, id_to_phoneme
            )
            
            per = eval_results['per']
            logger.info(f"음소 오류율 (PER): {per:.4f}")
            logger.info(f"총 음소 수: {eval_results['total_phonemes']}")
            logger.info(f"총 오류 수: {eval_results['total_errors']}")
            logger.info(f"삽입: {eval_results['insertions']}")
            logger.info(f"삭제: {eval_results['deletions']}")
            logger.info(f"대체: {eval_results['substitutions']}")
            
            # 평가 결과 저장
            epoch_results['evaluation'] = {
                'per': per,
                'total_phonemes': eval_results['total_phonemes'],
                'total_errors': eval_results['total_errors'],
                'insertions': eval_results['insertions'],
                'deletions': eval_results['deletions'],
                'substitutions': eval_results['substitutions']
            }
            
            # PER 기준으로 최고 모델 저장
            if per < best_per:
                best_per = per
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), os.path.join(args.output_dir, f'best_per_phoneme_recognition.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_per_phoneme_recognition.pth'))
                logger.info(f"PER {per:.4f}로 새로운 최고 모델 저장")
        
        with open(os.path.join(args.result_dir, f'phoneme_recognition_epoch{epoch}.json'), 'w') as f:
            json.dump(epoch_results, f, indent=4)
        
        # 검증 손실 기준으로 최고 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(args.output_dir, f'best_val_phoneme_recognition.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_val_phoneme_recognition.pth'))
            logger.info(f"검증 손실 {val_loss:.4f}로 새로운 최고 모델 저장")
        
        # 마지막 모델 저장
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(args.output_dir, f'last_phoneme_recognition.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'last_phoneme_recognition.pth'))
    
    logger.info("학습 완료!")

if __name__ == "__main__":
    main()