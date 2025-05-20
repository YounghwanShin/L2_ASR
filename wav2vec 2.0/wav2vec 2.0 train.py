
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from wav2vec import NaiveWav2Vec2PhonemeModel
from data import PhonemeRecognitionDataset
import os
import sys
import json
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from wav2vec import NaiveWav2Vec2PhonemeModel
from data import PhonemeRecognitionDataset
from evaluate import evaluate_phoneme_recognition

'''
# 아래 주석처리가 원래 작성 코드 -> 학습이 되긴 하나 너무 가중치 낮아서 문제 이건 영환님거 중 필요한거 가져온거

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def phoneme_collate_fn(batch):
    waveforms, phoneme_labels, label_lengths, wav_files = zip(*batch)

    max_audio_len = max([waveform.shape[0] for waveform in waveforms])
    padded_waveforms = [
        torch.nn.functional.pad(w, (0, max_audio_len - w.shape[0])) for w in waveforms
    ]
    padded_waveforms = torch.stack(padded_waveforms)

    max_label_len = max([lbl.shape[0] for lbl in phoneme_labels])
    padded_labels = [
        torch.nn.functional.pad(l, (0, max_label_len - l.shape[0]), value=0) for l in phoneme_labels
    ]
    padded_labels = torch.stack(padded_labels)

    audio_lengths = torch.tensor([w.shape[0] for w in waveforms])
    label_lengths = torch.tensor(label_lengths)

    return padded_waveforms, padded_labels, audio_lengths, label_lengths, wav_files


def train(model, dataloader, criterion, optimizer, device, epoch, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, (waveforms, phoneme_labels, audio_lengths, label_lengths, _) in enumerate(pbar):
        waveforms, phoneme_labels = waveforms.to(device), phoneme_labels.to(device)
        audio_lengths, label_lengths = audio_lengths.to(device), label_lengths.to(device)

        attention_mask = (torch.arange(waveforms.shape[1]).unsqueeze(0).to(device) < audio_lengths.unsqueeze(1)).float()
        phoneme_logits = model(waveforms, attention_mask)

        log_probs = torch.log_softmax(phoneme_logits, dim=-1)
        input_seq_len, output_seq_len = waveforms.size(1), phoneme_logits.size(1)
        input_lengths = torch.floor((audio_lengths.float() / input_seq_len) * output_seq_len).long()
        input_lengths = torch.clamp(input_lengths, min=1, max=output_seq_len)

        loss = criterion(log_probs.transpose(0, 1), phoneme_labels, input_lengths, label_lengths)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=running_loss / (batch_idx + 1))

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (waveforms, phoneme_labels, audio_lengths, label_lengths, _) in enumerate(pbar):
            waveforms, phoneme_labels = waveforms.to(device), phoneme_labels.to(device)
            audio_lengths, label_lengths = audio_lengths.to(device), label_lengths.to(device)

            attention_mask = (torch.arange(waveforms.shape[1]).unsqueeze(0).to(device) < audio_lengths.unsqueeze(1)).float()
            phoneme_logits = model(waveforms, attention_mask)

            log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            input_seq_len, output_seq_len = waveforms.size(1), phoneme_logits.size(1)
            input_lengths = torch.floor((audio_lengths.float() / input_seq_len) * output_seq_len).long()
            input_lengths = torch.clamp(input_lengths, min=1, max=output_seq_len)

            loss = criterion(log_probs.transpose(0, 1), phoneme_labels, input_lengths, label_lengths)
            running_loss += loss.item()
            pbar.set_postfix(val_loss=running_loss / (batch_idx + 1))

    return running_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--phoneme_train_data', type=str, required=True)
    parser.add_argument('--phoneme_val_data', type=str, required=True)
    parser.add_argument('--phoneme_map', type=str, required=True)

    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--max_audio_length', type=int, default=None)
    parser.add_argument('--use_scheduler', action='store_true')

    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.result_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    with open(args.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)

    num_phonemes = len(phoneme_to_id)
    model = NaiveWav2Vec2PhonemeModel(num_phonemes=num_phonemes).to(args.device)

    train_dataset = PhonemeRecognitionDataset(args.phoneme_train_data, phoneme_to_id, max_length=args.max_audio_length)
    val_dataset = PhonemeRecognitionDataset(args.phoneme_val_data, phoneme_to_id, max_length=args.max_audio_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=phoneme_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=phoneme_collate_fn)

    criterion = CTCLoss(blank=0, reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    scheduler = None
    if args.use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"Epoch {epoch} / {args.num_epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, args.device, epoch)
        val_loss = validate(model, val_loader, criterion, args.device)

        if scheduler:
            scheduler.step(val_loss)

        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'last_model.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_model.pth'))
            logger.info("Saved best model.")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()


'''

# 첫번째 보던 코드드
# 2차 일어난 배치 사이즈 - 길이 문제는 해결 -> 다만 배치 줄여도 학습시 계속 out of memory error 발생

# 경로 설정
train_json_path = "/home/ellt/Workspace/wav2vec/wav2vec 2.0/split_data/train.json"
val_json_path = "/home/ellt/Workspace/wav2vec/wav2vec 2.0/split_data/val.json"
phoneme_map_path = "/home/ellt/Workspace/wav2vec/wav2vec 2.0/split_data/phoneme_to_id.json"
output_dir = "/home/ellt/Workspace/wav2vec/wav2vec 2.0/checkpoints"


# 하이퍼파라미터
epochs = 10
batch_size = 4 #배치 사이즈 조정해도 해결이 oom 해결이 안됨
learning_rate = 5e-5
device = "cuda" if torch.cuda.is_available() else "cpu"


# 서로 다른 라벨 및 오디오 묶는 방법
def collate_fn(batch):
    waveforms, labels, _, _ = zip(*batch)

    # 자르지 않음 → 원본 전체 waveform 사용
    #MAX_AUDIO_LENGTH = 160000
    #waveforms = [x[:MAX_AUDIO_LENGTH] for x in waveforms]

    label_lengths = [x.shape[0] for x in labels]
    input_lengths = [x.shape[0] for x in waveforms]

    max_audio_len = max(input_lengths)
    padded_waveforms = torch.stack([
        torch.nn.functional.pad(x, (0, max_audio_len - x.shape[0])) for x in waveforms
    ])

    max_label_len = max(label_lengths)
    padded_labels = torch.stack([
        torch.nn.functional.pad(x, (0, max_label_len - x.shape[0]), value=0) for x in labels
    ])

    return (
        padded_waveforms,
        padded_labels,
        torch.tensor(input_lengths, dtype=torch.long),
        torch.tensor(label_lengths, dtype=torch.long)
    )


# 학습 함수
def train():
    batch_size = 4
    with open(phoneme_map_path, 'r') as f:
        phoneme_to_id = json.load(f)
    num_phonemes = len(phoneme_to_id)

    print(f"Using device: {device}")
    
    #모델 불러오기
    model = NaiveWav2Vec2PhonemeModel(num_phonemes=num_phonemes).to(device)

    train_dataset = PhonemeRecognitionDataset(train_json_path, phoneme_to_id)
    val_dataset = PhonemeRecognitionDataset(val_json_path, phoneme_to_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    criterion = CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n🟢 Epoch {epoch} 시작 ------------------------")
        torch.cuda.empty_cache()
        model.train()
        total_train_loss = 0.0
        for waveforms, labels, input_lengths, label_lengths in train_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            #labels = torch.cat([label for label in labels]).to(device)  # 1D로 이어붙이기-> (좀 전에 수정정)
            optimizer.zero_grad()
            logits = model(waveforms)                           # (B, T, C)
            log_probs = torch.log_softmax(logits, dim=-1)
            #log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
            batch_size = log_probs.size(1)                      # ✅ 실제 배치 크기
            seq_len = log_probs.size(0)                         # ✅ 시퀀스 길이

            output_lengths = torch.full(
                size=(batch_size,),                             # ✅ 반드시 (B,) 크기
                fill_value=seq_len,
                dtype=torch.long
            ).to(device)

          
          #  print("log_probs shape:", log_probs.transpose(0, 1).shape)
          #  print("labels shape:", labels.shape)
          #  print("output_lengths shape:", output_lengths.shape)
          #  print("label_lengths shape:", label_lengths.shape)
          #  print("label_lengths:", label_lengths)
            
            # (T, B, C) ← CTCLoss가 요구하는 형식
            loss = criterion(log_probs, labels, output_lengths, label_lengths)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")
        torch.cuda.empty_cache()
      
        # 검증 단계
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for waveforms, labels, input_lengths, label_lengths in val_loader:
              waveforms = waveforms.to(device)
              labels = labels.to(device)
              input_lengths = input_lengths.to(device)
              label_lengths = label_lengths.to(device)
              #labels = torch.cat([label for label in labels]).to(device)
              logits = model(waveforms)  # logits: (B, T, C)
              log_probs = torch.log_softmax(logits, dim=-1)

              current_batch_size = log_probs.size(1)
              seq_len = log_probs.size(0)

              output_lengths = torch.full(
                  size=(log_probs.size(1),),  #현재 배치 크기
                  fill_value=log_probs.size(0),  #시퀀스 길이
                  dtype=torch.long
                ).to(device)


              loss = criterion(log_probs, labels, output_lengths, label_lengths)
              total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f}")

        torch.cuda.empty_cache()
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved (epoch {epoch}, val loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    train()
