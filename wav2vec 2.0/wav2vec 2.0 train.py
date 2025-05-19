import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from wav2vec import NaiveWav2Vec2PhonemeModel
from data import PhonemeRecognitionDataset

# 원래 배치 8로 하면 out of memory 문제 발생 -> 배치 자르면 에폭1 끝나고 인풋 길이랑 배치 사이즈
# 달라서 에폭 2로 못넘어감. MAx audio length 지정해주면 마찬가지로 막판에 길이 불일치 문제.
# 해결이 안된 상태.

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

            optimizer.zero_grad()
            logits = model(waveforms)                           # (B, T, C)
            log_probs = torch.log_softmax(logits, dim=-1)

            batch_size = log_probs.size(1)                      # ✅ 실제 배치 크기
            seq_len = log_probs.size(0)                         # ✅ 시퀀스 길이

            output_lengths = torch.full(
                size=(batch_size,),                             # ✅ 반드시 (B,) 크기
                fill_value=seq_len,
                dtype=torch.long
            ).to(device)

          
            print("log_probs shape:", log_probs.transpose(0, 1).shape)
            print("labels shape:", labels.shape)
            print("output_lengths shape:", output_lengths.shape)
            print("label_lengths shape:", label_lengths.shape)
            print("label_lengths:", label_lengths)
            
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

              logits = model(waveforms)  # logits: (B, T, C)
              log_probs = torch.log_softmax(logits, dim=-1)

              current_batch_size = log_probs.size(1)
              seq_len = log_probs.size(0)

              output_lengths = torch.full(
                  size=(log_probs.size(0),),  #현재 배치 크기
                  fill_value=log_probs.size(1),  #시퀀스 길이
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
