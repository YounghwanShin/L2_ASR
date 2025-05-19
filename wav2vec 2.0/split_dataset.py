import os
import shutil
import random

def split_l2arctic_speaker_folders(
    input_dir,
    output_dir,
    test_speaker_count=6,
    val_ratio=0.2,
    seed=42
):
    # 1. 화자 폴더 목록 불러오기
    all_speakers = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    # 2. 셔플 및 분할
    random.seed(seed)
    random.shuffle(all_speakers)

    test_speakers = all_speakers[:test_speaker_count]
    trainval_speakers = all_speakers[test_speaker_count:]

    val_count = int(len(trainval_speakers) * val_ratio)
    val_speakers = trainval_speakers[:val_count]
    train_speakers = trainval_speakers[val_count:]

    print(" 분할 결과:")
    print(f"  ▸ Train 화자 수: {len(train_speakers)}")
    print(f"  ▸ Val   화자 수: {len(val_speakers)}")
    print(f"  ▸ Test  화자 수: {len(test_speakers)}")
    
    # 3. 복사 함수
    def copy_speakers(speaker_list, split_name):
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for speaker in speaker_list:
            src = os.path.join(input_dir, speaker)
            dst = os.path.join(split_dir, speaker)
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # 4. 실제 복사
    copy_speakers(train_speakers, 'train')
    copy_speakers(val_speakers, 'val')
    copy_speakers(test_speakers, 'test')

    print("폴더 복사 완료!")
    print(f"  🔹 저장 위치: {output_dir}/train, val, test")

# 실행 예시
if __name__ == "__main__":
    split_l2arctic_speaker_folders(
        input_dir='/home/ellt/Workspace/wav2vec/wav2vec 2.0/data/l2arctic_dataset',
        output_dir='/home/ellt/Workspace/wav2vec/wav2vec 2.0/split_data',
        test_speaker_count=6,
        val_ratio=0.2,
        seed=42
    )
