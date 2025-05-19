import os
import json
import re
from glob import glob
import wave

# TextGrid에서 phoneme 추출
def extract_phonemes_from_textgrid(tg_path):
    phonemes = []
    try:
        with open(tg_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        in_phones = False
        for line in lines:
            if 'name = "phones"' in line:
                in_phones = True
            elif in_phones and 'item [' in line:
                break
            elif in_phones and 'text =' in line:
                m = re.search(r'text = "(.*)"', line)
                if m and m.group(1).strip():
                    phonemes.append(m.group(1).strip())
    except Exception as e:
        print(f"[Error] TextGrid 읽기 실패: {tg_path} - {e}")
    return " ".join(phonemes)


def get_duration(wav_path):
    try:
        with wave.open(wav_path, 'r') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return round(duration, 2)
    except Exception as e:
        print(f"[Error] duration 추출 실패: {wav_path} - {e}")
        return 0.0


# JSON 파일 생성 (절대경로 기반)
def make_phoneme_json(data_root, output_json_path):
    dataset = {}

    for spk_dir in os.listdir(data_root):
        spk_path = os.path.join(data_root, spk_dir)
        if not os.path.isdir(spk_path):
            continue

        wav_dir = os.path.join(spk_path, "wav")
        tg_dir = os.path.join(spk_path, "textgrid")

        if not os.path.exists(wav_dir) or not os.path.exists(tg_dir):
            print(f"[경고] {spk_dir}의 wav/textgrid 폴더가 없습니다.")
            continue

        for wav_path in glob(os.path.join(wav_dir, "*.wav")):
            base = os.path.splitext(os.path.basename(wav_path))[0]
            tg_path = os.path.join(tg_dir, f"{base}.TextGrid")
            if not os.path.exists(tg_path):
                continue

            phonemes = extract_phonemes_from_textgrid(tg_path)
            if not phonemes:
                continue

            abs_wav_path = os.path.abspath(wav_path)

            dataset[abs_wav_path] = {
                "wav": abs_wav_path,
                "perceived_train_target": phonemes
            }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"JSON 저장 완료: {output_json_path} (총 {len(dataset)}개)")


# 실제 경로 적용
make_phoneme_json(
    data_root="/home/ellt/Workspace/wav2vec/wav2vec 2.0/split_data/train",
    output_json_path="/home/ellt/Workspace/wav2vec/wav2vec 2.0/split_data/train.json"
)

make_phoneme_json(
    data_root="/home/ellt/Workspace/wav2vec/wav2vec 2.0/split_data/test",
    output_json_path="/home/ellt/Workspace/wav2vec/wav2vec 2.0/split_data/test.json"
)

make_phoneme_json(
    data_root="/home/ellt/Workspace/wav2vec/wav2vec 2.0/split_data/val",
    output_json_path="/home/ellt/Workspace/wav2vec/wav2vec 2.0/split_data/val.json"
)
