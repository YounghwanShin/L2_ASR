import json
import os
import re

def normalize_phonemes(phoneme_str):
    # 소문자로 변환하고 숫자 제거
    return re.sub(r'\d', '', phoneme_str.lower())

def process_json_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for key, item in data.items():
        original = item.get("perceived_train_target", "")
        normalized = normalize_phonemes(original)
        data[key]["perceived_train_target"] = normalized

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 저장 완료: {json_path}")

# 적용할 대상 JSON 목록
json_files = [
    "/home/ellt/Workspace/L2_ASR/wav2vec2/train.json",
    "/home/ellt/Workspace/L2_ASR/wav2vec2/val.json",
    "/home/ellt/Workspace/L2_ASR/wav2vec2/test.json"
]

for json_file in json_files:
    process_json_file(json_file)
