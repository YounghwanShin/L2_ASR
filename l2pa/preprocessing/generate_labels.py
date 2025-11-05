"""Error label generation from phoneme alignments."""
import json
from pathlib import Path


def needleman_wunsch_align(canonical, perceived):
  n, m = len(canonical), len(perceived)
  score = [[0] * (m + 1) for _ in range(n + 1)]
  for i in range(n + 1):
    score[i][0] = -i
  for j in range(m + 1):
    score[0][j] = -j
  
  for i in range(1, n + 1):
    for j in range(1, m + 1):
      match = score[i-1][j-1] + (0 if canonical[i-1] == perceived[j-1] else -1)
      delete = score[i-1][j] - 1
      insert = score[i][j-1] - 1
      score[i][j] = max(match, delete, insert)
  
  aligned_canonical, aligned_perceived = [], []
  i, j = n, m
  
  while i > 0 or j > 0:
    if i > 0 and j > 0 and score[i][j] == score[i-1][j-1] + (0 if canonical[i-1] == perceived[j-1] else -1):
      aligned_canonical.append(canonical[i-1])
      aligned_perceived.append(perceived[j-1])
      i -= 1
      j -= 1
    elif i > 0 and score[i][j] == score[i-1][j] - 1:
      aligned_canonical.append(canonical[i-1])
      aligned_perceived.append(None)
      i -= 1
    else:
      aligned_canonical.append(None)
      aligned_perceived.append(perceived[j-1])
      j -= 1
  
  return aligned_canonical[::-1], aligned_perceived[::-1]


def generate_error_labels(canonical_str, perceived_str):
  canonical = canonical_str.split()
  perceived = perceived_str.split()
  aligned_canonical, aligned_perceived = needleman_wunsch_align(canonical, perceived)
  
  error_labels = []
  deletion_pending = False
  
  for canon_phone, perc_phone in zip(aligned_canonical, aligned_perceived):
    if perc_phone is None:
      deletion_pending = True
      continue
    if deletion_pending:
      error_labels.append('D')
      deletion_pending = False
    elif canon_phone is None:
      error_labels.append('I')
    elif canon_phone == perc_phone:
      error_labels.append('C')
    else:
      error_labels.append('S')
  
  return ' '.join(error_labels)


def add_error_labels(input_path, output_path):
  print(f'Loading {input_path}...')
  with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
  
  print(f'Processing {len(data)} samples...')
  success = 0
  label_stats = {'C': 0, 'S': 0, 'I': 0, 'D': 0}
  
  for key, item in data.items():
    canonical = item.get('canonical_train_target', '')
    perceived = item.get('perceived_train_target', '')
    if not canonical or not perceived:
      continue
    error_labels = generate_error_labels(canonical, perceived)
    item['error_labels'] = error_labels
    for label in error_labels.split():
      if label in label_stats:
        label_stats[label] += 1
    success += 1
  
  print(f'\nProcessed {success}/{len(data)} samples')
  print(f'\nLabel statistics:')
  for label, count in label_stats.items():
    print(f'  {label}: {count}')
  
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
  print(f'\nSaved to {output_path}')
