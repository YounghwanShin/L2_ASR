import json
import logging
import torch
from collections import Counter
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
import librosa
import os

logger = logging.getLogger(__name__)

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples from {json_path}")
    return data

def build_vocabularies(train_data, hparams):
    phoneme_counter = Counter()
    error_counter = Counter()
    
    for item in train_data.values():
        if 'perceived_train_target' in item and item['perceived_train_target'].strip():
            phonemes = item['perceived_train_target'].split()
            phoneme_counter.update(phonemes)
        
        if 'error_labels' in item and item['error_labels'].strip():
            errors = item['error_labels'].split()
            error_counter.update(errors)
    
    phoneme_vocab = ['<blank>'] + sorted(list(phoneme_counter.keys()))
    phoneme_to_id = {p: i for i, p in enumerate(phoneme_vocab)}
    
    error_vocab = ['<blank>'] + sorted(list(error_counter.keys()))
    error_to_id = {e: i for i, e in enumerate(error_vocab)}
    
    logger.info(f"Phoneme vocabulary size: {len(phoneme_vocab)}")
    logger.info(f"Error vocabulary size: {len(error_vocab)}")
    
    return phoneme_to_id, error_to_id

def create_dataset_from_data(data_dict, phoneme_to_id, error_to_id, hparams, is_train=False):
    dataset_dict = {}
    valid_samples = 0
    skipped_samples = 0
    task = hparams["task"]
    
    for sample_id, item in data_dict.items():
        sample = {
            "wav": item["wav"],
            "duration": item["duration"]
        }
        
        if 'perceived_train_target' in item and item['perceived_train_target'].strip():
            perceived_phonemes = item['perceived_train_target'].split()
            perceived_ids = [phoneme_to_id.get(p, 0) for p in perceived_phonemes if p in phoneme_to_id]
            if perceived_ids:
                sample["phoneme_tokens"] = torch.LongTensor(perceived_ids)
        
        if 'error_labels' in item and item['error_labels'].strip():
            error_labels = item['error_labels'].split()
            error_ids = [error_to_id.get(e, 0) for e in error_labels if e in error_to_id]
            if error_ids:
                sample["error_tokens"] = torch.LongTensor(error_ids)
        
        has_phoneme = "phoneme_tokens" in sample
        has_error = "error_tokens" in sample
        
        if task == "phoneme" and has_phoneme:
            dataset_dict[sample_id] = sample
            valid_samples += 1
        elif task == "error" and has_error:
            dataset_dict[sample_id] = sample
            valid_samples += 1
        elif task == "both" and (has_phoneme or has_error):
            dataset_dict[sample_id] = sample
            valid_samples += 1
        else:
            skipped_samples += 1
    
    logger.info(f"Dataset: {valid_samples} valid, {skipped_samples} skipped")
    
    dataset = DynamicItemDataset(dataset_dict)
    
    if is_train and hparams.get("sorting", "ascending") != "random":
        if hparams["sorting"] == "ascending":
            dataset = dataset.filtered_sorted(sort_key="duration")
        elif hparams["sorting"] == "descending":
            dataset = dataset.filtered_sorted(sort_key="duration", reverse=True)
    elif not is_train:
        dataset = dataset.filtered_sorted(sort_key="duration")
    
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav_path):
        import librosa
        audio, sr = librosa.load(wav_path, sr=hparams["sample_rate"])
        sig = hparams["wav2vec2"].feature_extractor(
            audio,
            sampling_rate=hparams["sample_rate"],
        ).input_values[0]
        sig = torch.Tensor(sig)
        return sig
    
    dataset.add_dynamic_item(audio_pipeline)
    
    output_keys = ["id", "sig"]
    if task in ["phoneme", "both"]:
        output_keys.append("phoneme_tokens")
    if task in ["error", "both"]:
        output_keys.append("error_tokens")
    
    dataset.set_output_keys(output_keys)
    
    logger.info(f"Created dataset: {len(dataset)} samples")
    return dataset

def create_datasets(hparams):
    train_data = load_data(hparams["train_json"])
    valid_data = load_data(hparams["valid_json"])
    test_data = load_data(hparams["test_json"])
    
    phoneme_to_id, error_to_id = build_vocabularies(train_data, hparams)
    
    hparams["phoneme_to_id"] = phoneme_to_id
    hparams["error_to_id"] = error_to_id
    hparams["num_phonemes"] = len(phoneme_to_id)
    hparams["num_errors"] = len(error_to_id)
    
    train_dataset = create_dataset_from_data(train_data, phoneme_to_id, error_to_id, hparams, is_train=True)
    valid_dataset = create_dataset_from_data(valid_data, phoneme_to_id, error_to_id, hparams, is_train=False)
    test_dataset = create_dataset_from_data(test_data, phoneme_to_id, error_to_id, hparams, is_train=False)
    
    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    
    phoneme_vocab = list(phoneme_to_id.keys())
    special_labels = {
        "blank_label": hparams["blank_index"],
    }
    
    @sb.utils.data_pipeline.takes("phoneme_tokens")
    @sb.utils.data_pipeline.provides("phn_list_target")
    def token_to_phoneme_list(tokens):
        return [phoneme_vocab[idx] for idx in tokens if idx < len(phoneme_vocab)]
    
    train_dataset.add_dynamic_item(token_to_phoneme_list)
    
    save_folder = hparams.get("output_folder", "./results")
    lab_enc_file = os.path.join(save_folder, "label_encoder.txt")
    os.makedirs(save_folder, exist_ok=True)
    
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_dataset],
        output_key="phn_list_target",
        special_labels=special_labels,
        sequence_input=True,
    )
    
    train_dataset.set_output_keys(["id", "sig", "phoneme_tokens", "error_tokens"])
    valid_dataset.set_output_keys(["id", "sig", "phoneme_tokens", "error_tokens"])
    test_dataset.set_output_keys(["id", "sig", "phoneme_tokens", "error_tokens"])
    
    return train_dataset, valid_dataset, test_dataset, label_encoder