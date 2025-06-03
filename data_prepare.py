import json
import os
import logging
from pathlib import Path
import torch
import torchaudio
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
import random

logger = logging.getLogger(__name__)


def prepare_data(data_folder, save_folder, train_json, val_json, test_json, **kwargs):
    pass


def create_dataset_from_json(json_path, phoneme_to_id, hparams):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    error_mapping = {'C': 2, 'I': 1}
    dataset_dict = {}
    valid_samples = 0
    skipped_samples = 0
    
    for sample_id, item in data.items():
        wav_file = item.get('wav', sample_id)
        
        if not os.path.exists(wav_file):
            logger.warning(f"Audio file not found: {wav_file}")
            skipped_samples += 1
            continue
        
        try:
            info = torchaudio.info(wav_file)
            duration = info.num_frames / info.sample_rate
            
            if duration < 0.5 or duration > 30.0:
                skipped_samples += 1
                continue
                
        except Exception as e:
            logger.warning(f"Could not process {wav_file}: {e}")
            skipped_samples += 1
            continue
        
        sample_id = os.path.basename(wav_file).replace('.wav', '')
        sample = {
            "wav_file": wav_file,
            "duration": duration,
            "spk_id": item.get('spk_id', 'unknown')
        }
        
        if 'error_labels' in item and item['error_labels'].strip():
            error_labels = item['error_labels'].split()
            error_ids = [error_mapping.get(label, 0) for label in error_labels]
            
            if any(eid > 0 for eid in error_ids):
                sample["error_tokens"] = error_ids
                sample["error_tokens_list"] = " ".join(map(str, error_ids))
        
        if 'perceived_train_target' in item and item['perceived_train_target'].strip():
            perceived_phonemes = item['perceived_train_target'].split()
            perceived_ids = [phoneme_to_id.get(p, 0) for p in perceived_phonemes]
            
            if perceived_ids:
                sample["phoneme_tokens"] = perceived_ids
                sample["phoneme_tokens_list"] = " ".join(map(str, perceived_ids))
        
        if 'canonical_aligned' in item and item['canonical_aligned'].strip():
            canonical_phonemes = item['canonical_aligned'].split()
            canonical_ids = [phoneme_to_id.get(p, 0) for p in canonical_phonemes]
            
            if canonical_ids:
                sample["canonical_tokens"] = canonical_ids
                sample["canonical_tokens_list"] = " ".join(map(str, canonical_ids))
        
        has_error = "error_tokens" in sample
        has_phoneme = "phoneme_tokens" in sample
        
        task = hparams["task"]
        if (task == "error" and has_error) or \
           (task == "phoneme" and has_phoneme) or \
           (task == "both" and (has_error or has_phoneme)):
            dataset_dict[sample_id] = sample
            valid_samples += 1
        else:
            skipped_samples += 1
    
    logger.info(f"Dataset from {json_path}: {valid_samples} valid, {skipped_samples} skipped")
    return dataset_dict


def audio_pipeline(wav_file, hparams):
    try:
        sig, sr = torchaudio.load(wav_file)
        
        if sig.shape[0] > 1:
            sig = torch.mean(sig, dim=0, keepdim=True)
        
        sample_rate = hparams["sample_rate"]
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            sig = resampler(sig)
        
        max_audio_length = hparams.get("max_audio_length")
        if max_audio_length and sig.shape[1] > max_audio_length:
            sig = sig[:, :max_audio_length]
        
        sig = sig.squeeze(0)
        return sig
        
    except Exception as e:
        logger.error(f"Error loading audio {wav_file}: {e}")
        return torch.zeros(1000)


def tokens_pipeline(tokens_list):
    if tokens_list and tokens_list.strip():
        try:
            tokens = [int(x) for x in tokens_list.split()]
            return torch.tensor(tokens, dtype=torch.long)
        except ValueError:
            return torch.tensor([], dtype=torch.long)
    return torch.tensor([], dtype=torch.long)


def dataio_prepare(hparams):
    with open(hparams["phoneme_map"], 'r') as f:
        phoneme_to_id = json.load(f)
    
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    error_types = {0: "", 1: "I", 2: "C"}
    
    hparams["phoneme_decoder"] = lambda x: id_to_phoneme.get(str(x), f"UNK_{x}")
    hparams["error_decoder"] = lambda x: error_types.get(x, f"UNK_{x}")
    
    datasets = {}
    data_loaders = {}
    
    for split, json_file in [("train", hparams["train_json"]), 
                            ("val", hparams["val_json"]), 
                            ("test", hparams["test_json"])]:
        
        if not os.path.exists(json_file):
            logger.warning(f"Dataset file not found: {json_file}")
            continue
        
        dataset_dict = create_dataset_from_json(json_file, phoneme_to_id, hparams)
        
        if not dataset_dict:
            logger.warning(f"No valid samples found in {json_file}")
            continue
        
        dataset = DynamicItemDataset(dataset_dict)
        
        dataset.add_dynamic_item(
            func=lambda wav_file: audio_pipeline(wav_file, hparams),
            takes="wav_file",
            provides="sig"
        )
        
        task = hparams["task"]
        if task in ['error', 'both']:
            dataset.add_dynamic_item(
                func=tokens_pipeline,
                takes="error_tokens_list",
                provides="error_tokens"
            )
        
        if task in ['phoneme', 'both']:
            dataset.add_dynamic_item(
                func=tokens_pipeline,
                takes="phoneme_tokens_list", 
                provides="phoneme_tokens"
            )
            
            dataset.add_dynamic_item(
                func=tokens_pipeline,
                takes="canonical_tokens_list",
                provides="canonical_tokens"
            )
        
        output_keys = ["id", "sig"]
        if task in ['error', 'both']:
            output_keys.append("error_tokens")
        if task in ['phoneme', 'both']:
            output_keys.extend(["phoneme_tokens", "canonical_tokens"])
        
        dataset.set_output_keys(output_keys)
        
        loader_opts = hparams.get(f"{split}_dataloader_opts", hparams["train_dataloader_opts"])
        
        data_loader = SaveableDataLoader(
            dataset,
            collate_fn=PaddedBatch,
            **loader_opts
        )
        
        datasets[split] = dataset
        data_loaders[split] = data_loader
        
        logger.info(f"Created {split} dataset: {len(dataset)} samples")
    
    train_loader = data_loaders.get("train")
    val_loader = data_loaders.get("val") 
    test_loader = data_loaders.get("test")
    
    return train_loader, val_loader, test_loader


class DatasetStats:
    @staticmethod
    def compute_stats(data_loader, phoneme_decoder, error_decoder):
        total_samples = 0
        total_duration = 0.0
        error_samples = 0
        phoneme_samples = 0
        
        phoneme_counts = {}
        error_counts = {}
        speaker_counts = {}
        
        for batch in data_loader:
            batch_size = len(batch.id)
            total_samples += batch_size
            
            for sample_id in batch.id:
                parts = sample_id.split('_')
                if len(parts) > 0:
                    spk = parts[0] if len(parts[0]) <= 4 else 'unknown'
                    speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
            
            if hasattr(batch, 'sig'):
                wavs, wav_lens = batch.sig
                durations = wav_lens * wavs.shape[1] / 16000
                total_duration += durations.sum().item()
            
            if hasattr(batch, 'error_tokens'):
                error_samples += batch_size
                if hasattr(batch.error_tokens, 'data'):
                    tokens_data = batch.error_tokens.data
                    lengths_data = batch.error_tokens.lengths
                else:
                    tokens_data = batch.error_tokens
                    lengths_data = [len(t) for t in tokens_data]
                
                for tokens, length in zip(tokens_data, lengths_data):
                    for token in tokens[:length]:
                        if isinstance(token, torch.Tensor):
                            token = token.item()
                        error_type = error_decoder(token)
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            if hasattr(batch, 'phoneme_tokens'):
                phoneme_samples += batch_size
                if hasattr(batch.phoneme_tokens, 'data'):
                    tokens_data = batch.phoneme_tokens.data
                    lengths_data = batch.phoneme_tokens.lengths
                else:
                    tokens_data = batch.phoneme_tokens
                    lengths_data = [len(t) for t in tokens_data]
                
                for tokens, length in zip(tokens_data, lengths_data):
                    for token in tokens[:length]:
                        if isinstance(token, torch.Tensor):
                            token = token.item()
                        phoneme = phoneme_decoder(token)
                        phoneme_counts[phoneme] = phoneme_counts.get(phoneme, 0) + 1
        
        stats = {
            "total_samples": total_samples,
            "total_duration": total_duration,
            "avg_duration": total_duration / total_samples if total_samples > 0 else 0,
            "error_samples": error_samples,
            "phoneme_samples": phoneme_samples,
            "num_speakers": len(speaker_counts),
            "speaker_distribution": speaker_counts,
            "error_distribution": error_counts,
            "phoneme_distribution": phoneme_counts,
            "num_phoneme_types": len(phoneme_counts),
            "num_error_types": len(error_counts)
        }
        
        return stats


def validate_l2arctic_data(data_folder):
    logger.info("Validating L2Arctic dataset...")
    
    required_files = ['train_data.json', 'val_data.json', 'eval.json', 'phoneme_to_id.json']
    missing_files = []
    
    for file in required_files:
        filepath = os.path.join(data_folder, file)
        if not os.path.exists(filepath):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    phoneme_file = os.path.join(data_folder, 'phoneme_to_id.json')
    with open(phoneme_file, 'r') as f:
        phoneme_to_id = json.load(f)
    
    logger.info(f"Phoneme vocabulary size: {len(phoneme_to_id)}")
    
    for split in ['train_data.json', 'val_data.json', 'eval.json']:
        filepath = os.path.join(data_folder, split)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        valid_samples = 0
        missing_audio = 0
        
        for sample_id, item in data.items():
            wav_file = item.get('wav', sample_id)
            if os.path.exists(wav_file):
                valid_samples += 1
            else:
                missing_audio += 1
                if missing_audio <= 5:
                    logger.warning(f"Missing audio: {wav_file}")
        
        logger.info(f"{split}: {valid_samples} valid, {missing_audio} missing audio files")
    
    logger.info("Dataset validation completed!")
    return True