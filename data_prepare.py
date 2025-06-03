import json
import logging
import torch
from collections import Counter
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset

logger = logging.getLogger(__name__)

def load_data(json_path):
    """Load data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples from {json_path}")
    return data

def build_vocabularies(train_data, hparams):
    """Build phoneme and error vocabularies"""
    
    # Phoneme vocabulary
    phoneme_counter = Counter()
    error_counter = Counter()
    
    for item in train_data.values():
        # Count phonemes from perceived_train_target
        if 'perceived_train_target' in item:
            phonemes = item['perceived_train_target'].split()
            phoneme_counter.update(phonemes)
        
        # Count errors
        if 'error_labels' in item:
            errors = item['error_labels'].split()
            error_counter.update(errors)
    
    # Create phoneme vocabulary (add blank for CTC)
    phoneme_vocab = ['<blank>'] + list(phoneme_counter.keys())
    phoneme_to_id = {p: i for i, p in enumerate(phoneme_vocab)}
    
    # Create error vocabulary (add blank for CTC)
    error_vocab = ['<blank>'] + list(error_counter.keys())
    error_to_id = {e: i for i, e in enumerate(error_vocab)}
    
    logger.info(f"Phoneme vocabulary size: {len(phoneme_vocab)}")
    logger.info(f"Error vocabulary size: {len(error_vocab)}")
    
    return phoneme_to_id, error_to_id

def create_dataset_from_data(data_dict, phoneme_to_id, error_to_id, hparams):
    """Create SpeechBrain dataset from data dictionary"""
    
    dataset_dict = {}
    valid_samples = 0
    skipped_samples = 0
    
    task = hparams["task"]
    
    for sample_id, item in data_dict.items():
        sample = {
            "wav": item["wav"],
            "duration": item["duration"]
        }
        
        # Process phoneme tokens (실제 발음)
        if 'perceived_train_target' in item and item['perceived_train_target'].strip():
            perceived_phonemes = item['perceived_train_target'].split()
            perceived_ids = [phoneme_to_id.get(p, 0) for p in perceived_phonemes]
            sample["phoneme_tokens"] = perceived_ids
        
        # Process error labels
        if 'error_labels' in item and item['error_labels'].strip():
            error_labels = item['error_labels'].split()
            error_ids = [error_to_id.get(e, 0) for e in error_labels]
            sample["error_tokens"] = error_ids
        
        # Filter based on task
        has_phoneme = "phoneme_tokens" in sample
        has_error = "error_tokens" in sample
        
        if task == "phoneme" and has_phoneme:
            dataset_dict[sample_id] = sample
            valid_samples += 1
        elif task == "error" and has_error:
            dataset_dict[sample_id] = sample
            valid_samples += 1
        elif task == "both":
            dataset_dict[sample_id] = sample
            valid_samples += 1
        else:
            skipped_samples += 1
    
    logger.info(f"Dataset: {valid_samples} valid, {skipped_samples} skipped")
    
    # Create SpeechBrain dataset
    dataset = DynamicItemDataset(dataset_dict)
    
    # Audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav_path):
        sig = sb.dataio.dataio.read_audio(wav_path)
        return sig
    
    dataset.add_dynamic_item(audio_pipeline)
    
    # Set output keys based on task
    output_keys = ["sig"]
    if task in ["phoneme", "both"] and any("phoneme_tokens" in item for item in dataset_dict.values()):
        output_keys.append("phoneme_tokens")
    if task in ["error", "both"] and any("error_tokens" in item for item in dataset_dict.values()):
        output_keys.append("error_tokens")
    
    dataset.set_output_keys(output_keys)
    
    logger.info(f"Created dataset: {len(dataset)} samples")
    return dataset

def create_datasets(hparams):
    """Create train, validation, and test datasets"""
    
    # Load data
    train_data = load_data(hparams["train_json"])
    valid_data = load_data(hparams["valid_json"])
    test_data = load_data(hparams["test_json"])
    
    # Build vocabularies from training data
    phoneme_to_id, error_to_id = build_vocabularies(train_data, hparams)
    
    # Save vocabularies
    hparams["phoneme_to_id"] = phoneme_to_id
    hparams["error_to_id"] = error_to_id
    hparams["num_phonemes"] = len(phoneme_to_id)
    hparams["num_errors"] = len(error_to_id)
    
    # Create datasets
    train_dataset = create_dataset_from_data(train_data, phoneme_to_id, error_to_id, hparams)
    valid_dataset = create_dataset_from_data(valid_data, phoneme_to_id, error_to_id, hparams)
    test_dataset = create_dataset_from_data(test_data, phoneme_to_id, error_to_id, hparams)
    
    return train_dataset, valid_dataset, test_dataset