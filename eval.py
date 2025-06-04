import os
import json
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from config import Config
from model import SimpleMultiTaskModel
from data_prepare import EvaluationDataset, evaluation_collate_fn
from evaluate import evaluate_error_detection, evaluate_phoneme_recognition

def load_model_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value
    
    return new_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, help='Override config values in format key=value')
    parser.add_argument('--eval_data', type=str, help='Override eval data path')
    parser.add_argument('--phoneme_map', type=str, help='Override phoneme map path')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    
    args = parser.parse_args()
    
    config = Config()
    
    if args.eval_data:
        config.eval_data = args.eval_data
    if args.phoneme_map:
        config.phoneme_map = args.phoneme_map
    if args.output_dir:
        config.result_dir = args.output_dir
    
    if args.config:
        for override in args.config.split(','):
            key, value = override.split('=')
            if hasattr(config, key):
                attr_type = type(getattr(config, key))
                setattr(config, key, attr_type(value))
    
    os.makedirs(config.result_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    with open(config.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    error_type_names = {0: 'blank', 1: 'incorrect', 2: 'correct'}
    
    eval_dataset = EvaluationDataset(
        config.eval_data, phoneme_to_id,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False, collate_fn=evaluation_collate_fn)
    
    model = SimpleMultiTaskModel(
        pretrained_model_name=config.pretrained_model,
        hidden_dim=config.hidden_dim,
        num_phonemes=config.num_phonemes,
        num_error_types=config.num_error_types
    )
    
    state_dict = load_model_checkpoint(args.model_checkpoint, config.device)
    model.load_state_dict(state_dict)
    model = model.to(config.device)
    
    results = {}
    
    logger.info("Evaluating error detection...")
    error_detection_results = evaluate_error_detection(model, eval_dataloader, config.device, error_type_names)
    
    logger.info("===== Error Detection Results =====")
    logger.info(f"Token Accuracy: {error_detection_results['token_accuracy']:.4f}")
    logger.info(f"Weighted F1: {error_detection_results['weighted_f1']:.4f}")
    
    for error_type, metrics in error_detection_results['class_metrics'].items():
        logger.info(f"{error_type}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
    results['error_detection'] = error_detection_results
    
    logger.info("Evaluating phoneme recognition...")
    phoneme_recognition_results = evaluate_phoneme_recognition(model, eval_dataloader, config.device, id_to_phoneme)
    
    logger.info("===== Phoneme Recognition Results =====")
    logger.info(f"Phoneme Error Rate (PER): {phoneme_recognition_results['per']:.4f}")
    logger.info(f"Phoneme Accuracy: {1.0 - phoneme_recognition_results['per']:.4f}")
    
    results['phoneme_recognition'] = {
        'per': phoneme_recognition_results['per'],
        'accuracy': 1.0 - phoneme_recognition_results['per'],
        'total_phonemes': phoneme_recognition_results['total_phonemes'],
        'total_errors': phoneme_recognition_results['total_errors']
    }
    
    results_path = os.path.join(config.result_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("=== SUMMARY ===")
    logger.info(f"Error Detection Token Accuracy: {error_detection_results['token_accuracy']:.4f}")
    logger.info(f"Phoneme Recognition PER: {phoneme_recognition_results['per']:.4f}")

if __name__ == "__main__":
    main()