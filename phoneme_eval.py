import os
import json
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from phoneme_data_prepare import PhonemeEvaluationDataset, phoneme_evaluation_collate_fn
from phoneme_evaluate import evaluate_phoneme_recognition

def get_phoneme_model_class(model_type):
    if model_type == 'simple':
        from models.phoneme_model import SimplePhonemeModel
        return SimplePhonemeModel
    elif model_type == 'transformer':
        from models.phoneme_model_transformer import TransformerPhonemeModel
        return TransformerPhonemeModel
    else:
        raise ValueError(f"Unknown phoneme model type: {model_type}")

def infer_model_type_from_path(checkpoint_path):
    path_parts = checkpoint_path.split('/')
    for part in path_parts:
        if 'phoneme_simple' in part:
            return 'simple'
        elif 'phoneme_transformer' in part:
            return 'transformer'
        elif 'simple' in part:
            return 'simple'
        elif 'transformer' in part:
            return 'transformer'
    return 'simple'

def main():
    parser = argparse.ArgumentParser(description='Evaluate Phoneme-only L2 Pronunciation Model')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--eval_data', type=str, help='Override evaluation data path')
    parser.add_argument('--phoneme_map', type=str, help='Override phoneme map path')
    parser.add_argument('--model_type', type=str, help='Force model type (simple/transformer)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--save_predictions', action='store_true', help='Save detailed predictions')
    
    args = parser.parse_args()
    
    config = Config()
    
    if args.eval_data:
        config.eval_data = args.eval_data
    if args.phoneme_map:
        config.phoneme_map = args.phoneme_map
    if args.batch_size:
        config.eval_batch_size = args.batch_size
    
    model_type = args.model_type
    if model_type is None:
        model_type = infer_model_type_from_path(args.model_checkpoint)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Checkpoint: {args.model_checkpoint}")
    
    with open(config.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    
    model_class = get_phoneme_model_class(model_type)
    model_config = config.model_configs[model_type]
    
    model = model_class(
        pretrained_model_name=config.pretrained_model,
        num_phonemes=config.num_phonemes,
        **model_config
    )
    
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            logger.info(f"Training metrics: {checkpoint['metrics']}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    eval_dataset = PhonemeEvaluationDataset(
        config.eval_data, phoneme_to_id,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.eval_batch_size, shuffle=False,
        collate_fn=phoneme_evaluation_collate_fn
    )
    
    logger.info("Starting phoneme-only evaluation...")
    
    logger.info("Evaluating phoneme recognition...")
    phoneme_recognition_results = evaluate_phoneme_recognition(model, eval_dataloader, device, id_to_phoneme)
    
    logger.info("\n" + "="*50)
    logger.info("PHONEME-ONLY EVALUATION RESULTS")
    logger.info("="*50)
    
    logger.info(f"Phoneme Error Rate (PER): {phoneme_recognition_results['per']:.4f}")
    logger.info(f"Phoneme Accuracy: {1.0 - phoneme_recognition_results['per']:.4f}")
    
    final_results = {
        'model_type': f"phoneme_{model_type}",
        'checkpoint_path': args.model_checkpoint,
        'phoneme_recognition': {
            'per': phoneme_recognition_results['per'],
            'accuracy': 1.0 - phoneme_recognition_results['per']
        }
    }
    
    if args.save_predictions:
        output_dir = os.path.dirname(args.model_checkpoint)
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        
        if 'predictions' in phoneme_recognition_results:
            predictions_path = os.path.join(output_dir, 'predictions.json')
            predictions_data = {
                'phoneme_predictions': phoneme_recognition_results.get('predictions', [])
            }
            with open(predictions_path, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            logger.info(f"Predictions saved to: {predictions_path}")

if __name__ == "__main__":
    main()