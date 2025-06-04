import os
import json
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data_prepare import EvaluationDataset, evaluation_collate_fn
from evaluate import evaluate_error_detection, evaluate_phoneme_recognition

def get_model_class(model_type):
    if model_type == 'simple':
        from model import SimpleMultiTaskModel
        return SimpleMultiTaskModel
    elif model_type == 'transformer':
        from model_transformer import TransformerMultiTaskModel
        return TransformerMultiTaskModel
    elif model_type == 'cross':
        from model_cross import CrossAttentionMultiTaskModel
        return CrossAttentionMultiTaskModel
    elif model_type == 'hierarchical':
        from model_hierarchical import HierarchicalMultiTaskModel
        return HierarchicalMultiTaskModel
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def infer_model_type_from_path(checkpoint_path):
    path_parts = checkpoint_path.split('/')
    for part in path_parts:
        if 'simple' in part:
            return 'simple'
        elif 'transformer' in part:
            return 'transformer'
        elif 'cross' in part:
            return 'cross'
        elif 'hierarchical' in part:
            return 'hierarchical'
    return 'simple'

def main():
    parser = argparse.ArgumentParser(description='Evaluate Multi-task L2 Pronunciation Model')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--eval_data', type=str, help='Override evaluation data path')
    parser.add_argument('--phoneme_map', type=str, help='Override phoneme map path')
    parser.add_argument('--model_type', type=str, help='Force model type (simple/transformer/cross/hierarchical)')
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
    error_type_names = {0: 'blank', 1: 'incorrect', 2: 'correct'}
    
    model_class = get_model_class(model_type)
    model_config = config.model_configs[model_type]
    
    model = model_class(
        pretrained_model_name=config.pretrained_model,
        num_phonemes=config.num_phonemes,
        num_error_types=config.num_error_types,
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
    
    eval_dataset = EvaluationDataset(
        config.eval_data, phoneme_to_id,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.eval_batch_size, shuffle=False,
        collate_fn=evaluation_collate_fn
    )
    
    logger.info("Starting evaluation...")
    
    logger.info("Evaluating error detection...")
    error_detection_results = evaluate_error_detection(model, eval_dataloader, device, error_type_names)
    
    logger.info("Evaluating phoneme recognition...")
    phoneme_recognition_results = evaluate_phoneme_recognition(model, eval_dataloader, device, id_to_phoneme)
    
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    
    logger.info(f"Error Token Accuracy: {error_detection_results['token_accuracy']:.4f}")
    logger.info(f"Error Weighted F1: {error_detection_results['weighted_f1']:.4f}")
    
    for error_type, metrics in error_detection_results['class_metrics'].items():
        if error_type != 'blank':
            logger.info(f"  {error_type}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    
    logger.info(f"Phoneme Error Rate (PER): {phoneme_recognition_results['per']:.4f}")
    logger.info(f"Phoneme Accuracy: {1.0 - phoneme_recognition_results['per']:.4f}")
    
    final_results = {
        'model_type': model_type,
        'checkpoint_path': args.model_checkpoint,
        'error_detection': {
            'token_accuracy': error_detection_results['token_accuracy'],
            'weighted_f1': error_detection_results['weighted_f1'],
            'class_metrics': error_detection_results['class_metrics']
        },
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
        
        if 'predictions' in error_detection_results:
            predictions_path = os.path.join(output_dir, 'predictions.json')
            predictions_data = {
                'error_predictions': error_detection_results['predictions'],
                'phoneme_predictions': phoneme_recognition_results.get('predictions', [])
            }
            with open(predictions_path, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            logger.info(f"Predictions saved to: {predictions_path}")

if __name__ == "__main__":
    main()