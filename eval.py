import os
import json
import argparse
import logging

import torch
from torch.utils.data import DataLoader

from model import MultiTaskModel
from data import EvaluationDataset, evaluation_collate_fn
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
    parser = argparse.ArgumentParser(description='Multi-task Model Final Evaluation')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_data', type=str, required=True)
    parser.add_argument('--phoneme_map', type=str, required=True)
    parser.add_argument('--model_checkpoint', type=str, required=True)
    
    parser.add_argument('--pretrained_model', type=str, default='facebook/wav2vec2-large-xlsr-53')
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_phonemes', type=int, default=42)
    parser.add_argument('--num_error_types', type=int, default=3)
    parser.add_argument('--use_cross_attention', action='store_true', default=True)
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_audio_length', type=int, default=None)
    
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--detailed', action='store_true')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading phoneme mapping from: {args.phoneme_map}")
    with open(args.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    error_type_names = {0: 'blank', 1: 'incorrect', 2: 'correct'}
    
    logger.info(f"Loading evaluation dataset: {args.eval_data}")
    eval_dataset = EvaluationDataset(args.eval_data, phoneme_to_id, max_length=args.max_audio_length)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=evaluation_collate_fn)
    
    logger.info("Initializing multi-task model")
    model = MultiTaskModel(
        pretrained_model_name=args.pretrained_model,
        hidden_dim=args.hidden_dim,
        num_phonemes=args.num_phonemes,
        num_error_types=args.num_error_types,
        use_cross_attention=args.use_cross_attention
    )
    
    logger.info(f"Loading model checkpoint: {args.model_checkpoint}")
    state_dict = load_model_checkpoint(args.model_checkpoint, args.device)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    
    results = {}
    
    logger.info("Evaluating error detection...")
    error_detection_results = evaluate_error_detection(model, eval_dataloader, args.device, error_type_names)
    
    logger.info("\n===== Error Detection Results =====")
    logger.info(f"Sequence Accuracy: {error_detection_results['sequence_accuracy']:.4f}")
    logger.info(f"Token Accuracy: {error_detection_results['token_accuracy']:.4f}")
    logger.info(f"Average Edit Distance: {error_detection_results['avg_edit_distance']:.4f}")
    logger.info(f"Weighted F1: {error_detection_results['weighted_f1']:.4f}")
    logger.info(f"Macro F1: {error_detection_results['macro_f1']:.4f}")
    
    logger.info("\nError Type Metrics:")
    for error_type, metrics in error_detection_results['class_metrics'].items():
        logger.info(f"  {error_type}:")
        logger.info(f"    Precision: {metrics['precision']:.4f}")
        logger.info(f"    Recall: {metrics['recall']:.4f}")
        logger.info(f"    F1 Score: {metrics['f1']:.4f}")
        logger.info(f"    Support: {metrics['support']}")
        
    results['error_detection'] = error_detection_results
    
    logger.info("Evaluating phoneme recognition...")
    phoneme_recognition_results = evaluate_phoneme_recognition(model, eval_dataloader, args.device, id_to_phoneme)
    
    logger.info("\n===== Phoneme Recognition Results =====")
    logger.info(f"Phoneme Error Rate (PER): {phoneme_recognition_results['per']:.4f}")
    logger.info(f"Phoneme Accuracy: {1.0 - phoneme_recognition_results['per']:.4f}")
    logger.info(f"Total Phonemes: {phoneme_recognition_results['total_phonemes']}")
    logger.info(f"Total Errors: {phoneme_recognition_results['total_errors']}")
    logger.info(f"Insertions: {phoneme_recognition_results['insertions']}")
    logger.info(f"Deletions: {phoneme_recognition_results['deletions']}")
    logger.info(f"Substitutions: {phoneme_recognition_results['substitutions']}")
    
    results['phoneme_recognition'] = {
        'per': phoneme_recognition_results['per'],
        'accuracy': 1.0 - phoneme_recognition_results['per'],
        'total_phonemes': phoneme_recognition_results['total_phonemes'],
        'total_errors': phoneme_recognition_results['total_errors'],
        'insertions': phoneme_recognition_results['insertions'],
        'deletions': phoneme_recognition_results['deletions'],
        'substitutions': phoneme_recognition_results['substitutions']
    }
    
    if args.detailed:
        per_sample_results_path = os.path.join(args.output_dir, 'per_sample_results.json')
        with open(per_sample_results_path, 'w') as f:
            json.dump(phoneme_recognition_results['per_sample'], f, indent=2)
        logger.info(f"Per-sample PER results saved to {per_sample_results_path}")
        
        detailed_results_path = os.path.join(args.output_dir, 'detailed_per_results.json')
        with open(detailed_results_path, 'w') as f:
            json.dump(phoneme_recognition_results['per_details'], f, indent=2)
        logger.info(f"Detailed PER results saved to {detailed_results_path}")
    
    results_path = os.path.join(args.output_dir, 'multitask_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Final evaluation results saved to {results_path}")
    
    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Error Detection Token Accuracy: {error_detection_results['token_accuracy']:.4f}")
    logger.info(f"Phoneme Recognition PER: {phoneme_recognition_results['per']:.4f}")
    logger.info(f"Phoneme Recognition Accuracy: {1.0 - phoneme_recognition_results['per']:.4f}")

if __name__ == "__main__":
    main()