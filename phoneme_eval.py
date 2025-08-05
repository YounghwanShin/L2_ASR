import os
import json
import argparse
import logging
import torch
from torch.utils.data import DataLoader

from config import Config
from utils import (
    get_phoneme_model_class,
    detect_phoneme_model_type_from_checkpoint,
    evaluate_phoneme_recognition,
    enable_wav2vec2_specaug,
    remove_module_prefix,
)
from data_prepare import BaseDataset, collate_fn

def main():
    parser = argparse.ArgumentParser(description='Evaluate Phoneme-only L2 Pronunciation Model')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--eval_data', type=str, help='Override evaluation data path')
    parser.add_argument('--phoneme_map', type=str, help='Override phoneme map path')
    parser.add_argument('--model_type', type=str, help='Force model type (simple/transformer)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--save_predictions', action='store_true', help='Save detailed predictions')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    config = Config()
    
    if args.eval_data:
        config.eval_data = args.eval_data
    if args.phoneme_map:
        config.phoneme_map = args.phoneme_map
    if args.batch_size:
        config.eval_batch_size = args.batch_size
    
    model_type = args.model_type
    if model_type is None:
        model_type = detect_phoneme_model_type_from_checkpoint(args.model_checkpoint)
        logger.info(f"Auto-detected model type from checkpoint: {model_type}")
    else:
        detected_type = detect_phoneme_model_type_from_checkpoint(args.model_checkpoint)
        if model_type != detected_type:
            logger.warning(f"Specified model type '{model_type}' doesn't match checkpoint '{detected_type}'. Using checkpoint type.")
            model_type = detected_type
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Checkpoint: {args.model_checkpoint}")
    
    with open(config.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    
    model_class, loss_class = get_phoneme_model_class(model_type)
    model_config = config.model_configs[model_type]
    
    model = model_class(
        pretrained_model_name=config.pretrained_model,
        num_phonemes=config.num_phonemes,
        **model_config
    )
    
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            logger.info(f"Training metrics: {checkpoint['metrics']}")
    else:
        state_dict = checkpoint
    
    state_dict = remove_module_prefix(state_dict)
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.error(f"Failed to load checkpoint. Error: {str(e)}")
        logger.error(f"Model architecture: {model_type}")
        logger.error("This usually means the checkpoint was saved with a different model architecture.")
        logger.error("Try running without --model_type to auto-detect, or check your checkpoint path.")
        raise
    
    model = model.to(device)
    model.eval()
    enable_wav2vec2_specaug(model, False)
    
    eval_dataset = BaseDataset(
        config.eval_data, phoneme_to_id,
        task_mode=config.task_mode['phoneme_eval'],
        max_length=config.max_length,
        sampling_rate=config.sampling_rate
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.eval_batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, task_mode=config.task_mode['phoneme_eval'])
    )
    
    logger.info("Starting phoneme-only evaluation...")
    
    logger.info("Evaluating phoneme recognition...")
    phoneme_recognition_results = evaluate_phoneme_recognition(model, eval_dataloader, device, id_to_phoneme)
    
    logger.info("\n" + "="*80)
    logger.info("COMPLETE PHONEME-ONLY EVALUATION RESULTS")
    logger.info("="*80)
    
    logger.info("\n--- PHONEME RECOGNITION RESULTS ---")
    logger.info(f"Phoneme Error Rate (PER): {phoneme_recognition_results['per']:.4f}")
    logger.info(f"Phoneme Accuracy: {1.0 - phoneme_recognition_results['per']:.4f}")
    
    if 'total_phonemes' in phoneme_recognition_results:
        logger.info(f"Total Phonemes: {phoneme_recognition_results['total_phonemes']}")
    if 'total_errors' in phoneme_recognition_results:
        logger.info(f"Total Errors: {phoneme_recognition_results['total_errors']}")
    if 'insertions' in phoneme_recognition_results:
        logger.info(f"Insertions: {phoneme_recognition_results['insertions']}")
    if 'deletions' in phoneme_recognition_results:
        logger.info(f"Deletions: {phoneme_recognition_results['deletions']}")
    if 'substitutions' in phoneme_recognition_results:
        logger.info(f"Substitutions: {phoneme_recognition_results['substitutions']}")
    
    logger.info("\n--- MISPRONUNCIATION DETECTION METRICS ---")
    logger.info(f"Precision: {phoneme_recognition_results['mispronunciation_precision']:.4f}")
    logger.info(f"Recall: {phoneme_recognition_results['mispronunciation_recall']:.4f}")
    logger.info(f"F1-Score: {phoneme_recognition_results['mispronunciation_f1']:.4f}")
    
    logger.info("\n--- CONFUSION MATRIX ---")
    cm = phoneme_recognition_results['confusion_matrix']
    logger.info(f"True Acceptance (TA): {cm['true_acceptance']}")
    logger.info(f"False Rejection (FR): {cm['false_rejection']}")
    logger.info(f"False Acceptance (FA): {cm['false_acceptance']}")
    logger.info(f"True Rejection (TR): {cm['true_rejection']}")
    
    logger.info("\n--- SUMMARY ---")
    logger.info(f"Overall Phoneme Recognition Performance: {1.0 - phoneme_recognition_results['per']:.4f} (Accuracy)")
    logger.info(f"Mispronunciation Detection F1: {phoneme_recognition_results['mispronunciation_f1']:.4f}")
    
    logger.info("\n--- BY COUNTRY RESULTS ---")
    for country in sorted(phoneme_recognition_results.get('by_country', {}).keys()):
        logger.info(f"\n{country}:")
        phoneme_country = phoneme_recognition_results['by_country'][country]
        logger.info(f"  Phoneme Accuracy: {1.0 - phoneme_country['per']:.4f}")
        logger.info(f"  Mispronunciation F1: {phoneme_country['mispronunciation_f1']:.4f}")
    
    experiment_dir_name = os.path.basename(os.path.dirname(os.path.dirname(args.model_checkpoint)))
    
    config_info = {
        'model_type': f"phoneme_{model_type}",
        'checkpoint_path': args.model_checkpoint,
        'experiment_name': experiment_dir_name,
        'evaluation_date': __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_config': config.model_configs.get(model_type, {})
    }
    
    final_results = {
        'config': config_info,
        'evaluation_results': {
            'phoneme_recognition': {
                'per': phoneme_recognition_results['per'],
                'accuracy': 1.0 - phoneme_recognition_results['per'],
                'mispronunciation_precision': phoneme_recognition_results['mispronunciation_precision'],
                'mispronunciation_recall': phoneme_recognition_results['mispronunciation_recall'],
                'mispronunciation_f1': phoneme_recognition_results['mispronunciation_f1'],
                'confusion_matrix': phoneme_recognition_results['confusion_matrix'],
                'by_country': phoneme_recognition_results.get('by_country', {})
            }
        }
    }
    
    for key in ['total_phonemes', 'total_errors', 'insertions', 'deletions', 'substitutions']:
        if key in phoneme_recognition_results:
            final_results['evaluation_results']['phoneme_recognition'][key] = phoneme_recognition_results[key]
    
    evaluation_results_dir = 'evaluation_results'
    os.makedirs(evaluation_results_dir, exist_ok=True)
    
    results_filename = f"{experiment_dir_name}_eval_results.json"
    results_path = os.path.join(evaluation_results_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"\nComplete evaluation results saved to: {results_path}")
    logger.info(f"Results include: config and full evaluation metrics with country breakdown")

if __name__ == "__main__":
    main()
