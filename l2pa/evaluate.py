"""Evaluation script for pronunciation assessment model.

This script evaluates a trained model on test data and generates comprehensive
metrics for canonical phoneme recognition, perceived phoneme recognition,
and error detection with per-speaker analysis.
"""

import os
import json
import logging
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import pytz

from .config import Config
from .data.dataset import PronunciationDataset, collate_batch
from .models.unified_model import UnifiedModel
from .evaluation.evaluator import ModelEvaluator
from .training.utils import detect_model_type_from_checkpoint, remove_module_prefix

logger = logging.getLogger(__name__)


def evaluate_model(checkpoint_path, config, save_predictions=False):
    """Main evaluation function.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        config: Configuration object.
        save_predictions: Whether to save detailed predictions.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Model type: {config.model_type}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Load phoneme mapping
    with open(config.phoneme_map, 'r') as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {str(v): k for k, v in phoneme_to_id.items()}
    error_type_names = config.get_error_type_names()

    # Create model
    model_config = config.model_configs[config.model_type]
    model = UnifiedModel(
        pretrained_model_name=config.pretrained_model,
        num_phonemes=config.num_phonemes,
        num_error_types=config.num_error_types,
        **model_config
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        logger.info(f"Loading model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint

    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load evaluation dataset
    eval_dataset = PronunciationDataset(
        config.eval_data, phoneme_to_id,
        training_mode=config.training_mode,
        max_length=config.max_length,
        sampling_rate=config.sampling_rate,
        device=device
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.eval_batch_size, shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, training_mode=config.training_mode)
    )

    # Create evaluator
    evaluator = ModelEvaluator(device)

    logger.info("Starting evaluation...")

    # Evaluate canonical phoneme recognition (multitask only)
    canonical_results = None
    if config.has_canonical_component():
        logger.info("Evaluating canonical phoneme recognition...")
        canonical_results = evaluator.evaluate_canonical_recognition(
            model=model,
            dataloader=eval_dataloader,
            training_mode=config.training_mode,
            id_to_phoneme=id_to_phoneme
        )

    # Evaluate perceived phoneme recognition
    logger.info("Evaluating perceived phoneme recognition...")
    perceived_results = evaluator.evaluate_perceived_recognition(
        model=model,
        dataloader=eval_dataloader,
        training_mode=config.training_mode,
        id_to_phoneme=id_to_phoneme
    )

    # Evaluate error detection
    error_results = None
    if config.has_error_component():
        logger.info("Evaluating error detection...")
        error_results = evaluator.evaluate_error_detection(
            model=model,
            dataloader=eval_dataloader,
            training_mode=config.training_mode,
            error_type_names=error_type_names
        )

    # Print results
    logger.info("\n" + "="*80)
    logger.info("Evaluation Results")
    logger.info("="*80)

    if canonical_results:
        logger.info("\n--- Canonical Phoneme Recognition ---")
        logger.info(f"PER: {canonical_results['per']:.4f}")
        logger.info(f"Accuracy: {1.0 - canonical_results['per']:.4f}")
    
    logger.info("\n--- Perceived Phoneme Recognition ---")
    logger.info(f"PER: {perceived_results['per']:.4f}")
    logger.info(f"Accuracy: {1.0 - perceived_results['per']:.4f}")
    
    if error_results:
        logger.info("\n--- Error Detection ---")
        logger.info(f"Token Accuracy: {error_results['token_accuracy']:.4f}")
        logger.info(f"Weighted F1: {error_results['weighted_f1']:.4f}")

    # Save results
    results_dir = 'evaluation_results'
    os.makedirs(results_dir, exist_ok=True)
    
    experiment_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))
    results_filename = f"{experiment_name}_eval_results.json"
    results_path = os.path.join(results_dir, results_filename)

    eval_results = {
        'config': {
            'training_mode': config.training_mode,
            'model_type': config.model_type,
            'checkpoint_path': checkpoint_path,
            'evaluation_date': datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
        },
        'canonical_recognition': canonical_results,
        'perceived_recognition': perceived_results,
        'error_detection': error_results
    }

    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")
