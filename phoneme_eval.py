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

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def detect_phoneme_model_type_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    state_dict = remove_module_prefix(state_dict)
    keys = list(state_dict.keys())
    
    if any('transformer_encoder' in key for key in keys):
        return 'transformer'
    elif any('shared_encoder' in key for key in keys):
        return 'simple'

def get_phoneme_sample_predictions(model, eval_dataloader, device, id_to_phoneme, num_samples=5):
    model.eval()
    sample_predictions = []
    samples_collected = 0
    
    with torch.no_grad():
        for batch_data in eval_dataloader:
            if samples_collected >= num_samples:
                break
                
            (waveforms, perceived_phoneme_ids, canonical_phoneme_ids, 
             audio_lengths, perceived_lengths, canonical_lengths, wav_files) = batch_data
            
            waveforms = waveforms.to(device)
            audio_lengths = audio_lengths.to(device)
            
            from phoneme_evaluate import make_attn_mask, get_wav2vec2_output_lengths_official, decode_ctc
            
            input_lengths = get_wav2vec2_output_lengths_official(model, audio_lengths)
            wav_lens_norm = audio_lengths.float() / waveforms.shape[1]
            attention_mask = make_attn_mask(waveforms, wav_lens_norm)
            
            outputs = model(waveforms, attention_mask)
            phoneme_logits = outputs['phoneme_logits']
            
            phoneme_input_lengths = torch.clamp(input_lengths, min=1, max=phoneme_logits.size(1))
            phoneme_log_probs = torch.log_softmax(phoneme_logits, dim=-1)
            phoneme_predictions = decode_ctc(phoneme_log_probs, phoneme_input_lengths)
            
            for i in range(min(waveforms.shape[0], num_samples - samples_collected)):
                phoneme_actual = [id_to_phoneme.get(str(int(pid)), f"UNK_{pid}") 
                                for pid in perceived_phoneme_ids[i][:perceived_lengths[i]]]
                phoneme_pred = [id_to_phoneme.get(str(int(pid)), f"UNK_{pid}") 
                              for pid in phoneme_predictions[i]]
                
                canonical_actual = [id_to_phoneme.get(str(int(pid)), f"UNK_{pid}") 
                                  for pid in canonical_phoneme_ids[i][:canonical_lengths[i]]]
                
                sample_predictions.append({
                    'file': wav_files[i],
                    'phoneme_actual': phoneme_actual,
                    'phoneme_predicted': phoneme_pred,
                    'canonical_phonemes': canonical_actual
                })
                
                samples_collected += 1
                if samples_collected >= num_samples:
                    break
            
            if samples_collected >= num_samples:
                break
    
    return sample_predictions

def infer_model_type_from_path(checkpoint_path):
    path_parts = checkpoint_path.split('/')
    for part in path_parts:
        if 'phoneme_simple' in part:
            return 'simple'
        elif 'phoneme_transformer' in part:
            return 'transformer'
        elif 'simple' in part:
            return 'simple'
        elif 'transformer' in part or 'trm' in part:
            return 'transformer'
    return 'simple'

def enable_wav2vec2_specaug(model, enable=True):
    actual_model = model.module if hasattr(model, 'module') else model
    if hasattr(actual_model.encoder.wav2vec2, 'config'):
        actual_model.encoder.wav2vec2.config.apply_spec_augment = enable

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
    
    model_class = get_phoneme_model_class(model_type)
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
    if 'mpd_f1' in phoneme_recognition_results:
        logger.info(f"MPD F1 Score: {phoneme_recognition_results['mpd_f1']:.4f}")
    
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
    if 'mpd_f1' in phoneme_recognition_results:
        logger.info(f"Original MPD F1: {phoneme_recognition_results['mpd_f1']:.4f}")
    logger.info(f"Mispronunciation Detection F1: {phoneme_recognition_results['mispronunciation_f1']:.4f}")
    
    logger.info("Collecting sample predictions...")
    sample_predictions = get_phoneme_sample_predictions(model, eval_dataloader, device, id_to_phoneme)
    
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
        'sample_predictions': sample_predictions,
        'evaluation_results': {
            'phoneme_recognition': {
                'per': phoneme_recognition_results['per'],
                'accuracy': 1.0 - phoneme_recognition_results['per'],
                'mpd_f1': phoneme_recognition_results['mpd_f1'],
                'mispronunciation_precision': phoneme_recognition_results['mispronunciation_precision'],
                'mispronunciation_recall': phoneme_recognition_results['mispronunciation_recall'],
                'mispronunciation_f1': phoneme_recognition_results['mispronunciation_f1'],
                'confusion_matrix': phoneme_recognition_results['confusion_matrix']
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
    logger.info(f"Results include: config, {len(sample_predictions)} sample predictions, and full evaluation metrics")

if __name__ == "__main__":
    main()