import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import MultiheadAttention
import torch.nn.functional as F


class MultiTaskModel(nn.Module):
    def __init__(self, hidden_dim, num_phonemes, num_error_types, use_cross_attention=True, dropout=0.1, task="both"):
        super().__init__()
        self.task = task
        
        self.feature_projection = Linear(
            input_size=1024,  # Wav2Vec2 output dimension
            n_neurons=hidden_dim
        )
        
        self.layer_norm = LayerNorm(input_size=hidden_dim)
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim // 4,
                kernel_size=k,
                padding=k//2
            ) for k in [1, 3, 5, 7]
        ])
        
        self.error_branch = nn.Sequential(
            Linear(input_size=hidden_dim, n_neurons=hidden_dim),
            LayerNorm(input_size=hidden_dim),
            Swish(),
            nn.Dropout(dropout)
        )
        
        self.phoneme_branch = nn.Sequential(
            Linear(input_size=hidden_dim, n_neurons=hidden_dim),
            LayerNorm(input_size=hidden_dim),
            Swish(),
            nn.Dropout(dropout)
        )
        
        if use_cross_attention:
            self.cross_attention = MultiheadAttention(
                nhead=8,
                d_model=hidden_dim,
                dropout=dropout
            )
        
        self.error_head = nn.Sequential(
            Linear(input_size=hidden_dim, n_neurons=hidden_dim // 2),
            Swish(),
            nn.Dropout(dropout),
            Linear(input_size=hidden_dim // 2, n_neurons=num_error_types)
        )
        
        self.phoneme_head = nn.Sequential(
            Linear(input_size=hidden_dim, n_neurons=hidden_dim // 2),
            Swish(),
            nn.Dropout(dropout),
            Linear(input_size=hidden_dim // 2, n_neurons=num_phonemes)
        )
    
    def forward(self, features, length=None):
        x = self.feature_projection(features)
        x = self.layer_norm(x)
        
        x_conv = x.transpose(1, 2)
        scale_features = []
        
        for conv in self.conv_layers:
            feat = conv(x_conv)
            feat = torch.relu(feat)
            scale_features.append(feat)
        
        x = torch.cat(scale_features, dim=1).transpose(1, 2)
        
        error_feats = self.error_branch(x)
        phoneme_feats = self.phoneme_branch(x)
        
        if hasattr(self, 'cross_attention'):
            error_enhanced, _ = self.cross_attention(
                error_feats, phoneme_feats, phoneme_feats
            )
            phoneme_enhanced, _ = self.cross_attention(
                phoneme_feats, error_feats, error_feats
            )
            
            error_feats = error_feats + error_enhanced
            phoneme_feats = phoneme_feats + phoneme_enhanced
        
        outputs = {}
        
        if self.task in ['error', 'both']:
            error_logits = self.error_head(error_feats)
            outputs['error_logits'] = error_logits
        
        if self.task in ['phoneme', 'both']:
            phoneme_logits = self.phoneme_head(phoneme_feats)
            outputs['phoneme_logits'] = phoneme_logits
        
        return outputs


class MultiTaskBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        feats = self.modules.wav2vec2(wavs, wav_lens)
        predictions = self.modules.model(feats, wav_lens)
        
        return predictions
    
    def compute_objectives(self, predictions, batch, stage):
        total_loss = torch.tensor(0.0, device=self.device)
        
        if 'error_logits' in predictions and hasattr(batch, 'error_tokens'):
            error_log_probs = predictions['error_logits'].log_softmax(dim=-1)
            
            input_lens = batch.sig[1]
            seq_len = error_log_probs.shape[1]
            input_lengths = (input_lens * seq_len).long()
            
            error_loss = sb.nnet.losses.ctc_loss(
                log_probs=error_log_probs,
                targets=batch.error_tokens.data,
                input_lens=input_lengths,
                target_lens=batch.error_tokens.lengths,
                blank_index=0
            )
            total_loss += self.hparams.error_weight * error_loss
            
            if stage != sb.Stage.TRAIN:
                self.error_metrics.append(
                    ids=batch.id,
                    predictions=error_log_probs,
                    targets=batch.error_tokens.data,
                    target_len=batch.error_tokens.lengths,
                    ind2lab=self.hparams.error_decoder
                )
        
        if 'phoneme_logits' in predictions and hasattr(batch, 'phoneme_tokens'):
            phoneme_log_probs = predictions['phoneme_logits'].log_softmax(dim=-1)
            
            input_lens = batch.sig[1]
            seq_len = phoneme_log_probs.shape[1]
            input_lengths = (input_lens * seq_len).long()
            
            phoneme_loss = sb.nnet.losses.ctc_loss(
                log_probs=phoneme_log_probs,
                targets=batch.phoneme_tokens.data,
                input_lens=input_lengths,
                target_lens=batch.phoneme_tokens.lengths,
                blank_index=0
            )
            total_loss += self.hparams.phoneme_weight * phoneme_loss
            
            if stage != sb.Stage.TRAIN:
                self.phoneme_metrics.append(
                    ids=batch.id,
                    predictions=phoneme_log_probs,
                    targets=batch.phoneme_tokens.data,
                    target_len=batch.phoneme_tokens.lengths,
                    ind2lab=self.hparams.phoneme_decoder
                )
        
        return total_loss
    
    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            self.phoneme_metrics = self.hparams.phoneme_stats()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss, "epoch": epoch}
        
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            
        elif stage == sb.Stage.VALID:
            if hasattr(self, 'error_metrics'):
                stage_stats["error_rate"] = self.error_metrics.summarize("error_rate")
                stage_stats["error_accuracy"] = 1.0 - stage_stats["error_rate"]
            
            if hasattr(self, 'phoneme_metrics'):
                stage_stats["PER"] = self.phoneme_metrics.summarize("error_rate")
                stage_stats["phoneme_accuracy"] = 1.0 - stage_stats["PER"]
            
            if hasattr(self.hparams, 'evaluate_every_epoch') and self.hparams.evaluate_every_epoch:
                if hasattr(self, 'test_loader'):
                    self._detailed_evaluation(epoch, stage_stats)
            
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            
            min_keys = ["loss"]
            if "error_rate" in stage_stats:
                min_keys.append("error_rate")
            if "PER" in stage_stats:
                min_keys.append("PER")
            
            self.checkpointer.save_and_keep_only(
                meta=stage_stats,
                min_keys=min_keys,
                max_keys=[]
            )
            
            self.valid_stats = stage_stats
            
        elif stage == sb.Stage.TEST:
            self.test_stats = stage_stats
        
        self._save_epoch_stats(stage_stats, epoch)
    
    def _detailed_evaluation(self, epoch, stage_stats):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} - Detailed Evaluation")
        print(f"{'='*60}")
        
        try:
            import evaluate
            
            if self.hparams.task in ['error', 'both']:
                error_results = evaluate.evaluate_error_detection(
                    self, self.test_loader, self.hparams.error_decoder, self.device
                )
                
                if error_results:
                    current_error_acc = error_results.get('token_accuracy', 0)
                    stage_stats["detailed_error_token_acc"] = current_error_acc
                    stage_stats["detailed_error_seq_acc"] = error_results.get('sequence_accuracy', 0)
                    stage_stats["detailed_error_f1"] = error_results.get('weighted_f1', 0)
                    
                    print(f"Error Detection - Token Acc: {current_error_acc:.4f}")
                    
                    if not hasattr(self, 'best_error_acc'):
                        self.best_error_acc = 0.0
                    if current_error_acc > self.best_error_acc:
                        self.best_error_acc = current_error_acc
                        print(f"NEW BEST Error Accuracy: {current_error_acc:.4f}")
            
            if self.hparams.task in ['phoneme', 'both']:
                phoneme_results = evaluate.evaluate_phoneme_recognition(
                    self, self.test_loader, self.hparams.phoneme_decoder, self.device
                )
                
                if phoneme_results:
                    current_per = phoneme_results.get('per', float('inf'))
                    stage_stats["detailed_PER"] = current_per
                    stage_stats["detailed_phoneme_acc"] = phoneme_results.get('accuracy', 0)
                    
                    print(f"Phoneme Recognition - PER: {current_per:.4f}")
                    
                    if not hasattr(self, 'best_per'):
                        self.best_per = float('inf')
                    if current_per < self.best_per:
                        self.best_per = current_per
                        print(f"NEW BEST PER: {current_per:.4f}")
            
            if hasattr(self.hparams, 'show_samples') and self.hparams.show_samples:
                sample_results = evaluate.show_sample_predictions(
                    self, self.test_loader, 
                    self.hparams.phoneme_decoder, self.hparams.error_decoder,
                    num_samples=getattr(self.hparams, 'num_sample_show', 3),
                    device=self.device
                )
                
                print(f"\n{'-'*40}")
                print("Sample Predictions:")
                print(f"{'-'*40}")
                
                for i, sample in enumerate(sample_results, 1):
                    print(f"\nSample {i}: {sample['id']}")
                    
                    if 'error' in sample['predictions']:
                        error_pred = sample['predictions']['error']
                        print(f"  Error - Pred: {' '.join(error_pred['predicted'])}")
                        print(f"         True: {' '.join(error_pred['target'])}")
                    
                    if 'phoneme' in sample['predictions']:
                        phoneme_pred = sample['predictions']['phoneme']
                        pred_clean = [p for p in phoneme_pred['predicted'] if p != 'sil']
                        true_clean = [p for p in phoneme_pred['target'] if p != 'sil']
                        print(f"  Phoneme - Pred: {' '.join(pred_clean)}")
                        print(f"            True: {' '.join(true_clean)}")
                        
        except Exception as e:
            print(f"Error in detailed evaluation: {e}")
    
    def _save_epoch_stats(self, stage_stats, epoch):
        if hasattr(self.hparams, 'save_folder'):
            import os, json
            epoch_file = os.path.join(self.hparams.save_folder, f'epoch_{epoch}_stats.json')
            os.makedirs(self.hparams.save_folder, exist_ok=True)
            with open(epoch_file, 'w') as f:
                json.dump(stage_stats, f, indent=2, default=str)
    
    def init_optimizers(self):
        wav2vec_params = []
        other_params = []
        
        for name, param in self.modules.named_parameters():
            if 'wav2vec2' in name:
                wav2vec_params.append(param)
            else:
                other_params.append(param)
        
        optimizers = []
        
        if wav2vec_params:
            wav2vec_optimizer = self.hparams.wav2vec_opt_class(wav2vec_params)
            optimizers.append(wav2vec_optimizer)
        
        if other_params:
            other_optimizer = self.hparams.adam_opt_class(other_params)
            optimizers.append(other_optimizer)
        
        if len(optimizers) == 1:
            return optimizers[0]
        else:
            return optimizers
    
    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        if should_step and hasattr(self.hparams, 'grad_clipping'):
            torch.nn.utils.clip_grad_norm_(
                self.modules.parameters(),
                self.hparams.grad_clipping
            )