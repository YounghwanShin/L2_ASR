import torch
import torch.nn as nn
import speechbrain as sb
from transformers import Wav2Vec2Model
import torch.nn.functional as F
import os
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.utils.data_utils import undo_padding
import logging

logger = logging.getLogger(__name__)

def make_attn_mask(wavs, wav_lens):
    abs_lens = (wav_lens * wavs.shape[1]).long()
    attn_mask = wavs.new(wavs.shape).zero_().long()
    for i in range(len(abs_lens)):
        attn_mask[i, :abs_lens[i]] = 1
    return attn_mask

class Wav2Vec2Encoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.output_dim = self.wav2vec2.config.hidden_size
    
    def forward(self, x, attention_mask=None):
        outputs = self.wav2vec2(x, attention_mask=attention_mask)
        return outputs.last_hidden_state

class MultiTaskHead(nn.Module):
    def __init__(self, input_dim, num_phonemes, num_errors):
        super().__init__()
        
        self.phoneme_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_phonemes)
        )
        
        self.error_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_errors)
        )
    
    def forward(self, x):
        phoneme_logits = self.phoneme_head(x)
        error_logits = self.error_head(x)
        return phoneme_logits, error_logits

class SimpleMultiTaskBrain(sb.Brain):
    def __init__(self, modules, opt_class, hparams, run_opts=None, checkpointer=None):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer)
        
        self.best_valid_loss = float('inf')
        self.best_phoneme_per = float('inf')
        self.best_error_acc = 0.0
    
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        
        wavs, wav_lens = batch.sig
        
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs, wav_lens)
        
        if hasattr(self.modules.wav2vec2.wav2vec2, 'feature_extractor'):
            if getattr(self.modules.wav2vec2.wav2vec2.feature_extractor, 'return_attention_mask', False):
                attn_mask = make_attn_mask(wavs, wav_lens)
            else:
                attn_mask = None
        else:
            attn_mask = None
        
        wav_features = self.modules.wav2vec2(wavs, attention_mask=attn_mask)
        
        phoneme_logits, error_logits = self.modules.model(wav_features)
        
        if hasattr(self.hparams, "log_softmax"):
            phoneme_log_probs = self.hparams.log_softmax(phoneme_logits)
            error_log_probs = self.hparams.log_softmax(error_logits)
        else:
            phoneme_log_probs = F.log_softmax(phoneme_logits, dim=-1)
            error_log_probs = F.log_softmax(error_logits, dim=-1)
        
        return phoneme_log_probs, error_log_probs, wav_lens
    
    def compute_objectives(self, predictions, batch, stage):
        phoneme_log_probs, error_log_probs, wav_lens = predictions
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        if hasattr(self.hparams, 'task'):
            task = self.hparams.task
            phoneme_weight = self.hparams.phoneme_weight
            error_weight = self.hparams.error_weight
            blank_index = self.hparams.blank_index
        else:
            task = self.hparams.get("task", "both")
            phoneme_weight = self.hparams.get("phoneme_weight", 1.0)
            error_weight = self.hparams.get("error_weight", 1.0)
            blank_index = self.hparams.get("blank_index", 0)
        
        if task in ["phoneme", "both"] and hasattr(batch, 'phoneme_tokens'):
            phoneme_targets, phoneme_target_lens = batch.phoneme_tokens
            
            phoneme_loss = self.hparams.ctc_cost(
                phoneme_log_probs, phoneme_targets, wav_lens, phoneme_target_lens
            )
            total_loss += phoneme_weight * phoneme_loss
        
        if task in ["error", "both"] and hasattr(batch, 'error_tokens'):
            error_targets, error_target_lens = batch.error_tokens
            
            error_loss = self.hparams.ctc_cost(
                error_log_probs, error_targets, wav_lens, error_target_lens
            )
            total_loss += error_weight * error_loss
        
        if stage == sb.Stage.TEST:
            if task in ["phoneme", "both"] and hasattr(batch, 'phoneme_tokens'):
                phoneme_sequence = sb.decoders.ctc_greedy_decode(
                    phoneme_log_probs, wav_lens, blank_id=blank_index
                )
                
                phoneme_targets, phoneme_target_lens = batch.phoneme_tokens
                
                self.phoneme_metrics.append(
                    ids=batch.id,
                    predict=phoneme_sequence,
                    target=phoneme_targets,
                    predict_len=None,
                    target_len=phoneme_target_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
        
        return total_loss
    
    def on_stage_start(self, stage, epoch):
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules.wav2vec2, 'wav2vec2'):
                if hasattr(self.modules.wav2vec2.wav2vec2, 'config'):
                    if hasattr(self.modules.wav2vec2.wav2vec2.config, 'apply_spec_augment'):
                        self.modules.wav2vec2.wav2vec2.config.apply_spec_augment = True
        
        if stage != sb.Stage.TRAIN:
            if hasattr(self.modules.wav2vec2, 'wav2vec2'):
                if hasattr(self.modules.wav2vec2.wav2vec2, 'config'):
                    if hasattr(self.modules.wav2vec2.wav2vec2.config, 'apply_spec_augment'):
                        self.modules.wav2vec2.wav2vec2.config.apply_spec_augment = False
            
            if hasattr(self.hparams, "per_stats"):
                self.phoneme_metrics = self.hparams.per_stats()
            else:
                per_stats = self.hparams.get("per_stats", None) if hasattr(self.hparams, 'get') else None
                if per_stats:
                    self.phoneme_metrics = per_stats()
                else:
                    from speechbrain.utils.metric_stats import ErrorRateStats
                    self.phoneme_metrics = ErrorRateStats()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            print(f"Epoch {epoch}, Train Loss: {stage_loss:.4f}")
        
        elif stage == sb.Stage.VALID:
            print(f"Epoch {epoch}, Valid Loss: {stage_loss:.4f}")
            
            if stage_loss < self.best_valid_loss:
                self.best_valid_loss = stage_loss
                print(f"  NEW BEST Valid Loss: {stage_loss:.4f}")
                self.save_checkpoint("best_loss")
                
                if hasattr(self, 'test_data') and self.test_data is not None:
                    self.run_test_evaluation(self.test_data, epoch)
        
        elif stage == sb.Stage.TEST:
            per = self.phoneme_metrics.summarize("error_rate")
            
            print("=== FINAL TEST RESULTS ===")
            print(f"Test Loss: {stage_loss:.4f}")
            print(f"Phoneme PER: {per:.4f}")
    
    def evaluate_batch(self, batch, stage):
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage)
            loss = self.compute_objectives(predictions, batch, stage)
            
            return loss
    
    def init_optimizers(self):
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.modules.model.parameters()
        )
        
        self.optimizer = self.adam_optimizer
        
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("wav2vec_opt", self.wav2vec_optimizer)
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
    
    def zero_grad(self, set_to_none=False):
        if hasattr(self, 'wav2vec_optimizer') and self.wav2vec_optimizer is not None:
            self.wav2vec_optimizer.zero_grad(set_to_none=set_to_none)
        if hasattr(self, 'adam_optimizer') and self.adam_optimizer is not None:
            self.adam_optimizer.zero_grad(set_to_none=set_to_none)
    
    def on_fit_start(self):
        super().on_fit_start()
        
        if not hasattr(self, 'wav2vec_optimizer') or self.wav2vec_optimizer is None:
            self.init_optimizers()
    
    def fit_batch(self, batch):
        if hasattr(self.hparams, 'gradient_accumulation'):
            gradient_accumulation = self.hparams.gradient_accumulation
        else:
            gradient_accumulation = self.hparams.get("gradient_accumulation", 1)
        
        if self.auto_mix_prec:
            self.wav2vec_optimizer.zero_grad()
            self.adam_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.wav2vec_optimizer)
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients():
                self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.adam_optimizer)

            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            
            if gradient_accumulation > 1:
                (loss / gradient_accumulation).backward()
                
                if self.step % gradient_accumulation == 0:
                    if self.check_gradients():
                        self.wav2vec_optimizer.step()
                        self.adam_optimizer.step()
                    
                    self.wav2vec_optimizer.zero_grad()
                    self.adam_optimizer.zero_grad()
            else:
                loss.backward()
                
                if self.check_gradients():
                    self.wav2vec_optimizer.step()
                    self.adam_optimizer.step()
                
                self.wav2vec_optimizer.zero_grad()
                self.adam_optimizer.zero_grad()

        return loss.detach().cpu()
    
    def save_checkpoint(self, name):
        if hasattr(self.hparams, 'output_folder'):
            output_folder = self.hparams.output_folder
        else:
            output_folder = self.hparams.get("output_folder", "./results")
        
        save_dir = os.path.join(output_folder, "save")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{name}.ckpt")
        
        checkpoint = {
            'model_state_dict': self.modules.state_dict(),
            'best_valid_loss': self.best_valid_loss,
            'best_phoneme_per': self.best_phoneme_per,
            'best_error_acc': self.best_error_acc,
        }
        
        torch.save(checkpoint, save_path)
        print(f"  Saved: {save_path}")
    
    def run_test_evaluation(self, test_data, epoch):
        
        self.on_stage_start(sb.Stage.TEST, epoch)
        self.modules.eval()
        
        with torch.no_grad():
            if hasattr(self.hparams, 'test_dataloader_opts'):
                test_dataloader_opts = self.hparams.test_dataloader_opts
            else:
                batch_size = self.hparams.get("batch_size", 4) if hasattr(self.hparams, 'get') else 4
                test_dataloader_opts = {
                    "batch_size": batch_size,
                    "shuffle": False,
                    "num_workers": 1
                }
            
            test_loader = self.make_dataloader(
                test_data, sb.Stage.TEST, **test_dataloader_opts
            )
            
            total_loss = 0.0
            num_batches = 0
            
            for batch in test_loader:
                loss = self.evaluate_batch(batch, sb.Stage.TEST)
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
        
        self.on_interim_test_end(avg_loss, epoch)
        self.modules.train()
    
    def on_interim_test_end(self, stage_loss, epoch):
        per = self.phoneme_metrics.summarize("error_rate")
        
        print(f"  Test Loss: {stage_loss:.4f}")
        print(f"  Phoneme PER: {per:.4f}")
        
        if per < self.best_phoneme_per:
            self.best_phoneme_per = per
            print(f"  NEW BEST Phoneme PER: {per:.4f}")
            self.save_checkpoint("best_phoneme_per")
        
        self.save_checkpoint(f"epoch_{epoch}")