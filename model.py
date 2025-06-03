import torch
import torch.nn as nn
import speechbrain as sb
from transformers import Wav2Vec2Model
import torch.nn.functional as F
import os

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
        
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
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
        
        predictions = {}
        if self.hparams.task in ["phoneme", "both"]:
            predictions["phoneme_logits"] = phoneme_logits
            predictions["phoneme_log_probs"] = F.log_softmax(phoneme_logits, dim=-1)
        if self.hparams.task in ["error", "both"]:
            predictions["error_logits"] = error_logits
            predictions["error_log_probs"] = F.log_softmax(error_logits, dim=-1)
        
        predictions["wav_lens"] = wav_lens
        
        return predictions
    
    def compute_objectives(self, predictions, batch, stage):
        total_loss = torch.tensor(0.0, device=self.device)
        
        if "phoneme_log_probs" in predictions and hasattr(batch, 'phoneme_tokens'):
            phoneme_log_probs = predictions["phoneme_log_probs"].transpose(0, 1)
            
            batch_size = phoneme_log_probs.shape[1]
            wav_lens = predictions["wav_lens"]
            input_lengths = (wav_lens * phoneme_log_probs.shape[0]).long()
            
            all_targets = []
            target_lengths = []
            
            for tokens in batch.phoneme_tokens:
                if isinstance(tokens, torch.Tensor):
                    token_list = tokens.cpu().tolist()
                else:
                    token_list = tokens
                all_targets.extend(token_list)
                target_lengths.append(len(token_list))
            
            targets = torch.tensor(all_targets, dtype=torch.long, device=self.device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=self.device)
            
            phoneme_loss = F.ctc_loss(
                log_probs=phoneme_log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                blank=0,
                reduction='mean',
                zero_infinity=True
            )
            
            total_loss += self.hparams.phoneme_weight * phoneme_loss
        
        if "error_log_probs" in predictions and hasattr(batch, 'error_tokens'):
            error_log_probs = predictions["error_log_probs"].transpose(0, 1)
            
            batch_size = error_log_probs.shape[1]
            wav_lens = predictions["wav_lens"]
            input_lengths = (wav_lens * error_log_probs.shape[0]).long()
            
            all_targets = []
            target_lengths = []
            
            for tokens in batch.error_tokens:
                if isinstance(tokens, torch.Tensor):
                    token_list = tokens.cpu().tolist()
                else:
                    token_list = tokens
                all_targets.extend(token_list)
                target_lengths.append(len(token_list))
            
            targets = torch.tensor(all_targets, dtype=torch.long, device=self.device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=self.device)
            
            error_loss = F.ctc_loss(
                log_probs=error_log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                blank=0,
                reduction='mean',
                zero_infinity=True
            )
            
            total_loss += self.hparams.error_weight * error_loss
        
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
            
            self.phoneme_predictions = []
            self.phoneme_targets = []
            self.error_predictions = []
            self.error_targets = []
    
    def on_stage_end(self, stage, stage_loss, epoch):
        
        if stage == sb.Stage.TRAIN:
            print(f"Epoch {epoch}, Train Loss: {stage_loss:.4f}")
        
        elif stage == sb.Stage.VALID:
            metrics = self.calculate_metrics()
            
            print(f"Epoch {epoch}, Valid Loss: {stage_loss:.4f}")
            if 'phoneme_per' in metrics:
                print(f"  Phoneme PER: {metrics['phoneme_per']:.4f}")
            if 'error_accuracy' in metrics:
                print(f"  Error Accuracy: {metrics['error_accuracy']:.4f}")
            
            self.save_best_models(stage_loss, metrics, epoch)
        
        elif stage == sb.Stage.TEST:
            metrics = self.calculate_metrics()
            print("=== FINAL TEST RESULTS ===")
            print(f"Test Loss: {stage_loss:.4f}")
            if 'phoneme_per' in metrics:
                print(f"Phoneme PER: {metrics['phoneme_per']:.4f}")
            if 'error_accuracy' in metrics:
                print(f"Error Accuracy: {metrics['error_accuracy']:.4f}")
    
    def evaluate_batch(self, batch, stage):
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage)
            loss = self.compute_objectives(predictions, batch, stage)
            
            if stage != sb.Stage.TRAIN:
                self.collect_predictions(predictions, batch)
            
            return loss
    
    def collect_predictions(self, predictions, batch):
        
        if "phoneme_log_probs" in predictions and hasattr(batch, 'phoneme_tokens'):
            phoneme_log_probs = predictions["phoneme_log_probs"]
            wav_lens = predictions["wav_lens"]
            
            batch_predictions = sb.decoders.ctc_greedy_decode(
                phoneme_log_probs, wav_lens, blank_id=0
            )
            
            batch_targets = []
            for tokens in batch.phoneme_tokens:
                if isinstance(tokens, torch.Tensor):
                    token_list = tokens.cpu().tolist()
                else:
                    token_list = tokens
                batch_targets.append(token_list)
            
            self.phoneme_predictions.extend(batch_predictions)
            self.phoneme_targets.extend(batch_targets)
        
        if "error_log_probs" in predictions and hasattr(batch, 'error_tokens'):
            error_log_probs = predictions["error_log_probs"]
            wav_lens = predictions["wav_lens"]
            
            batch_predictions = sb.decoders.ctc_greedy_decode(
                error_log_probs, wav_lens, blank_id=0
            )
            
            batch_targets = []
            for tokens in batch.error_tokens:
                if isinstance(tokens, torch.Tensor):
                    token_list = tokens.cpu().tolist()
                else:
                    token_list = tokens
                batch_targets.append(token_list)
            
            self.error_predictions.extend(batch_predictions)
            self.error_targets.extend(batch_targets)
    
    def calculate_metrics(self):
        metrics = {}
        
        if self.phoneme_predictions and self.phoneme_targets:
            total_edits = 0
            total_length = 0
            
            for pred, target in zip(self.phoneme_predictions, self.phoneme_targets):
                edits = self.edit_distance(pred, target)
                total_edits += edits
                total_length += len(target)
            
            per = total_edits / max(total_length, 1)
            metrics['phoneme_per'] = per
        
        if self.error_predictions and self.error_targets:
            correct = 0
            total = 0
            
            for pred, target in zip(self.error_predictions, self.error_targets):
                min_len = min(len(pred), len(target))
                correct += sum(1 for i in range(min_len) if pred[i] == target[i])
                total += len(target)
            
            accuracy = correct / max(total, 1)
            metrics['error_accuracy'] = accuracy
        
        return metrics
    
    def edit_distance(self, pred, target):
        m, n = len(pred), len(target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def fit_batch(self, batch):
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
            
            if hasattr(self.hparams, 'gradient_accumulation'):
                (loss / self.hparams.gradient_accumulation).backward()
                
                if self.step % self.hparams.gradient_accumulation == 0:
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
    
    def save_best_models(self, valid_loss, metrics, epoch):
        
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            print(f"  NEW BEST Valid Loss: {valid_loss:.4f}")
            self.save_checkpoint("best_loss")
        
        if 'phoneme_per' in metrics and metrics['phoneme_per'] < self.best_phoneme_per:
            self.best_phoneme_per = metrics['phoneme_per']
            print(f"  NEW BEST Phoneme PER: {self.best_phoneme_per:.4f}")
            self.save_checkpoint("best_phoneme_per")
        
        if 'error_accuracy' in metrics and metrics['error_accuracy'] > self.best_error_acc:
            self.best_error_acc = metrics['error_accuracy']
            print(f"  NEW BEST Error Accuracy: {self.best_error_acc:.4f}")
            self.save_checkpoint("best_error_acc")
        
        self.save_checkpoint(f"epoch_{epoch}")
    
    def save_checkpoint(self, name):
        save_dir = os.path.join(self.hparams.output_folder, "save")
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