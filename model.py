import torch
import torch.nn as nn
import speechbrain as sb
from transformers import Wav2Vec2Model
import torch.nn.functional as F
import os

class Wav2Vec2Encoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.output_dim = self.wav2vec2.config.hidden_size
    
    def forward(self, x):
        # x shape: (batch_size, time_steps)
        outputs = self.wav2vec2(x)
        return outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

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
        
        # Initialize tracking variables
        self.best_valid_loss = float('inf')
        self.best_phoneme_per = float('inf')
        self.best_error_acc = 0.0
    
    def compute_forward(self, batch, stage):
        """Forward pass of the model"""
        batch = batch.to(self.device)
        
        # Extract features using Wav2Vec2
        wav_features = self.modules.wav2vec2(batch.sig[0])
        
        # Multi-task heads
        phoneme_logits, error_logits = self.modules.model(wav_features)
        
        predictions = {}
        if self.hparams.task in ["phoneme", "both"]:
            predictions["phoneme_logits"] = phoneme_logits
        if self.hparams.task in ["error", "both"]:
            predictions["error_logits"] = error_logits
        
        return predictions
    
    def compute_objectives(self, predictions, batch, stage):
        """Compute loss objectives"""
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Phoneme recognition loss (CTC)
        if "phoneme_logits" in predictions and hasattr(batch, 'phoneme_tokens'):
            phoneme_log_probs = F.log_softmax(predictions["phoneme_logits"], dim=-1)
            
            # Get input lengths
            input_lengths = (batch.sig[1] * phoneme_log_probs.shape[1]).long()
            
            # Get target sequences and lengths
            targets = []
            target_lengths = []
            
            for tokens in batch.phoneme_tokens:
                if isinstance(tokens, torch.Tensor):
                    token_list = tokens.cpu().tolist()
                else:
                    token_list = tokens
                targets.extend(token_list)
                target_lengths.append(len(token_list))
            
            targets = torch.tensor(targets, dtype=torch.long, device=self.device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=self.device)
            
            # CTC loss
            phoneme_loss = F.ctc_loss(
                log_probs=phoneme_log_probs.transpose(0, 1),
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                blank=0,
                reduction='mean'
            )
            
            total_loss += self.hparams.phoneme_weight * phoneme_loss
        
        # Error detection loss (CTC)
        if "error_logits" in predictions and hasattr(batch, 'error_tokens'):
            error_log_probs = F.log_softmax(predictions["error_logits"], dim=-1)
            
            # Get input lengths
            input_lengths = (batch.sig[1] * error_log_probs.shape[1]).long()
            
            # Get target sequences and lengths
            targets = []
            target_lengths = []
            
            for tokens in batch.error_tokens:
                if isinstance(tokens, torch.Tensor):
                    token_list = tokens.cpu().tolist()
                else:
                    token_list = tokens
                targets.extend(token_list)
                target_lengths.append(len(token_list))
            
            targets = torch.tensor(targets, dtype=torch.long, device=self.device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=self.device)
            
            # CTC loss
            error_loss = F.ctc_loss(
                log_probs=error_log_probs.transpose(0, 1),
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                blank=0,
                reduction='mean'
            )
            
            total_loss += self.hparams.error_weight * error_loss
        
        return total_loss
    
    def on_stage_start(self, stage, epoch):
        """Called at the beginning of each stage"""
        if stage != sb.Stage.TRAIN:
            # Initialize metrics for validation/test
            self.phoneme_predictions = []
            self.phoneme_targets = []
            self.error_predictions = []
            self.error_targets = []
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Called at the end of each stage"""
        
        if stage == sb.Stage.TRAIN:
            print(f"Epoch {epoch}, Train Loss: {stage_loss:.4f}")
        
        elif stage == sb.Stage.VALID:
            # Calculate metrics
            metrics = self.calculate_metrics()
            
            print(f"Epoch {epoch}, Valid Loss: {stage_loss:.4f}")
            if 'phoneme_per' in metrics:
                print(f"  Phoneme PER: {metrics['phoneme_per']:.4f}")
            if 'error_accuracy' in metrics:
                print(f"  Error Accuracy: {metrics['error_accuracy']:.4f}")
            
            # Save best models
            self.save_best_models(stage_loss, metrics, epoch)
        
        elif stage == sb.Stage.TEST:
            # Calculate and print final test metrics
            metrics = self.calculate_metrics()
            print("=== FINAL TEST RESULTS ===")
            print(f"Test Loss: {stage_loss:.4f}")
            if 'phoneme_per' in metrics:
                print(f"Phoneme PER: {metrics['phoneme_per']:.4f}")
            if 'error_accuracy' in metrics:
                print(f"Error Accuracy: {metrics['error_accuracy']:.4f}")
    
    def evaluate_batch(self, batch, stage):
        """Evaluate a single batch"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage)
            loss = self.compute_objectives(predictions, batch, stage)
            
            # Collect predictions for metrics calculation
            if stage != sb.Stage.TRAIN:
                self.collect_predictions(predictions, batch)
            
            return loss
    
    def collect_predictions(self, predictions, batch):
        """Collect predictions and targets for metric calculation"""
        
        # Phoneme predictions
        if "phoneme_logits" in predictions and hasattr(batch, 'phoneme_tokens'):
            phoneme_log_probs = F.log_softmax(predictions["phoneme_logits"], dim=-1)
            input_lengths = (batch.sig[1] * phoneme_log_probs.shape[1]).long()
            
            # Greedy CTC decoding
            batch_predictions = self.greedy_ctc_decode(phoneme_log_probs, input_lengths)
            
            # Get targets
            batch_targets = []
            for tokens in batch.phoneme_tokens:
                if isinstance(tokens, torch.Tensor):
                    token_list = tokens.cpu().tolist()
                else:
                    token_list = tokens
                batch_targets.append(token_list)
            
            self.phoneme_predictions.extend(batch_predictions)
            self.phoneme_targets.extend(batch_targets)
        
        # Error predictions
        if "error_logits" in predictions and hasattr(batch, 'error_tokens'):
            error_log_probs = F.log_softmax(predictions["error_logits"], dim=-1)
            input_lengths = (batch.sig[1] * error_log_probs.shape[1]).long()
            
            # Greedy CTC decoding
            batch_predictions = self.greedy_ctc_decode(error_log_probs, input_lengths)
            
            # Get targets
            batch_targets = []
            for tokens in batch.error_tokens:
                if isinstance(tokens, torch.Tensor):
                    token_list = tokens.cpu().tolist()
                else:
                    token_list = tokens
                batch_targets.append(token_list)
            
            self.error_predictions.extend(batch_predictions)
            self.error_targets.extend(batch_targets)
    
    def greedy_ctc_decode(self, log_probs, input_lengths):
        """Simple greedy CTC decoding"""
        batch_predictions = []
        
        for i, length in enumerate(input_lengths):
            # Get predictions for this sequence
            seq_log_probs = log_probs[i, :length]
            pred_indices = torch.argmax(seq_log_probs, dim=-1)
            
            # Remove blanks and duplicates
            decoded = []
            prev_idx = -1
            for idx in pred_indices:
                idx = idx.item()
                if idx != 0 and idx != prev_idx:  # 0 is blank
                    decoded.append(idx)
                prev_idx = idx
            
            batch_predictions.append(decoded)
        
        return batch_predictions
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        metrics = {}
        
        # Phoneme Error Rate (PER)
        if self.phoneme_predictions and self.phoneme_targets:
            total_edits = 0
            total_length = 0
            
            for pred, target in zip(self.phoneme_predictions, self.phoneme_targets):
                edits = self.edit_distance(pred, target)
                total_edits += edits
                total_length += len(target)
            
            per = total_edits / max(total_length, 1)
            metrics['phoneme_per'] = per
        
        # Error Detection Accuracy
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
        """Calculate edit distance between two sequences"""
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
    
    def init_optimizers(self):
        """Initialize optimizers for the model"""
        # Separate learning rates for wav2vec2 and model
        wav2vec_params = list(self.modules.wav2vec2.parameters())
        model_params = list(self.modules.model.parameters())
        
        # Create separate optimizers
        optimizer1 = torch.optim.AdamW(
            wav2vec_params,
            lr=getattr(self.hparams, "lr_wav2vec", 1e-5),
            weight_decay=getattr(self.hparams, "weight_decay", 1e-4)
        )
        
        optimizer2 = torch.optim.AdamW(
            model_params,
            lr=getattr(self.hparams, "lr", 1e-4),
            weight_decay=getattr(self.hparams, "weight_decay", 1e-4)
        )
        
        return [optimizer1, optimizer2]
    
    def save_best_models(self, valid_loss, metrics, epoch):
        """Save best models based on different criteria"""
        
        # Save best validation loss
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            print(f"  NEW BEST Valid Loss: {valid_loss:.4f}")
            self.save_checkpoint("best_loss")
        
        # Save best phoneme PER
        if 'phoneme_per' in metrics and metrics['phoneme_per'] < self.best_phoneme_per:
            self.best_phoneme_per = metrics['phoneme_per']
            print(f"  NEW BEST Phoneme PER: {self.best_phoneme_per:.4f}")
            self.save_checkpoint("best_phoneme_per")
        
        # Save best error accuracy
        if 'error_accuracy' in metrics and metrics['error_accuracy'] > self.best_error_acc:
            self.best_error_acc = metrics['error_accuracy']
            print(f"  NEW BEST Error Accuracy: {self.best_error_acc:.4f}")
            self.save_checkpoint("best_error_acc")
        
        # Save current epoch
        self.save_checkpoint(f"epoch_{epoch}")
    
    def save_checkpoint(self, name):
        """Save model checkpoint"""
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