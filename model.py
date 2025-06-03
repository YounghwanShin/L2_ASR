import torch
import torch.nn as nn
import speechbrain as sb
from transformers import Wav2Vec2Model
import torch.nn.functional as F


class SimpleWav2Vec2(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53", freeze=False):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        if freeze:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        self.output_size = self.wav2vec2.config.hidden_size
    
    def forward(self, wavs, wav_lens=None):
        if wavs.dim() == 3:
            wavs = wavs.squeeze(1)
        
        attention_mask = None
        if wav_lens is not None:
            max_len = wavs.shape[1]
            batch_size = wavs.shape[0]
            attention_mask = torch.arange(max_len).expand(batch_size, max_len).to(wavs.device)
            attention_mask = (attention_mask < (wav_lens * max_len).unsqueeze(1)).float()
        
        outputs = self.wav2vec2(wavs, attention_mask=attention_mask)
        return outputs.last_hidden_state


class SimpleMultiTaskModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        
        # Handle both dict and SimpleNamespace
        wav2vec2_hub = hparams.get("wav2vec2_hub") if hasattr(hparams, 'get') else hparams.wav2vec2_hub
        wav2vec2_freeze = hparams.get("wav2vec2_freeze") if hasattr(hparams, 'get') else hparams.wav2vec2_freeze
        hidden_dim = hparams.get("hidden_dim") if hasattr(hparams, 'get') else hparams.hidden_dim
        dropout = hparams.get("dropout") if hasattr(hparams, 'get') else hparams.dropout
        task = hparams.get("task") if hasattr(hparams, 'get') else hparams.task
        num_error_types = hparams.get("num_error_types") if hasattr(hparams, 'get') else hparams.num_error_types
        num_phonemes = hparams.get("num_phonemes") if hasattr(hparams, 'get') else hparams.num_phonemes
        
        self.wav2vec2 = SimpleWav2Vec2(
            model_name=wav2vec2_hub,
            freeze=wav2vec2_freeze
        )
        
        wav2vec_dim = self.wav2vec2.output_size
        
        self.projection = nn.Linear(wav2vec_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        if task in ['error', 'both']:
            self.error_head = nn.Linear(hidden_dim, num_error_types)
        
        if task in ['phoneme', 'both']:
            self.phoneme_head = nn.Linear(hidden_dim, num_phonemes)
    
    def forward(self, features, length=None):
        x = self.projection(features)
        x = F.relu(x)
        x = self.dropout(x)
        
        outputs = {}
        
        if hasattr(self, 'error_head'):
            outputs['error_logits'] = self.error_head(x)
        
        if hasattr(self, 'phoneme_head'):
            outputs['phoneme_logits'] = self.phoneme_head(x)
        
        return outputs


class SimpleMultiTaskBrain(sb.Brain):
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
            
            if hasattr(batch.error_tokens, 'data'):
                targets = batch.error_tokens.data
                target_lengths = batch.error_tokens.lengths
            else:
                targets = []
                target_lengths = []
                for tokens in batch.error_tokens:
                    if isinstance(tokens, list):
                        targets.append(torch.tensor(tokens, dtype=torch.long))
                        target_lengths.append(len(tokens))
                    else:
                        targets.append(tokens)
                        target_lengths.append(len(tokens))
                
                max_len = max(target_lengths)
                padded_targets = torch.zeros(len(targets), max_len, dtype=torch.long, device=self.device)
                for i, target in enumerate(targets):
                    padded_targets[i, :len(target)] = target.to(self.device)
                
                targets = padded_targets
                target_lengths = torch.tensor(target_lengths, dtype=torch.float, device=self.device) / max_len
            
            error_loss = sb.nnet.losses.ctc_loss(
                log_probs=error_log_probs,
                targets=targets,
                input_lens=batch.sig[1],
                target_lens=target_lengths,
                blank_index=0
            )
            total_loss += self.hparams.error_weight * error_loss
        
        if 'phoneme_logits' in predictions and hasattr(batch, 'phoneme_tokens'):
            phoneme_log_probs = predictions['phoneme_logits'].log_softmax(dim=-1)
            
            if hasattr(batch.phoneme_tokens, 'data'):
                targets = batch.phoneme_tokens.data
                target_lengths = batch.phoneme_tokens.lengths
            else:
                targets = []
                target_lengths = []
                for tokens in batch.phoneme_tokens:
                    if isinstance(tokens, list):
                        targets.append(torch.tensor(tokens, dtype=torch.long))
                        target_lengths.append(len(tokens))
                    else:
                        targets.append(tokens)
                        target_lengths.append(len(tokens))
                
                max_len = max(target_lengths)
                padded_targets = torch.zeros(len(targets), max_len, dtype=torch.long, device=self.device)
                for i, target in enumerate(targets):
                    padded_targets[i, :len(target)] = target.to(self.device)
                
                targets = padded_targets
                target_lengths = torch.tensor(target_lengths, dtype=torch.float, device=self.device) / max_len
            
            phoneme_loss = sb.nnet.losses.ctc_loss(
                log_probs=phoneme_log_probs,
                targets=targets,
                input_lens=batch.sig[1],
                target_lens=target_lengths,
                blank_index=0
            )
            total_loss += self.hparams.phoneme_weight * phoneme_loss
        
        return total_loss
    
    def on_stage_start(self, stage, epoch):
        pass
    
    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss, "epoch": epoch}
        
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            print(f"Epoch {epoch}, Train Loss: {stage_loss:.4f}")
            
        elif stage == sb.Stage.VALID:
            print(f"Epoch {epoch}, Valid Loss: {stage_loss:.4f}")
            self.valid_stats = stage_stats
            
        elif stage == sb.Stage.TEST:
            self.test_stats = stage_stats
    
    def init_optimizers(self):
        wav2vec_params = list(self.modules.wav2vec2.parameters())
        model_params = list(self.modules.model.parameters())
        
        wav2vec_optimizer = torch.optim.AdamW(
            wav2vec_params, 
            lr=self.hparams.lr_wav2vec, 
            weight_decay=self.hparams.weight_decay
        )
        
        model_optimizer = torch.optim.AdamW(
            model_params, 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        
        return [wav2vec_optimizer, model_optimizer]