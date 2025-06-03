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
        
        self.wav2vec2 = SimpleWav2Vec2(
            model_name=hparams["wav2vec2_hub"],
            freeze=hparams["wav2vec2_freeze"]
        )
        
        wav2vec_dim = self.wav2vec2.output_size
        hidden_dim = hparams["hidden_dim"]
        
        self.projection = nn.Linear(wav2vec_dim, hidden_dim)
        self.dropout = nn.Dropout(hparams["dropout"])
        
        if hparams["task"] in ['error', 'both']:
            self.error_head = nn.Linear(hidden_dim, hparams["num_error_types"])
        
        if hparams["task"] in ['phoneme', 'both']:
            self.phoneme_head = nn.Linear(hidden_dim, hparams["num_phonemes"])
    
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
            
            error_loss = sb.nnet.losses.ctc_loss(
                log_probs=error_log_probs,
                targets=batch.error_tokens.data,
                input_lens=batch.sig[1],
                target_lens=batch.error_tokens.lengths,
                blank_index=0
            )
            total_loss += self.hparams["error_weight"] * error_loss
        
        if 'phoneme_logits' in predictions and hasattr(batch, 'phoneme_tokens'):
            phoneme_log_probs = predictions['phoneme_logits'].log_softmax(dim=-1)
            
            phoneme_loss = sb.nnet.losses.ctc_loss(
                log_probs=phoneme_log_probs,
                targets=batch.phoneme_tokens.data,
                input_lens=batch.sig[1],
                target_lens=batch.phoneme_tokens.lengths,
                blank_index=0
            )
            total_loss += self.hparams["phoneme_weight"] * phoneme_loss
        
        return total_loss
    
    def on_stage_start(self, stage, epoch):
        pass
    
    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss, "epoch": epoch}
        
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            
        elif stage == sb.Stage.VALID:
            print(f"Epoch {epoch}, Valid Loss: {stage_loss:.4f}")
            self.valid_stats = stage_stats
            
        elif stage == sb.Stage.TEST:
            self.test_stats = stage_stats