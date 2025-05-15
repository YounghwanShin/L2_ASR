import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class NaiveWav2Vec2PhonemeModel(nn.Module):
    def __init__(self, pretrained_model_name="facebook/wav2vec2-base-960h", num_phonemes=42):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_phonemes)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, D]
        logits = self.classifier(hidden_states)     # [B, T, num_phonemes]
        return logits.permute(1, 0, 2)  # CTC: [T, B, C]
