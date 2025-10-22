import torch.nn as nn
from transformers import HubertModel

class HubertEncoder(nn.Module):
    def __init__(self, model_name="facebook/hubert-base-ls960", feature_layer=6, freeze=True):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_name)
        self.feature_layer = feature_layer
        if freeze:
            for param in self.hubert.parameters():
                param.requires_grad = False
            self.hubert.eval()

    def forward(self, waveform):
        # waveform shape: [batch, sequence_length]
        outputs = self.hubert(waveform, output_hidden_states=True)
        # hidden_states shape: [batch, num_frames, hidden_dim]
        return outputs.hidden_states[self.feature_layer]