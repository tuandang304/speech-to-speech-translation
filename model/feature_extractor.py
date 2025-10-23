import torch
from transformers import HubertModel
from .utils_audio import load_audio

class FeatureExtractor:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = HubertModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print(f"Feature Extractor (HuBERT) đã được tải từ '{model_path}' lên {self.device}.")

    @torch.no_grad()
    def extract(self, audio_path):
        waveform = load_audio(audio_path, target_sr=16000).to(self.device)
        input_values = waveform.unsqueeze(0) # Thêm batch dimension
        hidden_states = self.model(input_values).last_hidden_state
        return hidden_states.squeeze(0) # Bỏ batch dimension, trả về (T, D)

def load_model(checkpoint_dir, device='cpu'):
    return FeatureExtractor(model_path=checkpoint_dir, device=device)