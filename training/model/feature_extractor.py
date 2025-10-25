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
    def extract(self, audio_path, target_sr: int = 16000, use_amp: bool = True):
        """
        Extract hidden states in a single pass (no chunking).
        Pads very short inputs minimally to satisfy conv kernel.
        Returns Tensor (T, D)
        """
        waveform = load_audio(audio_path, target_sr=target_sr).to(self.device)  # 1D tensor (samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, time)

        # Minimal padding for extremely short clips to avoid conv kernel error
        MIN_SAMPLES = 400
        if waveform.size(-1) < MIN_SAMPLES:
            pad = MIN_SAMPLES - waveform.size(-1)
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        if self.device.type == 'cuda' and use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                hs = self.model(waveform).last_hidden_state  # (1, T, D)
        else:
            hs = self.model(waveform).last_hidden_state

        return hs.squeeze(0)  # (T, D)

def load_model(checkpoint_dir, device='cpu'):
    return FeatureExtractor(model_path=checkpoint_dir, device=device)


if __name__ == "__main__":
    test_extractor = load_model('checkpoints/feature_extractor', device='cpu')
    features = test_extractor.extract('data/test_vie/1.wav')

    print("Extracted features shape:", features.shape)