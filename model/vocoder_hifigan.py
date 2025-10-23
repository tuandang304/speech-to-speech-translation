import torch
import torchaudio
from .utils_audio import save_audio
import torch.nn as nn

class UnitToMel(nn.Module):
    """
    Một model nhỏ để chuyển đổi từ discrete units (đã được map về embedding)
    sang Mel-spectrogram mà HiFi-GAN có thể hiểu.
    """
    def __init__(self, unit_dim=256, mel_bins=80, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(unit_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mel_bins)
        )
    def forward(self, x):
        return self.net(x)

class Vocoder:
    def __init__(self, quantizer_model, unit_to_mel_path, device='cpu'):
        self.device = torch.device(device)
        
        # 1. Tải HiFi-GAN generator từ TorchHub
        self.generator = torch.hub.load('bshall/hifigan:main', 'hifigan_hubert_base').to(self.device)
        self.generator.eval()
        
        # 2. Tải quantizer để lấy lại codebook
        self.quantizer = quantizer_model
        self.codebook = self.quantizer.vq.embedding.weight.data.to(self.device)
        
        # 3. Tải model chuyển unit sang mel
        self.unit_to_mel = UnitToMel(unit_dim=self.codebook.shape[1]).to(self.device)
        self.unit_to_mel.load_state_dict(torch.load(unit_to_mel_path, map_location=device))
        self.unit_to_mel.eval()

        print(f"Vocoder (HiFi-GAN) và UnitToMel đã được tải lên {device}.")

    @torch.no_grad()
    def synthesize(self, unit_sequence):
        # unit_sequence: numpy array of indices
        unit_tensor = torch.LongTensor(unit_sequence).to(self.device)
        
        # 1. Map indices to codebook vectors
        embedded_units = F.embedding(unit_tensor, self.codebook).unsqueeze(0) # (1, T, D)
        
        # 2. Convert embedded units to mel-spectrogram
        mel_spec = self.unit_to_mel(embedded_units) # (1, T, mel_bins)
        
        # 3. Use HiFi-GAN to generate waveform
        # HiFi-GAN cần (B, mel_bins, T)
        mel_spec = mel_spec.transpose(1, 2)
        waveform = self.generator(mel_spec).squeeze(0).cpu() # (1, T*hop_length) -> (T*hop_length)
        
        return waveform

def load_model(quantizer_model, checkpoint_path, device='cpu'):
    # Vocoder cần quantizer để lấy codebook
    return Vocoder(quantizer_model, checkpoint_path, device)