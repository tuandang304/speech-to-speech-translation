import torch
import torchaudio
from .utils_audio import save_audio
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, quantizer_model, unit_to_mel_path, device='cpu', sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 160, n_mels: int = 80):
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # 1) Codebook từ quantizer để embed units
        self.quantizer = quantizer_model
        self.codebook = self.quantizer.vq.embedding.weight.data.to(self.device)

        # 2) Mô hình Unit->Mel do mình train
        self.unit_to_mel = UnitToMel(unit_dim=self.codebook.shape[1], mel_bins=self.n_mels).to(self.device)
        self.unit_to_mel.load_state_dict(torch.load(unit_to_mel_path, map_location=device))
        self.unit_to_mel.eval()

        # 3) Cố gắng tải HiFi-GAN qua torch.hub (đúng callable), nếu không sẽ fallback Griffin-Lim
        self.generator = None
        hifigan_sources = [
            # Thử các nguồn HiFi-GAN khác nhau
            ('bshall/hifigan:main', 'hifigan_hubert_discrete'),
            ('bshall/hifigan:main', 'hifigan'),
            # Có thể thêm các nguồn khác nếu cần
        ]
        
        for repo, model_name in hifigan_sources:
            try:
                print(f"Đang thử tải HiFi-GAN từ {repo}::{model_name}...")
                self.generator = torch.hub.load(repo, model_name, trust_repo=True).to(self.device)
                self.generator.eval()
                print(f"✓ HiFi-GAN tải thành công từ {repo}::{model_name}")
                break
            except Exception as e:
                print(f"✗ Không tải được từ {repo}::{model_name}: {e}")
                continue

        # Chuẩn bị fallback Vocoder: InverseMel + Griffin-Lim
        n_stft = self.n_fft // 2 + 1
        # Dùng driver="gelsy" (SVD-based) thay vì "gels" để xử lý ma trận ill-conditioned tốt hơn
        self.inv_mel = torchaudio.transforms.InverseMelScale(n_stft=n_stft, n_mels=self.n_mels, sample_rate=self.sample_rate, driver="gelsy").to(self.device)
        self.griffin = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft, hop_length=self.hop_length, n_iter=32
        ).to(self.device)

        if self.generator is None:
            print("[WARN] Không tải được HiFi-GAN từ torch.hub. Sẽ dùng fallback InverseMel+GriffinLim để synthesize.")
        else:
            print(f"Vocoder (HiFi-GAN) và UnitToMel đã được tải lên {device}.")
        # HiFi-GAN expected mel channels (e.g., 128). If None, fallback expects self.n_mels
        self.expected_mel_ch = self.generator.conv_pre.in_channels if self.generator is not None else self.n_mels

    @torch.no_grad()
    def synthesize(self, unit_sequence):
        """
        unit_sequence: numpy array of unit indices
        Pipeline: units -> embed -> mel (UnitToMel) -> linear spec (InverseMel) -> waveform (Griffin-Lim)
        """
        unit_tensor = torch.as_tensor(unit_sequence, dtype=torch.long, device=self.device)

        # 1) Embed units via codebook
        embedded_units = F.embedding(unit_tensor, self.codebook).unsqueeze(0)  # (1, T, D)

        # 2) Units -> Mel (linear mel magnitude)
        mel_bt = self.unit_to_mel(embedded_units)   # (1, T, mel_bins)
        mel_chw = mel_bt.transpose(1, 2)            # (1, mel_bins, T)

        if self.generator is not None:
            # Dùng HiFi-GAN nếu có
            # Đảm bảo số kênh mel khớp với kỳ vọng của HiFi-GAN (ví dụ 128)
            mel_ch = mel_chw.size(1)
            exp_ch = self.expected_mel_ch
            if mel_ch != exp_ch:
                if mel_ch < exp_ch:
                    pad = torch.zeros((mel_chw.size(0), exp_ch - mel_ch, mel_chw.size(2)), device=mel_chw.device, dtype=mel_chw.dtype)
                    mel_chw = torch.cat([mel_chw, pad], dim=1)
                else:
                    mel_chw = mel_chw[:, :exp_ch, :]
            with torch.no_grad():
                waveform = self.generator(mel_chw).squeeze(0).cpu()
            return waveform
        else:
            # Fallback: InverseMel + Griffin-Lim
            mel_mag = torch.relu(mel_chw.squeeze(0))   # (mel_bins, T)
            
            # Debug: kiểm tra mel magnitude
            print(f"[DEBUG] mel_mag shape: {mel_mag.shape}, min: {mel_mag.min().item():.4f}, max: {mel_mag.max().item():.4f}, mean: {mel_mag.mean().item():.4f}")
            
            # Clamp mel magnitude để tránh giá trị quá nhỏ/lớn
            mel_mag = torch.clamp(mel_mag, min=1e-5, max=100.0)
            
            # Thêm small epsilon để tránh singular matrix
            mel_mag = mel_mag + 1e-6
            
            try:
                linear_mag = self.inv_mel(mel_mag)         # (n_stft, T)
                waveform = self.griffin(linear_mag).cpu()  # (T)
            except torch._C._LinAlgError as e:
                print(f"[ERROR] InverseMelScale failed: {e}")
                print("[FALLBACK] Trả về waveform zero thay vì crash")
                waveform = torch.zeros(mel_mag.shape[1] * self.hop_length)
            
            return waveform

def load_model(quantizer_model, checkpoint_path, device='cpu'):
    # Vocoder cần quantizer để lấy codebook
    return Vocoder(quantizer_model, checkpoint_path, device)
