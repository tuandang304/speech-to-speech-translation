import torch
import torchaudio
import librosa
import soundfile as sf


def load_audio(path, target_sr=16000):
    """Tải audio, chuyển mono và resample về target_sr.
    Ưu tiên dùng librosa (tránh TorchCodec). Nếu lỗi, fallback sang torchaudio.
    Trả về tensor 1D float32, sample_rate = target_sr.
    """
    # Try librosa first to avoid TorchCodec dependency
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return torch.from_numpy(y).float()
    except Exception as e:
        # Fallback to torchaudio
        try:
            waveform, sr = torchaudio.load(path)
            # Convert to mono if multi-channel
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample if needed
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
            return waveform.squeeze(0).float()
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio '{path}' with librosa ({e}) and torchaudio ({e2}).")

def save_audio(waveform, path, sample_rate=16000):
    """Lưu audio ra file, tránh TorchCodec bằng soundfile.
    Chấp nhận Tensor (1D hoặc 2D) hoặc numpy array. Ghi ra mono nếu nhiều kênh.
    """
    if isinstance(waveform, torch.Tensor):
        wav = waveform.detach().cpu().float().numpy()
    else:
        wav = waveform
    # Đưa về 1D mono nếu nhiều kênh
    if wav.ndim == 2:
        # Giả định (C, T) hoặc (T, C) -> lấy trung bình để ra mono
        if wav.shape[0] <= wav.shape[1]:
            wav = wav.mean(axis=0)
        else:
            wav = wav.mean(axis=1)
    sf.write(path, wav, sample_rate)
