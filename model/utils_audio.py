import torch
import torchaudio
import librosa

def load_audio(path, target_sr=16000):
    """Tải và resample audio về target_sr."""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0) # Trả về tensor 1D

def save_audio(waveform, path, sample_rate=16000):
    """Lưu audio ra file."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform, sample_rate)