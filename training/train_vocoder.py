import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
import numpy as np
from tqdm import tqdm

from model.vocoder_hifigan import UnitToMel
from model.quantizer_vqvae import load_model as load_quantizer
from model.utils_audio import load_audio

def compute_mel_spec(waveform, sr=16000, n_mels=80, n_fft=400, hop_length=160):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    ).to(waveform.device)
    return mel_transform(waveform)

def main():
    # --- Configs ---
    unit_dir = 'cache/units/vie'
    audio_dir = 'data/vie'
    quantizer_checkpoint = 'checkpoints/quantizer.pt'
    checkpoint_path = 'checkpoints/vocoder.pt'
    epochs = 20
    lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(unit_dir):
        print("Vui lòng chạy 'python training/preprocess_data.py' trước.")
        return

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint đã tồn tại tại '{checkpoint_path}'. Bỏ qua.")
        return

    # --- Models ---
    quantizer = load_quantizer(quantizer_checkpoint, device=device)
    codebook = quantizer.vq.embedding.weight.data
    
    model = UnitToMel(unit_dim=codebook.shape[1], mel_bins=80).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss() # L1 loss is common for mels

    # --- Data ---
    unit_files = sorted([f for f in os.listdir(unit_dir) if f.endswith('.npy')])
    
    print("Bắt đầu huấn luyện Vocoder (Unit-to-Mel converter)...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(unit_files, desc=f"Epoch {epoch+1}/{epochs}")
        for unit_file in pbar:
            # Tải units
            units = torch.from_numpy(np.load(os.path.join(unit_dir, unit_file))).long().to(device)
            embedded_units = F.embedding(units, codebook).unsqueeze(0) # (1, T, D)
            
            # Tải audio và tính mel thật
            audio_path = os.path.join(audio_dir, unit_file.replace('.npy', '.wav'))
            waveform = load_audio(audio_path).to(device)
            target_mel = compute_mel_spec(waveform).squeeze(0).transpose(0, 1) # (T_mel, mel_bins)
            
            # Đảm bảo độ dài khớp nhau
            min_len = min(embedded_units.shape[1], target_mel.shape[0])
            embedded_units = embedded_units[:, :min_len, :]
            target_mel = target_mel[:min_len, :].unsqueeze(0) # (1, T_mel, mel_bins)

            optimizer.zero_grad()
            
            predicted_mel = model(embedded_units)
            loss = criterion(predicted_mel, target_mel)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(unit_files)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Vocoder (Unit-to-Mel) checkpoint đã được lưu vào '{checkpoint_path}'.")

if __name__ == "__main__":
    main()