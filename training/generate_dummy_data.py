import numpy as np
import soundfile as sf
import os

def generate_sine_wave(freq, duration, sample_rate=16000):
    """Tạo một sóng sin đơn giản."""
    t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2. * np.pi * freq * t)
    return data.astype(np.int16)

def main():
    """Tạo dữ liệu audio giả cho en và vie."""
    num_samples = 5
    duration = 2.0  # seconds
    sample_rate = 16000
    
    en_dir = 'data/en'
    vie_dir = 'data/vie'
    
    os.makedirs(en_dir, exist_ok=True)
    os.makedirs(vie_dir, exist_ok=True)
    
    print(f"Tạo {num_samples} cặp audio giả...")
    for i in range(num_samples):
        # Tiếng Anh: tần số thấp hơn
        en_freq = 440.0 + i * 20
        en_audio = generate_sine_wave(en_freq, duration, sample_rate)
        en_path = os.path.join(en_dir, f'sample_{i}.wav')
        sf.write(en_path, en_audio, sample_rate)
        
        # Tiếng Việt: tần số cao hơn
        vie_freq = 880.0 + i * 20
        vie_audio = generate_sine_wave(vie_freq, duration, sample_rate)
        vie_path = os.path.join(vie_dir, f'sample_{i}.wav')
        sf.write(vie_path, vie_audio, sample_rate)
        
    print("Tạo dữ liệu giả hoàn tất!")
    print(f"Dữ liệu được lưu tại '{en_dir}' và '{vie_dir}'.")

if __name__ == "__main__":
    main()