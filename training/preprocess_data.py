import os
import torch
import numpy as np
from tqdm import tqdm
import subprocess

# Tự động gọi các script train nếu checkpoint chưa có
if not os.path.exists('checkpoints/feature_extractor'):
    print("Checkpoint Feature Extractor không tìm thấy. Đang tự động 'huấn luyện'...")
    subprocess.run(["python", "training/train_feature_extractor.py"])

from model import feature_extractor, quantizer_vqvae

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Paths ---
    data_dir = 'data'
    feature_cache_dir = 'cache/features'
    unit_cache_dir = 'cache/units'
    
    os.makedirs(feature_cache_dir, exist_ok=True)
    os.makedirs(os.path.join(unit_cache_dir, 'en'), exist_ok=True)
    os.makedirs(os.path.join(unit_cache_dir, 'vie'), exist_ok=True)
    
    # --- Load Models ---
    print("Tải các model cần thiết cho tiền xử lý...")
    feat_extractor = feature_extractor.load_model('checkpoints/feature_extractor', device=device)
    
    # Tự động train quantizer nếu chưa có
    quantizer_checkpoint = 'checkpoints/quantizer.pt'
    if not os.path.exists(quantizer_checkpoint):
        print("\nCheckpoint Quantizer không tìm thấy. Cần trích xuất features trước.")
        # Step 1: Extract features for quantizer training
        print("Bước 1: Trích xuất features cho Quantizer...")
        for lang in ['en', 'vie']:
            lang_dir = os.path.join(data_dir, lang)
            for fname in tqdm(os.listdir(lang_dir), desc=f"Extracting features for {lang}"):
                if fname.endswith('.wav'):
                    audio_path = os.path.join(lang_dir, fname)
                    features = feat_extractor.extract(audio_path)
                    np.save(os.path.join(feature_cache_dir, fname.replace('.wav', '.npy')), features.cpu().numpy())
        print("\nBước 2: Tự động huấn luyện Quantizer...")
        subprocess.run(["python", "training/train_quantizer.py"])

    quantizer = quantizer_vqvae.load_model(quantizer_checkpoint, device=device)
    
    # --- Process Data ---
    print("\nBắt đầu tiền xử lý: Trích xuất discrete units...")
    for lang in ['en', 'vie']:
        lang_dir = os.path.join(data_dir, lang)
        pbar = tqdm(os.listdir(lang_dir), desc=f"Processing {lang}")
        for fname in pbar:
            if fname.endswith('.wav'):
                audio_path = os.path.join(lang_dir, fname)
                
                # 1. Extract features
                features = feat_extractor.extract(audio_path)
                
                # 2. Quantize to get units
                with torch.no_grad():
                    units = quantizer.encode(features)
                
                # 3. Save units
                save_path = os.path.join(unit_cache_dir, lang, fname.replace('.wav', '.npy'))
                np.save(save_path, units.cpu().numpy())
                
    print("\nTiền xử lý hoàn tất!")
    print(f"Features được lưu tại: {feature_cache_dir}")
    print(f"Units được lưu tại: {unit_cache_dir}")

if __name__ == "__main__":
    main()