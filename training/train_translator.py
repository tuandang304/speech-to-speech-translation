import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm

from model.translator_seq2seq import Translator

def main():
    # --- Configs ---
    input_dir = 'cache/units'
    checkpoint_path = 'checkpoints/translator.pt'
    epochs = 20
    lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(input_dir):
        print(f"Thư mục cache units '{input_dir}' không tồn tại.")
        print("Vui lòng chạy 'python training/preprocess_data.py' trước.")
        return

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint đã tồn tại tại '{checkpoint_path}'. Bỏ qua.")
        return

    # --- Model ---
    model = Translator(vocab_size=512, d_model=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # --- Data ---
    # Lấy các cặp file
    src_files = sorted([f for f in os.listdir(os.path.join(input_dir, 'en')) if f.endswith('.npy')])
    tgt_files = sorted([f for f in os.listdir(os.path.join(input_dir, 'vie')) if f.endswith('.npy')])
    
    print("Bắt đầu huấn luyện Translator...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(zip(src_files, tgt_files), total=len(src_files), desc=f"Epoch {epoch+1}/{epochs}")
        for src_file, tgt_file in pbar:
            src_units = torch.from_numpy(np.load(os.path.join(input_dir, 'en', src_file))).long().unsqueeze(0).to(device)
            tgt_units = torch.from_numpy(np.load(os.path.join(input_dir, 'vie', tgt_file))).long().unsqueeze(0).to(device)
            
            # Đảm bảo độ dài bằng nhau cho mô hình đơn giản này
            min_len = min(src_units.size(1), tgt_units.size(1))
            src_units, tgt_units = src_units[:, :min_len], tgt_units[:, :min_len]

            optimizer.zero_grad()
            
            logits = model(src_units) # (B, T, V)
            
            # Reshape for CrossEntropyLoss: (B*T, V) and (B*T)
            loss = criterion(logits.reshape(-1, model.vocab_size), tgt_units.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(src_files)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Translator checkpoint đã được lưu vào '{checkpoint_path}'.")

if __name__ == "__main__":
    main()