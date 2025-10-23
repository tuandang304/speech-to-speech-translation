import torch
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm

from model.quantizer_vqvae import Quantizer

def main():
    # --- Configs ---
    input_dir = 'cache/features'
    checkpoint_path = 'checkpoints/quantizer.pt'
    epochs = 10
    lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(input_dir):
        print(f"Thư mục cache features '{input_dir}' không tồn tại.")
        print("Vui lòng chạy 'python training/preprocess_data.py' trước.")
        return
        
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint đã tồn tại tại '{checkpoint_path}'. Bỏ qua.")
        return

    # --- Model ---
    model = Quantizer(input_dim=768, hidden_dim=256, num_embeddings=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Data ---
    feature_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    # --- Training Loop ---
    print("Bắt đầu huấn luyện Quantizer...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(feature_files, desc=f"Epoch {epoch+1}/{epochs}")
        for f_path in pbar:
            features = torch.from_numpy(np.load(f_path)).to(device)
            
            optimizer.zero_grad()
            _, _, loss = model(features)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(feature_files)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    # --- Save checkpoint ---
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Quantizer checkpoint đã được lưu vào '{checkpoint_path}'.")

if __name__ == "__main__":
    main()