import os
import torch
from transformers import HubertModel

def main():
    """
    Đây là script "huấn luyện" giả.
    Thay vì huấn luyện từ đầu, chúng ta tải một model HuBERT đã được pre-trained
    và lưu nó vào thư mục checkpoints để pipeline có thể sử dụng.
    """
    model_name = "facebook/hubert-base-ls960"
    save_directory = "checkpoints/feature_extractor"
    
    print(f"Tải pre-trained HuBERT model: '{model_name}'...")
    if not os.path.exists(save_directory):
        model = HubertModel.from_pretrained(model_name)
        os.makedirs(save_directory, exist_ok=True)
        model.save_pretrained(save_directory)
        print(f"Model đã được lưu vào '{save_directory}'.")
    else:
        print(f"Checkpoint đã tồn tại tại '{save_directory}'. Bỏ qua.")

if __name__ == "__main__":
    main()