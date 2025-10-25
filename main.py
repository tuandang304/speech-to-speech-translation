import argparse
from training.model.pipeline_inference import S2ST_Pipeline
import torch

def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline S2ST từ command line.")
    parser.add_argument('--input_audio', type=str, required=True, help="Đường dẫn đến file audio tiếng Anh đầu vào (.wav).")
    parser.add_argument('--output_audio', type=str, required=True, help="Đường dẫn để lưu file audio tiếng Việt đầu ra (.wav).")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Khởi tạo pipeline
    pipeline = S2ST_Pipeline(device=device)
    
    # Thực hiện dịch
    pipeline.translate(args.input_audio, args.output_audio)
    
    print("Hoàn thành!")

if __name__ == "__main__":
    main()