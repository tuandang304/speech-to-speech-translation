import torch
from . import feature_extractor, quantizer_vqvae, translator_seq2seq, vocoder_hifigan
from .utils_audio import save_audio
import os

class S2ST_Pipeline:
    def __init__(self, device='cpu'):
        self.device = device
        
        # Kiểm tra sự tồn tại của các checkpoints
        required_checkpoints = [
            'checkpoints/feature_extractor', 
            'checkpoints/quantizer.pt', 
            'checkpoints/translator.pt', 
            'checkpoints/vocoder.pt'
        ]
        for chk in required_checkpoints:
            if not os.path.exists(chk):
                raise FileNotFoundError(f"Checkpoint '{chk}' không tồn tại! Vui lòng chạy các script huấn luyện trước.")

        print("Tải pipeline S2ST...")
        self.feat_extractor = feature_extractor.load_model('checkpoints/feature_extractor', device)
        self.quantizer = quantizer_vqvae.load_model('checkpoints/quantizer.pt', device)
        self.translator = translator_seq2seq.load_model('checkpoints/translator.pt', device)
        # Vocoder cần quantizer model để truy cập codebook
        self.vocoder = vocoder_hifigan.load_model(self.quantizer, 'checkpoints/vocoder.pt', device)
        print("Pipeline đã sẵn sàng.")

    def translate(self, input_audio_path, output_audio_path):
        print("Bước 1: Trích xuất đặc trưng...")
        features = self.feat_extractor.extract(input_audio_path)
        
        print("Bước 2: Rời rạc hóa (Quantization)...")
        en_units = self.quantizer.encode(features).cpu().numpy()
        
        print("Bước 3: Dịch sang units tiếng Việt...")
        vi_units = self.translator.translate(en_units)
        
        print("Bước 4: Tổng hợp âm thanh (Vocoding)...")
        vi_waveform = self.vocoder.synthesize(vi_units)
        
        print(f"Lưu kết quả ra file: {output_audio_path}")
        save_audio(vi_waveform, output_audio_path)
        
        return output_audio_path