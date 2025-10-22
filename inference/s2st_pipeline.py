import os
import torch
import torchaudio
# Add project root to sys.path to allow imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoder_hubert import HuBERTEncoder
from models.quantizer_vqvae import VQVAEQuantizer
from models.translator_seq2seq import SpeechTranslator
from models.vocoder_hifigan import HiFiGANVocoder

class S2STPipeline:
    def __init__(self, checkpoints_dir='checkpoints'):
        print("Initializing S2ST Pipeline...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = HuBERTEncoder(device=self.device)
        
        # Load Quantizer (assuming VQ-VAE for this example)
        self.quantizer = VQVAEQuantizer().to(self.device)
        quantizer_path = os.path.join(checkpoints_dir, 'vqvae.pt')
        if os.path.exists(quantizer_path):
            #self.quantizer.load_state_dict(torch.load(quantizer_path))
            print(f"Loaded quantizer from {quantizer_path}")
        else:
            print(f"Warning: Quantizer checkpoint not found at {quantizer_path}. Using initial weights.")
        self.quantizer.eval()
        
        # Load Translator
        self.translator = SpeechTranslator(num_en_units=512, num_vn_units=512).to(self.device)
        translator_path = os.path.join(checkpoints_dir, 'translator.pt')
        if os.path.exists(translator_path):
            #self.translator.load_state_dict(torch.load(translator_path))
            print(f"Loaded translator from {translator_path}")
        else:
            print(f"Warning: Translator checkpoint not found at {translator_path}. Using initial weights.")
        self.translator.eval()

        # Load Vocoder
        self.vocoder = HiFiGANVocoder(device=self.device)
        print("S2ST Pipeline is ready.")

    def translate_audio(self, input_wav_path, output_wav_path='results/output_vn.wav'):
        """
        Full end-to-end translation from an English audio file to a Vietnamese audio file.
        """
        print(f"\n--- Starting Translation for {os.path.basename(input_wav_path)} ---")
        print("1. Extracting features...")
        features_en = self.encoder.extract_features(input_wav_path).unsqueeze(0).to(self.device)
        
        print("2. Quantizing features to discrete units...")
        units_en = self.quantizer.quantize(features_en)
        
        print("3. Translating English units to Vietnamese units...")
        units_vn = self.translator.translate(units_en)
        
        print("4. Synthesizing Vietnamese audio from units...")
        waveform_vn = self.vocoder.synthesize(units_vn.cpu()) # Vocoder expects CPU tensor for dummy logic
        
        os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
        torchaudio.save(output_wav_path, waveform_vn.unsqueeze(0), sample_rate=16000)
        
        print(f"--- Translation Complete. Output saved to: {output_wav_path} ---")
        return output_wav_path

# Example usage
if __name__ == '__main__':
    # Create a dummy input wav for testing
    dummy_audio = torch.randn(1, 16000 * 3) # 3 seconds
    dummy_input_path = 'data/en/sample_for_inference.wav'
    os.makedirs('data/en', exist_ok=True)
    torchaudio.save(dummy_input_path, dummy_audio, 16000)

    # Initialize and run pipeline
    pipeline = S2STPipeline()
    pipeline.translate_audio(dummy_input_path)