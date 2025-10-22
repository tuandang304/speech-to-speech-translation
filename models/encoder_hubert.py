import os
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm

class HuBERTEncoder:
    def __init__(self, model_name="facebook/hubert-large-ls960-ft", device=None):
        """
        Initializes the HuBERT encoder.
        Args:
            model_name (str): The name of the pre-trained HuBERT model.
            device (str): 'cuda' or 'cpu'.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name).to(self.device)
        print(f"HuBERT model loaded on {self.device}")

    def extract_features(self, audio_path, target_sr=16000):
        """
        Extracts HuBERT features from a single audio file.
        """
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)

        # Process and extract features
        inputs = self.processor(waveform.squeeze(0), sampling_rate=target_sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Last hidden state is our feature representation
        return outputs.last_hidden_state.squeeze(0).cpu()

    def batch_extract(self, input_dir, output_dir):
        """
        Extracts features for all audio files in a directory and saves them.
        """
        os.makedirs(output_dir, exist_ok=True)
        audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
        
        for filename in tqdm(audio_files, desc=f"Extracting features from {input_dir}"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.wav', '.pt'))
            
            if os.path.exists(output_path):
                continue

            try:
                features = self.extract_features(input_path)
                torch.save(features, output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage to pre-process data
if __name__ == '__main__':
    encoder = HuBERTEncoder()
    os.makedirs('data/processed/en_features', exist_ok=True)
    os.makedirs('data/processed/vn_features', exist_ok=True)
    # Process English audio
    encoder.batch_extract('data/en', 'data/processed/en_features')
    # Process Vietnamese audio
    encoder.batch_extract('data/vn', 'data/processed/vn_features')
    print("Feature extraction complete.")