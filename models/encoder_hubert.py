import os
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# torchaudio.set_audio_backend("sox_io")

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

    def safe_load_audio(self, path, target_sr=16000):
        try:
            waveform, sample_rate = torchaudio.load(path)
        except Exception:
            waveform, sample_rate = sf.read(path, dtype='float32')
            waveform = torch.tensor(waveform).unsqueeze(0)
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        return waveform, target_sr

    def extract_features(self, audio_path, target_sr=16000):
        """
        Extracts HuBERT features from a single audio file.
        """
        # waveform, sample_rate = torchaudio.load(audio_path)
        waveform, sample_rate = self.safe_load_audio(audio_path)
        
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
                break

# Example usage to pre-process data
if __name__ == '__main__':
    encoder = HuBERTEncoder()

    processed_dir = 'data/mini_processed'

    os.makedirs(f'{processed_dir}/en_features', exist_ok=True)
    os.makedirs(f'{processed_dir}/vn_features', exist_ok=True)

    # Process English audio
    # encoder.batch_extract('data/en', 'data/processed/en_features')
    encoder.batch_extract('data/mini_data_en', f'{processed_dir}/en_features')
    # Process Vietnamese audio
    # encoder.batch_extract('data/vn', 'data/processed/vn_features')
    encoder.batch_extract('data/mini_data_vie', f'{processed_dir}/vn_features')

    print("Feature extraction complete.")