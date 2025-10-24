import os
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

class HuBERTEncoder:
    def __init__(self, model_name="facebook/hubert-base-ls960", device=None):
        """
        Initializes the HuBERT encoder.
        Args:
            model_name (str): The name of the pre-trained HuBERT model.
            device (str): 'cuda' or 'cpu'.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
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

    def extract_features(self, audio_path, output_dir=None, target_sr=16000, chunk_length_sec=10):
        """
        Extracts HuBERT features from a single audio file.
        Automatically splits long files into smaller chunks.
        """
        waveform, sample_rate = self.safe_load_audio(audio_path)

        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)

        total_samples = waveform.size(1)
        chunk_size = int(chunk_length_sec * target_sr)

        filename = os.path.splitext(os.path.basename(audio_path))[0]

        # Nếu file ngắn hơn chunk size thì xử lý 1 lần
        if total_samples <= chunk_size:
            features = self._process_waveform(waveform, target_sr)
            if output_dir:
                torch.save(features, os.path.join(output_dir, f"{filename}.pt"))
            return [features]

        # Nếu file dài, chia thành nhiều phần
        print(f"Splitting {filename} into chunks...")
        feature_list = []
        part = 1

        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            chunk = waveform[:, start:end]

            try:
                features = self._process_waveform(chunk, target_sr)
                feature_list.append(features)

                if output_dir:
                    part_name = f"{filename}_{part}.pt"
                    torch.save(features, os.path.join(output_dir, part_name))
                part += 1

            except RuntimeError as e:
                print(f"Error processing chunk {part} of {filename}: {e}")
                break

            torch.cuda.empty_cache()

        return feature_list

    def _process_waveform(self, waveform, sample_rate):
        """
        Internal helper to run HuBERT inference on one waveform chunk.
        """
        inputs = self.processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0).cpu()

    def batch_extract(self, input_dir, output_dir):
        """
        Extracts features for all audio files in a directory and saves them.
        Automatically splits long files.
        """
        os.makedirs(output_dir, exist_ok=True)
        audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

        for filename in tqdm(audio_files, desc=f"Extracting features from {input_dir}"):
            input_path = os.path.join(input_dir, filename)
            base_output = os.path.splitext(filename)[0] + ".pt"
            output_path = os.path.join(output_dir, base_output)

            # Skip if already processed
            if os.path.exists(output_path) or any(
                os.path.exists(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{i}.pt")) for i in range(1, 50)
            ):
                continue

            try:
                self.extract_features(input_path, output_dir)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue


if __name__ == '__main__':
    encoder = HuBERTEncoder()

    processed_dir = 'data/mini_processed'
    os.makedirs(f'{processed_dir}/en_features', exist_ok=True)
    os.makedirs(f'{processed_dir}/vn_features', exist_ok=True)

    # Process English audio
    encoder.batch_extract('data/mini_data_en', f'{processed_dir}/en_features')

    # Process Vietnamese audio
    encoder.batch_extract('data/mini_data_vie', f'{processed_dir}/vn_features')

    print("Feature extraction complete.")
