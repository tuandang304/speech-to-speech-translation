import torch
import torchaudio
import argparse
import yaml
from joblib import load as joblib_load

from src.models.hubert import HubertEncoder
from src.models.quantizer import VQVAEQuantizer
from src.models.translator import Seq2SeqTranslator

def inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load models
    hubert = HubertEncoder().to(device).eval()
    
    with open(args.quantizer_config, 'r') as f:
        q_config = yaml.safe_load(f)['model']
    with open(args.translator_config, 'r') as f:
        t_config = yaml.safe_load(f)['model']

    if args.quantizer_type == 'vqvae':
        quantizer = VQVAEQuantizer(
            input_dim=q_config['quantizer']['input_dim'],
            num_embeddings=q_config['quantizer']['num_embeddings'],
            commitment_cost=q_config['quantizer']['commitment_cost']
        ).to(device)
        quantizer.load_state_dict(torch.load(args.quantizer_path))
        quantizer.eval()
    else: # kmeans
        quantizer = joblib_load(args.quantizer_path)

    translator = Seq2SeqTranslator(
        vocab_size=t_config['vocab_size'], d_model=t_config['d_model'], nhead=t_config['nhead'],
        num_encoder_layers=t_config['num_encoder_layers'], num_decoder_layers=t_config['num_decoder_layers'],
        dim_feedforward=t_config['dim_feedforward'], pad_idx=t_config['pad_idx']
    ).to(device).eval()
    translator.load_state_dict(torch.load(args.translator_path))

    # 2. Process input audio
    wav, sr = torchaudio.load(args.input_audio)
    if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.to(device)

    # 3. Pipeline: HuBERT -> Quantizer -> Translator
    with torch.no_grad():
        features = hubert(wav)
        if args.quantizer_type == 'vqvae':
            src_units = quantizer.get_indices(features)
        else: # kmeans
            features_np = features.cpu().numpy().reshape(-1, features.shape[-1])
            src_units = torch.from_numpy(quantizer.predict(features_np)).unsqueeze(0).to(device)

        # Autoregressive decoding
        sos_idx, eos_idx = t_config['sos_idx'], t_config['eos_idx']
        tgt_units = torch.tensor([[sos_idx]], device=device)
        
        for _ in range(src_units.shape[1] + 50): # Max output length
            output = translator(src_units, tgt_units)
            next_token = output.argmax(2)[:, -1].item()
            if next_token == eos_idx:
                break
            tgt_units = torch.cat([tgt_units, torch.tensor([[next_token]], device=device)], dim=1)

        translated_units = tgt_units[0, 1:] # Remove SOS
        print(f"Generated Unit Sequence: {translated_units.tolist()}")

    # 4. Vocoder (Synthesis)
    print("\n--- INFERENCE COMPLETE ---")
    print("Next step: Use a pre-trained unit-based vocoder (e.g., HiFi-GAN) to synthesize audio from the generated units.")
    #
    # PSEUDO-CODE for Vocoder step:
    # vocoder = load_hifigan_vocoder_trained_on_units()
    # unit_embeddings = vocoder.embedding_layer(translated_units)
    # synthesized_wav = vocoder(unit_embeddings)
    # torchaudio.save(args.output_audio, synthesized_wav, 16000)
    # print(f"Synthesized audio saved to {args.output_audio}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S2ST Inference Pipeline")
    parser.add_argument('--input-audio', type=str, required=True)
    parser.add_argument('--output-audio', type=str, required=True, help="Path for final synthesized audio")
    parser.add_argument('--quantizer-type', type=str, required=True, choices=['vqvae', 'kmeans'])
    parser.add_argument('--quantizer-path', type=str, required=True)
    parser.add_argument('--translator-path', type=str, required=True)
    parser.add_argument('--quantizer-config', type=str, required=True)
    parser.add_argument('--translator-config', type=str, required=True)
    args = parser.parse_args()
    inference(args)