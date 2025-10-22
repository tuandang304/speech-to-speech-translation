import torch
import torchaudio
import argparse
import os
import yaml
from tqdm import tqdm
from joblib import load as joblib_load

from src.models.hubert import HubertEncoder
from src.models.quantizer import VQVAEQuantizer

def extract_units(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    hubert = HubertEncoder().to(device)
    hubert.eval()

    # Load Quantizer
    if args.quantizer_type == 'vqvae':
        with open(args.config, 'r') as f:
            q_config = yaml.safe_load(f)['model']
        quantizer = VQVAEQuantizer(
            input_dim=q_config['quantizer']['input_dim'],
            num_embeddings=q_config['quantizer']['num_embeddings'],
            commitment_cost=q_config['quantizer']['commitment_cost']
        ).to(device)
        quantizer.load_state_dict(torch.load(args.quantizer_path))
        quantizer.eval()
        print("VQ-VAE quantizer loaded.")
    elif args.quantizer_type == 'kmeans':
        quantizer = joblib_load(args.quantizer_path)
        print("K-means quantizer loaded.")
    else:
        raise ValueError("Invalid quantizer type")

    output_unit_dir = 'data/units'
    os.makedirs(output_unit_dir, exist_ok=True)
    
    print("--- Extracting units ---")
    with open(args.output_manifest, 'w') as f_out:
        with open(args.parallel_manifest, 'r') as f_in:
            for line in tqdm(f_in):
                uid, src_path, tgt_path = line.strip().split('\t')
                
                # Process source and target audio
                for role, path in [('src', src_path), ('tgt', tgt_path)]:
                    wav, sr = torchaudio.load(path)
                    if sr != 16000:
                        wav = torchaudio.functional.resample(wav, sr, 16000)
                    wav = wav.to(device)
                    
                    with torch.no_grad():
                        features = hubert(wav)
                        if args.quantizer_type == 'vqvae':
                            indices = quantizer.get_indices(features)
                        else: # kmeans
                            features_np = features.cpu().numpy().reshape(-1, features.shape[-1])
                            indices = torch.from_numpy(quantizer.predict(features_np)).unsqueeze(0).to(device)

                    unit_path = os.path.join(output_unit_dir, f"{uid}_{role}.pt")
                    torch.save(indices.cpu(), unit_path)
                
                src_unit_path = os.path.join(output_unit_dir, f"{uid}_src.pt")
                tgt_unit_path = os.path.join(output_unit_dir, f"{uid}_tgt.pt")
                f_out.write(f"{src_unit_path}\t{tgt_unit_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract discrete units from audio")
    parser.add_argument('--parallel-manifest', type=str, required=True)
    parser.add_argument('--quantizer-type', type=str, required=True, choices=['vqvae', 'kmeans'])
    parser.add_argument('--quantizer-path', type=str, required=True)
    parser.add_argument('--config', type=str, help="Quantizer config for VQ-VAE")
    parser.add_argument('--output-manifest', type=str, required=True)
    args = parser.parse_args()
    extract_units(args)