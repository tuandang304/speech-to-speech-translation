import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.hubert import HubertEncoder
from src.models.quantizer import VQVAEQuantizer
from src.data.dataloader import AudioDataset

def train_quantizer(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    hubert = HubertEncoder(
        config['model']['hubert_model'], 
        config['model']['feature_layer']
    ).to(device)
    quantizer_model = VQVAEQuantizer(
        input_dim=config['model']['quantizer']['input_dim'],
        num_embeddings=config['model']['quantizer']['num_embeddings'],
        commitment_cost=config['model']['quantizer']['commitment_cost']
    ).to(device)

    dataset = AudioDataset(args.data_manifest)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)

    optimizer = optim.Adam(quantizer_model.parameters(), lr=config['training']['learning_rate'])
    recon_criterion = nn.MSELoss()

    print("--- Starting Quantizer Training ---")
    for epoch in range(config['training']['epochs']):
        quantizer_model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for wav in pbar:
            wav = wav.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                features = hubert(wav)

            reconstructed_features, _, vq_loss = quantizer_model(features)
            recon_loss = recon_criterion(reconstructed_features, features)
            
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=loss.item(), recon_loss=recon_loss.item(), vq_loss=vq_loss.item())

    # Save model
    save_path = os.path.join(config['training']['output_dir'], 'vqvae_quantizer.pt')
    torch.save(quantizer_model.state_dict(), save_path)
    print(f"Quantizer model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ-VAE Quantizer")
    parser.add_argument('--config', type=str, required=True, help="Path to quantizer config file")
    parser.add_argument('--data-manifest', type=str, required=True, help="Path to unilingual audio manifest")
    args = parser.parse_args()
    train_quantizer(args)