import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import os
from tqdm import tqdm

from src.models.translator import Seq2SeqTranslator
from src.data.dataloader import create_unit_dataloader

def train_translator(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config['training']['output_dir'], exist_ok=True)

    model = Seq2SeqTranslator(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        pad_idx=config['model']['pad_idx']
    ).to(device)

    dataloader = create_unit_dataloader(
        args.train_manifest,
        batch_size=config['training']['batch_size'],
        sos_idx=config['model']['sos_idx'],
        eos_idx=config['model']['eos_idx'],
        pad_idx=config['model']['pad_idx']
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=config['model']['pad_idx'])

    print("--- Starting Translator Training ---")
    for epoch in range(config['training']['epochs']):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        total_loss = 0
        for src, tgt_in, tgt_out in pbar:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt_in)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} average loss: {total_loss / len(dataloader):.4f}")

    save_path = os.path.join(config['training']['output_dir'], 'seq2seq_translator.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Translator model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Seq2Seq Translator on units")
    parser.add_argument('--config', type=str, required=True, help="Path to translator config file")
    parser.add_argument('--train-manifest', type=str, required=True, help="Path to manifest of unit pairs")
    args = parser.parse_args()
    train_translator(args)