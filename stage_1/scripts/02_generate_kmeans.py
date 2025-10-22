import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
from joblib import dump
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from src.models.hubert import HubertEncoder
from src.data.dataloader import AudioDataset

def generate_kmeans(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    hubert = HubertEncoder().to(device)
    dataset = AudioDataset(args.data_manifest)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("--- Extracting HuBERT features ---")
    all_features = []
    for wav in tqdm(dataloader):
        wav = wav.to(device)
        with torch.no_grad():
            features = hubert(wav)
            all_features.append(features.cpu().numpy().reshape(-1, features.shape[-1]))
    
    all_features = np.concatenate(all_features, axis=0)
    print(f"Extracted {all_features.shape[0]} feature vectors.")

    print(f"--- Fitting MiniBatchKMeans with {args.num_clusters} clusters ---")
    kmeans = MiniBatchKMeans(
        n_clusters=args.num_clusters,
        random_state=0,
        batch_size=2048,
        verbose=1,
        max_iter=100
    ).fit(all_features)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    dump(kmeans, args.output_path)
    print(f"K-means model saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K-means model from HuBERT features")
    parser.add_argument('--data-manifest', type=str, required=True, help="Path to unilingual audio manifest")
    parser.add_argument('--num-clusters', type=int, default=512, help="Number of K-means clusters (codebook size)")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument('--output-path', type=str, required=True, help="Path to save the K-means model")
    args = parser.parse_args()
    generate_kmeans(args)