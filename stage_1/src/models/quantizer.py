import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

class VQVAEQuantizer(nn.Module):
    def __init__(self, input_dim, num_embeddings, commitment_cost):
        super().__init__()
        # VQ-VAE's internal embedding dim is the input_dim
        self.quantizer = VectorQuantize(
            dim=input_dim,
            codebook_size=num_embeddings,
            commitment_weight=commitment_cost
        )
        # A simple decoder for reconstruction loss during training
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, features):
        quantized, indices, vq_loss_dict = self.quantizer(features)
        reconstructed_features = self.decoder(quantized)
        return reconstructed_features, indices, vq_loss_dict['loss']
    
    def get_indices(self, features):
        _, indices, _ = self.quantizer(features)
        return indices