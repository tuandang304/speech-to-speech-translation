import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Basic Vector Quantizer module from VQ-VAE.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs shape: (B, T, D)
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding: find the closest embedding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(encoding_indices.view(-1))
        
        # Reshape to original input shape
        quantized = quantized.view(input_shape)
        
        # VQ-VAE loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.view(inputs.shape[0], inputs.shape[1])

class VQVAEQuantizer(nn.Module):
    def __init__(self, input_dim=1024, num_embeddings=512, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.encoder_proj = nn.Linear(input_dim, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        print("VQ-VAE Quantizer Initialized.")

    def forward(self, features):
        # Project HuBERT features to embedding dimension
        projected_features = self.encoder_proj(features)
        quantized, loss, indices = self.vq(projected_features)
        return quantized, loss, indices

    def quantize(self, features):
        """For inference, just get the indices."""
        self.eval()
        with torch.no_grad():
            projected_features = self.encoder_proj(features)
            _, _, indices = self.vq(projected_features)
        return indices