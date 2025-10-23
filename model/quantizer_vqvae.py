import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x):
        # x: (T, D)
        flat_x = x.view(-1, self.embedding.embedding_dim)
        
        # Tính khoảng cách
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True) 
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t())
                     + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t())
        
        # Tìm index gần nhất
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), flat_x)
        q_latent_loss = F.mse_loss(quantized, flat_x.detach())
        loss = q_latent_loss + 0.25 * e_latent_loss
        
        # Straight-through estimator
        quantized = flat_x + (quantized - flat_x).detach()
        
        return quantized.view_as(x), encoding_indices, loss

class Quantizer(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_embeddings=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.vq = VectorQuantizer(num_embeddings, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, features):
        encoded = self.encoder(features)
        quantized, indices, vq_loss = self.vq(encoded)
        decoded = self.decoder(quantized)
        reconstruction_loss = F.mse_loss(decoded, features)
        return decoded, indices, vq_loss + reconstruction_loss

    def encode(self, features):
        encoded = self.encoder(features)
        _, indices, _ = self.vq(encoded)
        return indices

def load_model(checkpoint_path, device='cpu'):
    # Bạn cần biết các tham số khi tạo model
    model = Quantizer(input_dim=768, hidden_dim=256, num_embeddings=512).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Quantizer đã được tải từ '{checkpoint_path}' lên {device}.")
    return model