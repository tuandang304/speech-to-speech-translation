import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Translator(nn.Module):
    def __init__(self, vocab_size=512, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src_units):
        # src_units: (B, T)
        src_embedded = self.embedding(src_units) * math.sqrt(self.d_model)
        src_pos = self.pos_encoder(src_embedded.transpose(0, 1)).transpose(0, 1) # (B, T, D)
        
        output = self.transformer_encoder(src_pos)
        return self.output_layer(output)

    @torch.no_grad()
    def translate(self, src_units, max_len=200):
        self.eval()
        # Đơn giản là trả về output của encoder, trong 1 hệ thống thật sẽ phức tạp hơn (cần decoder)
        # Ở đây ta giả lập một model "unit-to-unit" đơn giản
        src_tensor = torch.LongTensor(src_units).unsqueeze(0).to(next(self.parameters()).device)
        logits = self(src_tensor)
        predicted_units = torch.argmax(logits, dim=-1).squeeze(0)
        return predicted_units.cpu().numpy()

def load_model(checkpoint_path, device='cpu'):
    model = Translator(vocab_size=512, d_model=256, nhead=8, num_layers=4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Translator đã được tải từ '{checkpoint_path}' lên {device}.")
    return model