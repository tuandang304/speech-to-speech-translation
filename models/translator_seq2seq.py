import torch
import torch.nn as nn

class SpeechTranslator(nn.Module):
    def __init__(self, num_en_units, num_vn_units, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        """
        A simple Transformer-based Seq2Seq model for unit translation.
        """
        super().__init__()
        self.d_model = d_model
        self.en_embedding = nn.Embedding(num_en_units, d_model)
        self.vn_embedding = nn.Embedding(num_vn_units, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5000, d_model)) # Positional Encoding
        
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, num_vn_units)
        print("SpeechTranslator model initialized.")

    def forward(self, src_units, tgt_units):
        src_emb = self.en_embedding(src_units) * (self.d_model**0.5)
        tgt_emb = self.vn_embedding(tgt_units) * (self.d_model**0.5)
        
        src_emb += self.pos_encoder[:, :src_units.size(1), :]
        tgt_emb += self.pos_encoder[:, :tgt_units.size(1), :]
        
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_units.size(1)).to(src_units.device)
        
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        
        return self.fc_out(output)

    def translate(self, src_units, max_len=100, start_symbol=1, end_symbol=2):
        """
        Greedy decoding for inference.
        """
        self.eval()
        with torch.no_grad():
            src_emb = self.en_embedding(src_units) * (self.d_model**0.5)
            src_emb += self.pos_encoder[:, :src_units.size(1), :]
            
            memory = self.transformer.encoder(src_emb)
            
            # Start with the start symbol
            ys = torch.ones(1, 1).fill_(start_symbol).type_as(src_units.data)
            
            for _ in range(max_len-1):
                tgt_emb = self.vn_embedding(ys) * (self.d_model**0.5)
                tgt_emb += self.pos_encoder[:, :ys.size(1), :]
                
                tgt_mask = self.transformer.generate_square_subsequent_mask(ys.size(1)).type_as(memory.data)
                
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask)
                prob = self.fc_out(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()
                
                ys = torch.cat([ys, torch.ones(1, 1).type_as(src_units.data).fill_(next_word)], dim=1)
                if next_word == end_symbol:
                    break
                    
        return ys