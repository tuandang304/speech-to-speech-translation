import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2SeqTranslator(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = nn.Identity() # Placeholder for PositionalEncoding
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src_units, tgt_units):
        src_padding_mask = (src_units == self.pad_idx)
        tgt_padding_mask = (tgt_units == self.pad_idx)

        tgt_mask = self._generate_square_subsequent_mask(tgt_units.size(1)).to(src_units.device)
        
        src_emb = self.pos_encoder(self.embedding(src_units))
        tgt_emb = self.pos_encoder(self.embedding(tgt_units))
        
        output = self.transformer(
            src_emb, tgt_emb, 
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_mask
        )
        return self.fc_out(output)