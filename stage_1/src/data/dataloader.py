import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class AudioDataset(Dataset):
    def __init__(self, manifest_path, sampling_rate=16000, max_duration_s=10):
        self.paths = [line.strip() for line in open(manifest_path)]
        self.sampling_rate = sampling_rate
        self.max_samples = max_duration_s * sampling_rate

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        waveform, sr = torchaudio.load(path)
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Trim or pad
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        return waveform.squeeze(0)

class UnitPairDataset(Dataset):
    def __init__(self, manifest_path, sos_idx, eos_idx):
        self.pairs = []
        with open(manifest_path, 'r') as f:
            for line in f:
                src_path, tgt_path = line.strip().split('\t')
                self.pairs.append((src_path, tgt_path))
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_unit_path, tgt_unit_path = self.pairs[idx]
        src_units = torch.load(src_unit_path).squeeze(0)
        tgt_units = torch.load(tgt_unit_path).squeeze(0)
        
        # Add SOS/EOS tokens
        tgt_units_in = torch.cat([torch.tensor([self.sos_idx]), tgt_units])
        tgt_units_out = torch.cat([tgt_units, torch.tensor([self.eos_idx])])
        
        return src_units, tgt_units_in, tgt_units_out

def create_unit_dataloader(manifest_path, batch_size, sos_idx, eos_idx, pad_idx, shuffle=True):
    dataset = UnitPairDataset(manifest_path, sos_idx, eos_idx)

    def collate_fn(batch):
        src_units, tgt_units_in, tgt_units_out = zip(*batch)
        
        src_padded = pad_sequence(src_units, batch_first=True, padding_value=pad_idx)
        tgt_in_padded = pad_sequence(tgt_units_in, batch_first=True, padding_value=pad_idx)
        tgt_out_padded = pad_sequence(tgt_units_out, batch_first=True, padding_value=pad_idx)

        return src_padded, tgt_in_padded, tgt_out_padded

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)