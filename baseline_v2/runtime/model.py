import math
import numpy as np
import torch
from torch import nn
from .data import features, event_features, to_global


class TemporalBlock(nn.Module):
    def __init__(self, h, dilation):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(h, h, 5, padding=2 * dilation, dilation=dilation) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(2)])
        self.dropout = nn.Dropout(0.15)

    def forward(self, x, mask):
        z = x
        for conv, norm in zip(self.convs, self.norms):
            z = conv(z.transpose(1, 2)).transpose(1, 2)
            z = self.dropout(torch.nn.functional.gelu(norm(z))) * mask
        return (x + z) * mask


def get_device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    return torch.device('cuda' if name == 'cuda' or name == 'auto' and torch.cuda.is_available() else 'cpu')


class SharedGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        h, n = config['hidden'], len(config['sessions'])
        self.input_weight = nn.Parameter(torch.randn(n, 1024, h) / math.sqrt(1024))
        if config.get('shared_input', False):
            self.input_shared = nn.Parameter(torch.randn(1024, h) / math.sqrt(1024))
            nn.init.zeros_(self.input_weight)
        self.input_bias = nn.Parameter(torch.zeros(n, h))
        self.session_emb = nn.Embedding(n, 16)
        self.condition_emb = nn.Embedding(len(config['conditions']), 16)
        self.norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(0.15)
        input_size = h + 16 + (28 if config['metadata'] else 0)
        if config.get('architecture') == 'shared_tcn_v2':
            self.temporal_project = nn.Linear(input_size, h)
            self.blocks = nn.ModuleList([TemporalBlock(h, d) for d in (1, 2, 4, 8, 16)])
        else:
            self.gru = nn.GRU(input_size, h, 2, batch_first=True, dropout=0.15,
                              bidirectional=config.get('bidirectional', False))
        output_size = h * (2 if config.get('bidirectional', False) else 1)
        self.head = nn.Linear(output_size, 63)
        self.out_weight = nn.Parameter(torch.zeros(n, output_size, 63))
        self.out_bias = nn.Parameter(torch.zeros(n, 63))

    def forward(self, x, meta, sid, cid, lengths=None):
        z = self.config.get('adapter_scale', 1.0) * torch.bmm(x, self.input_weight[sid]) + self.input_bias[sid, None]
        if self.config.get('shared_input', False):
            z = z + x @ self.input_shared
        z = self.dropout(torch.nn.functional.gelu(self.norm(z)))
        parts = [z, self.session_emb(sid)[:, None].expand(-1, x.shape[1], -1)]
        if self.config['metadata']:
            parts += [self.condition_emb(cid)[:, None].expand(-1, x.shape[1], -1), meta]
        z = torch.cat(parts, dim=-1)
        if self.config.get('architecture') == 'shared_tcn_v2':
            mask = torch.ones_like(z[:, :, :1]) if lengths is None else (torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None])[:, :, None]
            z = self.temporal_project(z) * mask
            for block in self.blocks:
                z = block(z, mask)
        elif self.config.get('bidirectional', False) and lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
            z, _ = self.gru(packed)
            z, _ = nn.utils.rnn.pad_packed_sequence(z, batch_first=True, total_length=x.shape[1])
        else:
            z, _ = self.gru(z)
        return self.head(z) + torch.bmm(z, self.out_weight[sid]) + self.out_bias[sid, None]


@torch.inference_mode()
def predict(model, item, stat, sid, cid, device):
    model.eval()
    x = torch.from_numpy(features(item['x'], stat['x_mean'], stat['x_std']))[None].to(device)
    meta = torch.from_numpy(event_features(item))[None].to(device)
    out = model(x, meta, torch.tensor([sid], device=device), torch.tensor([cid], device=device))[0].cpu().numpy()
    return to_global(out * stat['y_std'] + stat['y_mean'], 0).astype('float32')


def save_checkpoint(path, model, stats, config, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'config': config,
                'stats': {s: {k: torch.from_numpy(v) for k, v in stat.items()} for s, stat in stats.items()},
                'names': names}, path)


def load_checkpoint(path, device):
    c = torch.load(path, map_location='cpu', weights_only=True)
    model = SharedGRU(c['config']).to(device)
    model.load_state_dict(c['state'])
    stats = {s: {k: v.numpy() for k, v in stat.items()} for s, stat in c['stats'].items()}
    return model, stats, c['config'], c['names']
