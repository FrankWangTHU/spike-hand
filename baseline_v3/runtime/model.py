import math
import numpy as np
import torch
from torch import nn
from .data import features, model_events, decode_target


class ContextBlock(nn.Module):
    """Masked global attention plus local convolution, at a reduced frame rate."""
    def __init__(self, h):
        super().__init__()
        self.heads = 4
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(3)])
        self.qkv = nn.Linear(h, 3 * h)
        self.attention_out = nn.Linear(h, h)
        self.conv = nn.Sequential(nn.Conv1d(h, h, 7, padding=3, groups=h), nn.GELU(), nn.Conv1d(h, h, 1))
        self.ffn = nn.Sequential(nn.Linear(h, 4*h), nn.GELU(), nn.Dropout(.15), nn.Linear(4*h, h))
        self.dropout = nn.Dropout(.15)

    def forward(self, z, valid):
        batch, length, width = z.shape
        mask = valid[:, :, None].to(z.dtype)
        q, k, v = self.qkv(self.norms[0](z)).reshape(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :], dropout_p=.15 if self.training else 0.)
        z = (z + self.dropout(self.attention_out(a.transpose(1, 2).reshape(batch, length, width)))) * mask
        z = (z + self.dropout(self.conv((self.norms[1](z) * mask).transpose(1, 2)).transpose(1, 2))) * mask
        return (z + self.dropout(self.ffn(self.norms[2](z)))) * mask


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
        dim = config.get('feature_dim', 1024)
        self.input_weight = nn.Parameter(torch.randn(n, dim, h) / math.sqrt(dim))
        if config.get('shared_input', False):
            self.input_shared = nn.Parameter(torch.randn(dim, h) / math.sqrt(dim))
            nn.init.zeros_(self.input_weight)
        if config.get('condition_input', False):
            self.condition_input_weight = nn.Parameter(torch.zeros(len(config['conditions']), dim, h))
        self.input_bias = nn.Parameter(torch.zeros(n, h))
        self.session_emb = nn.Embedding(n, 16)
        self.condition_emb = nn.Embedding(len(config['conditions']), 16)
        self.norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(0.15)
        input_size = h + 16 + ((28 + 4 * bool(config.get('recording_clock', False))) if config['metadata'] else 0)
        if config.get('origin_head', False):
            self.origin_head = nn.Sequential(nn.Linear(4*input_size, h), nn.GELU(), nn.Dropout(.15), nn.Linear(h, 3))
            self.origin_bias = nn.Parameter(torch.zeros(n, 3))
        if config.get('architecture') == 'stacked_dual_v3':
            self.forward_layers = nn.ModuleList([nn.GRU(input_size, h, batch_first=True), nn.GRU(2*h, h, batch_first=True)])
            self.backward_layers = nn.ModuleList([nn.GRU(input_size, h, batch_first=True), nn.GRU(2*h, h, batch_first=True)])
        elif config.get('architecture') == 'attention_v3':
            self.temporal_project = nn.Linear(input_size, h)
            self.context_blocks = nn.ModuleList([ContextBlock(h) for _ in range(4)])
            self.context_fusion = nn.Sequential(nn.Linear(2*h, h), nn.LayerNorm(h), nn.GELU())
        elif config.get('architecture') == 'dual_gru_v3':
            self.forward_gru = nn.GRU(input_size, h, 2, batch_first=True, dropout=0.15)
            self.backward_gru = nn.GRU(input_size, h, 2, batch_first=True, dropout=0.15)
        elif config.get('architecture') == 'shared_tcn_v2':
            self.temporal_project = nn.Linear(input_size, h)
            self.blocks = nn.ModuleList([TemporalBlock(h, d) for d in (1, 2, 4, 8, 16)])
        else:
            self.gru = nn.GRU(input_size, h, 2, batch_first=True, dropout=0.15,
                              bidirectional=config.get('bidirectional', False))
        output_size = h * (2 if config.get('bidirectional', False) or config.get('architecture') in ('dual_gru_v3', 'stacked_dual_v3') else 1)
        self.head = nn.Linear(output_size, 63)
        self.out_weight = nn.Parameter(torch.zeros(n, output_size, 63))
        self.out_bias = nn.Parameter(torch.zeros(n, 63))
        if config.get('condition_head', False):
            self.condition_weight = nn.Parameter(torch.zeros(len(config['conditions']), output_size, 63))

    @staticmethod
    def reverse_valid(z, lengths):
        if lengths is None:
            return z.flip(1)
        index = torch.arange(z.shape[1], device=z.device)[None].expand(z.shape[0], -1)
        reverse = torch.where(index < lengths[:, None], lengths[:, None] - 1 - index, index)
        return z.gather(1, reverse[:, :, None].expand_as(z))

    def forward(self, x, meta, sid, cid, lengths=None, return_origin=False):
        z = self.config.get('adapter_scale', 1.0) * torch.bmm(x, self.input_weight[sid]) + self.input_bias[sid, None]
        if self.config.get('shared_input', False):
            z = z + x @ self.input_shared
        if self.config.get('condition_input', False):
            z = z + .1 * torch.bmm(x, self.condition_input_weight[cid])
        z = self.dropout(torch.nn.functional.gelu(self.norm(z)))
        parts = [z, self.session_emb(sid)[:, None].expand(-1, x.shape[1], -1)]
        if self.config['metadata']:
            parts += [self.condition_emb(cid)[:, None].expand(-1, x.shape[1], -1), meta]
        z = torch.cat(parts, dim=-1)
        origin = None
        if self.config.get('origin_head', False):
            valid = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device) if lengths is None else torch.arange(z.shape[1], device=z.device)[None] < lengths[:, None]
            def pool(mask):
                return (z * mask[:, :, None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
            mean = pool(valid)
            sd = (((z - mean[:, None]).square() * valid[:, :, None]).sum(1) / valid.sum(1)[:, None] + 1e-6).sqrt()
            origin = self.origin_head(torch.cat([mean, pool(valid & (meta[:, :, 2] < 0)), pool(valid & (meta[:, :, 2] > 1)), sd], dim=-1)) + self.origin_bias[sid]
        if self.config.get('architecture') == 'stacked_dual_v3':
            for layer, (forward_gru, backward_gru) in enumerate(zip(self.forward_layers, self.backward_layers)):
                forward, _ = forward_gru(z)
                backward, _ = backward_gru(self.reverse_valid(z, lengths))
                z = torch.cat([forward, self.reverse_valid(backward, lengths)], dim=-1)
                if layer == 0:
                    z = self.dropout(z)
        elif self.config.get('architecture') == 'attention_v3':
            z = self.temporal_project(z)
            sizes = lengths.tolist() if lengths is not None else [z.shape[1]] * z.shape[0]
            lens = torch.tensor(sizes, device=z.device)
            valid = torch.arange(z.shape[1], device=z.device)[None] < lens[:, None]
            z = z * valid[:, :, None]
            stride = 3
            extra = (-z.shape[1]) % stride
            pooled = nn.functional.pad(z, (0, 0, 0, extra)).reshape(z.shape[0], -1, stride, z.shape[2]).sum(2)
            count = nn.functional.pad(valid, (0, extra)).reshape(z.shape[0], -1, stride).sum(2)
            coarse = pooled / count.clamp_min(1)[:, :, None]
            for block in self.context_blocks:
                coarse = block(coarse, count > 0)
            expanded = []
            for i, size in enumerate(sizes):
                c = coarse[i:i+1, :(size + stride - 1) // stride].transpose(1, 2)
                up = nn.functional.interpolate(c, scale_factor=stride, mode='linear', align_corners=False)[:, :, :size].transpose(1, 2)
                expanded.append(nn.functional.pad(up, (0, 0, 0, z.shape[1] - size)))
            z = self.context_fusion(torch.cat([z, torch.cat(expanded, dim=0)], dim=-1))
        elif self.config.get('architecture') == 'dual_gru_v3':
            forward, _ = self.forward_gru(z)
            backward, _ = self.backward_gru(self.reverse_valid(z, lengths))
            z = torch.cat([forward, self.reverse_valid(backward, lengths)], dim=-1)
        elif self.config.get('architecture') == 'shared_tcn_v2':
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
        result = self.head(z) + torch.bmm(z, self.out_weight[sid]) + self.out_bias[sid, None]
        if self.config.get('condition_head', False):
            result = result + .1 * torch.bmm(z, self.condition_weight[cid])
        if origin is not None:
            result = torch.cat([result[:, :, :3] + origin[:, None], result[:, :, 3:]], dim=-1)
        return (result, origin) if return_origin else result


@torch.inference_mode()
def predict(model, item, stat, sid, cid, device):
    model.eval()
    x = torch.from_numpy(features(item['x'], stat['x_mean'], stat['x_std'], model.config.get('feature_mode', 'ema'), stat.get('input_projection')))[None].to(device)
    meta = torch.from_numpy(model_events(item, stat, model.config))[None].to(device)
    out = model(x, meta, torch.tensor([sid], device=device), torch.tensor([cid], device=device))[0].cpu().numpy()
    return decode_target(out * stat['y_std'] + stat['y_mean'], stat)


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
