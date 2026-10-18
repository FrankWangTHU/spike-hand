import numpy as np
import torch
from torch import nn
from .data import features, to_global


def get_device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device('cuda' if name == 'cuda' or (name == 'auto' and torch.cuda.is_available()) else 'cpu')


class PoseGRU(nn.Module):
    def __init__(self, config, basis=None):
        super().__init__()
        h, k = config['hidden'], config['latent']
        self.project = nn.Sequential(nn.Linear(1024, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(h, h, num_layers=2, dropout=0.1, batch_first=True)
        self.factor = nn.Linear(h, k)
        self.residual = nn.Linear(h, 63)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.register_buffer('basis', torch.zeros(k, 63) if basis is None else torch.from_numpy(basis))

    def forward(self, x):
        h, _ = self.gru(self.project(x))
        factors = self.factor(h)
        return factors @ self.basis + 0.25 * self.residual(h), factors


@torch.inference_mode()
def predict(model, item, stats, wrist, device):
    model.eval()
    x = features(item['x'], stats['x_mean'], stats['x_std'])
    z, _ = model(torch.from_numpy(x).unsqueeze(0).to(device))
    local = z[0].cpu().numpy() * stats['y_std'] + stats['y_mean']
    y = to_global(local, wrist).astype('float32')
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite prediction')
    return y


def save_checkpoint(path, model, stats, config, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'stats': {k: torch.from_numpy(v) for k, v in stats.items()},
                'config': config, 'names': names}, path)


def load_checkpoint(path, device):
    c = torch.load(path, map_location='cpu', weights_only=True)
    model = PoseGRU(c['config']).to(device)
    model.load_state_dict(c['state'])
    return model, {k: v.numpy() for k, v in c['stats'].items()}, c['config'], c['names']
