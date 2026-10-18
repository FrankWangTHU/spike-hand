"""Training-only event-aligned augmentation; never reads validation trials."""
import torch


def mix_trial(reference, partner, coefficient):
    if reference[3:] != partner[3:]:
        raise ValueError('Mixup requires the same session and condition')
    if not 0 <= coefficient <= 1:
        raise ValueError('Invalid mixup coefficient')
    target_phase = reference[1][:, 2].contiguous()
    source_phase = partner[1][:, 2].contiguous()
    right = torch.searchsorted(source_phase, target_phase).clamp(1, len(source_phase) - 1)
    left = right - 1
    weight = ((target_phase - source_phase[left]) / (source_phase[right] - source_phase[left]).clamp_min(1e-8)).clamp(0, 1)[:, None]
    def interpolate(value):
        return value[left] * (1 - weight) + value[right] * weight
    x = coefficient * reference[0] + (1 - coefficient) * interpolate(partner[0])
    y = coefficient * reference[2] + (1 - coefficient) * interpolate(partner[2])
    return (x, reference[1], y, *reference[3:])
