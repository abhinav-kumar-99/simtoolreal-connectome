"""Fixed camera luminance sampling at annotated MaleCNS optic columns."""
import torch
from torch import nn
from torch.nn import functional as F


class RetinalPopulationEncoder(nn.Module):
    def __init__(self, proprioception, positions, grid, *, image_start, height, width, gain=1.0, fused=False):
        super().__init__()
        self.proprioception = proprioception
        self.register_buffer('positions', torch.as_tensor(positions, dtype=torch.long))
        self.register_buffer('grid', torch.as_tensor(grid, dtype=torch.float32).reshape(1, -1, 1, 2))
        self.image_start = int(image_start)
        self.height, self.width = int(height), int(width)
        self.gain = float(gain)
        self.fused = bool(fused)
        if self.height < 2 or self.width < 2 or not torch.isfinite(self.grid).all() or self.grid.abs().max() > 1.00001:
            raise ValueError('Invalid retinal image/grid geometry')

    def forward(self, observations):
        drive = self.proprioception(observations)
        if self.fused:
            from simtoolreal_shared.vision_ops import retinal_scatter
            return retinal_scatter(observations, self.grid, self.positions, drive,
                                   self.image_start, self.height, self.width, self.gain)
        end = self.image_start + self.height * self.width
        image = observations[:, self.image_start:end].float().reshape(-1, 1, self.height, self.width)
        luminance = F.grid_sample(image, self.grid.expand(image.shape[0], -1, -1, -1),
                                  mode='bilinear', padding_mode='border', align_corners=True).flatten(1)
        # L1/L2 luminance drive: decreases with light increments. This fixed
        # dimensionless contrast code does not claim measured phototransduction.
        drive.index_copy_(1, self.positions, torch.tanh(self.gain * (1 - 2 * luminance)))
        return drive

    def spike_rates(self, observations):
        raise ValueError('The visual profile currently requires tanh dynamics')
