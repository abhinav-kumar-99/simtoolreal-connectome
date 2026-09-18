"""Optional CUDA kernels for fixed vision encoders, selected in YAML."""
import torch
import triton
import triton.language as tl


@triton.jit
def _luminance(RGBA, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = tl.load(RGBA + i * 4, i < N, 0).to(tl.float32) / 255.0
    g = tl.load(RGBA + i * 4 + 1, i < N, 0).to(tl.float32) / 255.0
    b = tl.load(RGBA + i * 4 + 2, i < N, 0).to(tl.float32) / 255.0
    tl.store(OUT + i, r * .299 + g * .587 + b * .114, i < N)


def rgba_luminance(rgba, out=None):
    if rgba.dtype != torch.uint8 or not rgba.is_cuda or not rgba.is_contiguous() or rgba.shape[-1] != 4:
        raise ValueError('Expected contiguous CUDA uint8 RGBA images')
    if out is None:
        out = torch.empty(rgba.shape[:-1], dtype=torch.float32, device=rgba.device)
    if (out.shape != rgba.shape[:-1] or out.dtype != torch.float32
            or out.device != rgba.device or not out.is_contiguous()):
        raise ValueError('Luminance output must be contiguous CUDA FP32 with matching geometry')
    n = out.numel()
    _luminance[(triton.cdiv(n, 256),)](rgba, out, n, 256, enable_fp_fusion=False)
    return out


@triton.jit
def _retinal(OBS, GRID, POS, OUT, STRIDE: tl.constexpr, START: tl.constexpr,
             H: tl.constexpr, W: tl.constexpr, CELLS: tl.constexpr,
             PORTS: tl.constexpr, GAIN: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = (tl.load(GRID + 2 * j, j < CELLS, 0) + 1.) * .5 * (W - 1)
    y = (tl.load(GRID + 2 * j + 1, j < CELLS, 0) + 1.) * .5 * (H - 1)
    x = tl.minimum(tl.maximum(x, 0.), W - 1.)
    y = tl.minimum(tl.maximum(y, 0.), H - 1.)
    x0, y0 = x.to(tl.int32), y.to(tl.int32)
    x1, y1 = tl.minimum(x0 + 1, W - 1), tl.minimum(y0 + 1, H - 1)
    base = OBS + batch * STRIDE + START
    a = tl.load(base + y0 * W + x0, j < CELLS, 0)
    b = tl.load(base + y0 * W + x1, j < CELLS, 0)
    c = tl.load(base + y1 * W + x0, j < CELLS, 0)
    d = tl.load(base + y1 * W + x1, j < CELLS, 0)
    dx, dy = x - x0, y - y0
    lum = a * (1-dx) * (1-dy) + b * dx * (1-dy) + c * (1-dx) * dy + d * dx * dy
    z = GAIN * (1. - 2. * lum)
    drive = 2. / (1. + tl.exp(-2. * z)) - 1.
    pos = tl.load(POS + j, j < CELLS, 0)
    tl.store(OUT + batch * PORTS + pos, drive, j < CELLS)


def retinal_scatter(observations, grid, positions, drive, start, height, width, gain):
    if torch.is_grad_enabled():
        raise RuntimeError('Fused retinal input requires no_grad')
    if any(not t.is_cuda or t.device != observations.device for t in (observations, grid, positions, drive)):
        raise ValueError('Fused retinal input requires tensors on one CUDA device')
    if any(t.dtype != torch.float32 for t in (observations, grid, drive)):
        raise ValueError('Fused retinal input requires FP32')
    if observations.stride(1) != 1:
        observations = observations.contiguous()
    _retinal[(observations.shape[0], triton.cdiv(positions.numel(), 256))](
        observations, grid, positions, drive, observations.stride(0), start,
        height, width, positions.numel(), drive.shape[1], gain, 256,
        enable_fp_fusion=False,
    )
    return drive
