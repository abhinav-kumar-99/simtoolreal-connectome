"""FP32 CSR recurrent-step fusion and its first-order training backward.

Selected through operator_backend: triton_fused; no standalone CLI.
Dense adapter GEMMs remain in PyTorch. Internal states are neuron-major so
batch lanes access consecutive memory; the public state remains (B, N).
"""

from contextlib import nullcontext

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError(
        "triton_fused requires the Triton version supplied with PyTorch"
    ) from exc


@triton.jit
def _sparse_step(
    CROW,
    COL,
    VAL,
    H,
    GI,
    GO,
    LEAK,
    BIAS,
    S,
    D,
    SMAP,
    DMAP,
    OUT,
    REC,
    Z,
    B: tl.constexpr,
    BETA: tl.constexpr,
    FUSED: tl.constexpr,
    KB: tl.constexpr = 32,
    EB: tl.constexpr = 32,
):
    row = tl.program_id(0)
    b = tl.program_id(1) * KB + tl.arange(0, KB)
    e = tl.arange(0, EB)
    start = tl.load(CROW + row)
    end = tl.load(CROW + row + 1)
    acc = tl.full((KB,), 0, tl.float32)
    for first in range(start, end, EB):
        edge = first + e
        src = tl.load(COL + edge, edge < end, 0)
        w = tl.load(VAL + edge, edge < end, 0)
        h = tl.load(
            H + src[:, None] * B + b[None, :],
            (edge[:, None] < end) & (b[None, :] < B),
            0,
        )
        if FUSED:
            gain = tl.load(GO + src, edge < end, 0)
            w = w * gain
        acc += tl.sum(w[:, None] * h, axis=0)
    if FUSED:
        si = tl.load(SMAP + row)
        di = tl.load(DMAP + row)
        drive = tl.load(S + si * B + b, (si >= 0) & (b < B), 0)
        drive += tl.load(D + di * B + b, (di >= 0) & (b < B), 0)
        pre = BETA * tl.load(GI + row) * acc + drive + tl.load(BIAS + row)
        z = 2.0 / (1.0 + tl.exp(2.0 * -pre)) - 1.0
        leak = tl.load(LEAK + row)
        h0 = tl.load(H + row * B + b, b < B, 0)
        out = (1.0 - leak) * h0 + leak * z
        tl.store(REC + row * B + b, acc, b < B)
        tl.store(Z + row * B + b, z, b < B)
    else:
        out = acc
    tl.store(OUT + row * B + b, out, b < B)


@triton.jit
def _point_backward(
    DY,
    H,
    Z,
    REC,
    GI,
    LEAK,
    DP,
    DR,
    DIRECT,
    DGI,
    DL,
    DB,
    B: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    b = tl.arange(0, BLOCK)
    offset = row * B + b
    dy = tl.load(DY + offset, b < B, 0)
    z = tl.load(Z + offset, b < B, 0)
    h = tl.load(H + offset, b < B, 0)
    rec = tl.load(REC + offset, b < B, 0)
    leak = tl.load(LEAK + row)
    dp = dy * leak * (1.0 - z * z)
    tl.store(DP + offset, dp, b < B)
    tl.store(DR + offset, dp * BETA * tl.load(GI + row), b < B)
    tl.store(DIRECT + offset, dy * (1.0 - leak), b < B)
    tl.store(DGI + row, tl.sum(dp * BETA * rec, 0))
    tl.store(DL + row, tl.sum(dy * (z - h), 0))
    tl.store(DB + row, tl.sum(dp, 0))


@triton.jit
def _edge_backward(
    ROWS,
    COL,
    DR,
    H,
    GO,
    PARTS,
    B: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr = 256,
):
    edge = tl.program_id(0)
    tile = tl.program_id(1)
    b = tile * BLOCK + tl.arange(0, BLOCK)
    dst = tl.load(ROWS + edge)
    src = tl.load(COL + edge)
    d = tl.load(DR + dst * B + b, b < B, 0)
    h = tl.load(H + src * B + b, b < B, 0)
    value = tl.sum(d * h, 0) * tl.load(GO + src)
    tl.store(PARTS + edge * TILES + tile, value)


class _Step(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values,
        hidden,
        gi,
        go,
        leak,
        bias,
        sensory,
        descending,
        graph,
        sensory_indices,
        descending_indices,
        beta,
    ):
        if not hidden.is_cuda:
            raise RuntimeError("operator_backend: triton_fused requires CUDA")
        if any(
            t.dtype != torch.float32
            for t in (values, hidden, gi, go, leak, bias, sensory, descending)
        ):
            raise ValueError("Triton recurrence requires FP32")
        h, s, d = (x.t().contiguous() for x in (hidden, sensory, descending))
        n, b = h.shape
        key = (sensory_indices.data_ptr(), descending_indices.data_ptr())
        if key not in graph.population_maps:
            maps = []
            for indices in (sensory_indices, descending_indices):
                mapping = torch.full((n,), -1, dtype=torch.int64, device=h.device)
                mapping[indices] = torch.arange(indices.numel(), device=h.device)
                maps.append(mapping)
            graph.population_maps[key] = maps
        smap, dmap = graph.population_maps[key]
        out, rec, z = (torch.empty_like(h) for _ in range(3))
        _sparse_step[(n, triton.cdiv(b, 32))](
            graph.crow,
            graph.col,
            values,
            h,
            gi,
            go,
            leak,
            bias,
            s,
            d,
            smap,
            dmap,
            out,
            rec,
            z,
            b,
            beta,
            True,
        )
        ctx.graph, ctx.beta = graph, beta
        ctx.save_for_backward(
            values, h, gi, go, leak, rec, z, sensory_indices, descending_indices
        )
        return out.t()

    @staticmethod
    def backward(ctx, grad):
        values, h, gi, go, leak, rec, z, si, di = ctx.saved_tensors
        graph = ctx.graph
        dy = grad.t().contiguous()
        n, b = h.shape
        dp, dr, direct = (torch.empty_like(h) for _ in range(3))
        dgi, dl, db = (torch.empty_like(gi) for _ in range(3))
        _point_backward[(n,)](
            dy,
            h,
            z,
            rec,
            gi,
            leak,
            dp,
            dr,
            direct,
            dgi,
            dl,
            db,
            b,
            ctx.beta,
            triton.next_power_of_2(b),
        )
        dh = dgo = dv = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[3]:
            dscaled = torch.empty_like(h)
            tv = values[graph.permutation]
            _sparse_step[(n, triton.cdiv(b, 32))](
                graph.tcrow,
                graph.tcol,
                tv,
                dr,
                gi,
                go,
                leak,
                gi,
                h,
                h,
                graph.rows,
                graph.rows,
                dscaled,
                rec,
                z,
                b,
                ctx.beta,
                False,
            )
            if ctx.needs_input_grad[1]:
                dh = (direct + dscaled * go[:, None]).t()
            if ctx.needs_input_grad[3]:
                dgo = (dscaled * h).sum(1)
        if ctx.needs_input_grad[0]:
            tiles = triton.cdiv(b, 256)
            parts = torch.empty((values.numel(), tiles), device=h.device, dtype=h.dtype)
            _edge_backward[(values.numel(), tiles)](
                graph.rows, graph.col, dr, h, go, parts, b, tiles
            )
            dv = parts.sum(1)
        return (
            dv,
            dh,
            dgi if ctx.needs_input_grad[2] else None,
            dgo,
            dl if ctx.needs_input_grad[4] else None,
            db if ctx.needs_input_grad[5] else None,
            dp[si].t(),
            dp[di].t(),
            None,
            None,
            None,
            None,
        )


def fused_step(
    graph,
    values,
    hidden,
    gi,
    go,
    leak,
    bias,
    sensory,
    descending,
    sensory_indices,
    descending_indices,
    beta,
):
    with torch.cuda.device(hidden.device) if hidden.is_cuda else nullcontext():
        return _Step.apply(
            values,
            hidden,
            gi,
            go,
            leak,
            bias,
            sensory,
            descending,
            graph,
            sensory_indices,
            descending_indices,
            beta,
        )
