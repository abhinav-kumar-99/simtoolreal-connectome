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
def _topology_uniform(SEEDS, canonical, batch_lane, B: tl.constexpr):
    """Wang-hash U(0,1) keyed only by int64 seed and canonical edge ID."""
    seed = tl.load(
        SEEDS + batch_lane,
        mask=(batch_lane >= 0) & (batch_lane < B),
        other=0,
    )
    low = seed.to(tl.uint64).to(tl.uint32)
    high = (seed.to(tl.uint64) >> 32).to(tl.uint32)
    rotated = (high << 16) | (high >> 16)
    x = canonical.to(tl.uint32) ^ low ^ rotated ^ 0x9E3779B9
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27D4EB2D
    x = x ^ (x >> 15)
    return (x.to(tl.float32) + 0.5) * 2.3283064365386963e-10


@triton.jit
def _sparse_step(
    CROW,
    COL,
    VAL,
    H,
    GI,
    GO,
    LEAK,
    A,
    BIAS,
    S,
    D,
    SMAP,
    DMAP,
    OUT,
    REC,
    Z,
    PRE_DRIVE,
    B: tl.constexpr,
    BETA: tl.constexpr,
    FUSED: tl.constexpr,
    KB: tl.constexpr = 32,
    EB: tl.constexpr = 32,
    SAVE_BACKWARD: tl.constexpr = True,
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
        pre_drive = BETA * tl.load(GI + row) * acc + drive
        a = tl.load(A + row)
        pre = a * pre_drive + tl.load(BIAS + row)
        z = 2.0 / (1.0 + tl.exp(2.0 * -pre)) - 1.0
        leak = tl.load(LEAK + row)
        h0 = tl.load(H + row * B + b, b < B, 0)
        out = (1.0 - leak) * h0 + leak * z
        if SAVE_BACKWARD:
            tl.store(REC + row * B + b, acc, b < B)
            tl.store(Z + row * B + b, z, b < B)
            tl.store(PRE_DRIVE + row * B + b, pre_drive, b < B)
    else:
        out = acc
    tl.store(OUT + row * B + b, out, b < B)


@triton.jit
def _gated_sparse_step(
    CROW,
    COL,
    CANONICAL_EDGE,
    VAL,
    LOGIT,
    TOPOLOGY_SEED,
    H,
    GI,
    GO,
    LEAK,
    A,
    BIAS,
    S,
    D,
    SMAP,
    DMAP,
    OUT,
    REC,
    Z,
    PRE_DRIVE,
    B: tl.constexpr,
    BETA: tl.constexpr,
    FUSED: tl.constexpr,
    KB: tl.constexpr = 32,
    EB: tl.constexpr = 32,
    SAVE_BACKWARD: tl.constexpr = True,
):
    row = tl.program_id(0)
    b = tl.program_id(1) * KB + tl.arange(0, KB)
    e = tl.arange(0, EB)
    start = tl.load(CROW + row)
    end = tl.load(CROW + row + 1)
    acc = tl.full((KB,), 0, tl.float32)
    for first in range(start, end, EB):
        edge = first + e
        valid_edge = edge < end
        src = tl.load(COL + edge, valid_edge, 0)
        canonical = tl.load(CANONICAL_EDGE + edge, valid_edge, 0)
        value = tl.load(VAL + edge, valid_edge, 0.0)
        logit = tl.load(LOGIT + edge, valid_edge, 0.0)
        probability = 1.0 / (1.0 + tl.exp(-logit))
        uniform = _topology_uniform(
            TOPOLOGY_SEED, canonical[:, None], b[None, :], B
        )
        gate = (uniform < probability[:, None]).to(tl.float32)
        h = tl.load(
            H + src[:, None] * B + b[None, :],
            valid_edge[:, None] & (b[None, :] < B),
            0.0,
        )
        weighted = value[:, None] * gate
        if FUSED:
            gain = tl.load(GO + src, valid_edge, 0.0)
            weighted = weighted * gain[:, None]
        acc += tl.sum(weighted * h, axis=0)
    if FUSED:
        si = tl.load(SMAP + row)
        di = tl.load(DMAP + row)
        drive = tl.load(S + si * B + b, (si >= 0) & (b < B), 0.0)
        drive += tl.load(D + di * B + b, (di >= 0) & (b < B), 0.0)
        pre_drive = BETA * tl.load(GI + row) * acc + drive
        a = tl.load(A + row)
        pre = a * pre_drive + tl.load(BIAS + row)
        z = 2.0 / (1.0 + tl.exp(-2.0 * pre)) - 1.0
        leak = tl.load(LEAK + row)
        h0 = tl.load(H + row * B + b, b < B, 0.0)
        out = (1.0 - leak) * h0 + leak * z
        if SAVE_BACKWARD:
            tl.store(REC + row * B + b, acc, b < B)
            tl.store(Z + row * B + b, z, b < B)
            tl.store(PRE_DRIVE + row * B + b, pre_drive, b < B)
    else:
        out = acc
    tl.store(OUT + row * B + b, out, b < B)


@triton.jit
def _sparse_lif_step(
    CROW,
    COL,
    VAL,
    MEMBRANE,
    REFRACTORY,
    SPIKES,
    GI,
    GO,
    BIAS,
    S,
    D,
    SMAP,
    DMAP,
    MEMBRANE_OUT,
    REFRACTORY_OUT,
    SPIKES_OUT,
    B: tl.constexpr,
    BETA: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
    MEMBRANE_DECAY: tl.constexpr,
    THRESHOLD: tl.constexpr,
    REFRACTORY_MS: tl.constexpr,
    DT_MS: tl.constexpr,
    KB: tl.constexpr = 32,
    EB: tl.constexpr = 32,
):
    row = tl.program_id(0)
    b = tl.program_id(1) * KB + tl.arange(0, KB)
    e = tl.arange(0, EB)
    start = tl.load(CROW + row)
    end = tl.load(CROW + row + 1)
    recurrent = tl.full((KB,), 0, tl.float32)
    for first in range(start, end, EB):
        edge = first + e
        src = tl.load(COL + edge, edge < end, 0)
        weight = tl.load(VAL + edge, edge < end, 0)
        source_spike = tl.load(
            SPIKES + src[:, None] * B + b[None, :],
            (edge[:, None] < end) & (b[None, :] < B),
            0.0,
        )
        gain = tl.load(GO + src, edge < end, 0.0)
        recurrent += tl.sum(weight[:, None] * gain[:, None] * source_spike, axis=0)

    sensory_index = tl.load(SMAP + row)
    descending_index = tl.load(DMAP + row)
    drive = tl.load(
        S + sensory_index * B + b,
        (sensory_index >= 0) & (b < B),
        0.0,
    )
    drive += tl.load(
        D + descending_index * B + b,
        (descending_index >= 0) & (b < B),
        0.0,
    )
    current = (
        BETA * tl.load(GI + row) * recurrent
        + INPUT_SCALE * drive
        + tl.load(BIAS + row)
    )
    offset = row * B + b
    membrane = tl.load(MEMBRANE + offset, b < B, 0.0)
    remaining = tl.maximum(
        tl.load(REFRACTORY + offset, b < B, 0.0) - DT_MS, 0.0
    )
    available = remaining <= 0.0
    candidate = MEMBRANE_DECAY * membrane + (1.0 - MEMBRANE_DECAY) * current
    fired = available & (candidate >= THRESHOLD)
    membrane_out = tl.where(available & ~fired, candidate, 0.0)
    refractory_out = tl.where(fired, REFRACTORY_MS, remaining)
    tl.store(MEMBRANE_OUT + offset, membrane_out, b < B)
    tl.store(REFRACTORY_OUT + offset, refractory_out, b < B)
    tl.store(SPIKES_OUT + offset, fired.to(tl.float32), b < B)


@triton.jit
def _point_backward(
    DY,
    H,
    Z,
    REC,
    PRE_DRIVE,
    GI,
    LEAK,
    A,
    DP,
    DR,
    DIRECT,
    DGI,
    DL,
    DA,
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
    pre_drive = tl.load(PRE_DRIVE + offset, b < B, 0)
    leak = tl.load(LEAK + row)
    a = tl.load(A + row)
    dp = dy * leak * (1.0 - z * z)
    d_pre_drive = dp * a
    tl.store(DP + offset, d_pre_drive, b < B)
    tl.store(DR + offset, d_pre_drive * BETA * tl.load(GI + row), b < B)
    tl.store(DIRECT + offset, dy * (1.0 - leak), b < B)
    tl.store(DGI + row, tl.sum(d_pre_drive * BETA * rec, 0))
    tl.store(DL + row, tl.sum(dy * (z - h), 0))
    tl.store(DA + row, tl.sum(dp * pre_drive, 0))
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


@triton.jit
def _gated_edge_backward(
    ROWS,
    COL,
    CANONICAL_EDGE,
    VAL,
    LOGIT,
    TOPOLOGY_SEED,
    DR,
    H,
    GO,
    VALUE_PARTS,
    LOGIT_PARTS,
    B: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr = 256,
):
    edge = tl.program_id(0)
    tile = tl.program_id(1)
    b = tile * BLOCK + tl.arange(0, BLOCK)
    dst = tl.load(ROWS + edge)
    src = tl.load(COL + edge)
    canonical = tl.load(CANONICAL_EDGE + edge)
    logit = tl.load(LOGIT + edge)
    probability = 1.0 / (1.0 + tl.exp(-logit))
    uniform = _topology_uniform(TOPOLOGY_SEED, canonical, b, B)
    gate = (uniform < probability).to(tl.float32)
    d = tl.load(DR + dst * B + b, b < B, 0.0)
    h = tl.load(H + src * B + b, b < B, 0.0)
    base = d * h * tl.load(GO + src)
    value = tl.load(VAL + edge)
    value_grad = tl.sum(base * gate, 0)
    logit_grad = tl.sum(base * value * probability * (1.0 - probability), 0)
    offset = edge * TILES + tile
    tl.store(VALUE_PARTS + offset, value_grad)
    tl.store(LOGIT_PARTS + offset, logit_grad)


class _Step(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values,
        hidden,
        gi,
        go,
        leak,
        intrinsic,
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
            for t in (
                values,
                hidden,
                gi,
                go,
                leak,
                intrinsic,
                bias,
                sensory,
                descending,
            )
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
        out, rec, z, pre_drive = (torch.empty_like(h) for _ in range(4))
        _sparse_step[(n, triton.cdiv(b, 32))](
            graph.crow,
            graph.col,
            values,
            h,
            gi,
            go,
            leak,
            intrinsic,
            bias,
            s,
            d,
            smap,
            dmap,
            out,
            rec,
            z,
            pre_drive,
            b,
            beta,
            True,
        )
        ctx.graph, ctx.beta = graph, beta
        ctx.save_for_backward(
            values,
            h,
            gi,
            go,
            leak,
            intrinsic,
            rec,
            z,
            pre_drive,
            sensory_indices,
            descending_indices,
        )
        return out.t()

    @staticmethod
    def backward(ctx, grad):
        values, h, gi, go, leak, intrinsic, rec, z, pre_drive, si, di = (
            ctx.saved_tensors
        )
        graph = ctx.graph
        dy = grad.t().contiguous()
        n, b = h.shape
        dp, dr, direct = (torch.empty_like(h) for _ in range(3))
        dgi, dl, da, db = (torch.empty_like(gi) for _ in range(4))
        _point_backward[(n,)](
            dy,
            h,
            z,
            rec,
            pre_drive,
            gi,
            leak,
            intrinsic,
            dp,
            dr,
            direct,
            dgi,
            dl,
            da,
            db,
            b,
            ctx.beta,
            triton.next_power_of_2(b),
        )
        dh = dgo = dv = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[3]:
            dscaled = torch.empty_like(h)
            tv = values[graph.permutation]
            # Transpose SpMM path: FUSED=False returns only the sparse product.
            # Dummy A/BIAS/PRE_DRIVE pointers satisfy the kernel signature.
            _sparse_step[(n, triton.cdiv(b, 32))](
                graph.tcrow,
                graph.tcol,
                tv,
                dr,
                gi,
                go,
                leak,
                intrinsic,
                gi,
                h,
                h,
                graph.rows,
                graph.rows,
                dscaled,
                rec,
                z,
                pre_drive,
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
            da if ctx.needs_input_grad[5] else None,
            db if ctx.needs_input_grad[6] else None,
            dp[si].t(),
            dp[di].t(),
            None,
            None,
            None,
            None,
        )


class _GatedStep(torch.autograd.Function):
    """Fused recurrence with lane-specific hard Bernoulli topology gates."""

    @staticmethod
    def forward(
        ctx,
        values,
        posterior_logits,
        topology_seeds,
        hidden,
        gi,
        go,
        leak,
        intrinsic,
        bias,
        sensory,
        descending,
        graph,
        sensory_indices,
        descending_indices,
        beta,
    ):
        if not hidden.is_cuda:
            raise RuntimeError("probabilistic Triton recurrence requires CUDA")
        if any(
            tensor.dtype != torch.float32
            for tensor in (
                values,
                posterior_logits,
                hidden,
                gi,
                go,
                leak,
                intrinsic,
                bias,
                sensory,
                descending,
            )
        ):
            raise ValueError("probabilistic Triton recurrence requires FP32")
        if topology_seeds.dtype != torch.int64:
            raise ValueError("topology seeds must be int64")
        h, s, d = (x.t().contiguous() for x in (hidden, sensory, descending))
        topology_seeds = topology_seeds.reshape(-1).contiguous()
        n, b = h.shape
        if topology_seeds.numel() != b:
            raise ValueError(
                f"expected {b} topology seeds, got {topology_seeds.numel()}"
            )
        if values.shape != graph.col.shape or posterior_logits.shape != values.shape:
            raise ValueError("gated values/logits must match the candidate graph")
        key = (sensory_indices.data_ptr(), descending_indices.data_ptr())
        if key not in graph.population_maps:
            maps = []
            for indices in (sensory_indices, descending_indices):
                mapping = torch.full(
                    (n,), -1, dtype=torch.int64, device=h.device
                )
                mapping[indices] = torch.arange(
                    indices.numel(), device=h.device
                )
                maps.append(mapping)
            graph.population_maps[key] = maps
        smap, dmap = graph.population_maps[key]
        out, rec, z, pre_drive = (torch.empty_like(h) for _ in range(4))
        _gated_sparse_step[(n, triton.cdiv(b, 32))](
            graph.crow,
            graph.col,
            graph.canonical_edge_ids,
            values,
            posterior_logits,
            topology_seeds,
            h,
            gi,
            go,
            leak,
            intrinsic,
            bias,
            s,
            d,
            smap,
            dmap,
            out,
            rec,
            z,
            pre_drive,
            b,
            beta,
            True,
        )
        ctx.graph, ctx.beta = graph, beta
        ctx.save_for_backward(
            values,
            posterior_logits,
            topology_seeds,
            h,
            gi,
            go,
            leak,
            intrinsic,
            rec,
            z,
            pre_drive,
            sensory_indices,
            descending_indices,
        )
        return out.t()

    @staticmethod
    def backward(ctx, grad):
        (
            values,
            posterior_logits,
            topology_seeds,
            h,
            gi,
            go,
            leak,
            intrinsic,
            rec,
            z,
            pre_drive,
            sensory_indices,
            descending_indices,
        ) = ctx.saved_tensors
        graph = ctx.graph
        dy = grad.t().contiguous()
        n, b = h.shape
        dp, dr, direct = (torch.empty_like(h) for _ in range(3))
        dgi, dleak, dintrinsic, dbias = (
            torch.empty_like(gi) for _ in range(4)
        )
        _point_backward[(n,)](
            dy,
            h,
            z,
            rec,
            pre_drive,
            gi,
            leak,
            intrinsic,
            dp,
            dr,
            direct,
            dgi,
            dleak,
            dintrinsic,
            dbias,
            b,
            ctx.beta,
            triton.next_power_of_2(b),
        )

        dh = dgo = None
        if ctx.needs_input_grad[3] or ctx.needs_input_grad[5]:
            dscaled = torch.empty_like(h)
            permutation = graph.permutation
            _gated_sparse_step[(n, triton.cdiv(b, 32))](
                graph.tcrow,
                graph.tcol,
                graph.tcanonical_edge_ids,
                values[permutation],
                posterior_logits[permutation],
                topology_seeds,
                dr,
                gi,
                go,
                leak,
                intrinsic,
                gi,
                h,
                h,
                graph.rows,
                graph.rows,
                dscaled,
                rec,
                z,
                pre_drive,
                b,
                ctx.beta,
                False,
            )
            if ctx.needs_input_grad[3]:
                dh = (direct + dscaled * go[:, None]).t()
            if ctx.needs_input_grad[5]:
                dgo = (dscaled * h).sum(1)

        dvalues = dlogits = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            tiles = triton.cdiv(b, 256)
            value_parts = torch.empty(
                (values.numel(), tiles), device=h.device, dtype=h.dtype
            )
            logit_parts = torch.empty_like(value_parts)
            _gated_edge_backward[(values.numel(), tiles)](
                graph.rows,
                graph.col,
                graph.canonical_edge_ids,
                values,
                posterior_logits,
                topology_seeds,
                dr,
                h,
                go,
                value_parts,
                logit_parts,
                b,
                tiles,
            )
            if ctx.needs_input_grad[0]:
                dvalues = value_parts.sum(1)
            if ctx.needs_input_grad[1]:
                dlogits = logit_parts.sum(1)

        return (
            dvalues,
            dlogits,
            None,
            dh,
            dgi if ctx.needs_input_grad[4] else None,
            dgo,
            dleak if ctx.needs_input_grad[6] else None,
            dintrinsic if ctx.needs_input_grad[7] else None,
            dbias if ctx.needs_input_grad[8] else None,
            dp[sensory_indices].t(),
            dp[descending_indices].t(),
            None,
            None,
            None,
            None,
        )


class FrozenTanhRunner:
    """Reusable neuron-major buffers for a fixed, inference-only recurrence.

    Return a copy so a later call cannot overwrite a caller's recurrent state.
    CUDA graph capture covers only the neural passes, not simulation or inputs.
    """

    def __init__(self, graph, values, hidden, gi, go, leak, intrinsic, bias,
                 sensory, descending, si, di, beta, updates, capture=False):
        self.graph, self.values = graph, values
        self.gi, self.go, self.leak = gi, go, leak
        self.intrinsic, self.bias = intrinsic, bias
        self.beta, self.updates = beta, updates
        self.h = torch.empty_like(hidden.t(), memory_format=torch.contiguous_format)
        self.other = torch.empty_like(self.h)
        self.s = torch.empty_like(sensory.t(), memory_format=torch.contiguous_format)
        self.d = torch.empty_like(descending.t(), memory_format=torch.contiguous_format)
        self.maps = []
        for indices in (si, di):
            mapping = torch.full((hidden.shape[1],), -1, dtype=torch.int64, device=hidden.device)
            mapping[indices] = torch.arange(indices.numel(), device=hidden.device)
            self.maps.append(mapping)
        self.cuda_graph = None
        self.h.copy_(hidden.t())
        self.s.copy_(sensory.t())
        self.d.copy_(descending.t())
        if capture:
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                self._passes()  # compile outside capture
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph, stream=stream):
                self._passes()

    def _passes(self):
        n, b = self.h.shape
        source, dest = self.h, self.other
        for _ in range(self.updates):
            _sparse_step[(n, triton.cdiv(b, 32))](
                self.graph.crow, self.graph.col, self.values, source,
                self.gi, self.go, self.leak, self.intrinsic, self.bias,
                self.s, self.d, self.maps[0], self.maps[1], dest, dest, dest,
                dest, b, self.beta, True, SAVE_BACKWARD=False,
            )
            source, dest = dest, source
        self.output = source

    def __call__(self, hidden, sensory, descending):
        if torch.is_grad_enabled():
            raise RuntimeError('FrozenTanhRunner requires no_grad')
        self.h.copy_(hidden.t())
        self.s.copy_(sensory.t())
        self.d.copy_(descending.t())
        if self.cuda_graph is None:
            self._passes()
        else:
            self.cuda_graph.replay()
        return self.output.t().clone()


def fused_step(
    graph,
    values,
    hidden,
    gi,
    go,
    leak,
    intrinsic,
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
            intrinsic,
            bias,
            sensory,
            descending,
            graph,
            sensory_indices,
            descending_indices,
            beta,
        )


def fused_gated_step(
    graph,
    conditional_values,
    posterior_logits,
    topology_seeds,
    hidden,
    gi,
    go,
    leak,
    intrinsic,
    bias,
    sensory,
    descending,
    sensory_indices,
    descending_indices,
    beta,
):
    """Advance one sampled-topology step with straight-through logit gradients."""
    with torch.cuda.device(hidden.device) if hidden.is_cuda else nullcontext():
        return _GatedStep.apply(
            conditional_values,
            posterior_logits,
            topology_seeds,
            hidden,
            gi,
            go,
            leak,
            intrinsic,
            bias,
            sensory,
            descending,
            graph,
            sensory_indices,
            descending_indices,
            beta,
        )


def fused_lif_step(
    graph,
    values,
    membrane,
    refractory,
    spikes,
    gi,
    go,
    bias,
    sensory,
    descending,
    sensory_indices,
    descending_indices,
    beta,
    input_scale,
    membrane_decay,
    threshold,
    refractory_ms,
    dt_ms,
):
    """Advance one hard-threshold LIF step without constructing autograd state."""
    if not membrane.is_cuda:
        raise RuntimeError("operator_backend: triton_fused requires CUDA")
    tensors = (
        values,
        membrane,
        refractory,
        spikes,
        gi,
        go,
        bias,
        sensory,
        descending,
    )
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("Triton LIF recurrence requires FP32")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError(
            "Hard LIF recurrence is inference-only; run it under torch.no_grad()"
        )
    with torch.cuda.device(membrane.device):
        membrane_t, refractory_t, spikes_t, sensory_t, descending_t = (
            tensor.t().contiguous()
            for tensor in (
                membrane,
                refractory,
                spikes,
                sensory,
                descending,
            )
        )
        neuron_count, batch_size = membrane_t.shape
        key = (sensory_indices.data_ptr(), descending_indices.data_ptr())
        if key not in graph.population_maps:
            maps = []
            for indices in (sensory_indices, descending_indices):
                mapping = torch.full(
                    (neuron_count,), -1, dtype=torch.int64, device=membrane.device
                )
                mapping[indices] = torch.arange(
                    indices.numel(), device=membrane.device
                )
                maps.append(mapping)
            graph.population_maps[key] = maps
        sensory_map, descending_map = graph.population_maps[key]
        membrane_out = torch.empty_like(membrane_t)
        refractory_out = torch.empty_like(refractory_t)
        spikes_out = torch.empty_like(spikes_t)
        _sparse_lif_step[(neuron_count, triton.cdiv(batch_size, 32))](
            graph.crow,
            graph.col,
            values,
            membrane_t,
            refractory_t,
            spikes_t,
            gi,
            go,
            bias,
            sensory_t,
            descending_t,
            sensory_map,
            descending_map,
            membrane_out,
            refractory_out,
            spikes_out,
            batch_size,
            beta,
            input_scale,
            membrane_decay,
            threshold,
            refractory_ms,
            dt_ms,
        )
        return (
            membrane_out.t(),
            refractory_out.t(),
            spikes_out.t(),
        )
