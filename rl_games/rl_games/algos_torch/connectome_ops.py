"""Transient graph plans and autograd bridge for the optional cuSPARSE backend.

No command-line interface: selected by the actor's YAML operator_backend.
The extension is built lazily into PyTorch's external extension cache.
"""

import threading
from pathlib import Path

import torch


class Graph:
    def __init__(self, crow, col):
        self.crow, self.col = crow, col
        self.n = crow.numel() - 1
        self.rows = torch.repeat_interleave(
            torch.arange(self.n, device=col.device), crow[1:] - crow[:-1]
        )
        self.permutation = torch.argsort(col * self.n + self.rows, stable=True)
        self.tcol = self.rows[self.permutation].contiguous()
        counts = torch.bincount(col, minlength=self.n)
        self.tcrow = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        self.plans = {}
        self.population_maps = {}

    def plan(self, batch, options, transpose=False):
        if self.col.device.type != "cuda":
            raise RuntimeError("operator_backend: cusparse requires CUDA")
        algorithm = options.get("cusparse_algorithm", "alg2")
        if algorithm not in {"alg1", "alg2", "alg3"}:
            raise ValueError("cusparse_algorithm must be alg1, alg2, or alg3")
        layout = options.get(
            "cusparse_layout", "column" if algorithm == "alg1" else "row"
        )
        if layout not in {"row", "column"}:
            raise ValueError("cusparse_layout must be row or column")
        stream = torch.cuda.current_stream(self.col.device).cuda_stream
        key = (threading.get_ident(), stream, batch, algorithm, layout, transpose)
        if key not in self.plans:
            self.plans[key] = _extension().Plan(
                self.tcrow if transpose else self.crow,
                self.tcol if transpose else self.col,
                batch,
                int(algorithm[-1]),
                layout == "column",
            )
        return self.plans[key], layout


_module = None


def _extension():
    global _module
    if _module is None:
        from torch.utils.cpp_extension import CUDA_HOME, load

        if CUDA_HOME is None:
            raise RuntimeError(
                "cusparse requires a CUDA toolkit matching PyTorch, including headers and a C++ compiler"
            )
        try:
            _module = load(
                name="connectome_cusparse_v1",
                sources=[str(Path(__file__).with_name("connectome_cusparse.cpp"))],
                extra_cflags=["-O3"],
                extra_ldflags=["-lcusparse"],
                with_cuda=True,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not build the connectome cuSPARSE extension; check CUDA_HOME, compiler, ninja, and matching PyTorch/CUDA libraries"
            ) from exc
    return _module


def _layout(x, layout):
    return x.t().contiguous().t() if layout == "column" else x.contiguous()


class _SpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, x, graph, options):
        if values.dtype != torch.float32 or x.dtype != torch.float32:
            raise ValueError("cuSPARSE recurrence requires FP32")
        plan, layout = graph.plan(x.shape[1], options)
        x = _layout(x, layout)
        ctx.graph, ctx.options = graph, options
        ctx.save_for_backward(values, x)
        return plan.mm(values.detach(), x.detach())

    @staticmethod
    def backward(ctx, grad):
        values, x = ctx.saved_tensors
        graph, options = ctx.graph, ctx.options
        dx = dv = None
        if ctx.needs_input_grad[1]:
            plan, layout = graph.plan(x.shape[1], options, transpose=True)
            dx = plan.mm(values[graph.permutation], _layout(grad, layout))
        if ctx.needs_input_grad[0]:
            plan, _ = graph.plan(x.shape[1], options)
            dv = plan.edge_grad(grad.contiguous(), x.contiguous())
        return dv, dx, None, None


def cusparse_mm(graph, values, x, options=None):
    return _SpMM.apply(values, x, graph, options or {})
