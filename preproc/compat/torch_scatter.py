"""Pure-PyTorch stand-in for the two ``torch_scatter`` ops DROID-SLAM uses.

``torch_scatter`` is a compiled extension pinned to an exact torch build, and no wheel
exists for torch>=2.14+cu130 -- the combination Blackwell (sm_120) requires. Building it
from source would mean compiling CUDA kernels we do not need: DROID-SLAM imports only
``scatter_sum`` and ``scatter_mean``, and upstream implements BOTH as composite ops in
plain PyTorch (``torch_scatter/composite``-style, on top of ``Tensor.scatter_add_``),
not as custom kernels. So this is the same computation, not an approximation of it.

The implementations below follow upstream's reference code so that edge cases match:
``dim_size=None`` sizing, the ``count<1 -> 1`` guard in the mean, and the index-dim
clamp for indices of lower rank than ``src``.

Put this directory on ``PYTHONPATH`` to use it. It SHADOWS a real ``torch_scatter``
install, so drop it from the path if you ever build the genuine package.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["scatter_sum", "scatter_mean", "broadcast"]


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """Expand `src` so it can index/divide `other` along `dim`."""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand_as(other)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is not None:
        return out.scatter_add_(dim, index, src)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1                      # empty bins: divide by 1, leaving the 0 sum
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out
