"""Correctness tests for the JIT ``moe_topk_sum`` kernel.

``moe_topk_sum`` computes ``out[M, K] = in[M, topk, K].sum(dim=1)`` for
contiguous bf16 tensors. Two failure modes are pinned:

* the math -- the kernel accumulates the ``topk`` rows in fp32 in row order
  before casting to bf16, so it must match a torch reference bit-for-bit; and
* the launch geometry -- the token dim ``M`` used to map 1:1 onto ``grid.y``,
  which overflows CUDA's 65535 ``gridDim.y`` limit once ``M > 65535`` and
  fails with ``CUDA error: invalid argument``. The kernel now splits ``M``
  across ``grid.y``/``grid.z``, so ``M`` on both sides of 65535 is covered.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"


def _run(m: int, topk: int, k: int):
    x = torch.randn(m, topk, k, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty(m, k, device=DEVICE, dtype=torch.bfloat16)
    return moe_topk_sum(x, out), x


# k spans a single 8-bf16 vector up to the deepseek-v4 hidden dim (7168), whose
# K/8 outgrows one 256-thread block and exercises a multi-block grid.x.
@pytest.mark.parametrize(
    "m, topk, k",
    [
        (1, 1, 8),
        (2, 16, 7168),
        (200, 8, 512),
    ],
)
def test_topk_sum_vs_torch(m, topk, k):
    out, x = _run(m, topk, k)
    # Bit-identical: both sides sum the topk rows in fp32 in row order, then
    # cast to bf16, so there is no room for tolerance.
    assert torch.equal(out, x.float().sum(1).to(torch.bfloat16))


# 65535 is the largest single-grid.y launch; 65536 forces the first grid.z
# split; 70000 approximates a DP-synced 60K-token prefill batch.
@pytest.mark.parametrize("m", [65535, 65536, 70000])
def test_topk_sum_large_m(m):
    out, x = _run(m, topk=4, k=512)
    assert torch.equal(out, x.float().sum(1).to(torch.bfloat16))


def test_topk_sum_zero_tokens():
    out, _ = _run(m=0, topk=2, k=512)
    assert out.shape == (0, 512)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
