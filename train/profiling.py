from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch


def nvtx_enabled() -> bool:
    return os.environ.get("MJAI_ENABLE_NVTX", "1") != "0"


@contextmanager
def profile_scope(label: str) -> Iterator[None]:
    with torch.autograd.profiler.record_function(label):
        pushed = False
        if nvtx_enabled():
            try:
                torch.cuda.nvtx.range_push(label)
                pushed = True
            except Exception:
                pushed = False
        try:
            yield
        finally:
            if pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass