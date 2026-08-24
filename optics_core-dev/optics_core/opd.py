from __future__ import annotations

import torch

from .rays import TraceResult
from .sampling import SamplingResult


def chief_ray_referenced_opd(result: TraceResult, sample: SamplingResult) -> torch.Tensor:
    """把追迹得到的绝对光程转换为主光线参考 OPD，单位 mm。"""
    if result.rays.opl is None:
        raise ValueError("chief_ray_referenced_opd requires traced OPL.")

    opl = torch.as_tensor(result.rays.opl, dtype=torch.float64)
    chief_opl = opl[..., int(sample.chief_ray_index)].unsqueeze(-1)
    return torch.where(torch.isfinite(opl), opl - chief_opl, torch.full_like(opl, torch.nan))
