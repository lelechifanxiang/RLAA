from __future__ import annotations

import pytest
import torch

import optics_core as oc
from optics_core.opd import chief_ray_referenced_opd


pytestmark = pytest.mark.contract


def test_chief_ray_referenced_opd_uses_last_ray_dimension() -> None:
    opl = torch.tensor([[[[10.2, 10.0, 10.5]]]], dtype=torch.float64)
    rays = oc.RayBundle(
        x=torch.zeros_like(opl),
        y=torch.zeros_like(opl),
        z=torch.zeros_like(opl),
        l=torch.zeros_like(opl),
        m=torch.zeros_like(opl),
        n=torch.ones_like(opl),
        wavelength_index=torch.zeros_like(opl, dtype=torch.int64),
        opl=opl,
    )
    result = oc.TraceResult(rays=rays, valid=torch.ones_like(opl, dtype=torch.bool))
    sample = oc.SamplingResult(chief_ray_index=1)

    opd = chief_ray_referenced_opd(result, sample)

    torch.testing.assert_close(opd, torch.tensor([[[[0.2, 0.0, 0.5]]]], dtype=torch.float64))
