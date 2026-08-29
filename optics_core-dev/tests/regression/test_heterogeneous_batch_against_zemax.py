from __future__ import annotations

import pytest
import torch

from tests.zemax.heterogeneous_spot import (
    ZMX_FILENAMES,
    build_heterogeneous_system,
    load_heterogeneous_specs,
    load_spot_rms_reference,
    run_heterogeneous_spot_rms,
    topology_signature,
)


pytestmark = [pytest.mark.regression, pytest.mark.zemax]


def test_five_design_parallel_spot_rms_matches_zemax() -> None:
    """验证五文件异构 batch 的 RMS 与 Zemax 一致。"""
    specs = load_heterogeneous_specs()
    assert len({topology_signature(spec) for spec in specs}) == 1

    reference = load_spot_rms_reference()
    system = build_heterogeneous_system(specs)
    batch_rms = run_heterogeneous_spot_rms(system, specs, reference).cpu()
    zemax_rms = torch.tensor(
        [item["zemax_rms_radius_um"] for item in reference["systems"]],
        dtype=torch.float64,
    )

    for design_index, name in enumerate(ZMX_FILENAMES):
        zemax_error = torch.abs(batch_rms[design_index] - zemax_rms[design_index])
        print(f"{name} Zemax RMS (um): {zemax_rms[design_index].tolist()}")
        print(f"{name} 并行 RMS (um): {batch_rms[design_index].tolist()}")
        print(f"{name} 并行/Zemax 最大差异 (um): {zemax_error.max().item():.3e}")

    torch.testing.assert_close(batch_rms, zemax_rms, atol=1e-3, rtol=0.0)
