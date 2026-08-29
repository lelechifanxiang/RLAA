from __future__ import annotations

import torch

from tests.zemax.heterogeneous_first_order import load_first_order_reference
from tests.zemax.heterogeneous_spot import build_heterogeneous_system, load_heterogeneous_specs


def test_five_design_first_order_matches_zemax_reference() -> None:
    """验证五文件异构 batch 的必需一阶量。"""
    reference = load_first_order_reference()
    systems = reference["systems"]
    result = build_heterogeneous_system(load_heterogeneous_specs()).prepare().first_order_data

    expected = {
        "effl": torch.tensor([item["effl_mm"] for item in systems], dtype=torch.float64),
        "working_f_number": torch.tensor([item["working_f_number"] for item in systems], dtype=torch.float64),
        "ttl": torch.tensor([item["ttl_mm"] for item in systems], dtype=torch.float64),
        "image_plane_distance": torch.tensor(
            [item["image_plane_distance_mm"] for item in systems], dtype=torch.float64
        ),
        "bfl": torch.tensor([item["bfl_mm"] for item in systems], dtype=torch.float64),
    }
    for design_index, item in enumerate(systems):
        print(
            f"{item['zmx_file']}: EFL={result.effl[design_index].item():.9f} mm, "
            f"WFNO={result.working_f_number[design_index].item():.9f}, "
            f"TTL={result.ttl[design_index].item():.9f} mm, "
            f"BFL={result.bfl[design_index].item():.9f} mm"
        )

    assert result.valid.tolist() == [True] * len(systems)
    torch.testing.assert_close(result.effl, expected["effl"], atol=1e-9, rtol=0.0)
    torch.testing.assert_close(result.working_f_number, expected["working_f_number"], atol=1e-9, rtol=0.0)
    torch.testing.assert_close(result.ttl, expected["ttl"], atol=5e-5, rtol=0.0)
    torch.testing.assert_close(result.image_plane_distance, expected["image_plane_distance"], atol=1e-12, rtol=0.0)
    torch.testing.assert_close(result.bfl, expected["bfl"], atol=5e-5, rtol=0.0)
