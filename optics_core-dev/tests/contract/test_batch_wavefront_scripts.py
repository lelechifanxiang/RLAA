from __future__ import annotations

from pathlib import Path

import torch

import examples.batch_wavefront as batch_wavefront


CASE2_ZMX_PATH = Path("tests/zemax/zmx_files/case2_3p_center.zmx")


def test_batch_wavefront_exports_finite_object_afocal_multi_cb_images(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """物像颠倒且包含多组 CB 的系统应能批量导出波前图。"""
    output_dir = tmp_path / "batch_wavefront"
    monkeypatch.setattr(batch_wavefront, "ZMX_PATH", CASE2_ZMX_PATH)
    monkeypatch.setattr(batch_wavefront, "COORDINATE_BREAK_PAIRS", ((1, 4), (7, 10)))

    batch_wavefront.run_batch_wavefront_analysis(
        device=torch.device("cpu"),
        design_count=2,
        random_seed=0,
        sample_count=8,
        output_dir=output_dir,
        field_indices=(0,),
        wavelength_indices=(0,),
    )

    image_paths = sorted(output_dir.glob("wavefront_design_*_field_0_wave_0.png"))
    assert len(image_paths) == 2
    assert all(path.stat().st_size > 0 for path in image_paths)
