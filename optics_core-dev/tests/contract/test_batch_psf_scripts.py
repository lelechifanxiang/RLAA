from __future__ import annotations

from pathlib import Path

import torch

from examples.batch_psf import run_batch_psf_analysis


def test_batch_psf_analysis_exports_field_images(tmp_path: Path) -> None:
    """批量 PSF 脚本应按 design 和 field 导出图片。"""
    output_dir = tmp_path / "batch_psf"
    summary = run_batch_psf_analysis(
        device=torch.device("cpu"),
        design_count=1,
        random_seed=0,
        pupil_sample_count=4,
        image_sample_count=4,
        output_dir=output_dir,
    )

    assert summary["analysis_type"] == "psf"
    assert summary["scan_type"] == "assembly_tolerance_monte_carlo"
    assert summary["design_count"] == 1
    assert summary["field_count"] == 3
    assert summary["saved_image_count"] == 3
    assert summary["psf_images_per_second"] > 0.0

    image_paths = sorted(output_dir.glob("psf_design_*_field_*.png"))
    assert len(image_paths) == 3
    assert all(path.stat().st_size > 0 for path in image_paths)
