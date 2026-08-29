from __future__ import annotations

from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import torch

from tests.zemax.common import zp
from tests.zemax.temp_structures import ZemaxWavefrontMapReference


def fetch_zemax_wavefront_map_from_spec(
    spec: Any,
    oss: Any,
    *,
    sample_count: int,
    field_index: int,
    wavelength_index: int,
) -> ZemaxWavefrontMapReference:
    """调用 Zemax Wavefront Map 获取像面波前图。"""
    resolved_wavelength_index = int(wavelength_index)
    if resolved_wavelength_index < 0 or resolved_wavelength_index >= len(spec.wavelengths_um):
        raise ValueError("wavelength_index is out of range.")

    analysis = oss.Analyses.New_WavefrontMap()
    try:
        settings = analysis.GetSettings()
        settings.Field.SetFieldNumber(int(field_index) + 1)
        settings.Wavelength.SetWavelengthNumber(resolved_wavelength_index + 1)
        settings.Surface.UseImageSurface()
        settings.ShowAs = zp.constants.Analysis.ShowAs.Surface
        settings.Rotation = zp.constants.Analysis.Settings.Rotations.Rotate_0
        settings.Sampling = getattr(zp.constants.Analysis.SampleSizes, f"S_{int(sample_count)}x{int(sample_count)}")
        settings.ReferenceToPrimary = False
        settings.UseExitPupil = False
        settings.RemoveTilt = False
        settings.Scale = 1.0
        settings.Subaperture_X = 0.0
        settings.Subaperture_Y = 0.0
        settings.Subaperture_R = 1.0

        analysis.ApplyAndWaitForCompletion()
        results = analysis.GetResults()
        text_metadata = _read_wavefront_text_metadata(results)
        if int(results.NumberOfDataGrids) < 1:
            raise ValueError("Zemax Wavefront Map did not return a data grid.")
        grid = results.GetDataGrid(0)
        opd = torch.tensor(np.array(grid.Values, dtype=float), dtype=torch.float64)
        x = torch.linspace(float(grid.MinX), float(grid.MinX) + float(grid.Dx) * (int(grid.Nx) - 1), int(grid.Nx), dtype=torch.float64)
        y = torch.linspace(float(grid.MinY), float(grid.MinY) + float(grid.Dy) * (int(grid.Ny) - 1), int(grid.Ny), dtype=torch.float64)
        grid_metadata = {
            "nx": int(grid.Nx),
            "ny": int(grid.Ny),
            "min_x": float(grid.MinX),
            "min_y": float(grid.MinY),
            "dx": float(grid.Dx),
            "dy": float(grid.Dy),
        }
    finally:
        analysis.Close()

    pupil_y, pupil_x = torch.meshgrid(y, x, indexing="ij")
    finite = torch.isfinite(opd)
    datagrid_rms = float(torch.sqrt(torch.mean(opd[finite] * opd[finite])).item())
    rms_wavefront = float(text_metadata.get("rms_wavefront", datagrid_rms))

    return ZemaxWavefrontMapReference(
        opd=opd,
        pupil_x=pupil_x,
        pupil_y=pupil_y,
        rms_wavefront=rms_wavefront,
        field_point=(
            float(spec.field_points[int(field_index)][0]),
            float(spec.field_points[int(field_index)][1]),
        ),
        wavelength_um=float(spec.wavelengths_um[resolved_wavelength_index]),
        metadata={
            "source": "zospy.WavefrontMap",
            "zmx_path": spec.zmx_path,
            "sample_count": int(sample_count),
            "field_index": int(field_index),
            "wavelength_index": resolved_wavelength_index,
            "grid_shape": tuple(int(value) for value in opd.shape),
            **grid_metadata,
            "pupil_grid_size": text_metadata.get("pupil_grid_size", f"{opd.shape[1]} by {opd.shape[0]}"),
            "center_point": text_metadata.get("center_point"),
            "datagrid_rms_wavefront": datagrid_rms,
            "surface": "Image",
            "use_exit_pupil": False,
            "remove_tilt": False,
            "reference_to_primary": False,
        },
    )


def _read_wavefront_text_metadata(results: Any) -> dict[str, str]:
    """从 Zemax Wavefront Map 文本输出中提取网格说明。"""
    path = Path(tempfile.gettempdir()) / "optics_core_wavefront_map.txt"
    metadata: dict[str, str] = {}
    try:
        if not bool(results.GetTextFile(str(path))):
            return metadata
        text = path.read_bytes().decode("utf-16", errors="ignore")
        for line in text.splitlines():
            normalized = " ".join(line.split())
            grid_match = re.search(r"Pupil grid size:\s*(\d+\s+by\s+\d+)", normalized, flags=re.IGNORECASE)
            center_match = re.search(r"Center point is:\s*(.+)", normalized, flags=re.IGNORECASE)
            if grid_match:
                metadata["pupil_grid_size"] = grid_match.group(1)
            if center_match:
                metadata["center_point"] = center_match.group(1)
            rms_match = re.search(r"RMS\s*=\s*([+-]?\d+(?:\.\d+)?(?:E[+-]?\d+)?)\s*waves", normalized, flags=re.IGNORECASE)
            if rms_match:
                metadata["rms_wavefront"] = rms_match.group(1)
        return metadata
    finally:
        path.unlink(missing_ok=True)
