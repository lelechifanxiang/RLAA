from __future__ import annotations

from typing import Any

import torch

from tests.zemax.common import zp
from tests.zemax.temp_structures import ZemaxHuygensPSFReference


def fetch_zemax_huygens_psf_from_spec(
    spec: Any,
    oss: Any,
    *,
    pupil_sample_count: int,
    image_sample_count: int,
    image_delta_um: float,
    field_index: int,
    wavelength_index: int,
    normalize: bool = False,
) -> ZemaxHuygensPSFReference:
    """调用 Zemax Huygens PSF 获取指定视场的未归一化参考网格。

    wavelength_index=-1 时获取 Zemax 的全波长混合结果。
    """
    resolved_wavelength_index = int(wavelength_index)
    if resolved_wavelength_index == -1:
        zemax_wavelength: str | int = "All"
        wavelength_um: float | tuple[float, ...] = tuple(float(value) for value in spec.wavelengths_um)
    else:
        if resolved_wavelength_index < 0 or resolved_wavelength_index >= len(spec.wavelengths_um):
            raise ValueError("wavelength_index is out of range.")
        zemax_wavelength = resolved_wavelength_index + 1
        wavelength_um = float(spec.wavelengths_um[resolved_wavelength_index])

    analysis = zp.analyses.psf.HuygensPSFAndStrehlRatio(
        pupil_sampling=f"{int(pupil_sample_count)}x{int(pupil_sample_count)}",
        image_sampling=f"{int(image_sample_count)}x{int(image_sample_count)}",
        image_delta=float(image_delta_um),
        wavelength=zemax_wavelength,
        field=int(field_index) + 1,
        psf_type="Linear",
        show_as="Surface",
        use_polarization=False,
        use_centroid=False,
        normalize=normalize,
    )
    result = analysis.run(oss)
    psf_frame = result.data.psf
    psf = torch.tensor(psf_frame.to_numpy(dtype=float), dtype=torch.float64)
    x_um = [float(value) for value in psf_frame.columns.tolist()]
    y_um = [float(value) for value in psf_frame.index.tolist()]
    resolved_image_delta_um = abs(x_um[1] - x_um[0])

    return ZemaxHuygensPSFReference(
        psf=psf,
        strehl_ratio=float(result.data.strehl_ratio),
        x_um=x_um,
        y_um=y_um,
        field_point=(
            float(spec.field_points[int(field_index)][0]),
            float(spec.field_points[int(field_index)][1]),
        ),
        wavelength_um=wavelength_um,
        metadata={
            "source": "zospy.HuygensPSFAndStrehlRatio",
            "zmx_path": spec.zmx_path,
            "pupil_sample_count": int(pupil_sample_count),
            "image_sample_count": int(image_sample_count),
            "image_delta_um": float(image_delta_um),
            "resolved_image_delta_um": resolved_image_delta_um,
            "field_index": int(field_index),
            "wavelength_index": resolved_wavelength_index,
            "normalize": bool(normalize),
        },
    )
