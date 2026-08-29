from __future__ import annotations

from math import sqrt
from typing import Any

import torch

from tests.zemax.common import get_merit_operand_value, zp
from tests.zemax.temp_structures import ZemaxHuygensMTFReference


def fetch_zemax_huygens_mtf_from_spec(
    spec: Any,
    oss: Any,
    *,
    pupil_sample_count: int,
    image_sample_count: int,
    image_delta_um: float,
    maximum_frequency_lp_per_mm: float,
    field_index: int,
    wavelength_index: int,
) -> ZemaxHuygensMTFReference:
    """调用 Zemax Huygens MTF 直接获取指定视场的 S/T 曲线。"""
    resolved_wavelength_index = int(wavelength_index)
    if resolved_wavelength_index == -1:
        zemax_wavelength: str | int = "All"
        wavelength_um: float | tuple[float, ...] = tuple(float(value) for value in spec.wavelengths_um)
        image_delta_wavelength_um = max(wavelength_um)
        working_f_number_wavelength_index = int(spec.primary_wavelength_index) + 1
    else:
        zemax_wavelength = resolved_wavelength_index + 1
        wavelength_um = float(spec.wavelengths_um[resolved_wavelength_index])
        image_delta_wavelength_um = wavelength_um
        working_f_number_wavelength_index = resolved_wavelength_index + 1

    analysis = zp.analyses.mtf.HuygensMTF(
        pupil_sampling=f"{int(pupil_sample_count)}x{int(pupil_sample_count)}",
        image_sampling=f"{int(image_sample_count)}x{int(image_sample_count)}",
        image_delta=float(image_delta_um),
        wavelength=zemax_wavelength,
        field=int(field_index) + 1,
        mtf_type="Modulation",
        maximum_frequency=float(maximum_frequency_lp_per_mm),
        use_polarization=False,
        use_dashes=False,
    )
    result = analysis.run(oss)
    mtf_frame = result.data
    if mtf_frame is None:
        raise ValueError("Zemax Huygens MTF did not return a data series.")

    series_labels = [str(column[-1] if isinstance(column, tuple) else column) for column in mtf_frame.columns]
    sagittal_column = next(index for index, label in enumerate(series_labels) if "sagittal" in label.lower())
    tangential_column = next(index for index, label in enumerate(series_labels) if "tangential" in label.lower())
    frequencies = torch.tensor(mtf_frame.index.to_numpy(dtype=float), dtype=torch.float64)
    sagittal = torch.tensor(mtf_frame.iloc[:, sagittal_column].to_numpy(dtype=float), dtype=torch.float64)
    tangential = torch.tensor(mtf_frame.iloc[:, tangential_column].to_numpy(dtype=float), dtype=torch.float64)

    resolved_image_delta_um = float(image_delta_um)
    if resolved_image_delta_um == 0.0:
        working_f_number = get_merit_operand_value(
            oss,
            zp.constants.Editors.MFE.MeritOperandType.WFNO,
            surface_index=0,
            wavelength_index=working_f_number_wavelength_index,
        )
        resolved_image_delta_um = (
            image_delta_wavelength_um
            * working_f_number
            / sqrt(float(pupil_sample_count))
        )

    return ZemaxHuygensMTFReference(
        frequencies_lp_per_mm=frequencies,
        sagittal=sagittal,
        tangential=tangential,
        field_point=(
            float(spec.field_points[field_index][0]),
            float(spec.field_points[field_index][1]),
        ),
        wavelength_um=wavelength_um,
        metadata={
            "source": "zospy.HuygensMTF",
            "zmx_path": spec.zmx_path,
            "pupil_sample_count": int(pupil_sample_count),
            "image_sample_count": int(image_sample_count),
            "image_delta_um": float(image_delta_um),
            "resolved_image_delta_um": resolved_image_delta_um,
            "maximum_frequency_lp_per_mm": float(maximum_frequency_lp_per_mm),
            "field_index": int(field_index),
            "wavelength_index": resolved_wavelength_index,
        },
    )
