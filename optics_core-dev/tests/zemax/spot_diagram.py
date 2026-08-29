from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from tests.zemax.common import loaded_sequential_system, zp
from tests.zemax.temp_structures import ZemaxSpotDiagramReference
from tests.zemax.zmx_loader import load_zmx_sequential_system_spec


def fetch_zemax_standard_spot_metrics(
    zmx_path: str | Path,
    *,
    pattern: Literal["hexapolar", "square"],
    ray_density: int,
) -> ZemaxSpotDiagramReference:
    """直接调用 Zemax Standard Spot 获取每个视场的 RMS/GEO 半径。"""

    spec = load_zmx_sequential_system_spec(zmx_path)
    with loaded_sequential_system(spec.zmx_path) as oss:
        return fetch_zemax_standard_spot_metrics_from_spec(
            spec,
            oss,
            pattern=pattern,
            ray_density=ray_density,
        )


def fetch_zemax_standard_spot_metrics_from_spec(
    spec: Any,
    oss: Any,
    *,
    pattern: Literal["hexapolar", "square"],
    ray_density: int,
) -> ZemaxSpotDiagramReference:
    """基于已加载的 spec / oss 调用 Zemax Standard Spot 获取每个视场的 RMS/GEO 半径。"""

    pattern_constant = _pattern_constant(pattern)
    analysis = oss.Analyses.New_StandardSpot()
    try:
        settings = analysis.GetSettings()
        settings.Field.UseAllFields()
        settings.Wavelength.UseAllWavelengths()
        settings.Surface.UseImageSurface()
        settings.Pattern = pattern_constant
        settings.RayDensity = int(ray_density)
        settings.ReferTo = zp.constants.Analysis.Settings.Spot.Reference.ChiefRay

        analysis.ApplyAndWaitForCompletion()
        spot_data = analysis.GetResults().SpotData
        field_count = int(spot_data.NumberOfFields)
        rms_radius_um = [
            float(spot_data.GetRMSSpotSizeFor(field_index, 0))
            for field_index in range(1, field_count + 1)
        ]
        geo_radius_um = [
            float(spot_data.GetGeoSpotSizeFor(field_index, 0))
            for field_index in range(1, field_count + 1)
        ]
    finally:
        analysis.Close()

    return ZemaxSpotDiagramReference(
        rms_radius_um=rms_radius_um,
        geo_radius_um=geo_radius_um,
        field_points=[(float(field_x), float(field_y)) for field_x, field_y in spec.field_points],
        metadata={
            "source": "ZOSAPI.StandardSpot.SpotData",
            "zmx_path": spec.zmx_path,
            "pattern": pattern,
            "ray_density": int(ray_density),
            "refer_to": "ChiefRay",
            "field_count": len(spec.field_points),
        },
    )


def _pattern_constant(pattern: Literal["hexapolar", "square"]):
    if pattern == "hexapolar":
        return zp.constants.Analysis.Settings.Spot.Patterns.Hexapolar
    if pattern == "square":
        return zp.constants.Analysis.Settings.Spot.Patterns.Square
    raise ValueError(f"Unsupported spot diagram pattern: {pattern!r}.")
