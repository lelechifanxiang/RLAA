from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SemiDiameterSolve = Literal["auto", "fixed"]
SurfaceApertureType = Literal["none", "floating"]


@dataclass(frozen=True)
class ZemaxSequentialSurfaceSpec:
    radius_mm: float
    thickness_mm: float
    semi_diameter_mm: float
    semi_diameter_solve: SemiDiameterSolve = "auto"
    aperture_type: SurfaceApertureType = "none"
    surface_type: str = "Standard"
    focal_length_mm: float | None = None
    material_name: str | None = None
    nd: float | None = None
    vd: float | None = None
    material_pickup_surface_number: int | None = None
    refractive_indices: tuple[float, ...] | None = None
    comment: str | None = None
    is_stop: bool = False
    decenter_x_mm: float = 0.0
    decenter_y_mm: float = 0.0
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0
    tilt_z_deg: float = 0.0
    order_flag: int = 0


@dataclass(frozen=True)
class ZemaxSequentialSystemSpec:
    name: str
    zmx_path: str
    surfaces: tuple[ZemaxSequentialSurfaceSpec, ...]
    wavelengths_um: tuple[float, ...]
    primary_wavelength_index: int
    field_type: str
    field_points: tuple[tuple[float, float], ...]
    aperture_kind: str
    aperture_value: float
    object_distance_mm: float
    afocal_image_space: bool
    stop_surface_index: int
    image_surface_index: int
