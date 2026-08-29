from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class ParaxialTraceReference:
    paraxial_x_mm: list[float]
    paraxial_y_mm: list[float]
    paraxial_z_mm: list[float]

    image_x_mm: list[float]
    image_y_mm: list[float]
    image_z_mm: list[float]

    direction_l: list[float]
    direction_m: list[float]
    direction_n: list[float]

    valid: list[bool]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultispectralSphereMaterialCaseSpec:
    radius_mm: float = 50.0
    thickness_mm: float = 5.0
    aperture_radius_mm: float = 12.0
    nd: float = 1.5168
    vd: float = 64.17
    dpgf: float = 0.0
    wavelengths_um: tuple[float, ...] = (0.5875618, 0.4861327, 0.6562725)
    primary_wavelength_index: int = 0
    sag_sample_xy_mm: tuple[float, float] = (3.0, 4.0)
    field_hx: float = 0.0
    field_hy: float = 0.0
    abs_tol: float = 1e-6
    refractive_index_abs_tol: float = 6e-5


@dataclass
class ZemaxSurfaceSagReference:
    surface_sag_mm: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZemaxSurfaceRefractiveIndicesReference:
    wavelengths_um: list[float]
    refractive_indices: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SphericalForwardSurfaceSpec:
    radius_mm: float
    thickness_mm: float
    aperture_radius_mm: float
    nd: float | None = None
    vd: float | None = None
    dpgf: float = 0.0
    comment: str | None = None


@dataclass(frozen=True)
class SphericalForwardTraceCaseSpec:
    surfaces: tuple[SphericalForwardSurfaceSpec, ...]
    wavelengths_um: tuple[float, ...] = (0.5875618,)
    primary_wavelength_index: int = 0
    stop_aperture_radius_mm: float | None = None
    field_points: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    pupil_coordinates: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    abs_tol: float = 1e-6


@dataclass
class SphericalForwardTraceReference:
    surface_indices: list[int]
    x_mm: list[list[float]]
    y_mm: list[list[float]]
    z_mm: list[list[float]]
    direction_l: list[list[float]]
    direction_m: list[list[float]]
    direction_n: list[list[float]]
    refractive_indices_by_surface: dict[int, list[float]]
    wavelengths_um: list[float]
    field_points: list[tuple[float, float]]
    pupil_coordinates: list[tuple[float, float]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SphericalClearApertureReference:
    surface_indices: list[int]
    semi_diameter_mm: list[float]
    field_points: list[tuple[float, float]]
    pupil_coordinates: list[tuple[float, float]]
    wavelengths_um: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZemaxFirstOrderReference:
    effective_focal_length_mm: float
    image_f_number: float
    working_f_number: float
    entrance_pupil_z_mm: float
    entrance_pupil_radius_mm: float
    exit_pupil_z_mm: float
    exit_pupil_radius_mm: float
    total_track_length_mm: float
    image_plane_distance_mm: float
    back_focal_length_mm: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ZemaxSequentialSurfaceSpec:
    radius_mm: float
    thickness_mm: float
    aperture_radius_mm: float
    surface_type: str = "Standard"
    focal_length_mm: float | None = None
    nd: float | None = None
    vd: float | None = None
    refractive_indices: tuple[float, ...] | None = None
    dpgf: float = 0.0
    comment: str | None = None
    is_stop: bool = False


@dataclass(frozen=True)
class ZemaxSequentialSystemSpec:
    name: str
    zmx_path: str
    surfaces: tuple[ZemaxSequentialSurfaceSpec, ...]
    wavelengths_um: tuple[float, ...]
    primary_wavelength_index: int
    field_points: tuple[tuple[float, float], ...]
    aperture_kind: str
    aperture_value: float
    stop_surface_index: int
    image_surface_index: int


@dataclass
class ZemaxDirectRaySet:
    x: Any
    y: Any
    z: Any
    l: Any
    m: Any
    n: Any
    wavelength_um: Any
    wavelength_indices: tuple[int, ...]
    field_points: tuple[tuple[float, float], ...]
    pupil_coordinates: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZemaxBatchTraceResult:
    x_mm: Any
    y_mm: Any
    z_mm: Any
    direction_l: Any
    direction_m: Any
    direction_n: Any
    valid: Any
    error_codes: Any
    vignette_codes: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZemaxBatchKernelBenchmarkResult:
    run_avg_ms: float
    add_rays_avg_ms: float
    read_results_avg_ms: float
    last_result: ZemaxBatchTraceResult


@dataclass
class ZemaxSpotDiagramReference:
    rms_radius_um: list[float]
    geo_radius_um: list[float]
    field_points: list[tuple[float, float]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZemaxHuygensPSFReference:
    psf: Any
    strehl_ratio: float
    x_um: list[float]
    y_um: list[float]
    field_point: tuple[float, float]
    wavelength_um: float | tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZemaxHuygensMTFReference:
    frequencies_lp_per_mm: Any
    sagittal: Any
    tangential: Any
    field_point: tuple[float, float]
    wavelength_um: float | tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZemaxWavefrontMapReference:
    opd: Any
    pupil_x: Any
    pupil_y: Any
    rms_wavefront: float
    field_point: tuple[float, float]
    wavelength_um: float
    metadata: dict[str, Any] = field(default_factory=dict)
