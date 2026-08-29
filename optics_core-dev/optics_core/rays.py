from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .types import ArrayLike, Scalar, TraceDirection


@dataclass(slots=True)
class NormalizedFieldPoint:
    hx: Scalar = 0.0
    hy: Scalar = 0.0


@dataclass(slots=True)
class NormalizedPupilPoint:
    px: Scalar = 0.0
    py: Scalar = 0.0


@dataclass(slots=True)
class RayBundle:
    x: ArrayLike
    y: ArrayLike
    z: ArrayLike
    l: ArrayLike
    m: ArrayLike
    n: ArrayLike
    wavelength_index: ArrayLike
    intensity: ArrayLike | None = None
    opl: ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


SurfaceStateSelection = Literal["all"] | tuple[int, ...]


@dataclass(slots=True)
class SurfaceTraceHistory:
    """按追迹顺序堆叠的表面出射光线状态。"""

    surface_indices: tuple[int, ...]
    x: ArrayLike
    y: ArrayLike
    z: ArrayLike
    l: ArrayLike
    m: ArrayLike
    n: ArrayLike
    opl: ArrayLike | None
    valid: ArrayLike


@dataclass(slots=True)
class TraceOptions:
    start_surface: int = 0
    stop_surface: int | None = None
    direction: TraceDirection = "forward"
    record_intersections: bool = True
    record_opd: bool = True
    record_surface_states: SurfaceStateSelection | None = None
    record_ray_angles: bool = False
    ignore_coordinate_breaks: bool = False
    ignore_apertures: bool = False
    warm_start: Any | None = None


@dataclass(slots=True)
class SurfaceIntersection:
    surface_index: int
    position: tuple[ArrayLike, ArrayLike, ArrayLike]
    normal: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None


@dataclass(slots=True)
class TraceResult:
    rays: RayBundle
    valid: ArrayLike
    intersections: tuple[SurfaceIntersection, ...] = ()
    surface_history: SurfaceTraceHistory | None = None
    ray_angles_in_deg: ArrayLike | None = None
    ray_angles_out_deg: ArrayLike | None = None
    cache: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RayAimingResult:
    entrance_pupil_z: ArrayLike | None = None
    entrance_pupil_radius: ArrayLike | None = None
    cache: dict[str, Any] = field(default_factory=dict)
