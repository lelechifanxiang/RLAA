from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ArrayLike = Any
Scalar = float | int

BackendName = Literal["numpy", "torch"]
FieldType = Literal["angle", "object_height", "image_height", "normalized"]
ApertureType = Literal[
    "entrance_pupil_diameter",
    "image_f_number",
    "object_na",
    "float_by_stop_size",
]
PupilDistribution = Literal[
    "gaussian",
    "square",
    "hexapolar",
    "random",
    "line_x",
    "line_y",
]
TraceDirection = Literal["forward", "backward"]


@dataclass(slots=True)
class BackendConfig:
    name: BackendName = "torch"
    device: str | None = None
    dtype: str | None = None
    enable_autodiff: bool = True


@dataclass(slots=True)
class UnitSystem:
    length: str = "mm"
    wavelength: str = "um"
    angle: str = "deg"


@dataclass(slots=True)
class RuntimeConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    units: UnitSystem = field(default_factory=UnitSystem)
    default_batch_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
