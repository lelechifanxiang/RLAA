from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from .types import ArrayLike, Scalar

if TYPE_CHECKING:
    from .system import MultiOpticalSystem


@dataclass(slots=True)
class Sensor:
    resolution: tuple[int, int]
    pixel_pitch_um: Scalar
    fill_factor: Scalar = 1.0
    spectral_response: ArrayLike | None = None


@dataclass(slots=True)
class ImageSimulationResult:
    irradiance: ArrayLike | None = None
    sensor_image: ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageFormationModel(ABC):
    @abstractmethod
    def render(
        self,
        system: MultiOpticalSystem,
        sensor: Sensor,
        scene: Any,
        *,
        wavelengths_um: Sequence[Scalar] | None = None,
    ) -> ImageSimulationResult:
        raise NotImplementedError
