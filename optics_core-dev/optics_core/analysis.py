from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, Sequence, TypeVar

from .types import ArrayLike, Scalar

if TYPE_CHECKING:
    from .system import MultiOpticalSystem


@dataclass(slots=True)
class AnalysisResult:
    pass


@dataclass(slots=True)
class SpotDiagramSettings:
    pattern: Literal["hexapolar", "square"] = "hexapolar"
    ray_density: int = 30
    save_path: str | None = None


@dataclass(slots=True)
class WavefrontSettings:
    field_indices: Sequence[int] | None = None
    wavelength_indices: Sequence[int] | None = None
    sample_count: int = 32
    save_path: str | None = None


@dataclass(slots=True)
class ZernikeSettings:
    maximum_term: int = 15
    indexing_scheme: Literal["noll", "ansi", "fringe"] = "noll"
    normalization: Literal["orthonormal", "unit_rms", "fringe"] = "orthonormal"
    remove_piston: bool = True
    remove_tilt: bool = True


@dataclass(slots=True)
class PSFSettings:
    pupil_sample_count: int = 32
    image_sample_count: int = 32
    image_delta_um: Scalar = 0.0
    field_index: int = 0
    wavelength_index: int | None = None
    save_path: str | None = None


@dataclass(slots=True)
class MTFSettings:
    pupil_sample_count: int = 32
    image_sample_count: int = 32
    image_delta_um: Scalar = 0.0
    frequencies_lp_per_mm: Sequence[Scalar] = field(default_factory=tuple)
    field_indices: Sequence[int] | None = None
    wavelength_index: int | None = None
    save_path: str | None = None


@dataclass(slots=True)
class DistortionSettings:
    sample_count: int = 21


@dataclass(slots=True)
class Operand:
    name: str
    weight: Scalar = 1.0
    target: Scalar | None = None
    reducer: str = "rms"


@dataclass(slots=True)
class SpotDiagramResult(AnalysisResult):
    rms_radius_um: ArrayLike | None = None
    geo_radius_um: ArrayLike | None = None
    valid_count: ArrayLike | None = None
    valid_fraction: ArrayLike | None = None
    field_points: tuple[tuple[float, float], ...] = ()
    figure: Any | None = None
    axes: Any | None = None
    save_path: str | None = None
    scatter_points: dict[str, Any] | None = None


@dataclass(slots=True)
class WavefrontResult(AnalysisResult):
    opd: ArrayLike | None = None
    rms_wavefront: ArrayLike | None = None
    valid_mask: ArrayLike | None = None
    valid_count: ArrayLike | None = None
    valid_fraction: ArrayLike | None = None
    pupil_x: ArrayLike | None = None
    pupil_y: ArrayLike | None = None
    field_indices: tuple[int, ...] = ()
    wavelength_indices: tuple[int, ...] = ()
    sample_count: int | None = None
    figure: Any | None = None
    axes: Any | None = None
    save_path: str | None = None
    detected_design_batch_size: int | None = None
    design_batch_size: int | None = None
    minibatch_count: int = 0


@dataclass(slots=True)
class ZernikeResult(AnalysisResult):
    coefficients: ArrayLike | None = None
    indices: ArrayLike | None = None
    residual_rms_wavefront: ArrayLike | None = None
    reconstructed_opd: ArrayLike | None = None
    pupil_mask: ArrayLike | None = None


@dataclass(slots=True)
class PSFResult(AnalysisResult):
    psf: ArrayLike | None = None
    strehl_ratio: ArrayLike | None = None
    psf_by_wavelength: ArrayLike | None = None
    strehl_by_wavelength: ArrayLike | None = None
    pixel_pitch_um: ArrayLike | None = None
    field_index: int | None = None
    wavelength_indices: tuple[int, ...] = ()
    figure: Any | None = None
    axes: Any | None = None
    save_path: str | None = None
    detected_design_batch_size: int | None = None
    design_batch_size: int | None = None
    minibatch_count: int = 0


@dataclass(slots=True)
class MTFResult(AnalysisResult):
    frequencies_lp_per_mm: ArrayLike | None = None
    sagittal: ArrayLike | None = None
    tangential: ArrayLike | None = None
    pixel_pitch_um: ArrayLike | None = None
    field_indices: tuple[int, ...] = ()
    wavelength_indices: tuple[int, ...] = ()
    figure: Any | None = None
    axes: Any | None = None
    save_path: str | None = None
    detected_design_batch_size: int | None = None
    design_batch_size: int | None = None
    minibatch_count: int = 0


@dataclass(slots=True)
class DistortionResult(AnalysisResult):
    field_coordinates: ArrayLike | None = None
    distortion_percent: ArrayLike | None = None


@dataclass(slots=True)
class FirstOrderResult(AnalysisResult):
    effl: ArrayLike
    working_f_number: ArrayLike
    ttl: ArrayLike
    image_plane_distance: ArrayLike
    bfl: ArrayLike
    valid: ArrayLike
    entrance_pupil_z: ArrayLike | None = None
    entrance_pupil_radius: ArrayLike | None = None
    stop_radius: ArrayLike | None = None
    exit_pupil_z: ArrayLike | None = None
    exit_pupil_radius: ArrayLike | None = None
    cra_deg: ArrayLike | None = None
    image_height: ArrayLike | None = None


@dataclass(slots=True)
class MeritFunctionResult(AnalysisResult):
    merit: ArrayLike | None = None
    constraints: dict[str, ArrayLike] = field(default_factory=dict)


@dataclass(slots=True)
class Layout2DSettings:
    save_path: str | None = None


@dataclass(slots=True)
class Layout2DResult(AnalysisResult):
    filtered_field_indices: tuple[int, ...] = ()
    filtered_field_points: tuple[tuple[float, float], ...] = ()
    trace_result: Any | None = None
    clear_aperture_result: Any | None = None
    figure: Any | None = None
    axes: Any | None = None
    save_path: str | None = None
    message: str | None = None


TResult = TypeVar("TResult", bound=AnalysisResult)


class Analysis(ABC, Generic[TResult]):
    def __init__(self, system: MultiOpticalSystem) -> None:
        self.system = system

    @abstractmethod
    def run(self) -> TResult:
        raise NotImplementedError


class SpotDiagram(Analysis[SpotDiagramResult]):
    def __init__(self, system: MultiOpticalSystem, settings: SpotDiagramSettings | None = None) -> None:
        super().__init__(system)
        self.settings = settings or SpotDiagramSettings()

    def run(self) -> SpotDiagramResult:
        from .spot_diagram import run_spot_diagram

        return run_spot_diagram(self.system, self.settings)


class WavefrontMap(Analysis[WavefrontResult]):
    def __init__(self, system: MultiOpticalSystem, settings: WavefrontSettings | None = None) -> None:
        super().__init__(system)
        self.settings = settings or WavefrontSettings()

    def run(self) -> WavefrontResult:
        from .wavefront import run_wavefront

        return run_wavefront(self.system, self.settings)


class ZernikeDecomposition(Analysis[ZernikeResult]):
    def __init__(self, system: MultiOpticalSystem, settings: ZernikeSettings | None = None) -> None:
        super().__init__(system)
        self.settings = settings or ZernikeSettings()

    def run(self) -> ZernikeResult:
        raise NotImplementedError


class PointSpreadFunction(Analysis[PSFResult]):
    def __init__(self, system: MultiOpticalSystem, settings: PSFSettings | None = None) -> None:
        super().__init__(system)
        self.settings = settings or PSFSettings()

    def run(self) -> PSFResult:
        from .huygens_psf import run_huygens_psf

        return run_huygens_psf(self.system, self.settings)


class ModulationTransferFunction(Analysis[MTFResult]):
    def __init__(self, system: MultiOpticalSystem, settings: MTFSettings | None = None) -> None:
        super().__init__(system)
        self.settings = settings or MTFSettings()

    def run(self) -> MTFResult:
        from .huygens_mtf import run_huygens_mtf

        return run_huygens_mtf(self.system, self.settings)


class Distortion(Analysis[DistortionResult]):
    def __init__(self, system: MultiOpticalSystem, settings: DistortionSettings | None = None) -> None:
        super().__init__(system)
        self.settings = settings or DistortionSettings()

    def run(self) -> DistortionResult:
        raise NotImplementedError


class FirstOrder(Analysis[FirstOrderResult]):
    def run(self) -> FirstOrderResult:
        first_order = self.system.first_order_data
        if first_order is None:
            raise ValueError("first_order analysis requires system.prepare() before run().")
        result = FirstOrderResult(
            effl=first_order.effl,
            working_f_number=first_order.working_f_number,
            ttl=first_order.ttl,
            image_plane_distance=first_order.image_plane_distance,
            bfl=first_order.bfl,
            valid=first_order.valid,
            entrance_pupil_z=first_order.entrance_pupil_z,
            entrance_pupil_radius=first_order.entrance_pupil_radius,
            stop_radius=first_order.stop_radius,
            exit_pupil_z=first_order.exit_pupil_z,
            exit_pupil_radius=first_order.exit_pupil_radius,
        )
        return result


class MeritFunction(Analysis[MeritFunctionResult]):
    def __init__(self, system: MultiOpticalSystem, operands: Sequence[Operand]) -> None:
        super().__init__(system)
        self.operands = tuple(operands)

    def run(self) -> MeritFunctionResult:
        raise NotImplementedError


class Layout2D(Analysis[Layout2DResult]):
    def __init__(self, system: MultiOpticalSystem, settings: Layout2DSettings | None = None) -> None:
        super().__init__(system)
        self.settings = settings or Layout2DSettings()

    def run(self) -> Layout2DResult:
        from .layout_2d import run_layout_2d

        return run_layout_2d(self.system, self.settings)


class AnalysisHub:
    def __init__(self, system: MultiOpticalSystem) -> None:
        self._system = system

    def spot_diagram(self, settings: SpotDiagramSettings | None = None) -> SpotDiagram:
        return SpotDiagram(self._system, settings=settings)

    def wavefront(self, settings: WavefrontSettings | None = None) -> WavefrontMap:
        return WavefrontMap(self._system, settings=settings)

    def zernike(self, settings: ZernikeSettings | None = None) -> ZernikeDecomposition:
        return ZernikeDecomposition(self._system, settings=settings)

    def psf(self, settings: PSFSettings | None = None) -> PointSpreadFunction:
        return PointSpreadFunction(self._system, settings=settings)

    def mtf(self, settings: MTFSettings | None = None) -> ModulationTransferFunction:
        return ModulationTransferFunction(self._system, settings=settings)

    def distortion(self, settings: DistortionSettings | None = None) -> Distortion:
        return Distortion(self._system, settings=settings)

    def first_order(self) -> FirstOrder:
        return FirstOrder(self._system)

    def merit(self, operands: Sequence[Operand]) -> MeritFunction:
        return MeritFunction(self._system, operands=operands)

    def layout_2d(self, settings: Layout2DSettings | None = None) -> Layout2D:
        return Layout2D(self._system, settings=settings)
