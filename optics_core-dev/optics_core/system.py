from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch

from ._material_batch import BatchedMaterialData, compile_batched_material_data
from .materials import MaterialLibrary
from .rays import TraceOptions, TraceResult
from .sampling import PupilSampler, RayAimer
from .system_state import FirstOrderData, FrameData
from .surfaces import Surface, CoordinateBreak
from .system_specs import (
    FieldPoint,
    FieldSequence,
    OpticalArchitecture,
    ParameterSchema,
    ParameterVectorBatch,
    ParameterVectorBatchRange,
    SystemAperture,
    Wavelength,
    WavelengthSequence,
)
from .tracing import SequentialSurfaceRayTracer
from .types import ApertureType, RuntimeConfig, Scalar


class MultiOpticalSystem:
    """公开系统容器，组合共享拓扑与运行时配置。"""

    def __init__(
        self,
        architecture: OpticalArchitecture,
        *,
        name: str | None = None,
        parameter_schema: ParameterSchema | None = None,
        parameters: ParameterVectorBatch | list[list[Any]] | None = None,
        config: RuntimeConfig | None = None,
        tracer: SequentialSurfaceRayTracer | None = None,
        materials: MaterialLibrary | None = None,
        fields: Sequence[FieldPoint] | FieldSequence = (),
        wavelengths: Sequence[Wavelength] | WavelengthSequence = (),
        aperture: SystemAperture | None = None,
    ) -> None:
        self.architecture = architecture
        self.name = name or architecture.name
        self.config = config or RuntimeConfig()

        self.materials = materials or architecture.materials

        resolved_schema = parameter_schema
        if resolved_schema is None and isinstance(parameters, ParameterVectorBatch):
            resolved_schema = parameters.schema
        if resolved_schema is None:
            resolved_schema = ParameterSchema()

        if parameters is None:
            self.parameters = ParameterVectorBatch(
                schema=resolved_schema,
                vectors=[resolved_schema.default_vector()],
                grid_shape=(1,),
            )
        elif isinstance(parameters, ParameterVectorBatch):
            self.parameters = parameters if parameter_schema is None else parameters.with_schema(resolved_schema)
        else:
            if parameter_schema is None:
                raise ValueError("parameter_schema is required when parameters are provided as raw vectors.")
            self.parameters = ParameterVectorBatch(schema=resolved_schema, vectors=parameters)

        self.aperture: SystemAperture | None = None
        self.fields = FieldSequence(fields.field_type if isinstance(fields, FieldSequence) else "angle")
        for field in fields:
            self.fields.append(field)

        self.wavelengths = WavelengthSequence()
        for wavelength in wavelengths:
            self.wavelengths.append(wavelength)

        self.surfaces = architecture.surfaces.bind(owner=self, materials=self.materials)
        if aperture is not None:
            self.set_aperture(aperture)

        self.tracer = tracer
        self.frame_data: FrameData | None = None
        self.first_order_data: FirstOrderData | Any | None = None
        self.clear_aperture_data: Any | None = None
        self._material_data: BatchedMaterialData | None = None
        self._analysis_hub: Any | None = None

    @property
    def system_count(self) -> int:
        return self.parameters.system_count

    @property
    def parameter_schema(self) -> ParameterSchema:
        return self.parameters.schema

    @property
    def parameter_vectors(self) -> list[list[Any]]:
        return [list(vector) for vector in self.parameters]

    @property
    def parameter_count(self) -> int:
        return self.parameters.parameter_count

    def _normalize_design_index(self, index: int) -> int:
        normalized_index = index if index >= 0 else self.system_count + index
        if normalized_index < 0 or normalized_index >= self.system_count:
            raise IndexError(f"design index {index} is out of range for {self.system_count} systems.")
        return normalized_index

    def _parameter_payload(self, design_index: int, *, parameter_name: str) -> Any:
        parameter_index = self.parameter_schema.index_of(parameter_name)
        return self.parameters[design_index][parameter_index]

    def _resolve_material_reference(self, material: Any) -> Any:
        if material is None:
            return None
        if isinstance(material, str):
            return self.materials.resolve(material)

        material_name = getattr(material, "name", None)
        if isinstance(material_name, str) and material_name in self.materials:
            return self.materials.resolve(material_name)
        return material

    def _apply_parameter_path(self, surfaces: list[Surface], *, path: str, value: Any) -> None:
        if not path.startswith("surface["):
            raise ValueError(f"Unsupported parameter path root: {path!r}")

        closing_bracket = path.find("]")
        if closing_bracket < 0 or closing_bracket + 1 >= len(path) or path[closing_bracket + 1] != ".":
            raise ValueError(f"Unsupported parameter path format: {path!r}")

        surface_index = int(path[len("surface[") : closing_bracket])
        target = surfaces[surface_index]
        attributes = path[closing_bracket + 2 :].split(".")
        if not attributes:
            raise ValueError(f"Unsupported parameter path format: {path!r}")

        for attribute in attributes[:-1]:
            target = getattr(target, attribute)

        final_attribute = attributes[-1]
        if final_attribute == "medium":
            value = self._resolve_material_reference(value)
        setattr(target, final_attribute, value)

    def _materialize_design_surfaces(self, design_index: int) -> list[Surface]:
        surfaces = copy.deepcopy(list(self.surfaces))
        for surface in surfaces:
            surface.gap.medium = self._resolve_material_reference(surface.gap.medium)

        for spec in self.parameter_schema:
            self._apply_parameter_path(
                surfaces,
                path=spec.path,
                value=self._parameter_payload(design_index, parameter_name=spec.name),
            )
        return surfaces

    def design_view(self, index: int) -> MultiOpticalSystem:
        """返回第 n 条参数向量对应的单设计系统视图。"""

        design_index = self._normalize_design_index(index)
        design_vector = list(self.parameters[design_index])
        design_surfaces = self._materialize_design_surfaces(design_index)
        design_architecture = OpticalArchitecture.from_sequence(
            name=f"{self.architecture.name}_design_{design_index}",
            surfaces=design_surfaces,
            materials=self.materials,
            object_distance_mm=self.architecture.object_distance_mm,
            afocal_image_space=self.architecture.afocal_image_space,
        )
        design_parameters = ParameterVectorBatch(
            schema=self.parameter_schema,
            vectors=[design_vector],
            grid_shape=(1,),
        )

        design = MultiOpticalSystem(
            architecture=design_architecture,
            name=f"{self.name}[{design_index}]",
            parameter_schema=self.parameter_schema,
            parameters=design_parameters,
            config=copy.deepcopy(self.config),
            tracer=self.tracer,
            materials=self.materials,
            fields=copy.deepcopy(list(self.fields)),
            wavelengths=copy.deepcopy(list(self.wavelengths)),
            aperture=None if self.aperture is None else self.aperture.design_slice(design_index, design_index + 1),
        )
        if self._material_data is not None:
            design._material_data = self._material_data.design_slice(design_index, design_index + 1)
        return design

    def design_batch_view(self, start: int, stop: int) -> MultiOpticalSystem:
        """返回共享结构和准备态 tensor 的连续 design 轻量视图。"""
        start = int(start)
        stop = int(stop)
        if start < 0 or stop <= start or stop > self.system_count:
            raise IndexError("design batch range is out of bounds.")
        if self.frame_data is None or self.first_order_data is None:
            raise ValueError("design_batch_view requires system.prepare() before slicing.")

        view = object.__new__(MultiOpticalSystem)
        view.architecture = self.architecture
        view.name = f"{self.name}[{start}:{stop}]"
        view.config = self.config
        view.materials = self.materials
        view.parameters = ParameterVectorBatchRange(self.parameters, start, stop)
        view.aperture = None if self.aperture is None else self.aperture.design_slice(start, stop)
        view.fields = self.fields
        view.wavelengths = self.wavelengths
        view.surfaces = self.surfaces.bind(owner=view, materials=self.materials)
        view.tracer = self.tracer
        view.frame_data = self.frame_data.design_slice(start, stop)
        view.first_order_data = self.first_order_data.design_slice(start, stop)
        view.clear_aperture_data = (
            None if self.clear_aperture_data is None else self.clear_aperture_data.design_slice(start, stop)
        )
        view._material_data = self._material_data.design_slice(start, stop)
        view._analysis_hub = None
        return view

    def set_aperture(
        self,
        aperture: SystemAperture | ApertureType,
        value: Scalar | torch.Tensor | None = None,
        *,
        stop_surface: int | None = None,
        label: str | None = None,
    ) -> SystemAperture:
        """绑定系统级孔径定义。"""

        if isinstance(aperture, SystemAperture):
            self.aperture = aperture
            return aperture
        if value is None:
            raise ValueError("value is required when aperture is given as a string literal.")
        self.aperture = SystemAperture(
            kind=aperture,
            value=value,
            stop_surface=stop_surface,
            label=label,
        )
        return self.aperture

    def set_tracer(self, tracer: SequentialSurfaceRayTracer) -> SequentialSurfaceRayTracer:
        self.tracer = tracer
        return tracer

    def prepare(
        self,
        *,
        recompute_materials: bool = True,
        recompute_frame: bool = True,
        recompute_first_order: bool = True,
        recompute_clear_apertures: bool = True,
    ) -> MultiOpticalSystem:
        """显式计算并缓存运行态数据。

        材料表、frame、一阶数据和自动口径可以按需更新。默认行为保持
        原有的完整 prepare() 语义；动态参数只改变其中一部分时，调用方
        可以复用其余缓存，避免重复编译静态数据。
        """
        from .first_order import compute_first_order_data
        from .frames import compute_frame_data
        from ._runtime import default_device

        if recompute_materials or self._material_data is None:
            self._material_data = compile_batched_material_data(self, device=default_device(self))
        if recompute_frame or self.frame_data is None:
            self.frame_data = compute_frame_data(self)
        if recompute_first_order or self.first_order_data is None:
            self.first_order_data = compute_first_order_data(self)
        if recompute_clear_apertures or self.clear_aperture_data is None:
            self._prepare_auto_clear_apertures()
        return self

    def _prepare_auto_clear_apertures(self) -> None:
        """计算 auto 面的净口径，供 layout 等后续分析复用。"""
        import torch

        from .apertures import calculate_clear_apertures
        from .sampling import ExplicitPupilSampler

        if self.tracer is None:
            raise ValueError("system.prepare() requires system.tracer before clear aperture calculation.")

        # 使用 32 条圆周边缘入瞳光线估计各表面的需求半口径。
        angles = torch.arange(32, dtype=torch.float64) * (2.0 * torch.pi / 32.0)
        sampler = ExplicitPupilSampler(
            pupil_coordinates=torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        )
        result = calculate_clear_apertures(
            self,
            sampler=sampler,
            keep_trace_result=False,
        )
        self.clear_aperture_data = result

        # 只回填 Automatic Semi-Diameter，Surface Aperture 是否裁剪由自身类型决定。
        for aperture_index, surface_index in enumerate(result.surface_indices):
            surface = self.surfaces[surface_index]
            if surface.semi_diameter_solve != "auto":
                continue
            # 参照Zemax, Coordinate Break 的孔径不参与自动估计，始终保持 0。
            if isinstance(surface, CoordinateBreak):
                surface.semi_diameter = 0.0
                continue
            surface.semi_diameter = result.semi_diameter[:, aperture_index].detach().clone()

    def trace(
        self,
        *,
        sampler: PupilSampler | None = None,
        aimer: RayAimer | None = None,
        options: TraceOptions | None = None,
    ) -> TraceResult:
        """对当前系统声明的 design x field x wavelength 批量执行追迹。"""

        return self.tracer.batched_trace(
            self,
            sampler=sampler,
            aimer=aimer,
            options=options,
        )

    @property
    def analysis(self):
        """按需构造绑定到当前系统的分析入口。"""

        if self._analysis_hub is None:
            from .analysis import AnalysisHub

            self._analysis_hub = AnalysisHub(self)
        return self._analysis_hub
