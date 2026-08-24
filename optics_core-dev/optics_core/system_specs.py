from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from math import prod
from typing import Any, Iterable, Iterator, Sequence

import torch

from .materials import AIR, MaterialLibrary
from .surfaces import Surface, SurfaceSequence
from .types import ApertureType, FieldType, Scalar


@dataclass(slots=True)
class Vignetting:
    """描述视场点在渐晕采样下的缩放与偏移。"""

    x_scale: Scalar = 1.0
    y_scale: Scalar = 1.0
    x_shift: Scalar = 0.0
    y_shift: Scalar = 0.0


@dataclass(slots=True)
class FieldPoint:
    """定义 tracing 与 analysis 使用的单个视场采样点。"""

    x: Scalar = 0.0
    y: Scalar = 0.0
    weight: Scalar = 1.0
    label: str | None = None
    vignetting: Vignetting | None = None


class FieldSequence(Iterable[FieldPoint]):
    """按顺序保存一组视场点。"""

    def __init__(self, field_type: FieldType = "angle") -> None:
        self.field_type = field_type
        self._items: list[FieldPoint] = []

    def set_type(self, field_type: FieldType) -> None:
        """切换当前视场序列的坐标语义。"""

        self.field_type = field_type

    def append(self, field: FieldPoint) -> None:
        self._items.append(field)

    def add(
        self,
        *,
        x: Scalar = 0.0,
        y: Scalar = 0.0,
        weight: Scalar = 1.0,
        label: str | None = None,
        vignetting: Vignetting | None = None,
    ) -> FieldPoint:
        field = FieldPoint(x=x, y=y, weight=weight, label=label, vignetting=vignetting)
        self.append(field)
        return field

    def __iter__(self) -> Iterator[FieldPoint]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> FieldPoint:
        return self._items[index]


@dataclass(slots=True)
class Wavelength:
    """描述单个波长采样及其权重。"""

    value_um: Scalar
    weight: Scalar = 1.0
    is_primary: bool = False
    label: str | None = None


class WavelengthSequence(Iterable[Wavelength]):
    """按顺序保存系统波长定义。"""

    def __init__(self) -> None:
        self._items: list[Wavelength] = []

    def append(self, wavelength: Wavelength) -> None:
        if wavelength.is_primary:
            for existing in self._items:
                existing.is_primary = False
        self._items.append(wavelength)

    def add(
        self,
        value_um: Scalar,
        *,
        weight: Scalar = 1.0,
        is_primary: bool = False,
        label: str | None = None,
    ) -> Wavelength:
        wavelength = Wavelength(
            value_um=value_um,
            weight=weight,
            is_primary=is_primary,
            label=label,
        )
        self.append(wavelength)
        return wavelength

    @property
    def primary_index(self) -> int:
        for index in range(len(self._items) - 1, -1, -1):
            if self._items[index].is_primary:
                return index
        if not self._items:
            raise ValueError("At least one wavelength is required.")
        return 0

    @property
    def primary(self) -> Wavelength:
        return self._items[self.primary_index]

    def __iter__(self) -> Iterator[Wavelength]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Wavelength:
        return self._items[index]


@dataclass(slots=True)
class SystemAperture:
    """描述系统孔径口径类型及可选的 stop 信息。"""

    kind: ApertureType
    value: Scalar | torch.Tensor
    stop_surface: int | None = None
    label: str | None = None

    def design_slice(self, start: int, stop: int) -> SystemAperture:
        """切片逐设计孔径值，标量孔径保持共享。"""
        value = self.value[start:stop] if isinstance(self.value, torch.Tensor) else self.value
        return SystemAperture(self.kind, value, self.stop_surface, self.label)


@dataclass(slots=True)
class ParameterSpec:
    """描述参数向量中的一个槽位如何映射到光学参数。"""

    name: str
    path: str
    default: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty.")
        if not self.path:
            raise ValueError("path must not be empty.")

    def default_value(self) -> Any:
        if self.default is None:
            return 0.0
        return self.default


class ParameterSchema(Iterable[ParameterSpec]):
    """集中管理参数向量到实际光设参数的映射表。"""

    def __init__(self, specs: Sequence[ParameterSpec] = ()) -> None:
        self._specs: list[ParameterSpec] = []
        self._spec_map: dict[str, ParameterSpec] = {}
        self._spec_indices: dict[str, int] = {}
        for spec in specs:
            self.add_spec(spec)

    def add(
        self,
        name: str,
        path: str,
        *,
        default: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ParameterSpec:
        return self.add_spec(
            ParameterSpec(
                name=name,
                path=path,
                default=default,
                metadata=dict(metadata or {}),
            )
        )

    def add_spec(self, spec: ParameterSpec) -> ParameterSpec:
        if spec.name in self._spec_map:
            raise ValueError(f"Duplicate parameter spec name: {spec.name!r}")

        self._specs.append(spec)
        self._spec_map[spec.name] = spec
        self._spec_indices[spec.name] = len(self._specs) - 1
        return spec

    @property
    def parameter_count(self) -> int:
        return len(self._specs)

    @property
    def spec_names(self) -> list[str]:
        return [spec.name for spec in self._specs]

    def spec(self, name: str) -> ParameterSpec:
        return self._spec_map[name]

    def index_of(self, name: str) -> int:
        return self._spec_indices[name]

    def slice_of(self, name: str) -> slice:
        index = self.index_of(name)
        return slice(index, index + 1)

    def default_vector(self) -> list[Any]:
        return [spec.default_value() for spec in self._specs]

    def vector_from_mapping(
        self,
        values_by_name: dict[str, Any],
        *,
        base_vector: list[Any] | None = None,
    ) -> list[Any]:
        vector = list(self.default_vector() if base_vector is None else base_vector)
        if len(vector) != self.parameter_count:
            raise ValueError("base parameter vector length must match schema.parameter_count.")
        for name, payload in values_by_name.items():
            vector[self.index_of(name)] = payload
        return vector

    def __iter__(self) -> Iterator[ParameterSpec]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)


@dataclass(slots=True)
class ParameterSweepAxis:
    """描述一个用于枚举参数向量的 helper 轴。"""

    parameter: str
    values: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.parameter:
            raise ValueError("parameter must not be empty.")
        if not self.values:
            raise ValueError("values must not be empty.")


@dataclass(slots=True)
class ParameterVectorBatch(Iterable[list[Any]]):
    """保存一组共享映射表下的参数向量。"""

    schema: ParameterSchema
    vectors: list[list[Any]] = field(default_factory=list)
    grid_shape: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_vectors = [list(vector) for vector in self.vectors]
        if not normalized_vectors:
            raise ValueError("At least one parameter vector is required.")
        for index, vector in enumerate(normalized_vectors):
            if len(vector) != self.schema.parameter_count:
                raise ValueError(f"parameter vector {index} length must match schema.parameter_count.")
        if self.grid_shape is not None and prod(self.grid_shape) != len(normalized_vectors):
            raise ValueError("grid_shape must match the number of parameter vectors.")
        self.vectors = normalized_vectors

    @property
    def system_count(self) -> int:
        return len(self.vectors)

    @property
    def parameter_count(self) -> int:
        return self.schema.parameter_count

    def with_schema(self, schema: ParameterSchema) -> ParameterVectorBatch:
        if schema.parameter_count != self.parameter_count:
            raise ValueError("schema.parameter_count must match the current vector length.")
        return ParameterVectorBatch(
            schema=schema,
            vectors=[list(vector) for vector in self.vectors],
            grid_shape=self.grid_shape,
            metadata=dict(self.metadata),
        )

    def __iter__(self) -> Iterator[list[Any]]:
        return iter(self.vectors)

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, index: int) -> list[Any]:
        return self.vectors[index]


@dataclass(slots=True)
class ParameterVectorBatchRange(Iterable[list[Any]]):
    """引用父参数批量中的连续 design 区间，不复制参数向量。"""

    parent: ParameterVectorBatch | ParameterVectorBatchRange
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start or self.stop > self.parent.system_count:
            raise IndexError("parameter batch range is out of bounds.")

    @property
    def schema(self) -> ParameterSchema:
        return self.parent.schema

    @property
    def system_count(self) -> int:
        return self.stop - self.start

    @property
    def parameter_count(self) -> int:
        return self.parent.parameter_count

    @property
    def metadata(self) -> dict[str, Any]:
        return self.parent.metadata

    def __iter__(self) -> Iterator[list[Any]]:
        for index in range(self.system_count):
            yield self.parent[self.start + index]

    def __len__(self) -> int:
        return self.system_count

    def __getitem__(self, index: int) -> list[Any]:
        normalized_index = index if index >= 0 else self.system_count + index
        if normalized_index < 0 or normalized_index >= self.system_count:
            raise IndexError("parameter vector index is out of range.")
        return self.parent[self.start + normalized_index]


def build_parameter_vector_grid(
    schema: ParameterSchema,
    axes: Sequence[ParameterSweepAxis],
    *,
    base_vector: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ParameterVectorBatch:
    """用 mesh-grid 风格的枚举轴生成参数向量列表。"""

    normalized_base = list(schema.default_vector() if base_vector is None else base_vector)
    if len(normalized_base) != schema.parameter_count:
        raise ValueError("base parameter vector length must match schema.parameter_count.")

    if not axes:
        return ParameterVectorBatch(
            schema=schema,
            vectors=[normalized_base],
            grid_shape=(1,),
            metadata=dict(metadata or {}),
        )

    vectors: list[list[Any]] = []
    for axis_values in product(*(axis.values for axis in axes)):
        vector = list(normalized_base)
        for axis, axis_value in zip(axes, axis_values, strict=True):
            vector[schema.index_of(axis.parameter)] = axis_value
        vectors.append(vector)
    return ParameterVectorBatch(
        schema=schema,
        vectors=vectors,
        grid_shape=tuple(len(axis.values) for axis in axes),
        metadata={
            "axes": tuple(axis.parameter for axis in axes),
            **(metadata or {}),
        },
    )


class OpticalArchitecture:
    """持有可被一个或多个系统复用的共享面拓扑。"""

    def __init__(
        self,
        name: str = "unnamed_architecture",
        *,
        surfaces: Sequence[Surface] = (),
        materials: MaterialLibrary | None = None,
        object_distance_mm: float = math.inf,
        afocal_image_space: bool = False,
    ) -> None:
        self.name = name
        self.metadata: dict[str, Any] = {}
        self.materials = materials or MaterialLibrary({"AIR": AIR})
        self.object_distance_mm = float(object_distance_mm)
        self.afocal_image_space = bool(afocal_image_space)
        self.surfaces = SurfaceSequence(owner=None, materials=self.materials)
        for surface in surfaces:
            self.surfaces.add(surface)

    @classmethod
    def from_sequence(
        cls,
        *,
        name: str = "unnamed_architecture",
        surfaces: Sequence[Surface] = (),
        materials: MaterialLibrary | None = None,
        object_distance_mm: float = math.inf,
        afocal_image_space: bool = False,
    ) -> OpticalArchitecture:
        """从独立的面序列构造共享光学拓扑。"""

        return cls(
            name=name,
            surfaces=surfaces,
            materials=materials,
            object_distance_mm=object_distance_mm,
            afocal_image_space=afocal_image_space,
        )

    @property
    def surface_count(self) -> int:
        return len(self.surfaces)

    def __len__(self) -> int:
        return len(self.surfaces)
