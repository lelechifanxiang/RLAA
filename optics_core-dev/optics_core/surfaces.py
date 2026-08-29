from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Literal

from .geometries import EvenAsphereGeometry, ParaxialGeometry, PlaneGeometry, StandardGeometry
from .materials import AIR, Material, MaterialLibrary
from .types import Scalar

SemiDiameterSolve = Literal["auto", "fixed"]
SurfaceApertureType = Literal["none", "floating"]


@dataclass(slots=True)
class CoordinateFrame:
    x: Scalar = 0.0
    y: Scalar = 0.0
    z: Scalar = 0.0
    rx: Scalar = 0.0
    ry: Scalar = 0.0
    rz: Scalar = 0.0


@dataclass(slots=True)
class Gap:
    thickness: Scalar = 0.0
    medium: Material | None = field(default_factory=lambda: AIR)
    comment: str | None = None


@dataclass(slots=True)
class Surface:
    geometry: Any
    gap: Gap = field(default_factory=Gap)
    semi_diameter: Scalar | None = None
    semi_diameter_solve: SemiDiameterSolve = "auto"
    aperture_type: SurfaceApertureType = "none"
    frame: CoordinateFrame = field(default_factory=CoordinateFrame)
    label: str | None = None
    is_stop: bool = False


def _init_surface(
    surface: Surface,
    *,
    geometry: Any,
    gap: Gap | None = None,
    thickness: Scalar = 0.0,
    medium: Material | None = None,
    semi_diameter: Scalar | None = None,
    semi_diameter_solve: SemiDiameterSolve = "auto",
    aperture_type: SurfaceApertureType = "none",
    frame: CoordinateFrame | None = None,
    label: str | None = None,
    is_stop: bool = False,
) -> None:
    """集中初始化 Surface 公共字段，减少各类表面的重复转发。"""
    Surface.__init__(
        surface,
        geometry=geometry,
        gap=gap or Gap(thickness=thickness, medium=medium),
        semi_diameter=semi_diameter,
        semi_diameter_solve=semi_diameter_solve,
        aperture_type=aperture_type,
        frame=frame or CoordinateFrame(),
        label=label,
        is_stop=is_stop,
    )


class ObjectSurface(Surface):
    def __init__(
        self,
        *,
        gap: Gap | None = None,
        semi_diameter: Scalar | None = None,
        semi_diameter_solve: SemiDiameterSolve = "auto",
        aperture_type: SurfaceApertureType = "none",
        label: str | None = None,
    ) -> None:
        _init_surface(
            self,
            geometry=PlaneGeometry(label="object"),
            gap=gap or Gap(thickness=float("inf")),
            semi_diameter=semi_diameter,
            semi_diameter_solve=semi_diameter_solve,
            aperture_type=aperture_type,
            label=label,
        )


class SphereSurface(Surface):
    def __init__(
        self,
        *,
        radius: Scalar,
        thickness: Scalar,
        medium: Material | None = None,
        semi_diameter: Scalar | None = None,
        semi_diameter_solve: SemiDiameterSolve = "auto",
        aperture_type: SurfaceApertureType = "none",
        conic: Scalar = 0.0,
        label: str | None = None,
        is_stop: bool = False,
    ) -> None:
        _init_surface(
            self,
            geometry=StandardGeometry(radius=radius, conic=conic),
            thickness=thickness,
            medium=medium,
            semi_diameter=semi_diameter,
            semi_diameter_solve=semi_diameter_solve,
            aperture_type=aperture_type,
            label=label,
            is_stop=is_stop,
        )


class EvenAsphereSurface(Surface):
    def __init__(
        self,
        *,
        radius: Scalar,
        thickness: Scalar,
        medium: Material | None = None,
        semi_diameter: Scalar | None = None,
        semi_diameter_solve: SemiDiameterSolve = "auto",
        aperture_type: SurfaceApertureType = "none",
        conic: Scalar = 0.0,
        coefficients: tuple[Scalar, ...] = (),
        label: str | None = None,
        is_stop: bool = False,
    ) -> None:
        _init_surface(
            self,
            geometry=EvenAsphereGeometry(radius=radius, conic=conic, coefficients=coefficients),
            thickness=thickness,
            medium=medium,
            semi_diameter=semi_diameter,
            semi_diameter_solve=semi_diameter_solve,
            aperture_type=aperture_type,
            label=label,
            is_stop=is_stop,
        )


class ParaxialSurface(Surface):
    def __init__(
        self,
        *,
        focal_length: Scalar,
        thickness: Scalar = 0.0,
        medium: Material | None = None,
        semi_diameter: Scalar | None = None,
        semi_diameter_solve: SemiDiameterSolve = "auto",
        aperture_type: SurfaceApertureType = "none",
        label: str | None = None,
        is_stop: bool = False,
    ) -> None:
        _init_surface(
            self,
            geometry=ParaxialGeometry(focal_length=focal_length),
            thickness=thickness,
            medium=medium,
            semi_diameter=semi_diameter,
            semi_diameter_solve=semi_diameter_solve,
            aperture_type=aperture_type,
            label=label,
            is_stop=is_stop,
        )


class ImageSurface(Surface):
    def __init__(self, *, label: str | None = None) -> None:
        _init_surface(
            self,
            geometry=PlaneGeometry(label="image"),
            label=label,
        )


class CoordinateBreak(Surface):
    def __init__(
        self,
        *,
        thickness: Scalar = 0.0,
        medium: Material | None = None,
        frame: CoordinateFrame | None = None,
        order_flag: int = 0,
        label: str | None = None,
    ) -> None:
        _init_surface(
            self,
            geometry=PlaneGeometry(label="coordinate_break"),
            thickness=thickness,
            medium=medium,
            frame=frame or CoordinateFrame(),
            label=label,
        )
        self.order_flag = int(order_flag)


class SurfaceSequence(Iterable[Surface]):
    def __init__(
        self,
        owner: Any | None = None,
        materials: MaterialLibrary | None = None,
        items: list[Surface] | None = None,
    ) -> None:
        self._owner = owner
        self._items: list[Surface] = items if items is not None else []
        self._materials = materials or MaterialLibrary({"AIR": AIR})

    def __iter__(self) -> Iterator[Surface]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Surface:
        return self._items[index]

    def bind(self, *, owner: Any | None = None, materials: MaterialLibrary | None = None) -> SurfaceSequence:
        return SurfaceSequence(
            owner=owner,
            materials=materials or self._materials,
            items=self._items,
        )

    def add(self, surface: Surface, *, index: int | None = None) -> Surface:
        if index is None:
            self._items.append(surface)
            index = len(self._items) - 1
        else:
            self._items.insert(index, surface)
        if surface.is_stop:
            self.set_stop(index)
        return surface

    def add_object(self, **kwargs: Any) -> ObjectSurface:
        return self.add(ObjectSurface(**kwargs))

    def add_sphere(self, **kwargs: Any) -> SphereSurface:
        kwargs["medium"] = self._materials.resolve(kwargs.get("medium"))
        return self.add(SphereSurface(**kwargs))

    def add_even_asphere(self, **kwargs: Any) -> EvenAsphereSurface:
        kwargs["medium"] = self._materials.resolve(kwargs.get("medium"))
        return self.add(EvenAsphereSurface(**kwargs))

    def add_paraxial(self, **kwargs: Any) -> ParaxialSurface:
        kwargs["medium"] = self._materials.resolve(kwargs.get("medium"))
        return self.add(ParaxialSurface(**kwargs))

    def add_image(self, *, label: str | None = None) -> ImageSurface:
        return self.add(ImageSurface(label=label))

    def add_coordinate_break(self, **kwargs: Any) -> CoordinateBreak:
        kwargs["medium"] = self._materials.resolve(kwargs.get("medium"))
        return self.add(CoordinateBreak(**kwargs))

    def set_stop(self, index: int) -> None:
        for surface_index, surface in enumerate(self._items):
            surface.is_stop = surface_index == index

    @property
    def stop_index(self) -> int | None:
        for surface_index, surface in enumerate(self._items):
            if surface.is_stop:
                return surface_index
        return None
