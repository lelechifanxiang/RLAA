from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from .rays import RayAimingResult
from .types import ArrayLike, PupilDistribution

if TYPE_CHECKING:
    from .system import MultiOpticalSystem
    from .system_specs import FieldPoint, Wavelength


@dataclass(slots=True)
class SamplingResult:
    pupil_coordinates: torch.Tensor | None = None
    weights: ArrayLike | None = None
    pattern: PupilDistribution | str = "gaussian"
    chief_ray_index: int | None = None
    sample_ray_count: int | None = None


def _append_reference_chief_ray(
    pupil_coordinates: torch.Tensor,
    weights: torch.Tensor,
    *,
    pattern: PupilDistribution | str,
) -> SamplingResult:
    """在末尾追加一条 reference chief ray，普通采样数量保持可追踪。"""
    coordinates = torch.as_tensor(pupil_coordinates, dtype=torch.float64)
    if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
        raise ValueError("pupil_coordinates must have shape (ray_count, 2).")
    weight_tensor = torch.as_tensor(weights, dtype=torch.float64, device=coordinates.device).reshape(-1)
    if weight_tensor.shape[0] != coordinates.shape[0]:
        raise ValueError("weights must have one value for each pupil coordinate.")

    sample_ray_count = coordinates.shape[0]

    # 准备额外的主光线
    chief_coordinate = torch.zeros((1, 2), dtype=torch.float64, device=coordinates.device)
    chief_weight = torch.zeros((1,), dtype=torch.float64, device=coordinates.device)

    # 主光线加入到采样末尾
    coordinates = torch.cat((coordinates, chief_coordinate), dim=0)
    weight_tensor = torch.cat((weight_tensor, chief_weight), dim=0)

    return SamplingResult(
        pupil_coordinates=coordinates,
        weights=weight_tensor,
        pattern=pattern,
        chief_ray_index=sample_ray_count,
        sample_ray_count=sample_ray_count,
    )

class PupilSampler(ABC):
    pattern: PupilDistribution | str

    @abstractmethod
    def sample(self) -> SamplingResult:
        raise NotImplementedError


@dataclass(slots=True)
class ExplicitPupilSampler(PupilSampler):
    """显式入瞳采样器，直接使用给定的坐标和权重。"""
    pupil_coordinates: torch.Tensor
    pattern: str = "fixed"

    def sample(self) -> SamplingResult:
        return SamplingResult(
            pupil_coordinates=self.pupil_coordinates,
            weights=torch.ones(self.pupil_coordinates.shape[0], dtype=torch.float64),
            pattern=self.pattern,
            chief_ray_index=None,
            sample_ray_count=self.pupil_coordinates.shape[0],
        )


@dataclass(slots=True)
class GaussianPupilSampler(PupilSampler):
    """高斯入瞳采样器，生成单位圆内的随机采样点。"""
    rings: int = 6
    arms: int = 8
    pattern: PupilDistribution = "gaussian"

    def sample(self) -> SamplingResult:
        raise NotImplementedError


@dataclass(slots=True)
class SquarePupilSampler(PupilSampler):
    """平方入瞳采样器，生成单位矩形内的网格采样点。"""
    nx: int = 16
    ny: int = 16
    pattern: PupilDistribution = "square"

    def sample(self) -> SamplingResult:
        px_grid, py_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.nx, dtype=torch.float64),
            torch.linspace(-1.0, 1.0, self.ny, dtype=torch.float64),
            indexing="ij",
        )
        coordinates = torch.stack((px_grid.reshape(-1), py_grid.reshape(-1)), dim=-1)
        weights = _square_pupil_area_weights(coordinates, nx=self.nx, ny=self.ny)
        return _append_reference_chief_ray(
            coordinates,
            weights,
            pattern=self.pattern,
        )


@dataclass(slots=True)
class HexapolarPupilSampler(PupilSampler):
    """六边入瞳采样器，生成单位圆内的同心多边形采样点。中心点 + rings 个同心环，每个环上有 arms * ring_index 个采样点。"""
    rings: int = 6
    arms: int = 6
    pattern: PupilDistribution = "hexapolar"

    def sample(self) -> SamplingResult:
        if self.rings < 0:
            raise ValueError("rings must be non-negative.")
        if self.arms <= 0:
            raise ValueError("arms must be positive.")

        coordinates = [(0.0, 0.0)]
        for ring_index in range(1, self.rings + 1):
            radius = float(ring_index) / float(self.rings) if self.rings > 0 else 0.0
            ray_count_on_ring = self.arms * ring_index
            for ray_index in range(ray_count_on_ring):
                angle = 2.0 * math.pi * float(ray_index) / float(ray_count_on_ring)
                coordinates.append((radius * math.cos(angle), radius * math.sin(angle)))

        coordinate_tensor = torch.tensor(coordinates, dtype=torch.float64)
        ray_count = coordinate_tensor.shape[0]
        return _append_reference_chief_ray(
            coordinate_tensor,
            torch.full((ray_count,), 1.0 / ray_count, dtype=torch.float64),
            pattern=self.pattern,
        )


@dataclass(slots=True)
class RandomPupilSampler(PupilSampler):
    """随机入瞳采样器，生成单位圆内的均匀随机采样点。"""
    ray_count: int = 1024
    seed: int | None = None
    pattern: PupilDistribution = "random"

    def sample(self) -> SamplingResult:
        if self.ray_count <= 0:
            raise ValueError("ray_count must be greater than zero.")

        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)

        accepted: list[torch.Tensor] = []
        accepted_count = 0
        while accepted_count < self.ray_count:
            candidates = torch.rand(
                (max(self.ray_count, 256), 2),
                generator=generator,
                dtype=torch.float64,
            ) * 2.0 - 1.0
            valid_mask = candidates[:, 0] * candidates[:, 0] + candidates[:, 1] * candidates[:, 1] <= 1.0
            valid_candidates = candidates[valid_mask]
            if valid_candidates.numel() == 0:
                continue
            accepted.append(valid_candidates)
            accepted_count += valid_candidates.shape[0]

        coordinates = torch.cat(accepted, dim=0)[: self.ray_count]
        return _append_reference_chief_ray(
            coordinates,
            torch.full((self.ray_count,), 1.0 / self.ray_count, dtype=torch.float64),
            pattern=self.pattern,
        )


def _square_pupil_area_weights(
    coordinates: torch.Tensor,
    *,
    nx: int,
    ny: int,
) -> torch.Tensor:
    """估算方形网格单元落在单位圆内的面积比例，并归一化为积分权重。"""
    half_dx = 1.0 / float(max(nx - 1, 1))
    half_dy = 1.0 / float(max(ny - 1, 1))
    abs_x = torch.abs(coordinates[:, 0])
    abs_y = torch.abs(coordinates[:, 1])

    nearest_x = torch.clamp(abs_x - half_dx, min=0.0)
    nearest_y = torch.clamp(abs_y - half_dy, min=0.0)
    nearest_radius_squared = nearest_x * nearest_x + nearest_y * nearest_y
    farthest_radius_squared = (abs_x + half_dx) ** 2 + (abs_y + half_dy) ** 2

    weights = torch.zeros(coordinates.shape[0], dtype=torch.float64, device=coordinates.device)
    inside = farthest_radius_squared <= 1.0
    boundary = (nearest_radius_squared < 1.0) & ~inside
    weights[inside] = 1.0

    # 只对圆周附近的网格单元做子采样，避免完整 pupil 产生大规模临时张量。
    if torch.any(boundary):
        sub_sample_count = 32
        sub_axis = (
            (torch.arange(sub_sample_count, dtype=torch.float64, device=coordinates.device) + 0.5)
            / sub_sample_count
            - 0.5
        )
        offset_x, offset_y = torch.meshgrid(
            sub_axis * (2.0 * half_dx),
            sub_axis * (2.0 * half_dy),
            indexing="ij",
        )
        offsets = torch.stack((offset_x.reshape(-1), offset_y.reshape(-1)), dim=-1)
        boundary_points = coordinates[boundary, None, :] + offsets[None, :, :]
        boundary_inside = torch.sum(boundary_points * boundary_points, dim=-1) <= 1.0
        weights[boundary] = boundary_inside.to(dtype=torch.float64).mean(dim=-1)

    return weights / weights.sum()


class RayAimer(ABC):
    @abstractmethod
    def aim(
        self,
        system: MultiOpticalSystem,
        fields: Sequence[FieldPoint],
        wavelengths: Sequence[Wavelength],
        sample: SamplingResult,
        *,
        cache: Mapping[str, Any] | None = None,
    ) -> RayAimingResult:
        raise NotImplementedError


@dataclass(slots=True)
class EntrancePupilAimer(RayAimer):
    max_iterations: int = 8
    tolerance: float = 1e-6
    cache_enabled: bool = True

    def aim(
        self,
        system: MultiOpticalSystem,
        fields: Sequence[FieldPoint],
        wavelengths: Sequence[Wavelength],
        sample: SamplingResult,
        *,
        cache: Mapping[str, Any] | None = None,
    ) -> RayAimingResult:
        raise NotImplementedError
