from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from .types import Scalar


def _to_fp64_tensor(value: torch.Tensor, *, label: str, device: torch.device | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor.")
    tensor = value.to(dtype=torch.float64)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device=device)
    return tensor


def _broadcast_xy_tensors(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_tensor = _to_fp64_tensor(x, label="x")
    y_tensor = _to_fp64_tensor(y, label="y", device=x_tensor.device)
    return torch.broadcast_tensors(x_tensor, y_tensor)


def _split_vector_components(vector: torch.Tensor, *, label: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将输入的向量分解为三个分量，并转换为FP64张量。"""
    tensor = _to_fp64_tensor(vector, label=label)
    if tensor.shape[-1] != 3:
        raise ValueError(f"{label} tensor must have a trailing dimension of size 3.")
    return tensor[..., 0], tensor[..., 1], tensor[..., 2]


@dataclass(slots=True)
class BaseGeometry(ABC):
    label: str | None = None

    @abstractmethod
    def sag(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算几何表面的矢高。"""
        raise NotImplementedError

    @abstractmethod
    def normal(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算表面法向量的三个分量，返回值应当已经归一化。"""
        raise NotImplementedError

    @abstractmethod
    def intersect(self, ray_origin: torch.Tensor, ray_direction: torch.Tensor) -> torch.Tensor:
        """计算射线与表面的交点位置，返回值为从ray_origin沿ray_direction方向到交点的距离。"""
        raise NotImplementedError


@dataclass(slots=True)
class StandardGeometry(BaseGeometry):
    """
    标准球面/圆锥面几何，使用半径和离心率参数化。
    radius: 球面曲率半径，单位与x和y坐标相同。radius=0表示平面。
    conic: 离心率，conic=0表示球面，conic<0表示椭球面，conic>0表示双曲面。
    """
    radius: Scalar = 0.0
    conic: Scalar = 0.0

    @staticmethod
    def _plane_sag(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_tensor, _ = _broadcast_xy_tensors(x, y)
        return torch.zeros_like(x_tensor, dtype=torch.float64)

    @staticmethod
    def _plane_normal(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_tensor, _ = _broadcast_xy_tensors(x, y)
        zeros = torch.zeros_like(x_tensor, dtype=torch.float64)
        ones = torch.ones_like(x_tensor, dtype=torch.float64)
        return zeros, zeros, ones

    @staticmethod
    def _plane_intersection(ray_origin: torch.Tensor, ray_direction: torch.Tensor) -> torch.Tensor:
        _, _, origin_z = _split_vector_components(ray_origin, label="ray_origin")
        _, _, direction_z = _split_vector_components(ray_direction, label="ray_direction")

        zeros = torch.zeros_like(origin_z, dtype=torch.float64)
        nan_tensor = torch.full_like(origin_z, torch.nan)
        origin_on_plane = torch.isclose(origin_z, zeros)
        parallel_to_plane = torch.isclose(direction_z, zeros)
        safe_direction_z = torch.where(parallel_to_plane, torch.ones_like(direction_z), direction_z)

        intersection = -origin_z / safe_direction_z
        intersection = torch.where(origin_on_plane, zeros, intersection)
        intersection = torch.where(parallel_to_plane & ~origin_on_plane, nan_tensor, intersection)
        intersection = torch.where(intersection >= 0.0, intersection, nan_tensor)
        intersection = torch.where(origin_on_plane, zeros, intersection)
        return intersection

    def _is_plane(self) -> bool:
        return float(self.radius) == 0.0

    def _surface_sqrt_term(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor) -> torch.Tensor:
        """sqrt_term辅助函数，sqrt(1 - (1+k) * (x^2 + y^2) / R^2)。"""
        radius_tensor = torch.as_tensor(self.radius, dtype=torch.float64, device=x_tensor.device)
        rho_sq = x_tensor * x_tensor + y_tensor * y_tensor
        return torch.sqrt(1.0 - (1.0 + float(self.conic)) * rho_sq / (radius_tensor * radius_tensor))

    def sag(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """矢高计算公式： 
        z = (x^2 + y^2) / (R * (1 + sqrt_term))
        其中sqrt_term = sqrt(1 - (1+k) * (x^2 + y^2) / R^2)。
        """
        if self._is_plane():
            return self._plane_sag(x, y)

        x_tensor, y_tensor = _broadcast_xy_tensors(x, y)
        radius_tensor = torch.as_tensor(self.radius, dtype=torch.float64, device=x_tensor.device)
        rho_sq = x_tensor * x_tensor + y_tensor * y_tensor
        sqrt_term = self._surface_sqrt_term(x_tensor, y_tensor)
        return rho_sq / (radius_tensor * (1.0 + sqrt_term))

    def normal(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """法向量计算公式：
        nx = -dz/dx = -x / (R * sqrt_term)
        ny = -dz/dy = -y / (R * sqrt_term)
        nz = 1 / sqrt_term
        其中 sqrt_term = sqrt(1 - (1+k) * (x^2 + y^2) / R^2)。最后需要将法向量归一化。
        """

        if self._is_plane():
            return self._plane_normal(x, y)

        x_tensor, y_tensor = _broadcast_xy_tensors(x, y)
        radius_tensor = torch.as_tensor(self.radius, dtype=torch.float64, device=x_tensor.device)
        sqrt_term = self._surface_sqrt_term(x_tensor, y_tensor)

        dz_dx = x_tensor / (radius_tensor * sqrt_term)
        dz_dy = y_tensor / (radius_tensor * sqrt_term)
        nx = -dz_dx
        ny = -dz_dy
        nz = torch.ones_like(nx, dtype=torch.float64)

        norm = torch.sqrt(nx * nx + ny * ny + nz * nz)
        return nx / norm, ny / norm, nz / norm

    def intersect(self, ray_origin: torch.Tensor, ray_direction: torch.Tensor) -> torch.Tensor:
        """求解射线与标准球面/圆锥面交点。

        局部面形的隐式方程为：
        x^2 + y^2 + (1 + k) z^2 - 2 R z = 0
        其中 R 为曲率半径，k 为 conic。
        """

        if self._is_plane():
            return self._plane_intersection(ray_origin, ray_direction)

        # 将输入分解为分量，并转换为FP64张量
        ox, oy, oz = _split_vector_components(ray_origin, label="ray_origin")
        dx, dy, dz = _split_vector_components(ray_direction, label="ray_direction")

        # 计算标准面隐式方程代入光线后的二次方程
        radius_tensor = torch.as_tensor(self.radius, dtype=torch.float64, device=ox.device)
        conic_factor = 1.0 + float(self.conic)
        quadratic_a = dx * dx + dy * dy + conic_factor * dz * dz
        half_b = ox * dx + oy * dy + (conic_factor * oz - radius_tensor) * dz
        quadratic_c = ox * ox + oy * oy + conic_factor * oz * oz - 2.0 * radius_tensor * oz

        # 求解二次方程
        discriminant = half_b * half_b - quadratic_a * quadratic_c
        sqrt_discriminant = torch.sqrt(torch.clamp_min(discriminant, 0.0))
        safe_a = torch.where(torch.isclose(quadratic_a, torch.zeros_like(quadratic_a)), torch.ones_like(quadratic_a), quadratic_a)

        root_near = (-half_b - sqrt_discriminant) / safe_a
        root_far = (-half_b + sqrt_discriminant) / safe_a

        # 选择非负的最小根作为交点距离，如果两个根都是负的，则返回NaN表示没有交点。
        positive_near = torch.where(root_near >= 0.0, root_near, torch.full_like(root_near, torch.inf))
        positive_far = torch.where(root_far >= 0.0, root_far, torch.full_like(root_far, torch.inf))
        intersection = torch.minimum(positive_near, positive_far)

        nan_tensor = torch.full_like(intersection, torch.nan)
        intersection = torch.where((discriminant >= 0.0) & (~torch.isclose(quadratic_a, torch.zeros_like(quadratic_a))), intersection, nan_tensor)
        intersection = torch.where(torch.isinf(intersection), nan_tensor, intersection)
        return intersection


@dataclass(slots=True, init=False)
class PlaneGeometry(StandardGeometry):
    def __init__(self, label: str | None = None) -> None:
        StandardGeometry.__init__(self, label=label, radius=0.0, conic=0.0)


@dataclass(slots=True)
class EvenAsphereGeometry(StandardGeometry):
    coefficients: tuple[Scalar, ...] = field(default_factory=tuple)

    def _has_asphere_coefficients(self) -> bool:
        return len(self.coefficients) > 0

    def sag(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._has_asphere_coefficients():
            raise NotImplementedError("EvenAsphereGeometry with non-empty coefficients is not implemented yet.")
        return StandardGeometry.sag(self, x, y)

    def normal(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._has_asphere_coefficients():
            raise NotImplementedError("EvenAsphereGeometry with non-empty coefficients is not implemented yet.")
        return StandardGeometry.normal(self, x, y)

    def intersect(self, ray_origin: torch.Tensor, ray_direction: torch.Tensor) -> torch.Tensor:
        if self._has_asphere_coefficients():
            raise NotImplementedError("EvenAsphereGeometry with non-empty coefficients is not implemented yet.")
        return StandardGeometry.intersect(self, ray_origin, ray_direction)


@dataclass(slots=True)
class ParaxialGeometry(BaseGeometry):
    focal_length: Scalar = 0.0

    def sag(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_tensor, _ = _broadcast_xy_tensors(x, y)
        return torch.zeros_like(x_tensor, dtype=torch.float64)

    def normal(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_tensor, _ = _broadcast_xy_tensors(x, y)
        zeros = torch.zeros_like(x_tensor, dtype=torch.float64)
        ones = torch.ones_like(x_tensor, dtype=torch.float64)
        return zeros, zeros, ones

    def intersect(self, ray_origin: torch.Tensor, ray_direction: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
