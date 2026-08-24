from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .materials import AIR, Material
from .types import TraceDirection

if TYPE_CHECKING:
    from .system import MultiOpticalSystem


@dataclass(slots=True)
class BatchedMaterialData:
    """追迹使用的逐设计材料索引和折射率表。"""

    material_index: torch.Tensor
    refractive_index_table: torch.Tensor
    wavelength_um: torch.Tensor
    wavelength_weights: torch.Tensor
    material_names: tuple[str, ...]
    device: torch.device

    def design_slice(self, start: int, stop: int) -> BatchedMaterialData:
        return BatchedMaterialData(
            material_index=self.material_index[start:stop],
            refractive_index_table=self.refractive_index_table,
            wavelength_um=self.wavelength_um,
            wavelength_weights=self.wavelength_weights,
            material_names=self.material_names,
            device=self.device,
        )

    def surface_indices(
        self,
        surface_index: int,
        *,
        direction: TraceDirection,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回当前面的入射和出射材料索引。"""
        current = self.material_index[:, surface_index]
        previous = torch.zeros_like(current) if surface_index == 0 else self.material_index[:, surface_index - 1]
        if direction == "forward":
            return previous, current
        return current, previous

    def refractive_index(
        self,
        material_index: torch.Tensor,
        wavelength_index: torch.Tensor,
    ) -> torch.Tensor:
        """按 design 和波长列号查询折射率。"""
        expanded_material = material_index.reshape(
            (material_index.shape[0],) + (1,) * (wavelength_index.ndim - 1)
        ).expand_as(wavelength_index)
        return self.refractive_index_table[expanded_material, wavelength_index]


def compile_batched_material_data(
    system: MultiOpticalSystem,
    *,
    device: torch.device,
) -> BatchedMaterialData:
    """把共享表面和材料参数编译为 GPU/CPU tensor。"""
    parameter_by_path = {
        spec.path: system.parameter_schema.index_of(spec.name)
        for spec in system.parameter_schema
    }
    materials: list[Material] = [AIR]
    material_number = {AIR.name: 0}
    rows: list[list[int]] = []

    # 1. 遍历全部系统的全部表面，收集材料索引
    for design_index in range(system.system_count):
        row: list[int] = []
        for surface_index, surface in enumerate(system.surfaces):
            parameter_index = parameter_by_path.get(f"surface[{surface_index}].gap.medium")
            medium = (
                surface.gap.medium
                if parameter_index is None
                else system.parameters[design_index][parameter_index]
            )
            # 从材料库中找到材料，添加到 materials 列表中，并记录材料索引
            resolved = system._resolve_material_reference(medium) or AIR
            if resolved.name not in material_number:
                material_number[resolved.name] = len(materials)
                materials.append(resolved)
            row.append(material_number[resolved.name])
        rows.append(row)

    # 2. 按系统波长顺序计算折射率表
    wavelength_um = torch.tensor(
        [float(wavelength.value_um) for wavelength in system.wavelengths],
        dtype=torch.float64,
        device=device,
    )

    refractive_index_table = torch.stack(
        [material.refractive_index(wavelength_um) for material in materials],
        dim=0,
    )

    # 3. 构建 BatchedMaterialData 对象
    return BatchedMaterialData(
        material_index=torch.tensor(rows, dtype=torch.int64, device=device),
        refractive_index_table=refractive_index_table,
        wavelength_um=wavelength_um,
        wavelength_weights=torch.tensor(
            [float(wavelength.weight) for wavelength in system.wavelengths],
            dtype=torch.float64,
            device=device,
        ),
        material_names=tuple(material.name for material in materials),
        device=device,
    )
