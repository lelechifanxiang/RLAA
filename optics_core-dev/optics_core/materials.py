from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

from .types import Scalar


@dataclass(slots=True)
class MaterialMetadata:
    density: Scalar | None = None
    cost: Scalar | None = None
    thermal_expansion: Scalar | None = None
    catalog: str | None = None


@dataclass(slots=True)
class Material(ABC):
    name: str
    metadata: MaterialMetadata = field(default_factory=MaterialMetadata)

    @abstractmethod
    def refractive_index(self, wavelength_um: torch.Tensor) -> torch.Tensor:
        """计算材料在给定波长下的折射率，输入输出均为FP64 tensor。"""
        raise NotImplementedError

    def extinction_coefficient(self, wavelength_um: torch.Tensor) -> torch.Tensor:
        """计算材料在给定波长下的消光系数，默认实现返回零张量。"""
        wavelength_tensor = _to_fp64_tensor(wavelength_um, label="wavelength_um")
        return torch.zeros_like(wavelength_tensor, dtype=torch.float64)


def _to_fp64_tensor(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor.")
    return value.to(dtype=torch.float64)


def _buchdahl_chromatic_coordinate_scalar(wavelength_um: float, reference_wavelength_um: float) -> float:
    delta = wavelength_um - reference_wavelength_um
    denominator = 1.0 + 2.5 * delta
    if denominator == 0.0:
        raise ValueError("Buchdahl chromatic coordinate is singular at this wavelength.")
    return delta / denominator


def _buchdahl_chromatic_coordinate(wavelength_um: torch.Tensor, reference_wavelength_um: float) -> torch.Tensor:
    delta = wavelength_um - float(reference_wavelength_um)
    denominator = 1.0 + 2.5 * delta
    if torch.any(denominator == 0.0).item():
        raise ValueError("Buchdahl chromatic coordinate is singular at this wavelength.")
    return delta / denominator


# bx, cx 对应zemax材料参数的Kx, Lx
def _sellmeier1_refractive_index(
    wavelength_um: torch.Tensor,
    b1: Scalar,
    b2: Scalar,
    b3: Scalar,
    c1: Scalar,
    c2: Scalar,
    c3: Scalar,
) -> torch.Tensor:
    wavelength_sq = wavelength_um * wavelength_um

    def sellmeier_term(b: Scalar, c: Scalar) -> torch.Tensor:
        denominator = wavelength_sq - float(c)
        if torch.any(denominator == 0.0).item():
            raise ValueError("Sellmeier 1 is singular at this wavelength.")
        return float(b) * wavelength_sq / denominator

    refractive_index_sq = 1.0 + sellmeier_term(b1, c1) + sellmeier_term(b2, c2) + sellmeier_term(b3, c3)
    if torch.any(refractive_index_sq < 0.0).item():
        raise ValueError("Sellmeier 1 produced a negative refractive-index square.")
    return torch.sqrt(refractive_index_sq)


def _schott_refractive_index(
    wavelength_um: torch.Tensor,
    a0: Scalar,
    a1: Scalar,
    a2: Scalar,
    a3: Scalar,
    a4: Scalar,
    a5: Scalar,
) -> torch.Tensor:
    wavelength_sq = wavelength_um * wavelength_um
    if torch.any(wavelength_sq == 0.0).item():
        raise ValueError("Schott formula is singular at zero wavelength.")

    inverse_sq = 1.0 / wavelength_sq
    refractive_index_sq = (
        float(a0)
        + float(a1) * wavelength_sq
        + float(a2) * inverse_sq
        + float(a3) * inverse_sq * inverse_sq
        + float(a4) * inverse_sq * inverse_sq * inverse_sq
        + float(a5) * inverse_sq * inverse_sq * inverse_sq * inverse_sq
    )
    if torch.any(refractive_index_sq < 0.0).item():
        raise ValueError("Schott formula produced a negative refractive-index square.")
    return torch.sqrt(refractive_index_sq)


def _abbe_dispersion_coefficient(nd: Scalar, vd: Scalar) -> tuple[float, float]:
    if vd == 0:
        raise ValueError("vd must not be zero.")

    wavelength_d, wavelength_f, wavelength_c = 0.5875618, 0.4861327, 0.6562725
    omega_f = _buchdahl_chromatic_coordinate_scalar(wavelength_f, wavelength_d)
    omega_c = _buchdahl_chromatic_coordinate_scalar(wavelength_c, wavelength_d)
    line_delta = omega_f - omega_c
    if line_delta == 0.0:
        raise ValueError("Fraunhofer line wavelengths must not collapse.")

    dispersion_delta = (float(nd) - 1.0) / float(vd)
    return wavelength_d, dispersion_delta / line_delta


@dataclass(slots=True)
class AbbeModelMaterial(Material):
    nd: Scalar = 1.5168
    vd: Scalar = 64.17

    def refractive_index(self, wavelength_um: torch.Tensor) -> torch.Tensor:
        wavelength_tensor = _to_fp64_tensor(wavelength_um, label="wavelength_um")
        wavelength_d, dispersion_coefficient = _abbe_dispersion_coefficient(self.nd, self.vd)
        return (
            torch.full_like(wavelength_tensor, float(self.nd), dtype=torch.float64)
            + dispersion_coefficient * _buchdahl_chromatic_coordinate(wavelength_tensor, wavelength_d)
        )


@dataclass(slots=True)
class RealMaterial(Material):
    formula: Literal["Sellmeier1", "Schott"] = "Sellmeier1"
    nd: Scalar | None = None
    vd: Scalar | None = None
    sellmeier_b1: Scalar = 0.0
    sellmeier_b2: Scalar = 0.0
    sellmeier_b3: Scalar = 0.0
    sellmeier_c1: Scalar = 0.0
    sellmeier_c2: Scalar = 0.0
    sellmeier_c3: Scalar = 0.0
    schott_a0: Scalar = 1.0
    schott_a1: Scalar = 0.0
    schott_a2: Scalar = 0.0
    schott_a3: Scalar = 0.0
    schott_a4: Scalar = 0.0
    schott_a5: Scalar = 0.0
    extinction: Scalar = 0.0

    def refractive_index(self, wavelength_um: torch.Tensor) -> torch.Tensor:
        wavelength_tensor = _to_fp64_tensor(wavelength_um, label="wavelength_um")
        if self.formula == "Schott":
            return _schott_refractive_index(
                wavelength_tensor,
                self.schott_a0,
                self.schott_a1,
                self.schott_a2,
                self.schott_a3,
                self.schott_a4,
                self.schott_a5,
            )
        return _sellmeier1_refractive_index(
            wavelength_tensor,
            self.sellmeier_b1,
            self.sellmeier_b2,
            self.sellmeier_b3,
            self.sellmeier_c1,
            self.sellmeier_c2,
            self.sellmeier_c3,
        )

    def extinction_coefficient(self, wavelength_um: torch.Tensor) -> torch.Tensor:
        wavelength_tensor = _to_fp64_tensor(wavelength_um, label="wavelength_um")
        return torch.full_like(wavelength_tensor, float(self.extinction), dtype=torch.float64)


class MaterialLibrary:
    def __init__(self, materials: Mapping[str, Material] | None = None) -> None:
        self._materials: dict[str, Material] = dict(materials or {})

    def register(self, material: Material) -> Material:
        self._materials[material.name] = material
        return material

    def get(self, name: str) -> Material:
        return self._materials[name]

    def resolve(self, material: Material | str | None) -> Material | None:
        if material is None:
            return None
        if isinstance(material, str):
            return self.get(material)
        return material

    def __contains__(self, name: str) -> bool:
        return name in self._materials

    def __iter__(self):
        return iter(self._materials.values())

    def load_real_materials_from_excel(self, path: str | Path) -> None:
        """从玻璃 Excel 表读取真实材料，并直接注册到当前材料库。"""

        resolved_path = Path(path)
        for row in _iter_glass_excel_rows(resolved_path):
            material = _real_material_from_glass_row(row, catalog=resolved_path.name)
            if material is not None:
                self.register(material)

    def load_builtin_real_materials(self) -> None:
        """读取项目自带的真实玻璃材料表。"""

        self.load_real_materials_from_excel(Path(__file__).resolve().parent / "material" / "GLASS.xlsx")


def _iter_glass_excel_rows(path: Path):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("读取 GLASS.xlsx 需要安装 pandas 和 openpyxl。") from exc

    table = pd.read_excel(path, header=11, engine="openpyxl")
    table = table.rename(columns=lambda column: str(column).strip())
    for _, row in table.iterrows():
        if _cell_text(row, "Name"):
            yield row


def _real_material_from_glass_row(row: Any, *, catalog: str) -> RealMaterial | None:
    name = _cell_text(row, "Name")
    formula = _cell_text(row, "公式")
    if formula not in {"Sellmeier1", "Schott"}:
        return None

    parameters = _glass_formula_parameters(row)
    metadata = MaterialMetadata(
        density=_cell_float_or_none(row, "密度"),
        cost=_cell_float_or_none(row, "成本"),
        catalog=catalog,
    )

    if formula == "Schott":
        return RealMaterial(
            name=name,
            formula="Schott",
            nd=_cell_float_or_none(row, "Nd"),
            vd=_cell_float_or_none(row, "Vd"),
            schott_a0=parameters["A0"],
            schott_a1=parameters["A1"],
            schott_a2=parameters["A2"],
            schott_a3=parameters["A3"],
            schott_a4=parameters["A4"],
            schott_a5=parameters["A5"],
            metadata=metadata,
        )

    return RealMaterial(
        name=name,
        formula="Sellmeier1",
        nd=_cell_float_or_none(row, "Nd"),
        vd=_cell_float_or_none(row, "Vd"),
        sellmeier_b1=parameters["K1"],
        sellmeier_c1=parameters["L1"],
        sellmeier_b2=parameters["K2"],
        sellmeier_c2=parameters["L2"],
        sellmeier_b3=parameters["K3"],
        sellmeier_c3=parameters["L3"],
        metadata=metadata,
    )


def _glass_formula_parameters(row: Any) -> dict[str, float]:
    parameters: dict[str, float] = {}
    for index in range(23, min(len(row) - 1, 35), 2):
        key = _cell_text_at(row, index)
        if key:
            parameters[key] = _required_cell_float_at(row, index + 1)
    return parameters


def _cell_text(row: Any, column: str) -> str:
    return _text_from_value(row.get(column))


def _cell_text_at(row: Any, index: int) -> str:
    return _text_from_value(row.iloc[index])


def _text_from_value(value: object) -> str:
    if _is_empty_cell(value):
        return ""
    return str(value).strip()


def _cell_float_or_none(row: Any, column: str) -> float | None:
    return _float_from_value(row.get(column))


def _cell_float_or_none_at(row: Any, index: int) -> float | None:
    return _float_from_value(row.iloc[index])


def _float_from_value(value: object) -> float | None:
    if _is_empty_cell(value):
        return None
    text = str(value).strip()
    if not text or text == "?":
        return None
    return float(value)


def _required_cell_float_at(row: Any, index: int) -> float:
    value = _cell_float_or_none_at(row, index)
    if value is None:
        raise ValueError(f"玻璃材料表缺少第 {index + 1} 列的公式参数。")
    return value


def _is_empty_cell(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


AIR = RealMaterial(name="AIR")
