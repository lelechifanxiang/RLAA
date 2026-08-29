from __future__ import annotations

import math
from typing import Any


def row_direction(row: Any) -> tuple[float, float, float]:
    """从 Zemax 单光线结果中读取方向余弦。"""

    for names in (
        ("L", "M", "N"),
        ("L-cosine", "M-cosine", "N-cosine"),
        ("X-cosine", "Y-cosine", "Z-cosine"),
        ("X-direction cosine", "Y-direction cosine", "Z-direction cosine"),
    ):
        if all(name in row for name in names):
            return float(row[names[0]]), float(row[names[1]]), float(row[names[2]])

    if "X-tangent" in row and "Y-tangent" in row:
        tangent_x = float(row["X-tangent"])
        tangent_y = float(row["Y-tangent"])
        norm = math.sqrt(tangent_x * tangent_x + tangent_y * tangent_y + 1.0)
        return tangent_x / norm, tangent_y / norm, 1.0 / norm

    raise ValueError("Zemax ray trace row does not contain direction columns.")


def read_surface_semi_diameters(oss: Any, surface_indices: list[int]) -> list[float]:
    """直接读取指定 Zemax 表面的 SemiDiameter。"""

    return [
        float(oss.LDE.GetSurfaceAt(surface_index).SemiDiameter)
        for surface_index in surface_indices
    ]
