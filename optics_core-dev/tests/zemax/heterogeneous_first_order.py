from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tests.zemax.heterogeneous_spot import ZMX_DIRECTORY, ZMX_FILENAMES, load_heterogeneous_specs


REFERENCE_PATH = Path(__file__).resolve().parent / "reference_data/same_arch_diff_materials_first_order.json"


def load_first_order_reference() -> dict[str, Any]:
    """读取并校验五文件一阶量基准。"""
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    for item in reference["systems"]:
        zmx_path = ZMX_DIRECTORY / item["zmx_file"]
        if hashlib.sha256(zmx_path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Zemax first-order reference is stale for {zmx_path.name}.")
    return reference


def generate_first_order_reference() -> dict[str, Any]:
    """调用 Zemax 生成五文件一阶量基准。"""
    from tests.zemax.common import loaded_sequential_system
    from tests.zemax.first_order import fetch_zemax_first_order_from_spec

    systems: list[dict[str, Any]] = []
    for name, spec in zip(ZMX_FILENAMES, load_heterogeneous_specs(), strict=True):
        with loaded_sequential_system(spec.zmx_path) as oss:
            first_order = fetch_zemax_first_order_from_spec(spec, oss)
        systems.append(
            {
                "zmx_file": name,
                "sha256": hashlib.sha256((ZMX_DIRECTORY / name).read_bytes()).hexdigest(),
                "effl_mm": first_order.effective_focal_length_mm,
                "working_f_number": first_order.working_f_number,
                "ttl_mm": first_order.total_track_length_mm,
                "image_plane_distance_mm": first_order.image_plane_distance_mm,
                "bfl_mm": first_order.back_focal_length_mm,
            }
        )
    return {"schema_version": 1, "source": "Zemax MFE + System Data", "systems": systems}


if __name__ == "__main__":
    REFERENCE_PATH.write_text(
        json.dumps(generate_first_order_reference(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
