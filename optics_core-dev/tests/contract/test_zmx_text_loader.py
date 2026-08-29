from __future__ import annotations

from pathlib import Path

import pytest

from zemax_utils import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


DOUBLE_GAUSS_ZMX_PATH = Path("tests/zemax/zmx_files/Double Gauss 28 degree field real.zmx")
PARAXIAL_SINGLE_LENS_ZMX_PATH = Path("tests/zemax/zmx_files/paraxial_single_lens.zmx")
K9_SINGLE_SPHERE_ZMX_PATH = Path("tests/zemax/zmx_files/single_sphere_k9.zmx")
MODEL_GLASS_ZMX_PATH = Path("tests/zemax/zmx_files/single_sphere_glass158_radius32.zmx")
COOKE_ZMX_PATH = Path("tests/zemax/zmx_files/Cooke 40 degree field.zmx")
HETEROGENEOUS_ZMX_PATH = Path("tests/zemax/zmx_files/same_arch_diff_materials/sg6_hfov34_f4p8.zmx")


def test_load_double_gauss_zmx_text_spec() -> None:
    """验证双高斯 zmx 可在纯文本路径下完成解析。"""

    spec = load_zmx_sequential_system_spec(DOUBLE_GAUSS_ZMX_PATH)

    assert spec.name == DOUBLE_GAUSS_ZMX_PATH.stem
    assert spec.aperture_kind == "EntrancePupilDiameter"
    assert spec.aperture_value == pytest.approx(33.33)
    assert spec.wavelengths_um == pytest.approx((0.4861, 0.5876, 0.6563))
    assert spec.primary_wavelength_index == 1
    assert spec.field_type == "angle"
    assert spec.field_points == ((0.0, 0.0), (0.0, 10.0), (0.0, 14.0))
    assert spec.object_distance_mm == float("inf")
    assert spec.afocal_image_space is False
    assert len(spec.surfaces) == 11
    assert spec.stop_surface_index == 5
    assert spec.image_surface_index == 12

    first_surface = spec.surfaces[0]
    assert first_surface.surface_type == "Standard"
    assert first_surface.radius_mm == pytest.approx(1.0 / 1.846611368299999958e-02)
    assert first_surface.thickness_mm == pytest.approx(8.74665785)
    assert first_surface.semi_diameter_solve == "auto"
    assert first_surface.semi_diameter_mm == pytest.approx(29.225306460683793)
    assert first_surface.aperture_type == "none"
    assert first_surface.refractive_indices is None


def test_load_paraxial_zmx_text_spec_and_build_system() -> None:
    """验证 Paraxial 面在纯文本路径下仍可构建 optics_core 系统。"""

    spec = load_zmx_sequential_system_spec(PARAXIAL_SINGLE_LENS_ZMX_PATH)

    assert len(spec.surfaces) == 1
    assert spec.stop_surface_index == 0
    assert spec.image_surface_index == 2

    paraxial_surface = spec.surfaces[0]
    assert paraxial_surface.surface_type == "Paraxial"
    assert paraxial_surface.focal_length_mm == pytest.approx(40.0)
    assert paraxial_surface.thickness_mm == pytest.approx(40.0)
    assert paraxial_surface.semi_diameter_solve == "fixed"
    assert paraxial_surface.semi_diameter_mm == pytest.approx(6.0)
    assert paraxial_surface.aperture_type == "none"
    assert paraxial_surface.comment == "paraxial lens"

    system = build_optics_core_system_from_zmx_spec(spec)
    assert len(system.surfaces) == 2
    assert system.surfaces.stop_index == 0
    assert len(system.fields) == 1
    assert len(system.wavelengths) == 1


def test_load_semi_diameter_and_surface_aperture_separately() -> None:
    """Fixed Semi-Diameter 不应被误解为 Floating Aperture。"""
    heterogeneous = load_zmx_sequential_system_spec(HETEROGENEOUS_ZMX_PATH)
    heterogeneous_stop = heterogeneous.surfaces[heterogeneous.stop_surface_index]
    cooke = load_zmx_sequential_system_spec(COOKE_ZMX_PATH)
    cooke_stop = cooke.surfaces[cooke.stop_surface_index]

    assert heterogeneous_stop.semi_diameter_solve == "fixed"
    assert heterogeneous_stop.aperture_type == "none"
    assert cooke_stop.semi_diameter_solve == "fixed"
    assert cooke_stop.aperture_type == "floating"


def test_load_image_f_number_zmx_and_resolve_pupil_radius(tmp_path: Path) -> None:
    """FNUM 应映射为 Image Space F/#，并由 EFFL 换算入瞳半径。"""
    zmx_path = tmp_path / "paraxial_fnum.zmx"
    zmx_text = PARAXIAL_SINGLE_LENS_ZMX_PATH.read_text(encoding="utf-16")
    zmx_path.write_text(zmx_text.replace("ENPD 12", "FNUM 4"), encoding="utf-16")

    spec = load_zmx_sequential_system_spec(zmx_path)
    system = build_optics_core_system_from_zmx_spec(spec).prepare()

    assert spec.aperture_kind == "ImageSpaceFNumber"
    assert spec.aperture_value == pytest.approx(4.0)
    assert system.aperture.kind == "image_f_number"
    assert system.first_order_data.effl.item() == pytest.approx(40.0, abs=1e-12)
    assert system.first_order_data.entrance_pupil_radius.item() == pytest.approx(5.0, abs=1e-12)


def test_load_real_material_name_from_zmx_text_spec() -> None:
    """验证 use_zemax_indices=False 时仍会保留真实材料名。"""

    spec = load_zmx_sequential_system_spec(K9_SINGLE_SPHERE_ZMX_PATH)

    assert len(spec.surfaces) == 1
    assert spec.surfaces[0].material_name == "H-K9L"
    assert spec.surfaces[0].refractive_indices is None


def test_load_blank_model_glass_without_confusing_it_with_pickup() -> None:
    """___BLANK model glass 应保留 nd/vd，只有 solve type 2 才表示材料 pickup。"""
    surface = load_zmx_sequential_system_spec(MODEL_GLASS_ZMX_PATH).surfaces[0]

    assert surface.material_name == "___BLANK"
    assert surface.material_pickup_surface_number is None
    assert surface.nd == pytest.approx(1.58)


def test_load_floa_finite_object_and_afocal_image_space(tmp_path: Path) -> None:
    """FLOA 系统应从 stop 读取口径，并保留有限物距和无焦像空间标志。"""
    zmx_path = tmp_path / "finite_object_afocal.zmx"
    zmx_path.write_text(
        """MODE SEQ
FLOA
FTYP 0 0 1 1 0 0 1 1
XFLN 0
YFLN 0
WAVM 1 0.55 1
PWAV 1
SURF 0
  TYPE STANDARD
  CURV 0
  DISZ 14
SURF 1
  TYPE STANDARD
  CURV 0
  DISZ 1
SURF 2
  STOP
  TYPE STANDARD
  CURV 0
  DISZ 10
  GLAS ___BLANK 2 1 1.5 40 0 0 0 0 0 0
  DIAM 1.5 1
SURF 3
  TYPE STANDARD
  CURV 0
  DISZ 0
""",
        encoding="utf-8",
    )

    spec = load_zmx_sequential_system_spec(zmx_path)
    system = build_optics_core_system_from_zmx_spec(spec)

    assert spec.aperture_kind == "FloatByStopSize"
    assert spec.aperture_value == pytest.approx(3.0)
    assert spec.object_distance_mm == pytest.approx(14.0)
    assert spec.afocal_image_space is True
    assert spec.surfaces[1].material_pickup_surface_number == 1
    assert system.surfaces[1].gap.medium is system.surfaces[0].gap.medium
    assert system.aperture.kind == "float_by_stop_size"
    assert system.architecture.object_distance_mm == pytest.approx(14.0)
    assert system.architecture.afocal_image_space is True
