import numpy as np
import yaml
from pydantic import BaseModel

from bolojax.compute.beam_correlation import compute_corr_curves
from bolojax.models.camera import CameraConfig
from bolojax.models.params import Var, VariableHolder


class ChannelModel(BaseModel):
    band_center: Var("GHz")
    bolo_resistance: Var("Ohm") = None


class LengthModel(BaseModel):
    length: Var("m")


def test_var_parses_quantity_strings() -> None:
    model = ChannelModel(
        band_center="90 GHz",
        bolo_resistance={"var_type": "gauss", "value": "1.0 Ohm", "errors": "0.1 Ohm"},
    )

    assert isinstance(model.band_center, VariableHolder)
    assert np.isclose(float(model.band_center.value), 90.0)
    assert str(model.band_center.unit) == "gigahertz"
    assert np.isclose(float(model.bolo_resistance.value), 1.0)
    assert np.isclose(float(model.bolo_resistance.errors), 0.1)


def test_var_respects_mapping_unit_override() -> None:
    model = LengthModel(length={"unit": "cm", "value": "25 mm"})

    assert np.isclose(float(model.length.value), 2.5)
    assert str(model.length.unit) == "centimeter"
    assert np.isclose(float(model.length.SI), 0.025)


def test_beam_model_preset_string() -> None:
    """beam_model accepts a preset name string."""
    cam = CameraConfig(beam_model="trunc_gauss")
    _, gamma_ap, _ = compute_corr_curves(cam.beam_model)
    # trunc_gauss preset: sigma=1.33, R=1.0
    assert gamma_ap[0] > 0.99  # gamma(0) ~ 1 (normalized)
    # Should differ from bolocalc preset
    _, gamma_ap_bc, _ = compute_corr_curves("bolocalc")
    assert not np.allclose(gamma_ap, gamma_ap_bc)


def test_beam_model_custom_dict() -> None:
    """beam_model accepts a custom parameter dict."""
    custom = {"model": "trunc_gauss", "sigma": 2.0, "R": 0.8}
    cam = CameraConfig(beam_model=custom)
    _, gamma_ap, _ = compute_corr_curves(cam.beam_model)
    assert gamma_ap[0] > 0.99
    # Should differ from the trunc_gauss preset (different sigma, R)
    _, gamma_ap_preset, _ = compute_corr_curves("trunc_gauss")
    assert not np.allclose(gamma_ap, gamma_ap_preset)


def test_beam_model_from_yaml() -> None:
    """beam_model dict survives YAML round-trip."""
    yaml_str = """
    beam_model:
      model: he11
      R: 1.2
      R_taper: 1.1
    """
    data = yaml.safe_load(yaml_str)
    cam = CameraConfig(**data)
    assert isinstance(cam.beam_model, dict)
    assert cam.beam_model["model"] == "he11"
    _, gamma_ap, _ = compute_corr_curves(cam.beam_model)
    assert gamma_ap[0] > 0.99
