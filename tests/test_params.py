import numpy as np
from pydantic import BaseModel

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
