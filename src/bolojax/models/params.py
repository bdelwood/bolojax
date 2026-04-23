"""Configuration field types for bolojax.

Provides ParamHolder, VariableHolder, and the ``Var`` pydantic
annotation helper. Unit conversion is handled by pint.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Annotated, Any, Literal

import numpy as np
import pint
import scipy.stats as sps
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

from bolojax.models.interp import FreqInterp
from bolojax.models.pdf import ChoiceDist
from bolojax.models.unit import ureg
from bolojax.models.utils import cfg_path, is_none

PintUnit = Annotated[
    pint.Unit | None, BeforeValidator(lambda v: None if v is None else ureg.Unit(v))
]
ValueArray = Annotated[
    np.ndarray,
    BeforeValidator(lambda v: np.asarray(np.nan if is_none(v) else v, dtype=float)),
]
FloatArray = Annotated[np.ndarray, BeforeValidator(partial(np.asarray, dtype=float))]
BoolArray = Annotated[np.ndarray, BeforeValidator(partial(np.asarray, dtype=bool))]


class ParamHolder(BaseModel):
    """Container for a parameter value with unit conversion via pint."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    unit: PintUnit = None
    value: ValueArray = Field(default_factory=lambda: np.asarray(np.nan, dtype=float))
    errors: FloatArray = Field(default_factory=lambda: np.asarray(np.nan, dtype=float))
    bounds: FloatArray = Field(default_factory=lambda: np.asarray(np.nan, dtype=float))
    scale: FloatArray = Field(default_factory=lambda: np.asarray(1.0, dtype=float))
    free: BoolArray = Field(default_factory=lambda: np.asarray(False, dtype=bool))

    @property
    def scaled(self) -> np.ndarray:
        """Return value * scale."""
        return self.value * self.scale

    @property
    def SI(self) -> np.ndarray:
        """Return the value in SI base units."""
        return self()

    def __call__(self) -> np.ndarray:
        """Return the value converted to SI base units."""
        base_val = self.scaled
        if self.unit is None:
            return base_val
        q = ureg.Quantity(base_val, self.unit)
        return np.asarray(q.to_base_units().magnitude, dtype=float)

    def set_from_SI(self, val: np.ndarray) -> None:
        """Set the value from an SI-unit quantity."""
        if self.unit is None:
            self.value = val
            return
        # Convert from SI base units back to the declared unit
        base_unit = ureg.Quantity(1.0, self.unit).to_base_units().units
        q = ureg.Quantity(val, base_unit)
        self.value = np.asarray(q.to(self.unit).magnitude, dtype=float)


class VariableHolder(ParamHolder):
    """ParamHolder with frequency-dependent sampling support."""

    fname: str | None = None
    var_type: Literal["const", "gauss", "pdf", "dist"] = "const"

    _sampled_values: np.ndarray | None = PrivateAttr(default=None)
    _cached_interps: np.ndarray | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _parse_mapping_strings(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        parsed = dict(data)
        raw_unit = parsed.get("unit")
        unit = None if raw_unit is None else ureg.Unit(raw_unit)
        if unit is not None:
            parsed["unit"] = unit
        if is_none(parsed.get("var_type")):
            parsed["var_type"] = "const"
        for key, val in list(parsed.items()):
            if key in {"value", "errors", "bounds", "scale"} and isinstance(val, str):
                q = ureg.Quantity(val)
                parsed[key] = float(q.to(unit).magnitude if unit else q.magnitude)
        return parsed

    @staticmethod
    def _channel_value(arr: np.ndarray | float, chan_idx: int) -> np.ndarray | float:
        """Pick the value for a particular channel."""
        if not isinstance(arr, np.ndarray):
            return arr
        if not arr.shape:
            return arr
        if len(arr) < 2:
            return arr[0]
        return arr[chan_idx]

    def _cache_interps(self, freqs: np.ndarray | None = None) -> None:
        """Cache the interpolator object."""
        if self.var_type == "const":
            self._cached_interps = None
            return
        if self.var_type == "gauss":
            if np.isnan(self.errors).any():
                self._cached_interps = None
                return
            self._cached_interps = np.array(
                [
                    sps.norm(loc=val_, scale=sca_)
                    for val_, sca_ in zip(
                        np.atleast_1d(self.value),
                        np.atleast_1d(self.errors),
                        strict=False,
                    )
                ]
            )
            return
        tokens = self.fname.split(",")
        if self.var_type == "pdf":
            self._cached_interps = np.array(
                [ChoiceDist(cfg_path(token)) for token in tokens]
            )
            self.value = np.array([pdf.mean() for pdf in self._cached_interps])
            return
        self._cached_interps = np.array(
            [FreqInterp(cfg_path(token)) for token in tokens]
        )
        if freqs is None:
            self.value = np.array([pdf.mean_trans() for pdf in self._cached_interps])
            return
        self.value = np.array([pdf.rvs(freqs) for pdf in self._cached_interps])

    def rvs(
        self, nsamples: int, freqs: np.ndarray | None = None, chan_idx: int = 0
    ) -> np.ndarray | float:
        """Sample values (does NOT store them)."""
        self._cache_interps(freqs)
        val = self._channel_value(self.value, chan_idx)
        if self._cached_interps is None or not nsamples:
            return val
        interp = self._channel_value(self._cached_interps, chan_idx)
        if self.var_type == "gauss":
            return interp.rvs(nsamples).reshape((nsamples, 1))
        if self.var_type == "pdf":
            return interp.rvs(nsamples).reshape((nsamples, 1))
        return interp.rvs(freqs, nsamples).reshape((nsamples, len(freqs)))

    def sample(
        self, nsamples: int, freqs: np.ndarray | None = None, chan_idx: int = 0
    ) -> np.ndarray:
        """Sample values and store them."""
        self._sampled_values = self.rvs(nsamples, freqs, chan_idx)
        return self.SI

    def unsample(self) -> None:
        """Remove stored sampled values."""
        self._sampled_values = None

    @property
    def scaled(self) -> np.ndarray:
        """Return the product of the value and the scale."""
        if self._sampled_values is not None:
            return self._sampled_values * self.scale
        return super().scaled


def Var(unit: str | None = None) -> type[VariableHolder]:
    """Type annotation for a Variable field with optional pint unit.

    Accepts scalars, unit strings (``"25 mm"``), dicts
    (``{value: ..., var_type: gauss, ...}``), or None.

    Usage::

        band_center: Var("GHz") = None       # optional, defaults to NaN
        temperature: Var("K")                # required
        frac_bw: Var() = 0.35                # dimensionless with default
    """
    pint_unit = ureg.Unit(unit) if unit else None

    def _validate(v: Any) -> VariableHolder | dict[str, Any]:
        if isinstance(v, VariableHolder):
            return v
        if isinstance(v, Mapping):
            kw = dict(v)
        elif is_none(v):
            kw = {"value": np.nan}
        elif isinstance(v, str):
            # Try bare numeric string first (YAML parses "3.6E7" as str)
            try:
                kw = {"value": float(v)}
            except ValueError:
                kw = {"value": v}
        else:
            kw = {"value": v}
        if pint_unit is not None:
            kw.setdefault("unit", pint_unit)
        return kw

    return Annotated[VariableHolder, BeforeValidator(_validate)]
