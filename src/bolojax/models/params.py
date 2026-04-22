"""Configuration field types for bolojax.

Provides ParamHolder, VariableHolder, and the ``Var`` pydantic
annotation helper. Unit conversion is handled by pint.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

import numpy as np
import pint
import scipy.stats as sps
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from bolojax.models.interp import FreqInterp
from bolojax.models.pdf import ChoiceDist
from bolojax.models.unit import ureg
from bolojax.models.utils import cfg_path, is_none


class ParamHolder:
    """Container for a parameter value with unit conversion via pint."""

    def __init__(self, **kwargs: Any) -> None:
        value = kwargs.get("value", np.nan)
        if is_none(value):
            value = np.nan
        self.value: np.ndarray = np.asarray(value, dtype=float)
        self.errors: np.ndarray = np.asarray(kwargs.get("errors", np.nan), dtype=float)
        self.bounds: np.ndarray = np.asarray(kwargs.get("bounds", np.nan), dtype=float)
        self.scale: np.ndarray = np.asarray(kwargs.get("scale", 1.0), dtype=float)
        self.free: np.ndarray = np.asarray(kwargs.get("free", False), dtype=bool)
        unit = kwargs.get("unit")
        if isinstance(unit, str):
            unit = ureg.Unit(unit)
        self.unit: pint.Unit | None = unit

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

    def __init__(self, *args: float | np.ndarray, **kwargs: Any) -> None:
        if args:
            kwargs["value"] = args[0]
        self.fname: str | None = kwargs.pop("fname", None)
        vt: str | None = kwargs.pop("var_type", "const")
        if vt is None or is_none(vt):
            vt = "const"
        if vt not in ("pdf", "dist", "gauss", "const"):
            msg = f"var_type must be one of pdf/dist/gauss/const, got {vt}"
            raise ValueError(msg)
        self.var_type: Literal["const", "gauss", "pdf", "dist"] = vt
        super().__init__(**kwargs)
        self._sampled_values: np.ndarray | None = None
        self._cached_interps: np.ndarray | None = None

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


class _VarValidator:
    """Pydantic annotation that converts raw YAML values to VariableHolder.

    Handles three input forms:
    - Scalar (int/float): treated as a value in the declared unit
    - String with units (``"25 mm"``): parsed by pint, converted to declared unit
    - Dict (``{value: ..., var_type: gauss, ...}``): full VariableHolder config
    """

    def __init__(self, unit: str | None = None) -> None:
        self._unit: pint.Unit | None = ureg.Unit(unit) if unit else None

    def _parse_value(self, v: Any) -> float:
        """Parse a scalar or unit-string value into the declared unit."""
        if isinstance(v, str):
            q = ureg.Quantity(v)
            if self._unit is not None:
                q = q.to(self._unit)
            return float(q.magnitude)
        return v

    def __get_pydantic_core_schema__(
        self, source_type: type, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        unit = self._unit
        parse = self._parse_value

        def _validate(v: Any) -> VariableHolder:
            if isinstance(v, VariableHolder):
                return v
            kw: dict[str, Any] = {}
            if unit is not None:
                kw["unit"] = unit
            if isinstance(v, Mapping):
                kw.update(v)
                if "value" in kw and isinstance(kw["value"], str):
                    kw["value"] = parse(kw["value"])
            elif is_none(v):
                kw.setdefault("value", np.nan)
            elif isinstance(v, str):
                kw["value"] = parse(v)
            else:
                kw["value"] = v
            return VariableHolder(**kw)

        return core_schema.no_info_plain_validator_function(_validate)


def Var(unit: str | None = None) -> type[VariableHolder]:
    """Type annotation for a Variable field.

    Usage::

        band_center: Var("GHz") = None       # optional, defaults to NaN
        temperature: Var()                    # required (no default)
        frac_bw: Var() = 0.35                # optional with value default
    """
    return Annotated[VariableHolder, _VarValidator(unit)]
