"""Configuration field types for bolojax.

Provides ParamHolder, VariableHolder, OutputField, OutputHolder, and the
``Var`` / ``Out`` pydantic annotation helpers.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Annotated, Any

import numpy as np
import scipy.stats as sps
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from bolojax.models.interp import FreqInterp
from bolojax.models.pdf import ChoiceDist
from bolojax.models.unit import Unit
from bolojax.models.utils import cfg_path, is_none


class ParamHolder:
    """Container for a parameter value with unit conversion."""

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
        if isinstance(unit, str | int | float):
            unit = Unit(unit)
        self.unit: Unit | None = unit

    @property
    def scaled(self) -> np.ndarray:
        """Return value * scale."""
        return self.value * self.scale

    @property
    def SI(self) -> np.ndarray:
        """Return the value in SI units."""
        return self()

    def __call__(self) -> np.ndarray:
        """Return the value converted to SI units."""
        base_val = self.scaled
        if self.unit is None:
            return base_val
        return self.unit(base_val)

    def set_from_SI(self, val: np.ndarray) -> None:
        """Set the value from an SI-unit quantity."""
        if self.unit is None:
            self.value = val
            return
        self.value = self.unit.inverse(val)


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
        self.var_type: str = vt
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
    """Pydantic annotation that converts raw YAML values to VariableHolder."""

    def __init__(self, unit: str | None = None) -> None:
        self._unit: Unit | None = Unit(unit) if unit else None

    def __get_pydantic_core_schema__(
        self, source_type: type, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        unit = self._unit

        def _validate(v: Any) -> VariableHolder:
            if isinstance(v, VariableHolder):
                return v
            kw: dict[str, Any] = {}
            if unit is not None:
                kw["unit"] = unit
            if isinstance(v, Mapping):
                kw.update(v)
            elif is_none(v):
                kw.setdefault("value", np.nan)
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


class OutputHolder(ParamHolder):
    """ParamHolder for computed output values."""


class OutputField:
    """Descriptor for computed output values on the Sensitivity class."""

    def __init__(self, unit: Unit | str | None = None) -> None:
        if isinstance(unit, Unit):
            self._unit_str: str | None = unit.name or None
            self._unit: Unit | None = unit
        elif isinstance(unit, str):
            self._unit_str = unit
            self._unit = Unit(unit)
        else:
            self._unit_str = None
            self._unit = None
        self.public_name: str | None = None
        self.private_name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self.public_name = name
        self.private_name = "_" + name
        # Build class-level registry of output fields
        if (
            not hasattr(owner, "_output_fields")
            or "_output_fields" not in owner.__dict__
        ):
            owner._output_fields = {}
        owner._output_fields[name] = self

    def __get__(
        self, obj: object, objtype: type | None = None
    ) -> OutputField | OutputHolder | None:
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    def __set__(self, obj: object, value: OutputHolder) -> None:
        setattr(obj, self.private_name, value)

    def make_holder(self) -> OutputHolder:
        """Create an empty OutputHolder for this field."""
        kw: dict[str, Unit] = {}
        if self._unit is not None:
            kw["unit"] = self._unit
        return OutputHolder(**kw)

    def summarize(self, obj: object) -> StatsSummary:
        """Compute and return summary statistics."""
        val = getattr(obj, self.private_name).value
        return StatsSummary(self.public_name, val)

    def summarize_by_element(self, obj: object) -> StatsSummary:
        """Compute and return per-element summary statistics."""
        val = getattr(obj, self.private_name).value
        val = val.reshape((val.shape[0], np.prod(val.shape[1:])))
        return StatsSummary(self.public_name, val, axis=1)


class StatsSummary:
    """Summarise the statistical properties of a computed parameter."""

    def __init__(
        self, name: str, vals: np.ndarray, unit_name: str = "", axis: int | None = None
    ) -> None:
        self._name: str = name
        self._unit_name: str = unit_name
        self._mean: np.ndarray = np.mean(vals, axis=axis)
        self._median: np.ndarray = np.median(vals, axis=axis)
        self._std: np.ndarray = np.std(vals, axis=axis)
        self._quantiles: np.ndarray = np.quantile(
            vals, [0.023, 0.159, 0.841, 0.977], axis=axis
        )
        self._deltas: np.ndarray = np.abs(self._quantiles - self._median)

    def element_string(self, idx: int) -> str:
        """Pretty representation of the stats for one element."""
        return f"{self._mean[idx]:0.4f} +- [{self._deltas[1, idx]:0.4f} {self._deltas[2, idx]:0.4f}] {self._unit_name}"

    def __str__(self) -> str:
        """Pretty representation of the stats."""
        return f"{self._mean:0.4f} +- [{self._deltas[1]:0.4f} {self._deltas[2]:0.4f}] {self._unit_name}"

    def todict(self) -> OrderedDict[str, np.ndarray]:
        """Put the summary stats into a dictionary."""
        o_dict: OrderedDict[str, np.ndarray] = OrderedDict()
        for vn in ["_mean", "_median", "_std"]:
            o_dict[f"{self._name}{vn}"] = np.atleast_1d(self.__dict__[vn])
        for idx, vn in enumerate(["_n_2_sig", "_n_1_sig", "_p_1_sig", "_p_2_sig"]):
            o_dict[f"{self._name}{vn}"] = np.atleast_1d(self._deltas[idx])
        return o_dict
