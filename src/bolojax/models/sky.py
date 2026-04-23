# ruff: noqa: ARG002
"""Sky model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import am
import numpy as np
import xarray as xr
from joblib import Memory
from pydantic import PrivateAttr

from bolojax.compute import physics
from bolojax.models.base import BolojaxModel
from bolojax.models.params import Var
from bolojax.models.utils import cfg_path, is_not_none

if TYPE_CHECKING:
    from bolojax.models.instrument import InstrumentConfig

GHz_to_Hz = 1.0e09


def _interp_to_hz(
    freqs_hz: np.ndarray, freq_ghz: np.ndarray, vals: np.ndarray
) -> np.ndarray:
    """Interpolate values from a GHz grid onto an Hz grid."""
    return np.interp(freqs_hz, freq_ghz * GHz_to_Hz, vals)


class AtmBackend(ABC):
    """Atmosphere model base class.

    Subclasses implement :meth:`raw_spectra` to return raw
    ``(freq_ghz, temp, trans)`` arrays for a single set of atmospheric
    parameters.  The base class handles interpolation and batching.

    Parameters are passed as keyword arguments (e.g. ``pwv=0.6,
    elevation=60``).  The set of parameter names depends on the
    backend and the ``.amc`` configuration file.
    """

    @abstractmethod
    def raw_spectra(
        self, freqs: np.ndarray, **params: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (freq_ghz, brightness_temp_K, transmission) arrays."""

    def temp(self, freqs: np.ndarray, **params: float) -> np.ndarray:
        """Brightness temperature [K] at given freqs [Hz]."""
        freq_ghz, temp, _ = self.raw_spectra(freqs, **params)
        return _interp_to_hz(freqs, freq_ghz, temp)

    def trans(self, freqs: np.ndarray, **params: float) -> np.ndarray:
        """Transmission [0-1] at given freqs [Hz]."""
        freq_ghz, _, trans = self.raw_spectra(freqs, **params)
        return _interp_to_hz(freqs, freq_ghz, trans)

    def batch(
        self, freqs: np.ndarray, **param_arrays: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate temp and trans for arrays of parameters.

        Each kwarg is an array of sampled values for that parameter.
        All arrays must broadcast to the same shape.
        Returns (temp, trans) each of shape (n_samples, n_freq).
        """
        names = list(param_arrays.keys())
        arrays = list(param_arrays.values())
        temps, transs = [], []
        for vals in np.broadcast(*arrays):
            point = dict(zip(names, (float(v) for v in vals), strict=True))
            fg, t, x = self.raw_spectra(freqs, **point)
            temps.append(_interp_to_hz(freqs, fg, t))
            transs.append(_interp_to_hz(freqs, fg, x))
        return np.array(temps), np.array(transs)


class AtmProfile(AtmBackend):
    """Single fixed atmosphere profile from a text file."""

    def __init__(self, path: str | Path) -> None:
        self.freq_ghz, self.temps, self.transmission = np.loadtxt(
            path, unpack=True, usecols=[0, 2, 3], dtype=np.float64
        )

    def raw_spectra(
        self, freqs: np.ndarray, **params: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.freq_ghz, self.temps, self.transmission


def _make_grid(values: np.ndarray, step: float) -> list[float]:
    """Build a regular grid covering the range of *values* at *step* spacing."""
    lo = step * np.floor(np.min(values) / step)
    hi = step * np.ceil(np.max(values) / step)
    return list(np.arange(lo, hi + step / 2, step).round(6))


@dataclass
class DerivedParam:
    """A derived am argument computed from a grid coordinate.

    For example, ``zenith`` is derived from ``elevation`` via
    ``90 - elevation``.  The grid is built over ``source`` values;
    the ``transform`` converts to the am argument at evaluation time.
    """

    keyword: str
    source: str
    transform: Callable
    grid_step: float = 1.0

    def resolve(self, point: dict[str, float]) -> float:
        """Evaluate the transform for a given grid point."""
        return self.transform(point)


def _compute_am_grid(
    path: str,
    amc_args: list[str | float],
    derived: dict[str, DerivedParam],
    grid_coords: dict[str, list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute an N-dimensional atmosphere grid via am.ModelGrid.

    Module-level function for ``joblib.Memory`` caching.
    """
    params = xr.Dataset(
        coords={k: np.array(v, dtype=float) for k, v in grid_coords.items()}
    )

    def args_fn(**point: float) -> list[str | float]:
        return [
            derived[t].resolve(point)
            if t in derived
            else point.get(t, t)
            if isinstance(t, str)
            else t
            for t in amc_args
        ]

    result = am.ModelGrid(path, params, args_fn).compute()
    freq_ghz = result.coords["frequency"].values
    tb = result["tb_planck"].values if "tb_planck" in result else result["tb_rj"].values
    return freq_ghz, tb, result["transmittance"].values


class AmAtm(AtmBackend):
    """Atmosphere via am-python with automatic disk caching.

    On first use, computes a grid of atmosphere profiles over the
    dynamic parameters using ``am.ModelGrid`` in parallel and caches
    the result to disk via ``joblib``.  Subsequent calls load from
    cache instantly.

    Dynamic parameters are string entries in ``amc_args`` that match
    either a derived parameter name or a direct grid coordinate name.
    Numeric entries are static.
    """

    def __init__(
        self,
        path: str,
        amc_args: list[str | float],
        profile_pwv_mm: float = 0.425,
        cache_dir: str | None = None,
        **grid_overrides: list[float],
    ) -> None:
        self.path = path
        self.amc_args = amc_args
        self.profile_pwv_mm = profile_pwv_mm

        # Built-in derived parameters
        derived_list = [
            DerivedParam("zenith", "elevation", lambda p: 90.0 - p["elevation"], 1.0),
            DerivedParam(
                "pwv_scale",
                "pwv",
                lambda p: (
                    max(p["pwv"] / profile_pwv_mm, 1e-6) if p["pwv"] > 0 else 1e-6
                ),
                0.1,
            ),
        ]
        self.by_keyword: dict[str, DerivedParam] = {d.keyword: d for d in derived_list}
        self.by_source: dict[str, DerivedParam] = {d.source: d for d in derived_list}

        # Discover dynamic grid coordinates from amc_args string entries
        self.dynamic_params: list[str] = []
        for token in amc_args:
            if not isinstance(token, str):
                continue
            source = (
                self.by_keyword[token].source if token in self.by_keyword else token
            )
            if source not in self.dynamic_params:
                self.dynamic_params.append(source)

        self.grid_coords: dict[str, list[float] | None] = {
            p: grid_overrides.get(p) for p in self.dynamic_params
        }

        cache = Path(cache_dir) if cache_dir else Path(path).parent / ".bolojax_cache"
        self._cached_compute = Memory(cache, verbose=0).cache(
            _compute_am_grid, ignore=["derived"]
        )
        self.freq_ghz: np.ndarray | None = None
        self.tb_grid: np.ndarray | None = None
        self.tx_grid: np.ndarray | None = None

    def ensure_grid(self, **sampled_values: np.ndarray) -> None:
        """Ensure the grid is computed, inferring extent from sampled values."""
        if self.freq_ghz is not None:
            return
        for param in self.dynamic_params:
            if self.grid_coords[param] is None:
                if param not in sampled_values:
                    msg = f"Grid for '{param}' not configured and no sampled values provided"
                    raise ValueError(msg)
                d = self.by_source.get(param)
                step = d.grid_step if d else 1.0
                self.grid_coords[param] = _make_grid(
                    np.atleast_1d(sampled_values[param]), step
                )
        self.freq_ghz, self.tb_grid, self.tx_grid = self._cached_compute(
            self.path,
            self.amc_args,
            self.by_keyword,
            {k: v for k, v in self.grid_coords.items() if v is not None},
        )

    def raw_spectra(
        self, freqs: np.ndarray, **params: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Look up nearest grid point in N dimensions."""
        idx = tuple(
            np.argmin(np.abs(np.array(self.grid_coords[p]) - params[p]))
            for p in self.dynamic_params
        )
        return self.freq_ghz, self.tb_grid[idx], self.tx_grid[idx]

    def batch(
        self, freqs: np.ndarray, **param_arrays: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ensure grid covers all queried points, then delegate to base."""
        self.ensure_grid(**param_arrays)
        return super().batch(freqs, **param_arrays)


class Atmosphere(BolojaxModel):
    """Atmosphere model.

    The primary backend is ``AmAtm``, which computes atmosphere profiles
    from an ``.amc`` configuration file using am-python and caches the
    results to disk.  A fixed-profile text file can be used as a
    fallback via ``custom_atm_file`` on the InstrumentConfig.
    """

    amc_file: str | None = None
    amc_args: list[str | float] | None = None
    profile_pwv_mm: float = 0.425

    _telescope: InstrumentConfig | None = PrivateAttr(default=None)
    _sampled_params: dict[str, np.ndarray] = PrivateAttr(default_factory=dict)
    _nsamples: int = PrivateAttr(default=1)

    def set_telescope(self, value: InstrumentConfig) -> None:
        """Set the telescope (needed to sample parameter values)."""
        self._telescope = value

    @cached_property
    def cached_model(self) -> AtmBackend | None:
        """Cache the Atmosphere model backend."""
        if is_not_none(self._telescope.custom_atm_file):
            return AtmProfile(cfg_path(self._telescope.custom_atm_file))
        if is_not_none(self.amc_file):
            if self.amc_args is None:
                msg = "amc_args must be specified when using amc_file"
                raise ValueError(msg)
            return AmAtm(
                cfg_path(self.amc_file),
                self.amc_args,
                profile_pwv_mm=self.profile_pwv_mm,
            )
        return None

    def sample(self, nsamples: int) -> None:
        """Sample atmosphere parameters from instrument fields.

        Discovers which parameters to sample from the backend model's
        ``dynamic_params`` list, then looks up each on the instrument
        by name.
        """
        self._sampled_params = {}
        model = self.cached_model
        if model is None or not hasattr(model, "dynamic_params"):
            self._nsamples = max(nsamples, 1)
            return
        for param in model.dynamic_params:
            var = getattr(self._telescope, param, None)
            if var is not None and is_not_none(var):
                var.sample(nsamples)
                self._sampled_params[param] = np.atleast_1d(var())
        self._nsamples = max(nsamples, 1)

    def temp(
        self, freqs: np.ndarray, elevation: np.ndarray | None = None
    ) -> np.ndarray:
        """Brightness temperature [K] for sampled conditions."""
        nsamp = max(self._nsamples, 1)
        params = dict(self._sampled_params)
        if elevation is not None:
            params["elevation"] = elevation
        temps, _ = self.cached_model.batch(freqs, **params)
        return np.broadcast_to(temps, (nsamp, len(freqs))).reshape(
            (nsamp, 1, len(freqs))
        )

    def trans(
        self, freqs: np.ndarray, elevation: np.ndarray | None = None
    ) -> np.ndarray:
        """Transmission [0-1] for sampled conditions."""
        nsamp = max(self._nsamples, 1)
        params = dict(self._sampled_params)
        if elevation is not None:
            params["elevation"] = elevation
        _, transs = self.cached_model.batch(freqs, **params)
        return np.broadcast_to(transs, (nsamp, len(freqs))).reshape(
            (nsamp, 1, len(freqs))
        )


class Foreground(BolojaxModel):
    """Foreground model base class."""

    spectral_index: Var() = None
    scale_frequency: Var("GHz") = None
    emiss: Var() = 1.0


class Dust(Foreground):
    """Dust emission model."""

    amplitude: Var("MJy") = None
    scale_temperature: Var("K") = None

    _nsamples: int = PrivateAttr(default=1)
    _amp: np.ndarray | None = PrivateAttr(default=None)
    _scale_temp: np.ndarray | None = PrivateAttr(default=None)

    def sample(self, nsamples: int) -> None:
        """Sample this component."""
        self.amplitude.sample(nsamples)
        self.scale_temperature.sample(nsamples)
        self._amp = np.expand_dims(np.expand_dims(self.amplitude.SI, -1), -1)
        self._scale_temp = np.expand_dims(
            np.expand_dims(self.scale_temperature.SI, -1), -1
        )
        self._nsamples = max(self._amp.size, self._scale_temp.size)

    def temp(self, freqs: np.ndarray) -> np.ndarray:
        """Get sampled temperatures."""
        out_shape = (self._nsamples, 1, len(freqs))
        return self.__temp(
            freqs,
            self.emiss.SI,
            self._amp,
            self.scale_frequency.SI,
            self.spectral_index.SI,
            self._scale_temp,
        ).reshape(out_shape)

    @staticmethod
    def __temp(
        freqs: np.ndarray,
        emiss: float | np.ndarray,
        amp: np.ndarray,
        scale_frequency: float | np.ndarray,
        spectral_index: float | np.ndarray,
        scale_temp: np.ndarray,
    ) -> np.ndarray:  # pylint: disable=too-many-arguments
        """Return the galactic effective physical temperature."""
        # Passed amplitude [W/(m^2 sr Hz)] converted from [MJy]
        amp = emiss * amp
        # Frequency scaling: (freq / scale_freq)**dust_ind
        if np.isfinite(scale_frequency).all() and np.isfinite(spectral_index).all():
            freq_scale = (freqs / scale_frequency) ** (spectral_index)
        else:
            freq_scale = 1.0
        # Effective blackbody scaling: BB(freq, dust_temp) / BB(dust_freq, dust_temp)
        if np.isfinite(scale_temp).all() and np.isfinite(scale_frequency).all():
            spec_scale = physics.bb_spec_rad(freqs, scale_temp) / physics.bb_spec_rad(
                scale_frequency, scale_temp
            )
        else:
            spec_scale = 1.0
        # Convert [W/(m^2 sr Hz)] to brightness temperature [K_RJ]
        pow_spec_rad = amp * freq_scale * spec_scale
        return physics.Tb_from_spec_rad(freqs, pow_spec_rad)


class Synchrotron(Foreground):
    """Synchrotron emission model."""

    amplitude: Var("K_RJ") = None

    _nsamples: int = PrivateAttr(default=1)
    _amp: np.ndarray | None = PrivateAttr(default=None)

    def sample(self, nsamples: int) -> None:
        """Sample this component."""
        self.amplitude.sample(nsamples)
        self._amp = np.expand_dims(np.expand_dims(self.amplitude.SI, -1), -1)
        self._nsamples = self._amp.size

    def temp(self, freqs: np.ndarray) -> np.ndarray:
        """Get sampled temperatures."""
        out_shape = (self._nsamples, 1, len(freqs))
        return self.__temp(
            freqs,
            self.emiss.SI,
            self._amp,
            self.scale_frequency.SI,
            self.spectral_index.SI,
        ).reshape(out_shape)

    @staticmethod
    def __temp(
        freqs: np.ndarray,
        emiss: float | np.ndarray,
        amp: np.ndarray,
        scale_frequency: float | np.ndarray,
        spectral_index: float | np.ndarray,
    ) -> np.ndarray:
        """Return the effective physical temperature."""
        bright_temp = emiss * amp
        # Frequency scaling (freq / sync_freq)**sync_ind
        freq_scale = (freqs / scale_frequency) ** spectral_index
        scaled_bright_temp = bright_temp * freq_scale
        # Convert brightness temperature [K_RJ] to physical temperature [K]
        return physics.Tb_from_Trj(freqs, scaled_bright_temp)


class Universe(BolojaxModel):
    """Collection of emission models."""

    dust: Dust | None = None
    synchrotron: Synchrotron | None = None
    atmosphere: Atmosphere | None = None

    sources: ClassVar[list[str]] = ["cmb", "dust", "synchrotron", "atmosphere"]

    def sample(self, nsamples: int) -> None:
        """Sample the sky component."""
        self.dust.sample(nsamples)
        self.synchrotron.sample(nsamples)
        self.atmosphere.sample(nsamples)

    def temp(
        self, freqs: np.ndarray, elevation: np.ndarray | None = None
    ) -> OrderedDict[str, float | np.ndarray]:
        """Get sampled temperatures."""
        ret: OrderedDict[str, float | np.ndarray] = OrderedDict()
        ret["cmb"] = physics.Tcmb
        ret["dust"] = self.dust.temp(freqs)
        ret["synchrotron"] = self.synchrotron.temp(freqs)
        ret["atmosphere"] = self.atmosphere.temp(freqs, elevation=elevation)
        return ret

    def trans(
        self, freqs: np.ndarray, elevation: np.ndarray | None = None
    ) -> OrderedDict[str, np.ndarray]:
        """Get sampled transmission coefs."""
        ret: OrderedDict[str, np.ndarray] = OrderedDict()
        ret["atmosphere"] = self.atmosphere.trans(freqs, elevation=elevation)
        return ret
