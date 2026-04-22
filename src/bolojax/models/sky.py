# ruff: noqa: ARG002
"""Sky model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
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
    from bolojax.models.instrument import Instrument

GHz_to_Hz = 1.0e09


def _interp_to_hz(
    freqs_hz: np.ndarray, freq_ghz: np.ndarray, vals: np.ndarray
) -> np.ndarray:
    """Interpolate values from a GHz grid onto an Hz grid."""
    return np.interp(freqs_hz, freq_ghz * GHz_to_Hz, vals)


class AtmBackend(ABC):
    """Atmosphere model base class.

    Subclasses implement :meth:`raw_spectra` to return raw
    ``(freq_ghz, temp, trans)`` arrays for a single condition.
    The base class handles interpolation and batching.
    """

    @abstractmethod
    def raw_spectra(
        self, freqs: np.ndarray, pwv: float, elevation: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (freq_ghz, brightness_temp_K, transmission) arrays."""

    def temp(self, freqs: np.ndarray, pwv: float, elevation: float) -> np.ndarray:
        """Brightness temperature [K] at given freqs [Hz], pwv [m], elevation [deg]."""
        freq_ghz, temp, _ = self.raw_spectra(freqs, pwv, elevation)
        return _interp_to_hz(freqs, freq_ghz, temp)

    def trans(self, freqs: np.ndarray, pwv: float, elevation: float) -> np.ndarray:
        """Transmission [0-1] at given freqs [Hz], pwv [m], elevation [deg]."""
        freq_ghz, _, trans = self.raw_spectra(freqs, pwv, elevation)
        return _interp_to_hz(freqs, freq_ghz, trans)

    def batch(
        self, freqs: np.ndarray, pwv: np.ndarray, elevation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate temp and trans for arrays of (pwv, elevation).

        Returns (temp, trans) each of shape (n_samples, n_freq).
        """
        temps, transs = [], []
        for p, e in np.broadcast(pwv, elevation):
            fg, t, x = self.raw_spectra(freqs, float(p), float(e))
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
        self, freqs: np.ndarray, pwv: float, elevation: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.freq_ghz, self.temps, self.transmission


def _compute_am_grid(
    path: str,
    amc_args: list[str | float],
    profile_pwv_mm: float,
    pwv_mm: list[float],
    elevation: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a (PWV, elevation) atmosphere grid via am.ModelGrid.

    Module-level function for ``joblib.Memory`` caching.
    Returns ``(freq_ghz, tb, tx)`` with shapes ``(n_pwv, n_elev, n_freq)``.
    """
    params = xr.Dataset(
        coords={
            "pwv_mm": np.array(pwv_mm, dtype=float),
            "elevation": np.array(elevation, dtype=float),
        }
    )

    def args_fn(pwv_mm: float, elevation: float) -> list[str | float]:
        subs = {
            "zenith": 90.0 - elevation,
            "pwv_scale": max(pwv_mm / profile_pwv_mm, 1e-6) if pwv_mm > 0 else 1e-6,
        }
        return [subs.get(t, t) for t in amc_args]

    result = am.ModelGrid(path, params, args_fn).compute()
    freq_ghz = result.coords["frequency"].values
    tb = result["tb_planck"].values if "tb_planck" in result else result["tb_rj"].values
    return freq_ghz, tb, result["transmittance"].values


def _make_grid(values: np.ndarray, step: float) -> list[float]:
    """Build a regular grid covering the range of *values* at *step* spacing."""
    lo = step * np.floor(np.min(values) / step)
    hi = step * np.ceil(np.max(values) / step)
    return list(np.arange(lo, hi + step / 2, step).round(6))


class AmAtm(AtmBackend):
    """Atmosphere via am-python with automatic disk caching.

    On first use, computes a grid of atmosphere profiles over
    (PWV, elevation) using ``am.ModelGrid`` in parallel and caches
    the result to disk via ``joblib``.  Subsequent calls load from
    cache instantly.

    The grid extent is either set explicitly (``pwv_mm``, ``elevation``)
    or inferred lazily from the sampled conditions via
    :meth:`ensure_grid`.
    """

    def __init__(
        self,
        path: str,
        amc_args: list[str | float],
        profile_pwv_mm: float = 0.425,
        pwv_mm: list[float] | None = None,
        elevation: list[float] | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.path = path
        self.amc_args = amc_args
        self.profile_pwv_mm = profile_pwv_mm
        self.pwv_mm = pwv_mm
        self.elevation = elevation

        cache = Path(cache_dir) if cache_dir else Path(path).parent / ".bolojax_cache"
        self._compute = Memory(cache, verbose=0).cache(_compute_am_grid)
        self._freq_ghz: np.ndarray | None = None
        self._tb_grid: np.ndarray | None = None
        self._tx_grid: np.ndarray | None = None

    def ensure_grid(
        self,
        pwv_m: np.ndarray | None = None,
        elevation: np.ndarray | None = None,
        pwv_step_mm: float = 0.1,
        elev_step: float = 1.0,
    ) -> None:
        """Ensure the grid is computed, inferring extent from sampled values if needed."""
        if self._freq_ghz is not None:
            return
        if self.pwv_mm is None:
            if pwv_m is None:
                msg = "pwv_mm grid not configured and no sampled values to infer from"
                raise ValueError(msg)
            self.pwv_mm = _make_grid(np.atleast_1d(pwv_m) * 1e3, pwv_step_mm)
        if self.elevation is None:
            if elevation is None:
                msg = (
                    "elevation grid not configured and no sampled values to infer from"
                )
                raise ValueError(msg)
            self.elevation = _make_grid(np.atleast_1d(elevation), elev_step)
        self._freq_ghz, self._tb_grid, self._tx_grid = self._compute(
            self.path,
            self.amc_args,
            self.profile_pwv_mm,
            self.pwv_mm,
            self.elevation,
        )

    def raw_spectra(
        self, freqs: np.ndarray, pwv: float, elevation: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Look up nearest grid point. *pwv* is in meters (SI)."""
        i_pwv = np.argmin(np.abs(np.array(self.pwv_mm) - pwv * 1e3))
        i_elev = np.argmin(np.abs(np.array(self.elevation) - elevation))
        return (
            self._freq_ghz,
            self._tb_grid[i_pwv, i_elev],
            self._tx_grid[i_pwv, i_elev],
        )

    def batch(
        self, freqs: np.ndarray, pwv: np.ndarray, elevation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ensure grid covers all queried points, then delegate to base."""
        self.ensure_grid(pwv_m=pwv, elevation=elevation)
        return super().batch(freqs, pwv, elevation)


class Atmosphere(BolojaxModel):
    """Atmosphere model.

    The primary backend is ``AmAtm``, which computes atmosphere profiles
    from an ``.amc`` configuration file using am-python and caches the
    results to disk.  A fixed-profile text file can be used as a
    fallback via ``custom_atm_file`` on the Instrument.
    """

    amc_file: str | None = None
    amc_args: list[str | float] | None = None
    profile_pwv_mm: float = 0.425

    _telescope: Instrument | None = PrivateAttr(default=None)
    _sampled_pwv: np.ndarray | None = PrivateAttr(default=None)
    _sampled_elev: np.ndarray | None = PrivateAttr(default=None)
    _nsamples: int = PrivateAttr(default=1)

    def set_telescope(self, value: Instrument) -> None:
        """Set the telescope (needed to sample elevation and PWV values)."""
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
        """Sample PWV and elevation for atmosphere evaluation."""
        self._telescope.pwv.sample(nsamples)
        self._telescope.elevation.sample(nsamples)
        self._sampled_pwv = 1e-3 * np.atleast_1d(self._telescope.pwv())  # mm → m
        self._sampled_elev = np.atleast_1d(self._telescope.elevation())
        self._nsamples = max(nsamples, 1)

    def temp(
        self, freqs: np.ndarray, elevation: np.ndarray | None = None
    ) -> np.ndarray:
        """Brightness temperature [K] for sampled conditions."""
        nsamp = max(self._nsamples, 1)
        elev = elevation if elevation is not None else self._sampled_elev
        temps, _ = self.cached_model.batch(freqs, self._sampled_pwv, elev)
        return temps.reshape((nsamp, 1, len(freqs)))

    def trans(
        self, freqs: np.ndarray, elevation: np.ndarray | None = None
    ) -> np.ndarray:
        """Transmission [0-1] for sampled conditions."""
        nsamp = max(self._nsamples, 1)
        elev = elevation if elevation is not None else self._sampled_elev
        _, transs = self.cached_model.batch(freqs, self._sampled_pwv, elev)
        return transs.reshape((nsamp, 1, len(freqs)))


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
