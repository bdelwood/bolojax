"""Sky model."""

from __future__ import annotations

from collections import OrderedDict as odict
from functools import cached_property
from typing import Any, ClassVar

import h5py as hp
import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from . import physics
from .cfg import Var
from .utils import cfg_path, is_not_none

GHz_to_Hz = 1.0e09
m_to_mm = 1.0e03
mm_to_um = 1.0e03


def interp_spectra(freqs, freq_grid, vals):
    """Interpolate a spectrum."""
    freq_grid = freq_grid * GHz_to_Hz
    return np.interp(freqs, freq_grid, vals)


class AtmModel:
    """Atmospheric model using tabulated values."""

    def __init__(self, fname, site):
        self._file = hp.File(fname, "r")
        self._data = self._file[site]

    @staticmethod
    def get_keys(pwv, elev):
        return [
            f"{int(round(pwv_ * m_to_mm, 1) * mm_to_um)},{int(round(elev_, 0))}"
            for pwv_, elev_ in np.broadcast(pwv, elev)
        ]

    def temp(self, keys, freqs):
        return np.array(
            [
                interp_spectra(freqs, self._data[key_][0], self._data[key_][2])
                for key_ in keys
            ]
        )

    def trans(self, keys, freqs):
        return np.array(
            [
                interp_spectra(freqs, self._data[key_][0], self._data[key_][3])
                for key_ in keys
            ]
        )


class CustomAtm:
    """Atmospheric model using custom value from a txt file."""

    def __init__(self, fname):
        self._freqs, self._temps, self._trans = np.loadtxt(
            fname, unpack=True, usecols=[0, 2, 3], dtype=np.float64
        )

    def temp(self, freqs):
        return interp_spectra(freqs, self._freqs, self._temps)

    def trans(self, freqs):
        return interp_spectra(freqs, self._freqs, self._trans)


class Atmosphere(BaseModel):
    """Atmosphere model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    atm_model_file: str | None = None

    _atm_model: Any = PrivateAttr(default=None)
    _telescope: Any = PrivateAttr(default=None)
    _sampled_keys: Any = PrivateAttr(default=None)
    _nsamples: int = PrivateAttr(default=1)

    def set_telescope(self, value):
        """Set the telescope (needed to sample elevation and PWV values)."""
        self._telescope = value

    @cached_property
    def cached_model(self):
        """Cache the Atmosphere model."""
        if is_not_none(self._telescope.custom_atm_file):
            return CustomAtm(cfg_path(self._telescope.custom_atm_file))
        if is_not_none(self.atm_model_file):
            return AtmModel(cfg_path(self.atm_model_file), self._telescope.site)
        return None

    def sample(self, nsamples):
        """Sample the atmosphere."""
        model = self.cached_model
        if isinstance(model, CustomAtm):
            self._sampled_keys = None
            self._nsamples = nsamples
            return
        self._telescope.pwv.sample(nsamples)
        self._telescope.elevation.sample(nsamples)
        self._sampled_keys = model.get_keys(
            1e-3 * np.atleast_1d(self._telescope.pwv()),
            np.atleast_1d(self._telescope.elevation()),
        )
        self._nsamples = max(nsamples, 1)

    def temp(self, freqs):
        """Get sampled temperatures."""
        model = self.cached_model
        nfreqs = len(freqs)
        out_shape = (max(self._nsamples, 1), 1, nfreqs)
        if self._sampled_keys is None:
            ones = np.ones((max(self._nsamples, 1), 1, 1))
            return (ones * model.temp(freqs)).reshape(out_shape)
        return model.temp(self._sampled_keys, freqs).reshape(out_shape)

    def trans(self, freqs):
        """Get sampled transmission coefs."""
        model = self.cached_model
        nfreqs = len(freqs)
        out_shape = (max(self._nsamples, 1), 1, nfreqs)
        if self._sampled_keys is None:
            ones = np.ones((max(self._nsamples, 1), 1, 1))
            return (ones * model.trans(freqs)).reshape(out_shape)
        return model.trans(self._sampled_keys, freqs).reshape(out_shape)


class Foreground(BaseModel):
    """Foreground model base class."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    spectral_index: Var() = None
    scale_frequency: Var("GHz") = None
    emiss: Var() = 1.0


class Dust(Foreground):
    """Dust emission model."""

    amplitude: Var("MJy") = None
    scale_temperature: Var("K") = None

    _nsamples: int = PrivateAttr(default=1)
    _amp: Any = PrivateAttr(default=None)
    _scale_temp: Any = PrivateAttr(default=None)

    def sample(self, nsamples):
        """Sample this component."""
        self.amplitude.sample(nsamples)
        self.scale_temperature.sample(nsamples)
        self._amp = np.expand_dims(np.expand_dims(self.amplitude.SI, -1), -1)
        self._scale_temp = np.expand_dims(
            np.expand_dims(self.scale_temperature.SI, -1), -1
        )
        self._nsamples = max(self._amp.size, self._scale_temp.size)

    def temp(self, freqs):
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
    def __temp(freqs, emiss, amp, scale_frequency, spectral_index, scale_temp):  # pylint: disable=too-many-arguments
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
    _amp: Any = PrivateAttr(default=None)

    def sample(self, nsamples):
        """Sample this component."""
        self.amplitude.sample(nsamples)
        self._amp = np.expand_dims(np.expand_dims(self.amplitude.SI, -1), -1)
        self._nsamples = self._amp.size

    def temp(self, freqs):
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
    def __temp(freqs, emiss, amp, scale_frequency, spectral_index):
        """Return the effective physical temperature."""
        bright_temp = emiss * amp
        # Frequency scaling (freq / sync_freq)**sync_ind
        freq_scale = (freqs / scale_frequency) ** spectral_index
        scaled_bright_temp = bright_temp * freq_scale
        # Convert brightness temperature [K_RJ] to physical temperature [K]
        return physics.Tb_from_Trj(freqs, scaled_bright_temp)


class Universe(BaseModel):
    """Collection of emission models."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    dust: Dust | None = None
    synchrotron: Synchrotron | None = None
    atmosphere: Atmosphere | None = None

    sources: ClassVar[list[str]] = ["cmb", "dust", "synchrotron", "atmosphere"]

    def sample(self, nsamples):
        """Sample the sky component."""
        self.dust.sample(nsamples)
        self.synchrotron.sample(nsamples)
        self.atmosphere.sample(nsamples)

    def temp(self, freqs):
        """Get sampled temperatures."""
        ret = odict()
        ret["cmb"] = physics.Tcmb
        ret["dust"] = self.dust.temp(freqs)
        ret["synchrotron"] = self.synchrotron.temp(freqs)
        ret["atmosphere"] = self.atmosphere.temp(freqs)
        return ret

    def trans(self, freqs):
        """Get sampled transmission coefs."""
        ret = odict()
        ret["atmosphere"] = self.atmosphere.trans(freqs)
        return ret
