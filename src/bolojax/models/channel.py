"""Channel model."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from bolojax.compute import noise, physics

from .params import Var
from .sky import Universe
from .utils import is_not_none


class Channel(BaseModel):  # pylint: disable=too-many-instance-attributes
    """Channel Model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    _min_tc_tb_diff: ClassVar[float] = 0.010

    band_center: Var("GHz") = None
    fractional_bandwidth: Var() = 0.35
    band_response: Var() = 1.0

    det_eff: Var() = 1.0
    squid_nei: Var("pA/rtHz") = 1.0
    bolo_resistance: Var("Ohm") = 1.0

    pixel_size: Var("mm") = 6.8
    waist_factor: Var() = 3.0

    Tc: Var("K") = 0.165
    Tc_fraction: Var() = None

    num_det_per_water: int = 542
    num_wafer_per_optics_tube: int = 1
    num_optics_tube: int = 3

    psat: Var() = None
    psat_factor: Var() = 3.0

    read_frac: Var() = 0.1
    carrier_index: Var() = 3
    G: Var("pW/K") = None
    Flink: Var() = None
    Yield: Var() = None
    response_factor: Var() = None
    nyquist_inductance: Var() = None

    @property
    def noise_calc(self):
        """Return the noise calculator from the parent camera."""
        return self._camera.noise_calc

    # Private runtime state
    _optical_effic: Any = PrivateAttr(default=None)
    _optical_emiss: Any = PrivateAttr(default=None)
    _optical_temps: Any = PrivateAttr(default=None)
    _sky_temp_dict: Any = PrivateAttr(default=None)
    _sky_tran_dict: Any = PrivateAttr(default=None)
    _det_effic: Any = PrivateAttr(default=None)
    _det_emiss: Any = PrivateAttr(default=None)
    _det_temp: Any = PrivateAttr(default=None)
    _camera: Any = PrivateAttr(default=None)
    _idx: Any = PrivateAttr(default=None)
    _freqs: Any = PrivateAttr(default=None)
    _flo: Any = PrivateAttr(default=None)
    _fhi: Any = PrivateAttr(default=None)
    _freq_mask: Any = PrivateAttr(default=None)
    _bandwidth: Any = PrivateAttr(default=None)
    _band_width_factor: float = PrivateAttr(default=1.0)

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        self._bandwidth = value

    def set_camera(self, camera, idx):
        """Set the parent camera and the channel index."""
        self._camera = camera
        self._idx = idx

    def sample(self, nsamples):
        """Sample PDF parameters."""
        self.det_eff.sample(nsamples)
        self.squid_nei.sample(nsamples)
        self.bolo_resistance.sample(nsamples)

    @property
    def camera(self):
        return self._camera

    @property
    def freqs(self):
        return self._freqs

    @property
    def flo(self):
        return self._flo

    @property
    def fhi(self):
        return self._fhi

    @property
    def ndet(self):
        return (
            self.num_det_per_water
            * self.num_wafer_per_optics_tube
            * self.num_optics_tube
        )

    @property
    def idx(self):
        return self._idx

    def photon_NEP(self, elem_power, elems=None, ap_names=None):
        """Return the photon NEP given the power in the element in the optical chain."""
        if elems is None:
            return self.noise_calc.photon_NEP(elem_power, self._freqs)
        det_pitch = self.pixel_size.SI / (
            self.camera.f_number.SI * physics.lamb(self.band_center.SI)
        )
        return self.noise_calc.photon_NEP(
            elem_power, self._freqs, elems=elems, det_pitch=det_pitch, ap_names=ap_names
        )

    def _resolve_psat(self, opt_pow):
        """Resolve Psat from explicit value or psat_factor.

        Explicit psat takes priority over psat_factor (matching BoloCalc's
        convention). Result is broadcast to match opt_pow shape.
        """
        if is_not_none(self.psat) and np.isfinite(self.psat.SI).all():
            return np.broadcast_to(self.psat.SI, np.shape(opt_pow))
        if is_not_none(self.psat_factor) and np.isfinite(self.psat_factor.SI):
            return opt_pow * self.psat_factor.SI
        return opt_pow * 3.0

    def bolo_Psat(self, opt_pow):
        """Return the PSAT used in the computation."""
        return self._resolve_psat(opt_pow)

    def bolo_G(self, opt_pow):
        """Return the Bolometric G factor used in the computation."""
        tb = self._camera.bath_temperature()
        tc = self.Tc.SI
        n = self.carrier_index.SI
        if is_not_none(self.G) and np.isfinite(self.G.SI).all():
            g = self.G.SI
        else:
            g = noise.G(self._resolve_psat(opt_pow), n, tb, tc)
        return g

    def bolo_Flink(self):
        """Return the Bolometric f-link used in the computation."""
        tb = self._camera.bath_temperature()
        tc = self.Tc.SI
        n = self.carrier_index.SI
        if is_not_none(self.Flink) and np.isfinite(self.Flink.SI):
            flink = self.Flink.SI
        else:
            flink = noise.Flink(n, tb, tc)
        return flink

    def bolo_NEP(self, opt_pow):
        """Return the bolometric NEP given the detector details."""
        tb = self._camera.bath_temperature()
        tc = self.Tc.SI
        n = self.carrier_index.SI
        if is_not_none(self.G) and np.isfinite(self.G.SI).all():
            g = self.G.SI
        else:
            g = noise.G(self._resolve_psat(opt_pow), n, tb, tc)
        if is_not_none(self.Flink) and np.isfinite(self.Flink.SI):
            flink = self.Flink.SI
        else:
            flink = noise.Flink(n, tb, tc)
        return noise.bolo_NEP(flink, g, tc)

    def read_NEP(self, opt_pow):
        """Return the readout NEP given the detector details."""
        if np.isnan(self.squid_nei.SI).any():
            return None
        if np.isnan(self.bolo_resistance.SI).any():
            return None
        p_sat = self._resolve_psat(opt_pow)
        p_bias = (p_sat - opt_pow).clip(0, np.inf)
        if (
            is_not_none(self.response_factor)
            and np.isfinite(self.response_factor.SI).all()
        ):
            s_fact = self.response_factor.SI
        else:
            s_fact = 1.0
        return noise.read_NEP(
            p_bias, self.bolo_resistance.SI.T, self.squid_nei.SI.T, s_fact
        )

    def compute_evaluation_freqs(self, freq_resol=None):
        """Compute and return the evaluation frequecies."""
        self.bandwidth = self.band_center.SI * self.fractional_bandwidth.SI
        if freq_resol is None:
            freq_resol = 0.05 * self.bandwidth
        else:
            freq_resol = freq_resol * 1e9
        self._flo = self.band_center.SI - 0.5 * self.bandwidth
        self._fhi = self.band_center.SI + 0.5 * self.bandwidth

        # Grid may extend beyond band edges when band_width_factor > 1
        # included to match BoloCalc's behavior
        # (uses 1.3, ie takes the grid 30% beyond the band edges)
        half = self._band_width_factor * 0.5 * self.bandwidth
        grid_lo = self.band_center.SI - half
        grid_hi = self.band_center.SI + half
        grid_width = grid_hi - grid_lo
        freq_step = np.ceil(grid_width / freq_resol).astype(int)

        self._freqs = np.linspace(grid_lo, grid_hi, freq_step + 1)
        self._freq_mask = (self._freqs >= self._flo) & (self._freqs <= self._fhi)
        band_mean_response = self.band_response.sample(0, self._freqs)
        if np.isscalar(band_mean_response):
            return self._freqs
        self._flo, self._fhi = physics.band_edges(self._freqs, band_mean_response)
        self.bandwidth = self._fhi - self._flo
        self._freq_mask = (self._freqs >= self._flo) & (self._freqs <= self._fhi)
        self._freqs = self._freqs[self._freq_mask]
        return self._freqs

    def eval_optical_chain(self, nsample=0, freq_resol=None):
        """Evaluate the performance of the optical chain for this channel."""
        self.compute_evaluation_freqs(freq_resol)
        self._optical_effic = []
        self._optical_emiss = []
        self._optical_temps = []
        for elem in self._camera.optics.values():
            effic, emiss, temps = elem.compute_channel(self, self._freqs, nsample)
            self._optical_effic.append(effic)
            self._optical_emiss.append(emiss)
            self._optical_temps.append(temps)

    def eval_det_response(self, nsample=0, freq_resol=None):
        """Evaluate the detector response for this channel."""
        self._freqs = self.compute_evaluation_freqs(freq_resol)
        self.band_response.sample(nsample, self._freqs)
        self.det_eff.sample(nsample)
        det_eff = self.band_response.SI * self.det_eff.SI
        if self._band_width_factor > 1.0:
            det_eff = det_eff * self._freq_mask.astype(float)
        self._det_effic = det_eff
        self._det_emiss = 0.0
        self._det_temp = self._camera.bath_temperature()

    def eval_sky(self, universe, freq_resol=None):
        """Evaluate the sky parameters for this channel.

        This is done here, b/c the frequencies we care about are channel dependent.
        """
        self._freqs = self.compute_evaluation_freqs(freq_resol)
        self._sky_temp_dict = universe.temp(self._freqs)
        self._sky_tran_dict = universe.trans(self._freqs)

    @property
    def optical_effic(self):
        return self._optical_effic

    @property
    def optical_emiss(self):
        return self._optical_emiss

    @property
    def optical_temps(self):
        return self._optical_temps

    @property
    def sky_names(self):
        return list(self._sky_temp_dict.keys())

    @property
    def sky_temps(self):
        return [self._sky_temp_dict.get(k) for k in Universe.sources]

    @property
    def sky_effic(self):
        return [self._sky_tran_dict.get(k, 1.0) for k in Universe.sources]

    @property
    def sky_emiss(self):
        return [1] * len(Universe.sources)
