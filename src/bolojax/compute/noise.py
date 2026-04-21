"""Computations for noise estimation."""

import jax.numpy as jnp
import numpy as np

from . import physics
from .beam_correlation import compute_corr_curves


def Flink(n, Tb, Tc):
    """
    Link factor for the bolo to the bath.

    Args:
    n (float): thermal carrier index
    Tb (float): bath temperature [K]
    Tc (float): transition temperature [K]
    """
    return (
        ((n + 1) / (2 * n + 3))
        * (1 - (Tb / Tc) ** (2 * n + 3))
        / (1 - (Tb / Tc) ** (n + 1))
    )


def G(psat, n, Tb, Tc):
    """
    Thermal conduction between the bolo and the bath.

    Args:
    psat (float): saturation power [W]
    n (float): thermal carrier index
    Tb (float): bath temperature [K]
    Tc (float): bolo transition temperature [K]
    """
    return psat * (n + 1) * (Tc**n) / ((Tc ** (n + 1)) - (Tb ** (n + 1)))


def calc_photon_NEP(popts, freqs, factors=None):
    """
    Calculate photon NEP [W/rtHz] for a detector.

    Args:
    popts (list): power from elements in the optical elements [W]
    freqs (list): frequencies of observation [Hz]
    """
    popt = jnp.sum(popts, axis=0)
    # No correlations
    if factors is None:
        popt2 = popt * popt
        nep = jnp.sqrt(
            jnp.trapezoid((2.0 * physics.h * freqs * popt + 2.0 * popt2), freqs)
        )
        return nep, nep

    popt2 = sum(
        [popts[i] * popts[j] for i in range(len(popts)) for j in range(len(popts))]
    )
    popt2arr = sum(
        [
            factors[i] * factors[j] * popts[i] * popts[j]
            for i in range(len(popts))
            for j in range(len(popts))
        ]
    )
    nep = jnp.sqrt(jnp.trapezoid((2.0 * physics.h * freqs * popt + 2.0 * popt2), freqs))
    neparr = jnp.sqrt(
        jnp.trapezoid((2.0 * physics.h * freqs * popt + 2.0 * popt2arr), freqs)
    )

    return nep, neparr


def bolo_NEP(flink, G_val, Tc):
    """
    Thermal carrier NEP [W/rtHz].

    Args:
    flink (float): link factor to the bolo bath
    G (float): thermal conduction between the bolo and the bath [W/K]
    Tc (float): bolo transition temperature [K]
    """
    return jnp.sqrt(4 * physics.kB * flink * (Tc**2) * G_val)


def read_NEP(pelec, boloR, nei, sfact=1.0):
    """
    Readout NEP [W/rtHz] for a voltage-biased bolo.

    Args:
    pelec (float): bias power [W]
    boloR (float): bolometer resistance [Ohms]
    nei (float): noise equivalent current [A/rtHz]
    """
    responsivity = sfact / jnp.sqrt(boloR * pelec)
    return nei / responsivity


def dPdT(eff, freqs):
    """
    Change in power on the detector with change in CMB temperature [W/K].

    Args:
    eff (float): detector efficiency
    freqs (float): observation frequencies [Hz]
    """
    temp = jnp.full_like(freqs, physics.Tcmb)
    return jnp.trapezoid(
        physics.ani_pow_spec(jnp.asarray(freqs), temp, jnp.asarray(eff)), freqs
    )


def NET_from_NEP(nep, freqs, sky_eff, opt_coup=1.0):
    """
    NET [K-rts] from NEP.

    Args:
    nep (float): NEP [W/rtHz]
    freqs (list): observation frequencies [Hz]
    sky_eff (float): efficiency between the detector and the sky
    opt_coup (float): optical coupling to the detector. Default to 1.
    """
    dpdt = opt_coup * dPdT(sky_eff, freqs)
    return nep / (jnp.sqrt(2.0) * dpdt)


def NET_arr(net, n_det, det_yield=1.0):
    """
    Array NET [K-rts] from NET per detector and num of detectors.

    Args:
    net (float): NET per detector
    n_det (int): number of detectors
    det_yield (float): detector yield. Defaults to 1.
    """
    return net / (jnp.sqrt(n_det * det_yield))


def map_depth(net_arr, fsky, tobs, obs_eff):
    """
    Sensitivity [K-arcmin] given array NET.

    Args:
    net_arr (float): array NET [K-rts]
    fsky (float): sky fraction
    tobs (float): observation time [s]
    """
    return jnp.sqrt(
        (4.0 * physics.PI * fsky * 2.0 * jnp.power(net_arr, 2.0)) / (tobs * obs_eff)
    ) * (10800.0 / physics.PI)


class Noise:  # pylint: disable=too-many-instance-attributes
    """
    Noise object calculates NEP, NET, mapping speed, and sensitivity.

    Computes Bose white-noise correlation factors following Hill & Kusaka,
    Appl. Opt. 63, 1654 (2024), arXiv:2309.01153.  The corr_facts method
    evaluates the array-averaged HBT coefficient (Eq. 67) which enters the
    array noise variance as a (1 + gamma^(2)) multiplier on the wave noise
    term (Eq. 68).

    Args:
        beam_preset: name of a beam correlation preset ("bolocalc",
            "trunc_gauss", "he11") or a dict with beam model parameters.
            Defaults to "bolocalc".
    """

    _det_p: np.ndarray
    _c_apert: np.ndarray
    _c_stop: np.ndarray

    def __init__(self, beam_preset="bolocalc"):

        # Aperture stop names
        self._ap_names = ["APERT", "STOP", "LYOT"]

        # Compute correlation curves from beam model
        p_grid, gamma_apert, gamma_stop = compute_corr_curves(beam_preset)
        self._det_p = p_grid
        self._c_apert = gamma_apert
        self._c_stop = gamma_stop

        # Geometric pitch factor
        self._geo_fact = 6  # Hex packing; 6 for temperature. More complicated for polarization, not a simple factor.

    def corr_facts(self, elems, det_pitch, ap_names, flamb_max=3.0):
        """
        Calculate the Bose white-noise correlation factor.

        Args:
        elems (list): optical elements in the camera
        det_pitch (float): detector pitch in f-lambda units
        flamb_max (float): the maximum detector pitch distance
        for which to calculate the correlation factor.
        Default is 3.
        """
        ndets = int(round(flamb_max / (det_pitch), 0))
        inds1 = [
            np.argmin(abs(np.array(self._det_p) - det_pitch * (n + 1)))
            for n in range(ndets)
        ]
        inds2 = [
            np.argmin(abs(np.array(self._det_p) - det_pitch * (n + 1) * np.sqrt(3.0)))
            for n in range(ndets)
        ]
        inds = np.sort(inds1 + inds2)
        at_det = False
        factors = []
        for elem_ in elems:
            if at_det:
                factors.append(1.0)
                continue
            if "CMB" in elem_.upper():
                use_abs = abs(self._c_apert)
            elif elem_ in ap_names:
                # Original BoloCalc uses c_apert and c_stop in place of i_apert and i_stop
                use_abs = abs(self._c_stop)
                at_det = True
            else:
                use_abs = abs(self._c_apert)
            factors.append(
                np.sqrt(1.0 + self._geo_fact * (np.sum([use_abs[ind] for ind in inds])))
            )

        return np.array(factors)

    def photon_NEP(self, popts, freqs, **kwargs):
        """
        Calculate photon NEP [W/rtHz] for a detector.

        Args:
        popts (list): power from elements in the optical elements [W]
        freqs (list): frequencies of observation [Hz]
        elems (list): optical elements
        det_pitch (float): detector pitch in f-lambda units. Default is None.
        """
        elems = kwargs.get("elems")
        det_pitch = kwargs.get("det_pitch")
        ap_names = kwargs.get("ap_names")
        if elems is None or det_pitch is None:
            factors = None
        else:
            factors = self.corr_facts(elems, det_pitch, ap_names)
        return calc_photon_NEP(popts, freqs, factors)
