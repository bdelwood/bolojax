"""Computations for noise estimation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from bolojax.compute import physics
from bolojax.compute.beam_correlation import compute_corr_curves


def Flink(n: ArrayLike, Tb: ArrayLike, Tc: ArrayLike) -> Array:
    """Link factor for the bolo to the bath.

    Args:
        n: thermal carrier index.
        Tb: bath temperature [K].
        Tc: transition temperature [K].

    Returns:
        Dimensionless link factor.
    """
    return (
        ((n + 1) / (2 * n + 3))
        * (1 - (Tb / Tc) ** (2 * n + 3))
        / (1 - (Tb / Tc) ** (n + 1))
    )


def G(psat: ArrayLike, n: ArrayLike, Tb: ArrayLike, Tc: ArrayLike) -> Array:
    """Thermal conduction between the bolo and the bath.

    Args:
        psat: saturation power [W].
        n: thermal carrier index.
        Tb: bath temperature [K].
        Tc: bolo transition temperature [K].

    Returns:
        Thermal conductance [W/K].
    """
    return psat * (n + 1) * (Tc**n) / ((Tc ** (n + 1)) - (Tb ** (n + 1)))


def calc_photon_NEP(
    popts: Array, freqs: Array, factors: np.ndarray | None = None
) -> tuple[Array, Array]:
    r"""Calculate photon NEP $[\mathrm{W}/\sqrt{\mathrm{Hz}}]$ for a detector.

    Args:
        popts: power from each optical element, shape ``(n_elem, n_freq)`` [W].
        freqs: frequencies of observation [Hz].
        factors: per-element correlation factors from ``corr_facts``.
            If ``None``, correlations are ignored.

    Returns:
        Tuple of (NEP, NEP_corr).  Without correlations both are identical.
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


def bolo_NEP(flink: ArrayLike, G_val: ArrayLike, Tc: ArrayLike) -> Array:
    r"""Thermal carrier NEP $[\mathrm{W}/\sqrt{\mathrm{Hz}}]$.

    Args:
        flink: link factor to the bolo bath.
        G_val: thermal conduction between the bolo and the bath [W/K].
        Tc: bolo transition temperature [K].

    Returns:
        Bolometer phonon NEP.
    """
    return jnp.sqrt(4 * physics.kB * flink * (Tc**2) * G_val)


def read_NEP(
    pelec: ArrayLike, boloR: ArrayLike, nei: ArrayLike, sfact: ArrayLike = 1.0
) -> Array:
    r"""Readout NEP $[\mathrm{W}/\sqrt{\mathrm{Hz}}]$ for a voltage-biased bolo.

    Args:
        pelec: bias power [W].
        boloR: bolometer resistance [Ohms].
        nei: noise equivalent current $[\mathrm{A}/\sqrt{\mathrm{Hz}}]$.
        sfact: responsivity scale factor. Defaults to 1.

    Returns:
        Readout NEP.
    """
    responsivity = sfact / jnp.sqrt(boloR * pelec)
    return nei / responsivity


def dPdT(eff: ArrayLike, freqs: Array) -> Array:
    """Change in power on the detector with change in CMB temperature [W/K].

    Args:
        eff: detector efficiency (scalar or frequency-dependent).
        freqs: observation frequencies [Hz].

    Returns:
        Integrated dP/dT [W/K].
    """
    temp = jnp.full_like(freqs, physics.Tcmb)
    return jnp.trapezoid(
        physics.ani_pow_spec(jnp.asarray(freqs), temp, jnp.asarray(eff)), freqs
    )


def NET_from_NEP(
    nep: ArrayLike, freqs: Array, sky_eff: ArrayLike, opt_coup: ArrayLike = 1.0
) -> Array:
    r"""NET $[\mathrm{K}\sqrt{\mathrm{s}}]$ from NEP.

    Args:
        nep: NEP $[\mathrm{W}/\sqrt{\mathrm{Hz}}]$.
        freqs: observation frequencies [Hz].
        sky_eff: efficiency between the detector and the sky.
        opt_coup: optical coupling to the detector. Defaults to 1.

    Returns:
        Noise-equivalent temperature per detector.
    """
    dpdt = opt_coup * dPdT(sky_eff, freqs)
    return nep / (jnp.sqrt(2.0) * dpdt)


def NET_arr(
    net: ArrayLike, n_det: int | ArrayLike, det_yield: ArrayLike = 1.0
) -> Array:
    r"""Array NET $[\mathrm{K}\sqrt{\mathrm{s}}]$ from NET per detector and number of detectors.

    Args:
        net: NET per detector.
        n_det: number of detectors.
        det_yield: detector yield. Defaults to 1.

    Returns:
        Array-level noise-equivalent temperature.
    """
    return net / (jnp.sqrt(n_det * det_yield))


def map_depth(
    net_arr: ArrayLike, fsky: ArrayLike, tobs: ArrayLike, obs_eff: ArrayLike
) -> Array:
    r"""Sensitivity [K-arcmin] given array NET.

    Args:
        net_arr: array NET $[\mathrm{K}\sqrt{\mathrm{s}}]$.
        fsky: sky fraction.
        tobs: observation time [s].
        obs_eff: observing efficiency.

    Returns:
        Map depth [K-arcmin].
    """
    return jnp.sqrt(
        (4.0 * physics.PI * fsky * 2.0 * jnp.power(net_arr, 2.0)) / (tobs * obs_eff)
    ) * (10800.0 / physics.PI)


class Noise:  # pylint: disable=too-many-instance-attributes
    r"""Noise object calculates NEP, NET, mapping speed, and sensitivity.

    Computes Bose white-noise correlation factors following Hill & Kusaka,
    Appl. Opt. 63, 1654 (2024), arXiv:2309.01153.  The corr_facts method
    evaluates the array-averaged HBT coefficient (Eq. 67) which enters the
    array noise variance as a $(1 + \gamma^{(2)})$ multiplier on the wave
    noise term (Eq. 68).

    Args:
        beam_preset: name of a beam correlation preset ("bolocalc",
            "trunc_gauss", "he11") or a dict with beam model parameters.
            Defaults to "bolocalc".
    """

    _det_p: Array
    _c_apert: Array
    _c_stop: Array

    def __init__(self, beam_preset: str | dict = "bolocalc") -> None:
        # Aperture stop names
        self._ap_names: list[str] = ["APERT", "STOP", "LYOT"]

        # Compute correlation curves from beam model
        p_grid, gamma_apert, gamma_stop = compute_corr_curves(beam_preset)
        self._det_p = p_grid
        self._c_apert = gamma_apert
        self._c_stop = gamma_stop

        # Geometric pitch factor
        self._geo_fact: int = 6  # Hex packing; 6 for temperature. More complicated for polarization, not a simple factor.

    def corr_facts(
        self,
        elems: list[str],
        det_pitch: float,
        ap_names: list[str],
        flamb_max: float = 3.0,
    ) -> np.ndarray:
        r"""Calculate the Bose white-noise correlation factor.

        Args:
            elems: optical element names in the camera.
            det_pitch: detector pitch in $F\lambda$ units.
            ap_names: names identifying aperture stop elements.
            flamb_max: maximum detector pitch distance for which to
                calculate the correlation factor. Defaults to 3.

        Returns:
            Per-element correlation factors.
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

    def photon_NEP(
        self,
        popts: Array,
        freqs: Array,
        *,
        elems: list[str] | None = None,
        det_pitch: float | None = None,
        ap_names: list[str] | None = None,
    ) -> tuple[Array, Array]:
        r"""Calculate photon NEP $[\mathrm{W}/\sqrt{\mathrm{Hz}}]$ for a detector.

        Args:
            popts: power from each optical element [W].
            freqs: frequencies of observation [Hz].
            elems: optical element names.
            det_pitch: detector pitch in $F\lambda$ units.
            ap_names: names identifying aperture stop elements.

        Returns:
            Tuple of (NEP, NEP_corr).
        """
        if elems is None or det_pitch is None or ap_names is None:
            factors = None
        else:
            factors = self.corr_facts(elems, det_pitch, ap_names)
        return calc_photon_NEP(popts, freqs, factors)
