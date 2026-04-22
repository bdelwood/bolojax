"""
Physics object calculates physical quantities

Attributes:
h (float): Planck constant [J/s]
kB (float): Boltzmann constant [J/K]
c (float): Speed of light [m/s]
PI (float): Pi
mu0 (float): Permability of free space [H/m]
ep0 (float): Permittivity of free space [F/m]
Z0 (float): Impedance of free space [Ohm]
Tcmb (float): CMB Temperature [K]
co (dict): CO Emission lines [Hz]
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

h = 6.62607015e-34  # Planck constant [J·s] (CODATA 2018 exact)
kB = 1.380649e-23  # Boltzmann constant [J/K] (CODATA 2018 exact)
c = 299792458.0  # speed of light [m/s] (exact by definition)
PI = math.pi
mu0 = 1.25663706212e-6  # permeability of free space [H/m]
ep0 = 8.8541878128e-12  # permittivity of free space [F/m]
Z0 = math.sqrt(mu0 / ep0)
Tcmb = 2.7255  # CMB temperature [K] (Fixsen 2009)


def lamb(freq: ArrayLike, ind: ArrayLike = 1.0) -> Array:
    """Convert from frequency [Hz] to wavelength [m].

    Args:
        freq: frequencies [Hz]
        ind: index of refraction. Defaults to 1.

    Returns:
        Wavelength [m].
    """
    return c / (freq * ind)


def band_edges(freqs: Array, tran: Array) -> tuple[Array, Array]:
    """Find the -3 dB points of an arbitrary band.

    Args:
        freqs: frequency grid [Hz].
        tran: transmission profile on ``freqs``.

    Returns:
        Tuple of (flo, fhi) at the half-max points.
    """
    max_tran = jnp.amax(tran)
    max_tran_loc = jnp.argmax(tran)
    lo_point = jnp.argmin(abs(tran[:max_tran_loc] - 0.5 * max_tran))
    hi_point = jnp.argmin(abs(tran[max_tran_loc:] - 0.5 * max_tran)) + max_tran_loc
    flo = freqs[lo_point]
    fhi = freqs[hi_point]
    return flo, fhi


def spill_eff(
    freq: ArrayLike, pixd: ArrayLike, fnum: ArrayLike, wf: ArrayLike = 3.0
) -> Array:
    """Pixel beam coupling efficiency.

    Args:
        freq: frequencies [Hz].
        pixd: pixel size [m].
        fnum: f-number.
        wf: waist factor. Defaults to 3.

    Returns:
        Spillover efficiency.
    """
    return 1.0 - jnp.exp(
        (-jnp.power(PI, 2) / 2.0) * jnp.power((pixd / (wf * fnum * (c / freq))), 2)
    )


def edge_taper(ap_eff: ArrayLike) -> Array:
    """Edge taper given an aperture efficiency.

    Args:
        ap_eff: aperture efficiency.

    Returns:
        Edge taper [dB].
    """
    return 10.0 * jnp.log10(1.0 - ap_eff)


def apert_illum(
    freq: ArrayLike, pixd: ArrayLike, fnum: ArrayLike, wf: ArrayLike = 3.0
) -> Array:
    """Aperture illumination efficiency.

    Args:
        freq: frequencies [Hz].
        pixd: pixel diameter [m].
        fnum: f-number.
        wf: beam waist factor.

    Returns:
        Aperture illumination efficiency.
    """
    lamb_val = lamb(freq)
    w0 = pixd / wf
    theta_stop = lamb_val / (PI * w0)
    theta_apert = jnp.arange(0.0, jnp.arctan(1.0 / (2.0 * fnum)), 0.01)
    V = jnp.exp(-jnp.power(theta_apert, 2.0) / jnp.power(theta_stop, 2.0))
    eff_num = jnp.power(jnp.trapezoid(V * jnp.tan(theta_apert / 2.0), theta_apert), 2.0)
    eff_denom = jnp.trapezoid(jnp.power(V, 2.0) * jnp.sin(theta_apert), theta_apert)
    eff_fact = 2.0 * jnp.power(jnp.tan(theta_apert / 2.0), -2.0)
    return (eff_num / eff_denom) * eff_fact


def ruze_eff(freq: ArrayLike, sigma: ArrayLike) -> Array:
    """Ruze efficiency given frequency and surface RMS roughness.

    Args:
        freq: frequencies [Hz].
        sigma: RMS surface roughness [m].

    Returns:
        Ruze efficiency.
    """
    return jnp.exp(-jnp.power(4 * PI * sigma / (c / freq), 2.0))


def ohmic_eff(freq: ArrayLike, sigma: ArrayLike) -> Array:
    """Ohmic efficiency given frequency and conductivity.

    Args:
        freq: frequencies [Hz].
        sigma: conductivity [S/m].

    Returns:
        Ohmic efficiency.
    """
    return 1.0 - 4.0 * jnp.sqrt(PI * freq * mu0 / sigma) / Z0


def Trj_over_Tb(freq: ArrayLike, Tb: ArrayLike) -> Array:
    r"""Ratio $dT_{\mathrm{RJ}} / dT_b$ for a given physical temperature and frequency.

    Args:
        freq: frequencies [Hz].
        Tb: physical temperature [K].

    Returns:
        Rayleigh-Jeans to physical temperature derivative.
    """
    x = (h * freq) / (Tb * kB)
    thermo_fact = jnp.power((jnp.exp(x) - 1.0), 2.0) / (jnp.power(x, 2.0) * jnp.exp(x))
    return 1.0 / thermo_fact


def Tb_from_spec_rad(freq: ArrayLike, pow_spec: ArrayLike) -> Array:
    r"""Physical temperature from spectral radiance $[\mathrm{W}/(\mathrm{m}^2\,\mathrm{sr}\,\mathrm{Hz})]$.

    Args:
        freq: frequencies [Hz].
        pow_spec: spectral radiance.

    Returns:
        Physical temperature [K].
    """
    return (h * freq / kB) / jnp.log((2 * h * (freq**3 / c**2) / pow_spec) + 1)


def Tb_from_Trj(freq: ArrayLike, Trj: ArrayLike) -> Array:
    """Physical temperature from Rayleigh-Jeans temperature.

    Args:
        freq: frequencies [Hz].
        Trj: Rayleigh-Jeans temperature [K].

    Returns:
        Physical temperature [K].
    """
    alpha = (h * freq) / kB
    return alpha / jnp.log((2 * alpha / Trj) + 1)


def inv_var(err: ArrayLike) -> Array:
    """Inverse variance weights based on input errors.

    Args:
        err: errors to generate weights.

    Returns:
        Inverse-variance-weighted combination.
    """
    return 1.0 / (jnp.sqrt(jnp.sum(1.0 / (jnp.power(jnp.asarray(err), 2.0)))))


def dielectric_loss(
    freq: ArrayLike, thick: ArrayLike, ind: ArrayLike, ltan: ArrayLike
) -> Array:
    """Dielectric loss of a substrate.

    Args:
        freq: frequencies [Hz].
        thick: substrate thickness [m].
        ind: index of refraction.
        ltan: loss tangent.

    Returns:
        Fractional dielectric loss.
    """
    return 1.0 - jnp.exp((-2.0 * PI * ind * ltan * thick) / (lamb(freq)))


def rj_temp(powr: ArrayLike, bw: ArrayLike, eff: ArrayLike = 1.0) -> Array:
    r"""Rayleigh-Jeans temperature given power, bandwidth, and efficiency.

    Returns temperature in $K_{\mathrm{RJ}}$.

    Args:
        powr: power [W].
        bw: bandwidth [Hz].
        eff: efficiency. Defaults to 1.

    Returns:
        Rayleigh-Jeans temperature [K].
    """
    return powr / (kB * bw * eff)


def n_occ(freq: ArrayLike, temp: ArrayLike) -> Array:
    """Photon occupation number given frequency and blackbody temperature.

    Args:
        freq: frequency [Hz].
        temp: blackbody temperature [K].

    Returns:
        Bose-Einstein occupation number.
    """
    fact = (h * freq) / (kB * temp)
    fact = jnp.where(fact > 100, 100, fact)
    return 1.0 / (jnp.exp(fact) - 1.0)


def a_omega(freq: ArrayLike) -> Array:
    r"""Throughput $[\mathrm{m}^2]$ for a diffraction-limited detector.

    Args:
        freq: frequencies [Hz].

    Returns:
        Throughput ($\lambda^2$).
    """
    return lamb(freq) ** 2


def bb_spec_rad(freq: ArrayLike, temp: ArrayLike, emis: ArrayLike = 1.0) -> Array:
    r"""Blackbody spectral radiance $[\mathrm{W}/(\mathrm{m}^2\,\mathrm{sr}\,\mathrm{Hz})]$.

    Args:
        freq: frequencies [Hz].
        temp: blackbody temperature [K].
        emis: blackbody emissivity. Defaults to 1.

    Returns:
        Spectral radiance.
    """
    return emis * (2 * h * (freq**3) / (c**2)) * n_occ(freq, temp)


def bb_pow_spec(freq: ArrayLike, temp: ArrayLike, emis: ArrayLike = 1.0) -> Array:
    """Blackbody power spectrum [W/Hz] on a diffraction-limited polarimeter.

    Args:
        freq: frequencies [Hz].
        temp: blackbody temperature [K].
        emis: blackbody emissivity. Defaults to 1.

    Returns:
        Power spectral density [W/Hz].
    """
    return 0.5 * a_omega(freq) * bb_spec_rad(freq, temp, emis)


def ani_pow_spec(freq: ArrayLike, temp: ArrayLike, emiss: ArrayLike = 1.0) -> Array:
    r"""Derivative of blackbody power spectrum, $dP/dT$ [W/K].

    Evaluated on a diffraction-limited detector given a frequency,
    blackbody temperature, and emissivity.

    Args:
        freq: frequency [Hz].
        temp: blackbody temperature [K].
        emiss: blackbody emissivity. Defaults to 1.

    Returns:
        Power derivative with respect to temperature [W/K].
    """
    return (
        emiss
        * kB
        * jnp.exp((h * freq) / (kB * temp))
        * (h * freq * n_occ(freq, temp) / (kB * temp)) ** 2
    )


def pow_frac(T1: ArrayLike, T2: ArrayLike, freqs: ArrayLike) -> Array:
    """Fractional power between two physical temperatures.

    Args:
        T1: first temperature [K].
        T2: second temperature [K].
        freqs: frequencies [Hz].

    Returns:
        Power ratio P(T1)/P(T2).
    """
    return bb_pow_spec(freqs, T1) / bb_pow_spec(freqs, T2)
