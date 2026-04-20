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

import math

import jax.numpy as jnp

h = 6.6261e-34
kB = 1.3806e-23
c = 299792458.0
PI = math.pi
mu0 = 1.256637e-6
ep0 = 8.854188e-12
Z0 = math.sqrt(mu0 / ep0)
Tcmb = 2.725


def lamb(freq, ind=1.0):
    """
    Convert from from frequency [Hz] to wavelength [m]

    Args:
    freq (float): frequencies [Hz]
    ind: index of refraction. Defaults to 1
    """
    return c / (freq * ind)


def band_edges(freqs, tran):
    """Find the -3 dB points of an arbitrary band."""
    max_tran = jnp.amax(tran)
    max_tran_loc = jnp.argmax(tran)
    lo_point = jnp.argmin(abs(tran[:max_tran_loc] - 0.5 * max_tran))
    hi_point = jnp.argmin(abs(tran[max_tran_loc:] - 0.5 * max_tran)) + max_tran_loc
    flo = freqs[lo_point]
    fhi = freqs[hi_point]
    return flo, fhi


def spill_eff(freq, pixd, fnum, wf=3.0):
    """
    Pixel beam coupling efficiency given a frequency [Hz],
    pixel diameter [m], f-number, and beam wasit factor

    Args:
    freq (float): frequencies [Hz]
    pixd (float): pixel size [m]
    fnum (float): f-number
    wf (float): waist factor. Defaults to 3.
    """
    return 1.0 - jnp.exp(
        (-jnp.power(PI, 2) / 2.0) * jnp.power((pixd / (wf * fnum * (c / freq))), 2)
    )


def edge_taper(ap_eff):
    """
    Edge taper given an aperture efficiency

    Args:
    ap_eff (float): aperture efficiency
    """
    return 10.0 * jnp.log10(1.0 - ap_eff)


def apert_illum(freq, pixd, fnum, wf=3.0):
    """
    Aperture illumination efficiency given a frequency [Hz],
    pixel diameter [m], f-number, and beam waist factor

    Args:
    freq (float): frequencies [Hz]
    pixd (float): pixel diameter [m]
    fnum (float): f-number
    wf (float): beam waist factor
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


def ruze_eff(freq, sigma):
    """
    Ruze efficiency given a frequency [Hz] and surface RMS roughness [m]

    Args:
    freq (float): frequencies [Hz]
    sigma (float): RMS surface roughness
    """
    return jnp.exp(-jnp.power(4 * PI * sigma / (c / freq), 2.0))


def ohmic_eff(freq, sigma):
    """
    Ohmic efficiency given a frequency [Hz] and conductivity [S/m]

    Args:
    freq (float): frequencies [Hz]
    sigma (float): conductivity [S/m]
    """
    return 1.0 - 4.0 * jnp.sqrt(PI * freq * mu0 / sigma) / Z0


def Trj_over_Tb(freq, Tb):
    """
    Brightness temperature [K_RJ] given a physical temperature [K]
    and frequency [Hz]. dTrj / dTb

    Args:
    freq (float): frequencies [Hz]
    Tb (float): physical temperature. Default to Tcmb
    """
    x = (h * freq) / (Tb * kB)
    thermo_fact = jnp.power((jnp.exp(x) - 1.0), 2.0) / (jnp.power(x, 2.0) * jnp.exp(x))
    return 1.0 / thermo_fact


def Tb_from_spec_rad(freq, pow_spec):
    """Physical temperature from spectral radiance [W/(m^2 sr Hz)]."""
    return (h * freq / kB) / jnp.log((2 * h * (freq**3 / c**2) / pow_spec) + 1)


def Tb_from_Trj(freq, Trj):
    """Physical temperature from Rayleigh-Jeans temperature."""
    alpha = (h * freq) / kB
    return alpha / jnp.log((2 * alpha / Trj) + 1)


def inv_var(err):
    """
    Inverse variance weights based on input errors

    Args:
    err (float): errors to generate weights
    """
    return 1.0 / (jnp.sqrt(jnp.sum(1.0 / (jnp.power(jnp.asarray(err), 2.0)))))


def dielectric_loss(freq, thick, ind, ltan):
    """
    The dielectric loss of a substrate given the frequency [Hz],
    substrate thickness [m], index of refraction, and loss tangent

    Args:
    freq (float): frequencies [Hz]
    thick (float): substrate thickness [m]
    ind (float): index of refraction
    ltan (float): loss tangent
    """
    return 1.0 - jnp.exp((-2.0 * PI * ind * ltan * thick) / (lamb(freq)))


def rj_temp(powr, bw, eff=1.0):
    """
    RJ temperature [K_RJ] given power [W], bandwidth [Hz], and efficiency

    Args:
    powr (float): power [W]
    bw (float): bandwidth [Hz]
    eff (float): efficiency
    """
    return powr / (kB * bw * eff)


def n_occ(freq, temp):
    """
    Photon occupation number given a frequency [Hz] and
    blackbody temperature [K]

    freq (float): frequency [Hz]
    temp (float): blackbody temperature [K]
    """
    fact = (h * freq) / (kB * temp)
    fact = jnp.where(fact > 100, 100, fact)
    return 1.0 / (jnp.exp(fact) - 1.0)


def a_omega(freq):
    """
    Throughput [m^2] for a diffraction-limited detector
    given the frequency [Hz]

    Args:
    freq (float): frequencies [Hz]
    """
    return lamb(freq) ** 2


def bb_spec_rad(freq, temp, emis=1.0):
    """
    Blackbody spectral radiance [W/(m^2 sr Hz)] given a frequency [Hz],
    blackbody temperature [K], and blackbody emissivity

    Args:
    freq (float): frequencies [Hz]
    temp (float): blackbody temperature [K]
    emiss (float): blackbody emissivity. Defaults to 1.
    """
    return emis * (2 * h * (freq**3) / (c**2)) * n_occ(freq, temp)


def bb_pow_spec(freq, temp, emis=1.0):
    """
    Blackbody power spectrum [W/Hz] on a diffraction-limited polarimeter
    for a frequency [Hz], blackbody temperature [K],
    and blackbody emissivity

    Args:
    freq (float): frequencies [Hz]
    temp (float): blackbody temperature [K]
    emiss (float): blackbody emissivity. Defaults to 1.
    """
    return 0.5 * a_omega(freq) * bb_spec_rad(freq, temp, emis)


def ani_pow_spec(freq, temp, emiss=1.0):
    """
    Derivative of blackbody power spectrum with respect to blackbody
    temperature, dP/dT, on a diffraction-limited detector [W/K] given
    a frequency [Hz], blackbody temperature [K], and blackbody
    emissivity

    Args:
    freq (float): frequency [Hz]
    temp (float): blackbody temperature [K]
    emiss (float): blackbody emissivity, Defaults to 1.
    """
    return (
        emiss
        * kB
        * jnp.exp((h * freq) / (kB * temp))
        * (h * freq * n_occ(freq, temp) / (kB * temp)) ** 2
    )


def pow_frac(T1, T2, freqs):
    """Fractional power between two physical temperatures."""
    return bb_pow_spec(freqs, T1) / bb_pow_spec(freqs, T2)
