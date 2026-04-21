"""Pure, jax-traceable sensitivity computation.

This module contains the core math for bolometer sensitivity calculations,
expressed entirely in jax.numpy operations. All functions are differentiable
via jax.grad / eqx.filter_grad.

The main entry point is :func:`compute_sensitivity`, which takes an
:class:`OpticsState` and :class:`BoloParams` and returns a
:class:`SensitivityResult`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from . import noise, physics

NSKY_SRC = 4


class OpticsState(eqx.Module):
    """Pre-computed optical chain state (from channel/instrument setup).

    These arrays are typically fixed for a given instrument configuration
    and not differentiated through. They describe the optical chain geometry.
    """

    freqs: jax.Array  # (n_freq,) frequency evaluation points [Hz]
    bandwidth: float  # integrated bandwidth [Hz]
    temps: jax.Array  # (n_elem, ...) broadcast temperatures [K]
    trans: jax.Array  # (n_elem+2, ...) padded transmissions
    emiss: jax.Array  # (n_elem, 1, 1, n_freq) emissivities
    corr_factors: jax.Array  # (n_elem,) Bose correlation factors


class BoloParams(eqx.Module):
    """Bolometer and observation parameters.

    Float fields are differentiable via eqx.filter_grad.
    Integer fields (ndet) are automatically treated as static.
    """

    # Bolometer thermal
    Tc: jax.Array
    bath_temp: jax.Array
    carrier_index: jax.Array
    psat: jax.Array
    psat_factor: jax.Array
    G: jax.Array
    Flink: jax.Array

    # Readout
    squid_nei: jax.Array
    bolo_R: jax.Array
    response_factor: jax.Array
    read_frac: jax.Array
    optical_coupling: jax.Array

    # Observation / array
    NET_scale: jax.Array
    ndet: int
    det_yield: jax.Array
    fsky: jax.Array
    obs_time: jax.Array
    obs_effic: jax.Array


class SensitivityResult(eqx.Module):
    """All computed sensitivity quantities as a JAX pytree."""

    effic: jax.Array
    opt_power: jax.Array
    P_sat: jax.Array
    G: jax.Array
    Flink: jax.Array
    tel_power: jax.Array
    sky_power: jax.Array
    tel_rj_temp: jax.Array
    sky_rj_temp: jax.Array
    elem_effic: jax.Array
    elem_cumul_effic: jax.Array
    elem_power_from_sky: jax.Array
    elem_power_to_det: jax.Array
    NEP_bolo: jax.Array
    NEP_read: jax.Array
    NEP_ph: jax.Array
    NEP_ph_corr: jax.Array
    NEP: jax.Array
    NEP_corr: jax.Array
    NET: jax.Array
    NET_corr: jax.Array
    NET_RJ: jax.Array
    NET_corr_RJ: jax.Array
    NET_arr: jax.Array
    NET_arr_RJ: jax.Array
    corr_fact: jax.Array
    map_depth: jax.Array
    map_depth_RJ: jax.Array


def resolve_psat(psat, psat_factor, opt_pow):
    """Resolve Psat from explicit value or psat_factor.

    Uses explicit psat where finite, otherwise falls back to
    opt_pow * psat_factor (matching BoloCalc's convention).

    NaN-safe: replaces NaN with a dummy value before computing to avoid
    gradient leakage through jnp.where's unused branch.
    """
    is_explicit = jnp.isfinite(psat)
    safe_psat = jnp.where(is_explicit, psat, 1.0)
    fallback = opt_pow * psat_factor
    return jnp.where(is_explicit, safe_psat, fallback)


def compute_G(G_explicit, psat, carrier_index, bath_temp, Tc):
    """Compute thermal conductance G [W/K].

    Uses explicit G where finite, otherwise computes from Psat.
    """
    is_explicit = jnp.isfinite(G_explicit)
    safe_G = jnp.where(is_explicit, G_explicit, 1.0)
    G_computed = noise.G(psat, carrier_index, bath_temp, Tc)
    return jnp.where(is_explicit, safe_G, G_computed)


def compute_Flink(Flink_explicit, carrier_index, bath_temp, Tc):
    """Compute link factor.

    Uses explicit Flink where finite, otherwise computes from
    carrier index and temperatures.
    """
    is_explicit = jnp.isfinite(Flink_explicit)
    safe_Flink = jnp.where(is_explicit, Flink_explicit, 1.0)
    Flink_computed = noise.Flink(carrier_index, bath_temp, Tc)
    return jnp.where(is_explicit, safe_Flink, Flink_computed)


def compute_read_nep(
    squid_nei, bolo_R, response_factor, psat, opt_power, read_frac, NEP_bolo, NEP_ph
):
    """Compute readout NEP with jnp.where fallback.

    Where squid_nei and bolo_R are finite, computes the full readout NEP
    for a voltage-biased bolometer. Otherwise falls back to the read_frac
    approximation (NEP_read ~ read_frac * sqrt(NEP_bolo^2 + NEP_ph^2)).
    """
    # Full readout NEP: nei / responsivity
    # Use safe values to avoid NaN gradient leakage through unused branch
    inputs_valid = jnp.isfinite(squid_nei) & jnp.isfinite(bolo_R)
    safe_nei = jnp.where(inputs_valid, squid_nei, 1.0)
    safe_R = jnp.where(inputs_valid, bolo_R, 1.0)

    p_bias = jnp.clip(psat - opt_power, 1e-30, jnp.inf)
    sfact = jnp.where(jnp.isfinite(response_factor), response_factor, 1.0)
    responsivity = sfact / jnp.sqrt(safe_R * p_bias)
    full_read_nep = safe_nei / responsivity

    # Fallback when squid_nei or bolo_R not provided
    fallback_nep = jnp.sqrt((1 + read_frac) ** 2 - 1.0) * jnp.sqrt(
        NEP_bolo * NEP_bolo + NEP_ph * NEP_ph
    )

    return jnp.where(inputs_valid, full_read_nep, fallback_nep)


def photon_nep(elem_power_to_det_by_freq, freqs, corr_factors):
    """Compute photon NEP and correlated photon NEP.

    The photon NEP includes both shot noise (h*nu*P) and wave noise (P^2)
    contributions, integrated across the band.

    The correlated NEP accounts for Bose white-noise correlations between
    neighbouring detectors (see arXiv:1806.04316).

    Args:
        elem_power_to_det_by_freq: (n_elem, ..., n_freq) power spectrum per element
        freqs: (n_freq,) frequency array
        corr_factors: (n_elem,) correlation factors from Noise.corr_facts

    Returns:
        (NEP_ph, NEP_ph_corr) tuple
    """
    popt = jnp.sum(elem_power_to_det_by_freq, axis=0)
    n_elem = elem_power_to_det_by_freq.shape[0]

    # popt2 = sum_ij P_i * P_j (uncorrelated total power^2)
    popt2 = jnp.sum(
        elem_power_to_det_by_freq[:, None] * elem_power_to_det_by_freq[None, :],
        axis=(0, 1),
    )

    # popt2_corr = sum_ij (f_i * f_j * P_i * P_j) (correlated power^2)
    factors_2d = corr_factors[:, None] * corr_factors[None, :]
    # Reshape for broadcasting: (n_elem, n_elem, 1, 1, ...) to match power dims
    shape = (n_elem, n_elem) + (1,) * (elem_power_to_det_by_freq.ndim - 1)
    factors_2d = factors_2d.reshape(shape)
    popt2_corr = jnp.sum(
        factors_2d
        * elem_power_to_det_by_freq[:, None]
        * elem_power_to_det_by_freq[None, :],
        axis=(0, 1),
    )

    nep = jnp.sqrt(jnp.trapezoid(2.0 * physics.h * freqs * popt + 2.0 * popt2, freqs))
    nep_corr = jnp.sqrt(
        jnp.trapezoid(2.0 * physics.h * freqs * popt + 2.0 * popt2_corr, freqs)
    )
    return nep, nep_corr


def trj_over_tcmb(freqs):
    """Compute the RJ-to-CMB temperature conversion factor.

    Integrates dTrj/dTcmb across the band and normalizes by bandwidth.
    """
    factor_spec = physics.Trj_over_Tb(freqs, physics.Tcmb)
    bw = freqs[-1] - freqs[0]
    return jnp.trapezoid(factor_spec, freqs) / bw


@eqx.filter_jit
def compute_sensitivity(optics: OpticsState, params: BoloParams) -> SensitivityResult:
    """Pure, jax-traceable sensitivity computation.

    All inputs and outputs are jax pytrees. No side effects.
    Differentiable via eqx.filter_grad with respect to params.

    Args:
        optics: Pre-computed optical chain state
        params: Bolometer and observation parameters

    Returns:
        SensitivityResult pytree with all computed quantities
    """
    freqs = optics.freqs
    bandwidth = optics.bandwidth
    temps = optics.temps
    trans = optics.trans
    emiss = optics.emiss

    # Total transmission efficiency of the channel as a function of frequency.
    # Pull out the padding from trans (first and last elements).
    chan_effic = jnp.prod(trans[1:-1], axis=0)

    # Efficiency of a particular element getting to the detector, as a function
    # of frequency. We take the cumulative product starting from the detector
    # side (hence the [::-1] reversal), then strip the padding.
    elem_cumul_effic_by_freq = jnp.cumprod(trans[::-1], axis=0)[::-1][2:]

    # Power emitted by a particular element (blackbody)
    elem_power_by_freq = physics.bb_pow_spec(freqs, temps, emiss)

    # Power from a particular element reaching the detector
    elem_power_to_det_by_freq = elem_power_by_freq * elem_cumul_effic_by_freq

    # Power accumulated from the sky side down to each element.
    # This is a sequential accumulation: at each element, add that element's
    # emitted power then multiply by the element's transmission.
    def _scan_fn(carry, inputs):
        elem_pow, elem_tr = inputs
        carry = (carry + elem_pow) * elem_tr
        return carry, carry

    _, elem_sky_power_by_freq = jax.lax.scan(
        _scan_fn,
        jnp.zeros_like(elem_power_by_freq[0]),
        (elem_power_by_freq, trans[:-2]),
    )

    # Integrate all frequency-dependent quantities across the band
    # using the trapezoid rule
    elem_power_to_det = jnp.trapezoid(elem_power_to_det_by_freq, freqs)
    elem_power_from_sky = jnp.trapezoid(elem_sky_power_by_freq, freqs)
    elem_effic = jnp.trapezoid(trans[1:-1], freqs) / bandwidth
    elem_cumul_effic = jnp.trapezoid(elem_cumul_effic_by_freq, freqs) / bandwidth
    effic = jnp.trapezoid(chan_effic, freqs) / bandwidth
    tel_effic = jnp.trapezoid(elem_cumul_effic_by_freq[NSKY_SRC], freqs) / bandwidth

    # From this point, everything is integrated across bands and given per channel
    opt_power = jnp.sum(elem_power_to_det, axis=0)
    tel_power = jnp.sum(elem_power_to_det[NSKY_SRC:], axis=0)
    sky_power = jnp.sum(elem_power_from_sky[:NSKY_SRC], axis=0)

    # RJ temperature equivalents
    tel_rj_temp = physics.rj_temp(tel_power, bandwidth, tel_effic)
    sky_rj_temp = physics.rj_temp(sky_power, bandwidth, tel_effic)

    # Bolometer thermal NEP
    psat = resolve_psat(params.psat, params.psat_factor, opt_power)
    G_val = compute_G(params.G, psat, params.carrier_index, params.bath_temp, params.Tc)
    flink = compute_Flink(
        params.Flink, params.carrier_index, params.bath_temp, params.Tc
    )
    NEP_bolo = noise.bolo_NEP(flink, G_val, params.Tc)

    # Photon NEP (shot + wave noise, with Bose correlations)
    NEP_ph, NEP_ph_corr = photon_nep(
        elem_power_to_det_by_freq, freqs, optics.corr_factors
    )

    # Readout NEP (full calculation or read_frac fallback)
    NEP_read = compute_read_nep(
        params.squid_nei,
        params.bolo_R,
        params.response_factor,
        psat,
        opt_power,
        params.read_frac,
        NEP_bolo,
        NEP_ph,
    )

    # Total NEP: quadrature sum of all noise sources
    NEP = jnp.sqrt(NEP_bolo * NEP_bolo + NEP_ph * NEP_ph + NEP_read * NEP_read)
    NEP_corr = jnp.sqrt(
        NEP_bolo * NEP_bolo + NEP_ph_corr * NEP_ph_corr + NEP_read * NEP_read
    )

    # Convert NEP to NET (noise equivalent temperature)
    NET = noise.NET_from_NEP(NEP, freqs, chan_effic, params.optical_coupling)
    NET_corr = noise.NET_from_NEP(NEP_corr, freqs, chan_effic, params.optical_coupling)

    # Convert from CMB temperature to RJ temperature
    Trj_factor = trj_over_tcmb(freqs)
    NET_RJ = Trj_factor * NET
    NET_corr_RJ = Trj_factor * NET_corr

    # Array NET: per-detector NET scaled by sqrt(n_det * yield)
    ndet_f = jnp.asarray(params.ndet, dtype=jnp.float64)
    NET_arr = params.NET_scale * noise.NET_arr(NET, ndet_f, params.det_yield)
    NET_arr_RJ = params.NET_scale * noise.NET_arr(NET_RJ, ndet_f, params.det_yield)

    # Correlation factor and map depth
    corr_fact = NET_corr / NET
    map_depth = noise.map_depth(NET_arr, params.fsky, params.obs_time, params.obs_effic)
    map_depth_RJ = noise.map_depth(
        NET_arr_RJ, params.fsky, params.obs_time, params.obs_effic
    )

    # Broadcast P_sat, G, Flink to match NET shape for output consistency
    to_shape = jnp.ones_like(NET_corr)
    P_sat = psat * to_shape
    G_out = G_val * to_shape
    Flink_out = flink * to_shape

    return SensitivityResult(
        effic=effic,
        opt_power=opt_power,
        P_sat=P_sat,
        G=G_out,
        Flink=Flink_out,
        tel_power=tel_power,
        sky_power=sky_power,
        tel_rj_temp=tel_rj_temp,
        sky_rj_temp=sky_rj_temp,
        elem_effic=elem_effic,
        elem_cumul_effic=elem_cumul_effic,
        elem_power_from_sky=elem_power_from_sky,
        elem_power_to_det=elem_power_to_det,
        NEP_bolo=NEP_bolo,
        NEP_read=NEP_read,
        NEP_ph=NEP_ph,
        NEP_ph_corr=NEP_ph_corr,
        NEP=NEP,
        NEP_corr=NEP_corr,
        NET=NET,
        NET_corr=NET_corr,
        NET_RJ=NET_RJ,
        NET_corr_RJ=NET_corr_RJ,
        NET_arr=NET_arr,
        NET_arr_RJ=NET_arr_RJ,
        corr_fact=corr_fact,
        map_depth=map_depth,
        map_depth_RJ=map_depth_RJ,
    )
