"""Pure, jax-traceable sensitivity computation.

Core math for bolometer sensitivity calculations.

:func:`compute_sensitivity` takes an :class:`~bolojax.compute.experiment.Experiment`
and returns a :class:`SensitivityResult`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, get_type_hints

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pint
import xarray as xr
import zodiax as zdx
from jax import Array

from bolojax.compute import noise, physics
from bolojax.compute.elements import SkySource
from bolojax.models.unit import ureg

if TYPE_CHECKING:
    from bolojax.compute.experiment import Experiment

# Annotated output types: pint.Unit metadata in the type hint
Power = Annotated[Array, ureg.Unit("pW")]
Conductance = Annotated[Array, ureg.Unit("pW/K")]
Temp = Annotated[Array, ureg.Unit("K")]
NoiseDensity = Annotated[Array, ureg.Unit("aW/rtHz")]
SensUnit = Annotated[Array, ureg.Unit("uK * s**0.5")]
MapDepth = Annotated[Array, ureg.Unit("uK * arcmin")]


class SensitivityResult(zdx.Base):
    """All computed sensitivity quantities as a JAX pytree.

    Fields annotated with a ``pint.Unit`` carry display-unit metadata.
    Call ``.to_dataset(element_names)`` for an xarray Dataset.
    """

    effic: Array
    opt_power: Power
    P_sat: Power
    G: Conductance
    Flink: Array
    tel_power: Power
    sky_power: Power
    tel_rj_temp: Temp
    sky_rj_temp: Temp
    elem_effic: Array
    elem_cumul_effic: Array
    elem_power_from_sky: Power
    elem_power_to_det: Power
    NEP_bolo: NoiseDensity
    NEP_read: NoiseDensity
    NEP_ph: NoiseDensity
    NEP_ph_corr: NoiseDensity
    NEP: NoiseDensity
    NEP_corr: NoiseDensity
    NET: SensUnit
    NET_corr: SensUnit
    NET_RJ: SensUnit
    NET_corr_RJ: SensUnit
    NET_arr: SensUnit
    NET_arr_RJ: SensUnit
    corr_fact: Array
    map_depth: MapDepth
    map_depth_RJ: MapDepth

    def to_dataset(self, element_names: list[str]) -> xr.Dataset:
        """Convert to an xarray Dataset with labeled dimensions and units.

        Unit metadata is read from ``pint.Unit`` annotations on each
        field's type hint. Element-level fields are inferred from array
        shape (leading dimension == number of elements).
        """
        n_elem = len(element_names)
        hints = get_type_hints(type(self), include_extras=True)
        data_vars = {}

        for name, hint in hints.items():
            arr = np.asarray(getattr(self, name))

            unit_obj = next(
                (
                    a
                    for a in getattr(hint, "__metadata__", [])
                    if isinstance(a, pint.Unit)
                ),
                None,
            )

            attrs = {}
            if unit_obj is not None:
                base_unit = ureg.Quantity(1.0, unit_obj).to_base_units().units
                arr = ureg.Quantity(arr, base_unit).to(unit_obj).magnitude
                attrs["units"] = str(unit_obj)

            has_elem_dim = arr.ndim >= 1 and arr.shape[0] == n_elem
            if has_elem_dim:
                dims = ["element"] + [f"dim_{i}" for i in range(arr.ndim - 1)]
                coords = {"element": element_names}
            else:
                dims = [f"dim_{i}" for i in range(arr.ndim)]
                coords = {}

            data_vars[name] = xr.DataArray(arr, dims=dims, coords=coords, attrs=attrs)

        return xr.Dataset(data_vars)


def resolve_psat(psat: Array, psat_factor: Array, opt_pow: Array) -> Array:
    """Resolve Psat from explicit value or psat_factor.

    Uses explicit psat where finite, otherwise falls back to
    opt_pow * psat_factor (matching BoloCalc's convention).

    NaN-safe: replaces NaN with a dummy value before computing to avoid
    gradient leakage through jnp.where's unused branch.
    """
    is_explicit = jnp.isfinite(psat)
    safe_psat = jnp.where(is_explicit, psat, 1.0)
    safe_factor = jnp.where(is_explicit, 1.0, psat_factor)
    fallback = opt_pow * safe_factor
    return jnp.where(is_explicit, safe_psat, fallback)


def compute_G(
    G_explicit: Array, psat: Array, carrier_index: Array, bath_temp: Array, Tc: Array
) -> Array:
    """Compute thermal conductance G [W/K].

    Uses explicit G where finite, otherwise computes from Psat.
    """
    is_explicit = jnp.isfinite(G_explicit)
    safe_G = jnp.where(is_explicit, G_explicit, 1.0)
    G_computed = noise.G(psat, carrier_index, bath_temp, Tc)
    return jnp.where(is_explicit, safe_G, G_computed)


def compute_Flink(
    Flink_explicit: Array, carrier_index: Array, bath_temp: Array, Tc: Array
) -> Array:
    """Compute link factor.

    Uses explicit Flink where finite, otherwise computes from
    carrier index and temperatures.
    """
    is_explicit = jnp.isfinite(Flink_explicit)
    safe_Flink = jnp.where(is_explicit, Flink_explicit, 1.0)
    Flink_computed = noise.Flink(carrier_index, bath_temp, Tc)
    return jnp.where(is_explicit, safe_Flink, Flink_computed)


def compute_read_nep(
    squid_nei: Array,
    bolo_R: Array,
    response_factor: Array,
    psat: Array,
    opt_power: Array,
    read_frac: Array,
    NEP_bolo: Array,
    NEP_ph: Array,
) -> Array:
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
    safe_read_frac = jnp.where(inputs_valid, 0.0, read_frac)
    fallback_nep = jnp.sqrt((1 + safe_read_frac) ** 2 - 1.0) * jnp.sqrt(
        NEP_bolo * NEP_bolo + NEP_ph * NEP_ph
    )

    return jnp.where(inputs_valid, full_read_nep, fallback_nep)


def photon_nep(
    elem_power_to_det_by_freq: Array, freqs: Array, corr_factors: Array
) -> tuple[Array, Array]:
    r"""Compute photon NEP and correlated photon NEP.

    The photon NEP includes both shot noise ($h\nu P$) and wave noise ($P^2$)
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


def trj_over_tcmb(freqs: Array) -> Array:
    """Compute the RJ-to-CMB temperature conversion factor.

    Integrates dTrj/dTcmb across the band and normalizes by bandwidth.
    """
    factor_spec = physics.Trj_over_Tb(freqs, physics.Tcmb)
    bw = freqs[-1] - freqs[0]
    return jnp.trapezoid(factor_spec, freqs) / bw


@eqx.filter_jit
def compute_sensitivity(experiment: Experiment) -> SensitivityResult:
    """Pure, jax-traceable sensitivity computation.

    Loops over instrument elements calling each element's
    ``emiss_trans(freqs)`` to compute emissivity and transmission,
    then accumulates power contributions through the optical chain.

    Args:
        experiment: ``Experiment`` pytree (instrument + survey params)

    Returns:
        SensitivityResult pytree with all computed quantities
    """
    inst = experiment.instrument
    freqs = inst.freqs
    bandwidth = inst.bandwidth
    n_sky = sum(isinstance(e, SkySource) for e in inst.elements.values())

    # Compute per-element emissivity and transmission via polymorphic dispatch.
    # The for-loop unrolls at trace time (pytree structure is static).
    emiss_list = []
    trans_list = []
    power_list = []
    for elem in inst.elements.values():
        e, t = elem.emiss_trans(freqs)
        emiss_list.append(e)
        trans_list.append(t)
        # Power emitted by a particular element (blackbody)
        power_list.append(physics.bb_pow_spec(freqs, elem.temperature, e))

    # Broadcast to common shape and stack (sky sources may carry batch dims
    # that optical elements don't, e.g. (1, 1, n_freq) vs (n_freq,))
    trans_inner = jnp.stack(jnp.broadcast_arrays(*trans_list))
    elem_power_by_freq = jnp.stack(jnp.broadcast_arrays(*power_list))

    # Pad transmission: [0, t1, t2, ..., tn, 1] for cumulative product
    pad_lo = jnp.zeros_like(trans_inner[:1])
    pad_hi = jnp.ones_like(trans_inner[:1])
    trans = jnp.concatenate([pad_lo, trans_inner, pad_hi], axis=0)

    # Channel efficiency: product of all element transmissions
    chan_effic = jnp.prod(trans[1:-1], axis=0)

    # Efficiency of a particular element getting to the detector, as a function
    # of frequency. We take the cumulative product starting from the detector
    # side (hence the [::-1] reversal), then strip the padding.
    elem_cumul_effic_by_freq = jnp.cumprod(trans[::-1], axis=0)[::-1][2:]

    # Power from each element reaching the detector
    elem_power_to_det_by_freq = elem_power_by_freq * elem_cumul_effic_by_freq

    # Cumulative sky-side power: at each element, accumulate power from
    # upstream elements, then multiply by this element's transmission.
    carry = jnp.zeros_like(elem_power_by_freq[0])
    sky_power_list = []
    for i in range(len(emiss_list)):
        carry = (carry + elem_power_by_freq[i]) * trans_inner[i]
        sky_power_list.append(carry)
    elem_sky_power_by_freq = jnp.stack(sky_power_list)

    # Integrate frequency-dependent quantities across the band
    elem_power_to_det = jnp.trapezoid(elem_power_to_det_by_freq, freqs)
    elem_power_from_sky = jnp.trapezoid(elem_sky_power_by_freq, freqs)
    elem_effic = jnp.trapezoid(trans_inner, freqs) / bandwidth
    elem_cumul_effic = jnp.trapezoid(elem_cumul_effic_by_freq, freqs) / bandwidth
    effic = jnp.trapezoid(chan_effic, freqs) / bandwidth
    tel_effic = jnp.trapezoid(elem_cumul_effic_by_freq[n_sky], freqs) / bandwidth

    # Integrated per-channel quantities
    opt_power = jnp.sum(elem_power_to_det, axis=0)
    tel_power = jnp.sum(elem_power_to_det[n_sky:], axis=0)
    sky_power = jnp.sum(elem_power_from_sky[:n_sky], axis=0)

    # RJ temperature equivalents
    tel_rj_temp = physics.rj_temp(tel_power, bandwidth, tel_effic)
    sky_rj_temp = physics.rj_temp(sky_power, bandwidth, tel_effic)

    # Bolometer thermal NEP
    psat = resolve_psat(inst.psat, inst.psat_factor, opt_power)
    G_val = compute_G(inst.G, psat, inst.carrier_index, inst.bath_temp, inst.Tc)
    flink = compute_Flink(inst.Flink, inst.carrier_index, inst.bath_temp, inst.Tc)
    NEP_bolo = noise.bolo_NEP(flink, G_val, inst.Tc)

    # Photon NEP (shot + wave noise, with Bose correlations)
    NEP_ph, NEP_ph_corr = photon_nep(
        elem_power_to_det_by_freq, freqs, inst.corr_factors
    )

    # Readout NEP (full calculation or read_frac fallback)
    NEP_read = compute_read_nep(
        inst.squid_nei,
        inst.bolo_R,
        inst.response_factor,
        psat,
        opt_power,
        inst.read_frac,
        NEP_bolo,
        NEP_ph,
    )

    # Total NEP: quadrature sum of all noise sources
    NEP = jnp.sqrt(NEP_bolo * NEP_bolo + NEP_ph * NEP_ph + NEP_read * NEP_read)
    NEP_corr = jnp.sqrt(
        NEP_bolo * NEP_bolo + NEP_ph_corr * NEP_ph_corr + NEP_read * NEP_read
    )

    # Convert NEP to NET (noise equivalent temperature)
    NET = noise.NET_from_NEP(NEP, freqs, chan_effic, inst.optical_coupling)
    NET_corr = noise.NET_from_NEP(NEP_corr, freqs, chan_effic, inst.optical_coupling)

    # Convert from CMB temperature to RJ temperature
    Trj_factor = trj_over_tcmb(freqs)
    NET_RJ = Trj_factor * NET
    NET_corr_RJ = Trj_factor * NET_corr

    # Array NET: per-detector NET scaled by sqrt(n_det * yield)
    ndet_f = jnp.asarray(inst.ndet, dtype=jnp.float64)
    NET_arr = experiment.NET_scale * noise.NET_arr(NET, ndet_f, inst.det_yield)
    NET_arr_RJ = experiment.NET_scale * noise.NET_arr(NET_RJ, ndet_f, inst.det_yield)

    # Correlation factor and map depth
    corr_fact = NET_corr / NET
    map_depth = noise.map_depth(
        NET_arr, experiment.fsky, experiment.obs_time, experiment.obs_effic
    )
    map_depth_RJ = noise.map_depth(
        NET_arr_RJ, experiment.fsky, experiment.obs_time, experiment.obs_effic
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
