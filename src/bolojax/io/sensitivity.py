"""Config-to-compute bridge.

``build_experiment`` converts a configured ``ChannelConfig`` into an
``Experiment`` zodiax pytree ready for ``.compute()`` and ``.to_dataset()``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from bolojax.compute import elements, physics
from bolojax.compute.experiment import Experiment, Instrument
from bolojax.models import optics
from bolojax.models.utils import is_not_none

if TYPE_CHECKING:
    from bolojax.models.channel import ChannelConfig


def _make_element(optic: optics.OpticalElement, chan_idx: int) -> elements.Element:
    """Convert a pydantic OpticalElement to a JAX compute Element.

    Uses pre-computed results from ``eval_instrument`` for sampled
    properties (temperature, reflection, scatter, spillover) and raw
    config values for physical properties (thickness, index,
    loss_tangent, conductivity).
    """
    r = optic.results[chan_idx]

    # TODO: fix this limitation! Somewhat bad form to silently convert elements to SkySource
    has_freq_dep = np.ndim(r.emiss) > 0 or np.ndim(r.effic) > 0
    if has_freq_dep:
        return elements.SkySource(
            temperature=jnp.asarray(r.temp, dtype=jnp.float64),
            emiss_spectrum=jnp.asarray(r.emiss, dtype=jnp.float64),
            trans_spectrum=jnp.asarray(r.effic, dtype=jnp.float64),
        )

    # Scalar properties: build a typed element for differentiability
    base = {
        "temperature": jnp.asarray(r.temp, dtype=jnp.float64),
        "reflection": jnp.float64(r.refl),
        "scatter_frac": jnp.float64(r.scat),
        "scatter_temp": jnp.float64(r.scat_temp),
        "spillover": jnp.float64(r.spil),
        "spillover_temp": jnp.float64(r.spil_temp),
    }

    if (
        isinstance(optic, optics.Dielectric)
        and is_not_none(optic.thickness)
        and np.isfinite(optic.thickness.SI).all()
        and is_not_none(optic.loss_tangent)
        and np.isfinite(optic.loss_tangent.SI).all()
    ):
        return elements.Dielectric(
            **base,
            thickness=jnp.float64(optic.thickness.SI),
            index=jnp.float64(optic.index.SI),
            loss_tangent=jnp.float64(optic.loss_tangent.SI),
        )
    if (
        isinstance(optic, optics.Mirror)
        and is_not_none(optic.conductivity)
        and np.isfinite(optic.conductivity.SI).all()
    ):
        return elements.Mirror(
            **base,
            conductivity=jnp.float64(optic.conductivity.SI),
            surface_rough=jnp.float64(optic.surface_rough.SI)
            if is_not_none(optic.surface_rough)
            and np.isfinite(optic.surface_rough.SI).all()
            else jnp.float64(0.0),
        )

    # Generic element: use pre-computed absorption
    return elements.Element(**base, absorption=jnp.float64(r.abso))


def build_experiment(channel: ChannelConfig) -> Experiment:
    """Extract JAX pytrees from a configured ChannelConfig.

    Call after ``eval_sky()`` and ``eval_instrument()`` have populated
    the channel's sky and optical chain data.

    Args:
        channel: a configured ChannelConfig instance.

    Returns:
        ``Experiment`` pytree ready for ``.compute()``.
    """
    camera = channel.camera
    inst_cfg = camera.instrument

    freqs = jnp.asarray(channel.freqs, dtype=jnp.float64)
    bandwidth = float(channel.bandwidth)

    chain = OrderedDict()

    # Sky sources: precomputed emissivity/transmission from the sky model
    for name, emiss, effic, temp in zip(
        channel.sky_names,
        channel.sky_emiss,
        channel.sky_effic,
        channel.sky_temps,
        strict=False,
    ):
        chain[name] = elements.SkySource(
            temperature=jnp.asarray(temp, dtype=jnp.float64),
            emiss_spectrum=jnp.asarray(emiss, dtype=jnp.float64),
            trans_spectrum=jnp.asarray(effic, dtype=jnp.float64),
        )
    # Optical elements: typed elements with physical properties
    for name, optic in camera.optics.items():
        chain[name] = _make_element(optic, channel.idx)

    # Detector
    chain["detector"] = elements.SkySource(
        temperature=jnp.asarray(channel._det_temp, dtype=jnp.float64),
        emiss_spectrum=jnp.asarray(channel._det_emiss, dtype=jnp.float64),
        trans_spectrum=jnp.asarray(channel._det_effic, dtype=jnp.float64),
    )

    # Pre-compute Bose white-noise correlation factors
    elem_names = list(chain.keys())
    ap_names = list(inst_cfg.optics.apertureStops.keys())
    det_pitch = channel.pixel_size.SI / (
        camera.f_number.SI * (physics.c / channel.band_center.SI)
    )
    corr_factors = jnp.asarray(
        channel.noise_calc.corr_facts(elem_names, float(det_pitch), ap_names)
    )

    instrument = Instrument(
        freqs=freqs,
        bandwidth=bandwidth,
        elements=chain,
        corr_factors=corr_factors,
        Tc=jnp.asarray(channel.Tc.SI, dtype=jnp.float64),
        bath_temp=jnp.asarray(camera.bath_temperature(), dtype=jnp.float64),
        carrier_index=jnp.asarray(channel.carrier_index.SI, dtype=jnp.float64),
        psat=jnp.asarray(channel.psat.SI, dtype=jnp.float64),
        psat_factor=jnp.asarray(channel.psat_factor.SI, dtype=jnp.float64),
        G=jnp.asarray(channel.G.SI, dtype=jnp.float64),
        Flink=jnp.asarray(channel.Flink.SI, dtype=jnp.float64),
        optical_coupling=jnp.asarray(camera.optical_coupling(), dtype=jnp.float64),
        read_frac=jnp.asarray(channel.read_frac(), dtype=jnp.float64),
        squid_nei=jnp.asarray(channel.squid_nei.SI, dtype=jnp.float64),
        bolo_R=jnp.asarray(channel.bolo_resistance.SI, dtype=jnp.float64),
        response_factor=jnp.asarray(channel.response_factor.SI, dtype=jnp.float64),
        ndet=int(channel.ndet),
        det_yield=jnp.asarray(channel.Yield(), dtype=jnp.float64),
    )

    return Experiment(
        instrument=instrument,
        fsky=jnp.asarray(inst_cfg.sky_fraction(), dtype=jnp.float64),
        obs_time=jnp.asarray(inst_cfg.obs_time(), dtype=jnp.float64),
        obs_effic=jnp.asarray(inst_cfg.obs_effic(), dtype=jnp.float64),
        NET_scale=jnp.asarray(inst_cfg.NET(), dtype=jnp.float64),
    )
