"""Sensitivity calculation.

Wraps the pure jax-traceable computation in :mod:`bolojax.compute` with
the OutputField descriptor pattern for summary/table output.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from typing import TYPE_CHECKING, ClassVar, TextIO

import jax.numpy as jnp
import numpy as np

from bolojax.compute import elements
from bolojax.compute.sensitivity import BoloParams, OpticsState, compute_sensitivity
from bolojax.models import optics
from bolojax.models.params import OutputField, StatsSummary
from bolojax.models.utils import is_not_none

if TYPE_CHECKING:
    from astropy.table import Table

    from bolojax.io.tables import TableDict
    from bolojax.models.channel import Channel


def _make_element(optic: optics.OpticalElement, chan_idx: int) -> elements.Element:
    """Convert a pydantic OpticalElement to a JAX compute Element.

    Uses pre-computed results from ``eval_instrument`` for sampled
    properties (temperature, reflection, scatter, spillover) and raw
    config values for physical properties (thickness, index,
    loss_tangent, conductivity).
    """
    r = optic.results[chan_idx]
    base = {
        "temperature": jnp.asarray(r.temp, dtype=jnp.float64),
        "reflection": float(np.mean(r.refl)),
        "scatter_frac": float(np.mean(r.scat)),
        "scatter_temp": float(np.mean(r.scat_temp)),
        "spillover": float(np.mean(r.spil)),
        "spillover_temp": float(np.mean(r.spil_temp)),
    }

    if (
        isinstance(optic, optics.Dielectric)
        and is_not_none(optic.thickness)
        and is_not_none(optic.loss_tangent)
    ):
        return elements.Dielectric(
            **base,
            thickness=float(optic.thickness.SI),
            index=float(optic.index.SI),
            loss_tangent=float(optic.loss_tangent.SI),
        )
    if isinstance(optic, optics.Mirror) and is_not_none(optic.conductivity):
        return elements.Mirror(
            **base,
            conductivity=float(optic.conductivity.SI),
            surface_rough=float(optic.surface_rough.SI)
            if is_not_none(optic.surface_rough)
            else 0.0,
        )
    if isinstance(optic, optics.ApertureStop):
        return elements.ApertureStop(**base)

    # Generic element (e.g. forebaffle): use pre-computed absorption
    abso = float(np.mean(r.abso)) if np.ndim(r.abso) > 0 else float(r.abso)
    return elements.Element(**base, absorption=abso)


def build_params(channel: Channel) -> tuple[OpticsState, BoloParams, list[str]]:
    """Extract JAX pytrees from a configured Channel.

    Call after ``instrument.eval_sky()`` and ``instrument.eval_instrument()``
    have populated the channel's sky and optical chain data.

    Args:
        channel: a configured Channel instance.

    Returns:
        ``(OpticsState, BoloParams, elem_names)`` ready for
        :func:`compute_sensitivity`.
    """
    camera = channel.camera
    instrument = camera.instrument

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
    n_sky = len(channel.sky_names)

    # Optical elements: typed elements with physical properties
    for name, optic in camera.optics.items():
        chain[name] = _make_element(optic, channel.idx)

    # Detector
    chain["detector"] = elements.SkySource(
        temperature=jnp.asarray(channel._det_temp, dtype=jnp.float64),
        emiss_spectrum=jnp.asarray(channel._det_emiss, dtype=jnp.float64),
        trans_spectrum=jnp.asarray(channel._det_effic, dtype=jnp.float64),
    )

    # Element names for correlation factor computation
    elem_names = list(chain.keys())

    # Pre-compute Bose white-noise correlation factors
    ap_names = list(instrument.optics.apertureStops.keys())
    det_pitch = channel.pixel_size.SI / (
        camera.f_number.SI * (299792458.0 / channel.band_center.SI)
    )
    corr_factors = jnp.asarray(
        channel.noise_calc.corr_facts(elem_names, float(det_pitch), ap_names)
    )

    optics = OpticsState(
        freqs=freqs,
        bandwidth=bandwidth,
        elements=chain,
        corr_factors=corr_factors,
        n_sky=n_sky,
    )

    params = BoloParams(
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
        NET_scale=jnp.asarray(instrument.NET(), dtype=jnp.float64),
        ndet=int(channel.ndet),
        det_yield=jnp.asarray(channel.Yield(), dtype=jnp.float64),
        fsky=jnp.asarray(instrument.sky_fraction(), dtype=jnp.float64),
        obs_time=jnp.asarray(instrument.obs_time(), dtype=jnp.float64),
        obs_effic=jnp.asarray(instrument.obs_effic(), dtype=jnp.float64),
    )

    return optics, params, elem_names


class Sensitivity:  # pylint: disable=too-many-instance-attributes
    """Sensitivity calculation.

    Extracts physical parameters from a Channel, calls the pure
    :func:`compute_sensitivity` function, and stores results in
    OutputField holders for summary/table output.
    """

    _output_fields: ClassVar[dict[str, OutputField]] = {}

    effic = OutputField()
    opt_power = OutputField(unit="pW")
    P_sat = OutputField(unit="pW")
    G = OutputField(unit="pW/K")
    Flink = OutputField()

    tel_power = OutputField(unit="pW")
    sky_power = OutputField(unit="pW")

    tel_rj_temp = OutputField(unit="K")
    sky_rj_temp = OutputField(unit="K")

    elem_effic = OutputField()
    elem_cumul_effic = OutputField()
    elem_power_from_sky = OutputField(unit="pW")
    elem_power_to_det = OutputField(unit="pW")

    NEP_bolo = OutputField(unit="aW/rtHz")
    NEP_read = OutputField(unit="aW/rtHz")
    NEP_ph = OutputField(unit="aW/rtHz")
    NEP_ph_corr = OutputField(unit="aW/rtHz")
    NEP = OutputField(unit="aW/rtHz")
    NEP_corr = OutputField(unit="aW/rtHz")

    NET = OutputField(unit="uK-rts")
    NET_corr = OutputField(unit="uK-rts")

    NET_RJ = OutputField(unit="uK-rts")
    NET_corr_RJ = OutputField(unit="uK-rts")

    NET_arr = OutputField(unit="uK-rts")
    NET_arr_RJ = OutputField(unit="uK-rts")

    corr_fact = OutputField()

    map_depth = OutputField(unit="uK-amin")
    map_depth_RJ = OutputField(unit="uK-amin")

    summary_fields: ClassVar[list[str]] = [
        "effic",
        "opt_power",
        "P_sat",
        "Flink",
        "G",
        "tel_rj_temp",
        "sky_rj_temp",
        "NEP_bolo",
        "NEP_read",
        "NEP_ph",
        "NEP",
        "NET",
        "NET_corr",
        "corr_fact",
        "NET_arr",
    ]

    optical_output_fields: ClassVar[list[str]] = [
        "elem_effic",
        "elem_cumul_effic",
        "elem_power_from_sky",
        "elem_power_to_det",
    ]

    def __init__(self, channel: Channel) -> None:
        # Initialize all output holders
        for field in type(self)._output_fields.values():
            setattr(self, field.private_name, field.make_holder())

        self._channel = channel
        self._camera = self._channel.camera
        self._instrument = self._camera.instrument
        self._channel_name = f"{self._camera.name}_{self._channel.idx}"
        self._summary = None
        self._optical_output = None

        # Extract JAX pytrees from the configured channel
        optics, params, self._elem_names = build_params(channel)

        # Call the pure jax computation
        results = compute_sensitivity(optics, params)

        # Unpack results into OutputField holders
        for key, field in type(self)._output_fields.items():
            val = getattr(results, key, None)
            if val is not None:
                getattr(self, field.private_name).set_from_SI(np.asarray(val))

        self.summarize()
        self.analyze_optical_chain()

    def summarize(self) -> OrderedDict[str, StatsSummary]:
        """Compute and cache summary statistics."""
        self._summary: OrderedDict[str, StatsSummary] = OrderedDict()
        for key in self.summary_fields:
            self._summary[key] = type(self)._output_fields[key].summarize(self)
        return self._summary

    def analyze_optical_chain(self) -> OrderedDict[str, StatsSummary]:
        """Compute and cache optical output statistics."""
        self._optical_output: OrderedDict[str, StatsSummary] = OrderedDict()
        for key in self.optical_output_fields:
            self._optical_output[key] = (
                type(self)._output_fields[key].summarize_by_element(self)
            )
        return self._optical_output

    def print_summary(self, stream: TextIO = sys.stdout) -> None:
        """Print summary statistics in human-readable format."""
        stream.writelines(
            f"{key.ljust(20)} : {val}\n" for key, val in self._summary.items()
        )

    def print_optical_output(self, stream: TextIO = sys.stdout) -> None:
        """Print optical output statistics in human-readable format."""
        elem_power_from_sky = self._optical_output["elem_power_from_sky"]
        elem_power_to_det = self._optical_output["elem_power_to_det"]
        elem_effic = self._optical_output["elem_effic"]
        elem_cumul_effic = self._optical_output["elem_cumul_effic"]
        stream.write(
            f"{'Element'.ljust(20)} | {'Power from Sky [pW]'.ljust(26)} | {'Power to Det [pW]'.ljust(26)} | {'Efficiency'.ljust(26)} | {'Cumul. Effic.'.ljust(26)}\n"
        )
        stream.writelines(
            f"{elem.ljust(20)} | {elem_power_from_sky.element_string(idx)} | {elem_power_to_det.element_string(idx)} | {elem_effic.element_string(idx)} | {elem_cumul_effic.element_string(idx)}\n"
            for idx, elem in enumerate(self._elem_names)
        )

    def make_sims_table(self, name: str, table_dict: TableDict) -> Table:
        """Make a table with per-simulation parameters."""
        o_dict = OrderedDict(
            [
                (key, type(self)._output_fields[key].__get__(self).value.flatten())
                for key in self.summary_fields
            ]
        )
        try:
            return table_dict.make_datatable(name, o_dict)
        except ValueError as msg:
            s = "Column shape mismatch: "
            for k, v in o_dict.items():
                s += f"{k} {v.size}, "
            raise ValueError(s) from msg

    def make_optical_table(self, name: str, table_dict: TableDict) -> Table:
        """Make a table with optical output parameters."""
        o_dict: OrderedDict[str, np.ndarray] = OrderedDict()
        for val in self._optical_output.values():
            o_dict.update(val.todict())
        o_dict["element"] = np.array(self._elem_names)
        o_dict["channel"] = np.array([self._channel_name] * len(self._elem_names))
        return table_dict.make_datatable(name, o_dict)

    def make_sum_table(self, name: str, table_dict: TableDict) -> Table:
        """Make a table with summary parameters."""
        o_dict: OrderedDict[str, np.ndarray] = OrderedDict()
        for val in self._summary.values():
            o_dict.update(val.todict())
        o_dict["channel"] = np.array([self._channel_name])
        return table_dict.make_datatable(name, o_dict)

    def make_tables(
        self, base_name: str, table_dict: TableDict, **kwargs: bool
    ) -> TableDict:
        """Make output tables."""
        if kwargs.get("save_sim", True):
            self.make_sims_table(f"{base_name}_sims", table_dict)
        if kwargs.get("save_summary", True):
            self.make_sum_table(f"{base_name}_summary", table_dict)
        if kwargs.get("save_optical", True):
            self.make_optical_table(f"{base_name}_optical", table_dict)
        return table_dict
