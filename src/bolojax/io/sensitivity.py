"""Sensitivity calculation.

Wraps the pure jax-traceable computation in :mod:`bolojax.compute` with
the OutputField descriptor pattern for summary/table output.
"""

from __future__ import annotations

import sys
from collections import OrderedDict as odict
from typing import ClassVar

import jax.numpy as jnp
import numpy as np

from bolojax.compute.sensitivity import BoloParams, OpticsState, compute_sensitivity
from bolojax.models.params import OutputField


def _bcast_list(array_list):
    """Broadcast a list of arrays and stack along axis 0."""
    arrays = [jnp.asarray(a, dtype=jnp.float64) for a in array_list]
    broadcasted = jnp.broadcast_arrays(*arrays)
    return jnp.stack(broadcasted, axis=0)


def build_params(channel):
    """Extract JAX pytrees from a configured Channel.

    Call after ``instrument.eval_sky()`` and ``instrument.eval_instrument()``
    have populated the channel's sky and optical chain data.

    Returns:
        (OpticsState, BoloParams) ready for :func:`compute_sensitivity`.
    """
    camera = channel.camera
    instrument = camera.instrument

    freqs = jnp.asarray(channel.freqs, dtype=jnp.float64)
    bandwidth = float(channel.bandwidth)

    temps_list = channel.sky_temps + channel.optical_temps + [channel._det_temp]
    temps = _bcast_list(temps_list)

    # Buffer both sides of the transmission array, because we will be taking
    # cumulative products that are offset by one (i.e., we want the product of
    # all the elements downstream of a particular element)
    trans_list = [
        0.0,
        *channel.sky_effic,
        *channel.optical_effic,
        channel._det_effic,
        1.0,
    ]
    trans = _bcast_list(trans_list)

    emiss_list = [*channel.sky_emiss, *channel.optical_emiss, channel._det_emiss]
    emiss = _bcast_list(emiss_list)

    # Normalize emiss shape to 4D for physics.bb_pow_spec broadcasting
    if emiss.ndim == 2:
        emiss = emiss.reshape((emiss.shape[0], 1, 1, emiss.shape[1]))
    elif emiss.ndim == 3:
        emiss = emiss.reshape((emiss.shape[0], 1, emiss.shape[1], emiss.shape[2]))

    # Element names (for correlation factor computation)
    elem_names = channel.sky_names + list(camera.optics.keys()) + ["detector"]

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
        temps=temps,
        trans=trans,
        emiss=emiss,
        corr_factors=corr_factors,
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

    def __init__(self, channel):
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

    def summarize(self):
        """Compute and cache summary statistics."""
        self._summary = odict()
        for key in self.summary_fields:
            self._summary[key] = type(self)._output_fields[key].summarize(self)
        return self._summary

    def analyze_optical_chain(self):
        """Compute and cache optical output statistics."""
        self._optical_output = odict()
        for key in self.optical_output_fields:
            self._optical_output[key] = (
                type(self)._output_fields[key].summarize_by_element(self)
            )
        return self._optical_output

    def print_summary(self, stream=sys.stdout):
        """Print summary statistics in human-readable format."""
        for key, val in self._summary.items():
            stream.write(f"{key.ljust(20)} : {val}\n")

    def print_optical_output(self, stream=sys.stdout):
        """Print optical output statistics in human-readable format."""
        elem_power_from_sky = self._optical_output["elem_power_from_sky"]
        elem_power_to_det = self._optical_output["elem_power_to_det"]
        elem_effic = self._optical_output["elem_effic"]
        elem_cumul_effic = self._optical_output["elem_cumul_effic"]
        stream.write(
            f"{'Element'.ljust(20)} | {'Power from Sky [pW]'.ljust(26)} | {'Power to Det [pW]'.ljust(26)} | {'Efficiency'.ljust(26)} | {'Cumul. Effic.'.ljust(26)}\n"
        )
        for idx, elem in enumerate(self._elem_names):
            stream.write(
                f"{elem.ljust(20)} | {elem_power_from_sky.element_string(idx)} | {elem_power_to_det.element_string(idx)} | {elem_effic.element_string(idx)} | {elem_cumul_effic.element_string(idx)}\n"
            )

    def make_sims_table(self, name, table_dict):
        """Make a table with per-simulation parameters."""
        o_dict = odict(
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

    def make_optical_table(self, name, table_dict):
        """Make a table with optical output parameters."""
        o_dict = odict()
        for val in self._optical_output.values():
            o_dict.update(val.todict())
        o_dict["element"] = np.array(self._elem_names)
        o_dict["channel"] = np.array([self._channel_name] * len(self._elem_names))
        return table_dict.make_datatable(name, o_dict)

    def make_sum_table(self, name, table_dict):
        """Make a table with summary parameters."""
        o_dict = odict()
        for val in self._summary.values():
            o_dict.update(val.todict())
        o_dict["channel"] = np.array([self._channel_name])
        return table_dict.make_datatable(name, o_dict)

    def make_tables(self, base_name, table_dict, **kwargs):
        """Make output tables."""
        if kwargs.get("save_sim", True):
            self.make_sims_table(f"{base_name}_sims", table_dict)
        if kwargs.get("save_summary", True):
            self.make_sum_table(f"{base_name}_summary", table_dict)
        if kwargs.get("save_optical", True):
            self.make_optical_table(f"{base_name}_optical", table_dict)
        return table_dict
