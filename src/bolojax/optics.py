"""Model of optical elements."""

from __future__ import annotations

from collections import OrderedDict as odict

import numpy as np
from pydantic import BaseModel, ConfigDict, create_model

from . import physics
from .cfg import Var, expand_dict
from .utils import is_not_none


class ChannelResults:
    """Performance parameters for one optical element for one channel."""

    def __init__(self):
        self.temp = None
        self.refl = None
        self.spil = None
        self.scat = None
        self.spil_temp = None
        self.scat_temp = None
        self.abso = None
        self.emiss = None
        self.effic = None

    @staticmethod
    def emission(freqs, abso, spil, spil_temp, scat, scat_temp, temp):  # pylint: disable=too-many-arguments
        """Compute the emission for this element."""
        return (
            abso
            + spil * physics.pow_frac(spil_temp, temp, freqs)
            + scat * physics.pow_frac(scat_temp, temp, freqs)
        )

    @staticmethod
    def efficiency(refl, abso, spil, scat):
        """Compute the transmission for this element."""
        return (1 - refl) * (1 - abso) * (1 - spil) * (1 - scat)

    def calculate(self, freqs):
        """Compute the results for the frequencies of interest for a given channel."""
        emiss_shape = np.broadcast(
            freqs,
            self.abso,
            self.spil,
            self.spil_temp,
            self.scat,
            self.scat_temp,
            self.temp,
        ).shape
        self.emiss = self.emission(
            freqs,
            self.abso,
            self.spil,
            self.spil_temp,
            self.scat,
            self.scat_temp,
            self.temp,
        ).reshape(emiss_shape)
        effic_shape = np.broadcast(self.refl, self.abso, self.spil, self.scat).shape
        self.effic = self.efficiency(
            self.refl, self.abso, self.spil, self.scat
        ).reshape(effic_shape)

    def __call__(self):
        """Return key parameters."""
        return (self.effic, self.emiss, self.temp)


class OpticalElement(BaseModel):
    """Model for a single optical element."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    temperature: Var() = None
    spillover_temp: Var("K") = None
    scatter_temp: Var("K") = None
    surface_rough: Var() = None

    absorption: Var() = None
    reflection: Var() = None
    spillover: Var() = None
    scatter_frac: Var() = None

    elem_name: str | None = None
    results: dict = {}

    def model_post_init(self, __context):
        # Ensure each instance gets its own results dict
        object.__setattr__(self, "results", {})

    def unsample(self):
        """Clear out the sampled parameters."""
        self.temperature.unsample()
        self.reflection.unsample()
        self.spillover.unsample()
        self.scatter_frac.unsample()

    def sample(self, freqs, nsample, chan_idx):
        """Sample input parameters for a given channel."""
        self.temperature.sample(nsample)
        results_ = ChannelResults()
        results_.temp = self.temperature.SI
        results_.refl = self.reflection.sample(nsample, freqs, chan_idx)
        results_.spil = self.spillover.sample(nsample, freqs, chan_idx)
        if is_not_none(self.surface_rough) and np.isfinite(self.surface_rough.SI).all():
            results_.scat = 1.0 - physics.ruze_eff(freqs, self.surface_rough.SI)
        else:
            results_.scat = self.scatter_frac.sample(nsample, freqs, chan_idx)
        if (
            is_not_none(self.spillover_temp)
            and np.isfinite(self.spillover_temp.SI).all()
        ):
            results_.spil_temp = self.spillover_temp.SI
        else:
            results_.spil_temp = results_.temp
        if is_not_none(self.scatter_temp) and np.isfinite(self.scatter_temp.SI).all():
            results_.scat_temp = self.scatter_temp.SI
        else:
            results_.scat_temp = results_.temp
        self.results[chan_idx] = results_
        return results_

    def compute_channel(self, channel, freqs, nsample):
        """Compute the results for the frequencies of interest for a given channel."""
        self.unsample()
        results_ = self.sample(freqs, nsample, channel.idx)
        results_.abso = self.calc_abso(channel, freqs, nsample)
        results_.calculate(freqs)
        return results_()

    def calc_abso(self, channel, freqs, nsample):
        """Compute the absorption for a given channel."""
        return self.absorption.sample(nsample, freqs, channel.idx)


class Mirror(OpticalElement):
    """OpticalElement sub-class for mirrors."""

    conductivity: Var() = None

    def calc_abso(self, channel, freqs, nsample):
        if is_not_none(self.conductivity) and np.isfinite(self.conductivity.SI).all():
            return 1.0 - physics.ohmic_eff(freqs, self.conductivity.SI)
        return super().calc_abso(channel, freqs, nsample)


class Dielectric(OpticalElement):
    """OpticalElement sub-class for dielectrics."""

    thickness: Var() = None
    index: Var() = None
    loss_tangent: Var() = None

    def calc_abso(self, channel, freqs, nsample):
        if (
            is_not_none(self.thickness)
            and is_not_none(self.index)
            and is_not_none(self.loss_tangent)
        ):
            return physics.dielectric_loss(
                freqs, self.thickness.SI, self.index.SI, self.loss_tangent.SI
            )
        return super().calc_abso(channel, freqs, nsample)


class ApertureStop(OpticalElement):
    """OpticalElement sub-class for apertures."""

    def calc_abso(self, channel, freqs, nsample):
        pixel_size = channel.pixel_size()
        f_number = channel.camera.f_number()
        waist_factor = channel.waist_factor()

        if (
            is_not_none(pixel_size)
            and is_not_none(f_number)
            and is_not_none(waist_factor)
        ):
            return 1.0 - physics.spill_eff(
                np.array(freqs), pixel_size, f_number, waist_factor
            )
        return super().calc_abso(channel, freqs, nsample)


class Optics_Base(BaseModel):
    """Base class for optical chains."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    def model_post_init(self, __context):
        elements = odict()
        mirrors = odict()
        dielectics = odict()
        apertureStops = odict()
        for key, val in self.__dict__.items():
            if isinstance(val, OpticalElement):
                val.elem_name = key
                elements[key] = val
            if isinstance(val, Mirror):
                mirrors[key] = val
            if isinstance(val, Dielectric):
                dielectics[key] = val
            if isinstance(val, ApertureStop):
                apertureStops[key] = val
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "mirrors", mirrors)
        object.__setattr__(self, "dielectics", dielectics)
        object.__setattr__(self, "apertureStops", apertureStops)


def build_optics_class(name="Optics", **kwargs):
    """Build a class that consists of a set of OpticalElements.

    Parameters
    ----------
    name : str
        The name of the new class
    kwargs : dict
        Hierarchical dictionary used to build elements

    Returns
    -------
    object
        Instance of the new class with all requested optical elements
    """
    type_dict = {
        None: OpticalElement,
        "Mirror": Mirror,
        "Dielectric": Dielectric,
        "ApertureStop": ApertureStop,
    }
    return _build_model(name, Optics_Base, [kwargs], [type_dict])


def _build_model(name, base_class, config_dicts, type_dicts, **kwargs):
    """Build a pydantic model class dynamically and return an instance."""
    kwcopy = kwargs.copy()
    field_definitions = {}
    for config_dict, type_dict in zip(config_dicts, type_dicts):
        expanded = expand_dict(config_dict)
        kwcopy.update(expanded)
        for field_name, field_config in expanded.items():
            cls = type_dict[field_config.pop("obj_type", None)]
            field_definitions[field_name] = (cls | None, None)
    new_class = create_model(name, __base__=base_class, **field_definitions)
    return new_class(**kwcopy)
