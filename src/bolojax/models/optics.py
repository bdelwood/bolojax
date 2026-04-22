"""Model of optical elements."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from bolojax.compute import physics
from bolojax.models.params import Var
from bolojax.models.utils import is_not_none

if TYPE_CHECKING:
    from bolojax.models.channel import Channel


class ChannelResults:
    """Performance parameters for one optical element for one channel."""

    def __init__(self) -> None:
        self.temp: float | np.ndarray | None = None
        self.refl: float | np.ndarray | None = None
        self.spil: float | np.ndarray | None = None
        self.scat: float | np.ndarray | None = None
        self.spil_temp: float | np.ndarray | None = None
        self.scat_temp: float | np.ndarray | None = None
        self.abso: float | np.ndarray | None = None
        self.emiss: float | np.ndarray | None = None
        self.effic: float | np.ndarray | None = None

    @staticmethod
    def emission(
        freqs: np.ndarray,
        abso: float | np.ndarray,
        spil: float | np.ndarray,
        spil_temp: float | np.ndarray,
        scat: float | np.ndarray,
        scat_temp: float | np.ndarray,
        temp: float | np.ndarray,
    ) -> np.ndarray:  # pylint: disable=too-many-arguments
        """Compute the emission for this element."""
        return (
            abso
            + spil * physics.pow_frac(spil_temp, temp, freqs)
            + scat * physics.pow_frac(scat_temp, temp, freqs)
        )

    @staticmethod
    def efficiency(
        refl: float | np.ndarray,
        abso: float | np.ndarray,
        spil: float | np.ndarray,
        scat: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute the transmission for this element."""
        return (1 - refl) * (1 - abso) * (1 - spil) * (1 - scat)

    def calculate(self, freqs: np.ndarray) -> None:
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

    def __call__(
        self,
    ) -> tuple[
        float | np.ndarray | None, float | np.ndarray | None, float | np.ndarray | None
    ]:
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
    results: dict = Field(default_factory=dict)

    def unsample(self) -> None:
        """Clear out the sampled parameters."""
        self.temperature.unsample()
        self.reflection.unsample()
        self.spillover.unsample()
        self.scatter_frac.unsample()

    def sample(self, freqs: np.ndarray, nsample: int, chan_idx: int) -> ChannelResults:
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

    def compute_channel(
        self, channel: Channel, freqs: np.ndarray, nsample: int
    ) -> tuple[
        float | np.ndarray | None, float | np.ndarray | None, float | np.ndarray | None
    ]:
        """Compute the results for the frequencies of interest for a given channel."""
        self.unsample()
        results_ = self.sample(freqs, nsample, channel.idx)
        results_.abso = self.calc_abso(channel, freqs, nsample)
        results_.calculate(freqs)
        return results_()

    def calc_abso(
        self, channel: Channel, freqs: np.ndarray, nsample: int
    ) -> float | np.ndarray:
        """Compute the absorption for a given channel."""
        return self.absorption.sample(nsample, freqs, channel.idx)


class Mirror(OpticalElement):
    """OpticalElement sub-class for mirrors."""

    conductivity: Var() = None

    def calc_abso(
        self, channel: Channel, freqs: np.ndarray, nsample: int
    ) -> float | np.ndarray:
        if is_not_none(self.conductivity) and np.isfinite(self.conductivity.SI).all():
            return 1.0 - physics.ohmic_eff(freqs, self.conductivity.SI)
        return super().calc_abso(channel, freqs, nsample)


class Dielectric(OpticalElement):
    """OpticalElement sub-class for dielectrics."""

    thickness: Var() = None
    index: Var() = None
    loss_tangent: Var() = None

    def calc_abso(
        self, channel: Channel, freqs: np.ndarray, nsample: int
    ) -> float | np.ndarray:
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

    def calc_abso(
        self, channel: Channel, freqs: np.ndarray, nsample: int
    ) -> float | np.ndarray:
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


_ELEMENT_TYPES: dict[str | None, type[OpticalElement]] = {
    None: OpticalElement,
    "Mirror": Mirror,
    "Dielectric": Dielectric,
    "ApertureStop": ApertureStop,
}


class Optics(BaseModel):
    """A collection of optical elements built from a config dict."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    elements: dict[str, OpticalElement] = Field(default_factory=dict)
    mirrors: dict[str, Mirror] = Field(default_factory=dict)
    dielectics: dict[str, Dielectric] = Field(default_factory=dict)
    apertureStops: dict[str, ApertureStop] = Field(default_factory=dict)


def build_optics(config: dict[str, Any]) -> Optics:
    """Build an Optics instance from a config dict.

    Parameters
    ----------
    config : dict
        Dictionary with ``default`` (optional) and ``elements`` keys.
        ``elements`` is a list of single-key dicts where the key is
        the element name and the value is its config::

            elements:
              - forebaffle: { temperature: 240.0 }
              - window: { obj_type: Dielectric, thickness: 0.001 }

        Element ordering is preserved (it defines the optical chain).

    Returns
    -------
    Optics
        Instance with all requested optical elements categorised.
    """
    defaults = config.get("default", {})
    elem_list = config["elements"]

    elements = OrderedDict()
    mirrors = OrderedDict()
    dielectics = OrderedDict()
    apertureStops = OrderedDict()

    for item in elem_list:
        ((name, props),) = item.items()
        cfg = {**defaults, **(props or {})}
        obj_type = cfg.pop("obj_type", None)
        cls = _ELEMENT_TYPES[obj_type]
        elem = cls(**cfg)
        elem.elem_name = name
        elements[name] = elem
        if isinstance(elem, Mirror):
            mirrors[name] = elem
        if isinstance(elem, Dielectric):
            dielectics[name] = elem
        if isinstance(elem, ApertureStop):
            apertureStops[name] = elem

    return Optics(
        elements=elements,
        mirrors=mirrors,
        dielectics=dielectics,
        apertureStops=apertureStops,
    )
