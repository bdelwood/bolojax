"""Optical element types for the JAX compute layer.

Each element type stores its physical properties and implements
``emiss_trans(freqs)`` to compute frequency-dependent emissivity and
transmission. These are pytrees where the leaves can be
selected with dot notation.
"""

from __future__ import annotations

import jax.numpy as jnp
import zodiax as zdx
from jax import Array

from bolojax.compute import physics


class Element(zdx.Base):
    """Generic optical element.

    Handles elements specified by direct absorption, reflection,
    scatter fraction, and spillover (e.g. forebaffle). Children
    override ``_calc_absorption`` to compute absorption from physical
    properties.
    """

    temperature: float | Array = 0.0
    absorption: float | Array = 0.0
    reflection: float | Array = 0.0
    scatter_frac: float | Array = 0.0
    scatter_temp: float | Array | None = None
    spillover: float | Array = 0.0
    spillover_temp: float | Array | None = None

    def _calc_absorption(self, freqs: Array) -> Array:  # noqa: ARG002
        return jnp.atleast_1d(jnp.asarray(self.absorption, dtype=jnp.float64))

    def emiss_trans(self, freqs: Array) -> tuple[Array, Array]:
        """Compute emissivity and transmission arrays.

        Emissivity accounts for absorption, plus scatter and spillover
        weighted by their temperature relative to the element temperature.

        Returns:
            (emissivity, transmission) arrays of shape ``(n_freq,)``.
        """
        abso = self._calc_absorption(freqs)

        # Scatter contribution, weighted by temperature ratio
        scat = self.scatter_frac
        scat_temp = (
            self.scatter_temp if self.scatter_temp is not None else self.temperature
        )
        scat_weight = scat * physics.pow_frac(scat_temp, self.temperature, freqs)

        # Spillover contribution, weighted by temperature ratio
        spil = self.spillover
        spil_temp = (
            self.spillover_temp if self.spillover_temp is not None else self.temperature
        )
        spil_weight = spil * physics.pow_frac(spil_temp, self.temperature, freqs)

        emiss = abso + scat_weight + spil_weight
        trans = (1 - self.reflection) * (1 - abso) * (1 - scat) * (1 - spil)
        return jnp.atleast_1d(emiss), jnp.atleast_1d(trans)


class Dielectric(Element):
    """Transmissive element: absorption from dielectric loss tangent."""

    thickness: float | Array = 0.0
    index: float | Array = 1.0
    loss_tangent: float | Array = 0.0

    def _calc_absorption(self, freqs: Array) -> Array:
        return physics.dielectric_loss(
            freqs, self.thickness, self.index, self.loss_tangent
        )


class Mirror(Element):
    """Reflective element: absorption from ohmic and Ruze losses."""

    conductivity: float | Array = 0.0
    surface_rough: float | Array = 0.0

    def _calc_absorption(self, freqs: Array) -> Array:
        safe_cond = jnp.where(self.conductivity > 0, self.conductivity, 1.0)
        safe_rough = jnp.where(self.surface_rough > 0, self.surface_rough, 1.0)
        ohmic = 1.0 - physics.ohmic_eff(freqs, safe_cond)
        ruze = 1.0 - physics.ruze_eff(freqs, safe_rough)
        abso = jnp.where(self.conductivity > 0, ohmic, 0.0)
        return abso + jnp.where(self.surface_rough > 0, ruze, 0.0)


class ApertureStop(Element):
    """Aperture stop element.

    Spillover is typically computed from pixel geometry at setup time
    and stored as a scalar.  Uses the base ``emiss_trans`` directly.
    """


class SkySource(zdx.Base):
    """Precomputed sky element (CMB, atmosphere, dust, synchrotron).

    Emissivity and transmission are frequency-dependent arrays produced
    by the sky model during setup. Not differentiated through.
    """

    temperature: float | Array = 0.0
    emiss_spectrum: Array = None
    trans_spectrum: Array = None

    def emiss_trans(self, freqs: Array) -> tuple[Array, Array]:  # noqa: ARG002
        return self.emiss_spectrum, self.trans_spectrum
