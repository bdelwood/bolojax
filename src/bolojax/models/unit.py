"""Pint unit registry for bolojax.

Registers custom units used in CMB instrument modeling that pint
doesn't include by default.
"""

from __future__ import annotations

import pint

ureg = pint.UnitRegistry()

# Jansky (flux density): 1 Jy = 1e-26 W/m^2/Hz
ureg.define("Jy = 1e-26 W / m^2 / Hz")
ureg.define("MJy = 1e6 Jy")

# Rayleigh-Jeans temperature (dimensionally kelvin)
ureg.define("K_RJ = K")

# rtHz shorthand for noise spectral densities
ureg.define("rtHz = Hz ** 0.5")

# Ohm (pint uses lowercase 'ohm')
ureg.define("@alias ohm = Ohm")

__all__ = ["ureg"]
