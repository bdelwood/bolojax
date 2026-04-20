"""Unit conversions"""

from cfgmdl import Unit

to_SI_dict = {
    "GHz": 1.0e09,
    "mm": 1.0e-03,
    "aW/rtHz": 1.0e-18,
    "pA/rtHz": 1.0e-12,
    "pW": 1.0e-12,
    "pW/K": 1.0e-12,
    "um": 1.0e-06,
    "pct": 1.0e-02,
    "uK": 1.0e-06,
    "uK-rts": 1.0e-06,
    "uK-amin": 1.0e-06,
    "uK^2": 1.0e-12,
    "yr": (365.25 * 24.0 * 60.0 * 60),
    "e-4": 1.0e-04,
    "e+6": 1.0e06,
    "um RMS": 1.0e-06,
    "MJy": 1.0e-20,
    "Ohm": 1.0,
    "W/Hz": 1.0,
    "Hz": 1.0,
    "m": 1.0,
    "W/rtHz": 1.0,
    "A/rtHz": 1.0,
    "W": 1.0,
    "K": 1.0,
    "K_RJ": 1.0,
    "K^2": 1.0,
    "s": 1.0,
    "deg": 1,
    "NA": 1.0,
}

Unit.update(to_SI_dict)
