"""Unit conversion system for bolojax."""

from __future__ import annotations

from typing import ClassVar

import numpy as np


class Unit:
    """Simple SI unit conversion.

    Maintains a class-level registry mapping unit names to SI conversion
    factors.  Calling an instance converts *to* SI; ``inverse`` converts
    *from* SI.
    """

    to_SI_dict: ClassVar[dict[str, float]] = {}

    def __init__(self, unit: str | float | None = None) -> None:
        if unit is None:
            self._SI = 1.0
            self._name = ""
            return
        if isinstance(unit, str):
            if unit not in self.to_SI_dict:
                msg = f"Passed unit '{unit}' not understood by Unit object"
                raise KeyError(msg)
            self._SI = self.to_SI_dict[unit]
            self._name = unit
            return
        self._SI = float(unit)
        self._name = "a.u."

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, val: float | np.ndarray | None) -> np.ndarray | None:
        """Convert *val* from native units to SI."""
        if val is None:
            return None
        return np.array(val) * self._SI

    def inverse(self, val: float | np.ndarray | None) -> np.ndarray | None:
        """Convert *val* from SI to native units."""
        if val is None:
            return None
        return np.array(val) / self._SI

    @classmethod
    def update(cls, a_dict: dict[str, float]) -> None:
        """Register additional unit conversions."""
        cls.to_SI_dict.update(a_dict)


_to_SI = {
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
    "yr": 365.25 * 24.0 * 60.0 * 60.0,
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
    "deg": 1.0,
    "NA": 1.0,
}

Unit.update(_to_SI)

__all__ = ["Unit"]
