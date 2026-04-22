"""
Interpolates quantities vs frequency
"""

from __future__ import annotations

import numpy as np

from bolojax.models.utils import is_none, read_txt_to_np

_rng = np.random.default_rng()


class FreqInterp:
    """
    Interpolates quantities vs frequency, with errors
    for detectors and optics
    """

    def __init__(self, fname: str) -> None:
        """Constructor"""
        self.fname: str = fname
        band_data: np.ndarray = read_txt_to_np(self.fname)
        if len(band_data) == 3:
            self.freq: np.ndarray
            self.tran: np.ndarray
            self.errs: np.ndarray | None
            self.freq, self.tran, self.errs = band_data
        elif len(band_data) == 2:
            self.freq, self.tran = band_data
            self.errs = None
        self.freq *= 1e9
        self.tran_interp: np.ndarray | None = None
        self.errs_interp: np.ndarray | None = None

    def mean(self) -> np.floating:
        """Return the weighted mean of the interpolation curve"""
        return np.sum(self.freq * self.tran) / np.sum(self.tran)

    def mean_trans(self) -> np.floating:
        """Return the value at the mean of the interpolation curve"""
        return np.interp(self.mean(), self.freq, self.tran)

    def cache_grid(self, freqs: np.ndarray | None) -> None:
        """Cache the values and errors from the interpolation grid"""
        if freqs is None:
            self.tran_interp = self.tran
            self.errs_interp = self.errs
            return
        mask = np.bitwise_and(freqs < self.freq[-1], freqs > self.freq[0])
        self.tran_interp = np.where(
            mask, np.interp(freqs, self.freq, self.tran), 0.0
        ).clip(0.0, 1.0)
        if is_none(self.errs):
            self.errs_interp = None
            return
        self.errs_interp = np.where(
            mask, np.interp(freqs, self.freq, self.errs), 0.0
        ).clip(1e-6, np.inf)
        return

    def rvs(self, freqs: np.ndarray | None, nsample: int = 0) -> np.ndarray:
        """Sample values"""

        self.cache_grid(freqs)
        if not nsample:
            return self.tran_interp

        if is_none(self.errs_interp):
            return (np.ones((nsample, 1)) * self.tran_interp).clip(0.0, 1.0)
        return _rng.normal(
            self.tran_interp, self.errs_interp, (nsample, len(self.tran_interp))
        ).clip(0.0, 1.0)
