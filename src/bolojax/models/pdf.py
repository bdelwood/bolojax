"""Simple implementation of PDFs for instrument parameters"""

from __future__ import annotations

import numpy as np

from bolojax.models.utils import read_txt_to_np

_rng = np.random.default_rng()


class ChoiceDist:
    """
    ChoiceDist object holds probability distribution functions (PDFs)
    for instrument parameters

    Parameters
    ----------
    inp (str or arr): file name for the input PDF or input data array

    """

    def __init__(self, inp: str | np.ndarray | list[list[float]]) -> None:
        """
        Parameters
        ----------
        inp (str or arr): file name for the input PDF or input data array
        """
        if isinstance(inp, str):
            self._fname: str | None = inp
            self._inp: np.ndarray = read_txt_to_np(inp)
        else:
            self._fname = None
            self._inp = np.array(inp)

        if len(self._inp.shape) != 2:
            msg = f"ChoiceDist requires 2 input arrays {self._inp.shape}"
            raise ValueError(msg)

        self.val: np.ndarray = self._inp[0]
        self.prob: np.ndarray = self._inp[1]
        self.prob /= np.sum(self.prob)
        self._cum: np.ndarray = np.cumsum(self.prob)

    # ***** Public Methods *****
    def rvs(self, nsample: int = 1) -> np.floating | np.ndarray:
        """
        Sample the distribution nsample times

        Args:
        nsample (int): the number of times to sample the distribution
        """
        if nsample == 1:
            return _rng.choice(self.val, size=nsample, p=self.prob)[0]
        return _rng.choice(self.val, size=nsample, p=self.prob)

    def change(self, new_avg: float) -> None:
        """Arithmetically shift the distribution to the new central value"""
        old_mean = self.mean()
        shift = new_avg - old_mean
        self.val += shift

    def mean(self) -> np.floating:
        """Return the mean of the distribution"""
        if self.prob is not None:
            return np.sum(self.prob * self.val)
        return np.mean(self.val)

    def std(self) -> np.floating:
        """Return the standard deviation of the distribution"""
        if self.prob is not None:
            mean = self.mean()
            return np.sqrt(np.sum(self.prob * ((self.val - mean) ** 2)))
        return np.std(self.val)

    def median(self) -> np.floating:
        """Return the median of the distribution"""
        if self.prob is not None:
            arg = np.argmin(abs(self._cum - 0.5))
            return self.val[arg]
        return np.median(self.val)

    def one_sigma(self) -> tuple[np.floating, np.floating]:
        """Return the 15.9% and 84.1% values"""
        med = self.median()
        if self.prob is not None:
            lo, hi = np.interp((0.159, 0.841), self._cum, self.val)
        else:
            lo, hi = np.percentile(self.val, [0.159, 0.841])
        return (hi - med, med - lo)

    def two_sigma(self) -> tuple[np.floating, np.floating]:
        """Return the 2.3% and 97.7% values"""
        med = self.median()
        if self.prob is not None:
            lo, hi = np.interp((0.023, 0.977), self._cum, self.val)
        else:
            lo, hi = np.percentile(self.val, [0.023, 0.977])
        return (hi - med, med - lo)
