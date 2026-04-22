r"""Beam correlation factors for photon noise calculations.

Computes the VCZT (van Cittert-Zernike theorem) amplitude coherence between
detector pixels as a function of separation, following the formalism of
Hill & Kusaka, "Photon noise correlations in millimeter-wave telescopes,"
Appl. Opt. 63, 1654 (2024), arXiv:2309.01153.

The aperture coherence $\gamma^\mathrm{np}_{\mathrm{ap},ij}$ (Hill Eq. 53) and
stop coherence $\gamma^\mathrm{np}_{\mathrm{stop},ij}$ (Eq. 56) are computed
via a Hankel transform of the beam illumination intensity $|G(r)|^2$ (Eq. 54).
The HBT intensity coherence $|\gamma|^2$ (Eq. 17) enters the array noise
covariance (Eq. 68) as a multiplicative factor on the Bose (wave) noise term.

For the "bolocalc" preset the poly_taper beam parameters were obtained by
fitting the Hankel transform against the 100-point coherentApertCorr.pkl
curve shipped with the original BoloCalc (RMS residual 0.02%).  The stop
correlation for this preset is loaded from a stored array (stop_corr.npy,
converted from the original coherentStopCorr.pkl) whose beam model and
generation method are undocumented and I couldn't reproduce.  For physically
motivated presets ("trunc_gauss", "he11"), both correlations derive from
the same beam model with different integration limits (aperture: 0 to R_ap,
stop: R_ap outward), following the decomposition in Eq. 51.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import bessel_jn
from jax.typing import ArrayLike

BeamModel = Literal["poly_taper", "trunc_gauss", "he11"]
BeamPreset = Literal["bolocalc", "trunc_gauss", "he11"]
BeamParams = dict[str, BeamModel | float]


def j0(x: ArrayLike) -> Array:
    r"""Zeroth-order Bessel function $J_0(x)$.

    Args:
        x: argument (scalar or array).

    Returns:
        $J_0(x)$, with the $x=0$ singularity handled.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    small = jnp.abs(x) < 0.01
    safe_x = jnp.where(small, 1.0, x)
    large_val = bessel_jn(safe_x, v=0, n_iter=50)[0]
    # for small values, use Taylor expansion, similar to what scipy does
    # see scipy/xsf/include/xsf/cephes/j0.h lines 169-173
    small_val = 1.0 - x**2 / 4.0
    return jnp.where(small, small_val, large_val)


def soft_edge(r: ArrayLike, R: float, softening: float = 0.01) -> Array:
    r"""Smooth sigmoid edge: 1 inside R, tapering to 0 outside.

    Uses tanh profile: $0.5\,(1 - \tanh((r - R) / \sigma))$ where
    $\sigma$ is the softening width.

    Args:
        r: radial coordinate.
        R: edge radius.
        softening: transition width (smaller = sharper edge).

    Returns:
        Smooth step from 1 to 0 around R.
    """
    return 0.5 * (1 - jnp.tanh((r - R) / softening))


def poly_taper(r: ArrayLike, a1: float, a2: float, n: float) -> Array:
    r"""Polynomial taper beam: $(1 - a_1 r - a_2 r^2)^n$ with soft edge.

    The "bolocalc" preset parameters (a1=1.0825, a2=-0.0413, n=1.300) were
    obtained by least-squares fitting the Hankel transform of this beam model
    against the 100-point coherentApertCorr.pkl curve distributed with the
    original BoloCalc. The fit reproduces the pickle to 0.02% RMS.

    Args:
        r: radial coordinate (in units of aperture radius).
        a1: linear coefficient.
        a2: quadratic coefficient.
        n: taper exponent.
        softening: edge transition width.

    Returns:
        Beam illumination intensity.
    """
    inner = jnp.maximum(1 - a1 * r - a2 * r**2, 0.0)
    return inner**n


def trunc_gauss(r: ArrayLike, sigma: float, R: float, softening: float = 0.01) -> Array:
    """Truncated Gaussian beam with soft edge at R.

    Generalization of the Gaussian illumination in Hill Eq. 54.

    Args:
        r: radial coordinate.
        sigma: Gaussian width parameter.
        R: truncation radius.
        softening: edge transition width.

    Returns:
        Beam illumination intensity.
    """
    return jnp.exp(-2 * sigma**2 * r**2) * soft_edge(r, R, softening)


def he11(r: ArrayLike, R: float, R_taper: float, softening: float = 0.01) -> Array:
    r"""Corrugated horn HE11 mode with soft edge at $R_{\mathrm{taper}}$.

    $J_0(2.405 \cdot r/R)^2$ with smooth cutoff at $R_{\mathrm{taper}}$.
    Alternative to the Gaussian illumination of Hill Eq. 54 for corrugated
    feedhorn optics.

    Args:
        r: radial coordinate.
        R: horn radius parameter.
        R_taper: soft cutoff radius.
        softening: edge transition width.

    Returns:
        Beam illumination intensity.
    """
    u01 = 2.4048255577  # first zero of J0
    return j0(u01 * r / R) ** 2 * soft_edge(r, R_taper, softening)


def beam_coherence(
    p: ArrayLike,
    beam_func: Callable[[Array], Array],
    r_min: float = 0.0,
    r_max: float = 1.0,
    n_pts: int = 10000,
) -> Array:
    r"""Compute amplitude coherence $\gamma(p)$ via Hankel transform.

    Evaluates the normalized zeroth-order Hankel transform of the beam
    intensity pattern $|G(r)|^2$ over the radial range [r_min, r_max],
    implementing the circularly-symmetric form of Hill Eq. 53 (aperture)
    and Eq. 56 (stop):

    .. math::

        \gamma(p) = \frac{1}{\eta}
        \int_{r_\min}^{r_\max} |G(r)|^2 \, J_0(2\pi p r) \, 2\pi r \, dr

    where $\eta$ is the integrated power in the region.  The flat-illumination
    limit recovers the Bessel/jinc result of Eq. 55.

    Args:
        p: detector separations in $F\lambda$ units (array).
        beam_func: callable returning $|G(r)|^2$ given r in units of
            $D_\mathrm{ap}/2$.
        r_min: inner integration bound (0 for aperture, $R_\mathrm{ap}$ for
            stop).
        r_max: outer integration bound.
        n_pts: number of integration points.

    Returns:
        $\gamma(p)$ array, normalized so $\gamma(0) = 1$.
    """
    p = jnp.atleast_1d(jnp.asarray(p, dtype=jnp.float64))
    r = jnp.linspace(max(r_min, 1e-10), r_max, n_pts)
    G2 = jnp.maximum(beam_func(r), 0.0)
    norm = 2 * jnp.pi * jnp.trapezoid(G2 * r, r)

    def _single_p(pp: Array) -> Array:
        integrand = G2 * j0(2 * jnp.pi * pp * r) * r * 2 * jnp.pi
        return jnp.trapezoid(integrand, r) / norm

    return jax.vmap(_single_p)(p)


# Preset parameters. The "bolocalc" values were fitted to reproduce
# the aperture correlation pickle distributed with BoloCalc.
PRESETS: dict[str, BeamParams] = {
    "bolocalc": {
        "model": "poly_taper",
        "a1": 1.0825,
        "a2": -0.0413,
        "n": 1.300,
        "R_zero": 0.961,  # root of (1 - a1*r - a2*r**2), beam vanishes here
    },
    "trunc_gauss": {
        "model": "trunc_gauss",
        "sigma": 1.33,
        "R": 1.0,
    },
    "he11": {
        "model": "he11",
        "R": 1.05,
        "R_taper": 1.05,
    },
}


def load_bolocalc_stop() -> tuple[Array, Array]:
    """Load the stored stop correlation curve.

    The stop correlation for the "bolocalc" preset is loaded from a stored
    array (stop_corr.npy), converted from coherentStopCorr.pkl originally
    distributed with BoloCalc. The beam model and computation
    method used to generate this curve are undocumented.  Exhaustive fitting
    against Gaussian, Airy, sinc, polynomial, and annular beam models, as
    well as 2D FFT and Monte Carlo simulations, could not reproduce the
    sidelobe structure to better than ~2% RMS.

    Returns:
        Tuple of (pitch, correlation_values).
    """
    data_dir = Path(__file__).parent / "data"
    data = np.load(data_dir / "stop_corr.npy")  # (2, 100): [pitch, values]
    return jnp.asarray(data[0]), jnp.asarray(data[1])


def compute_corr_curves(
    preset: BeamPreset | BeamParams = "bolocalc", p_grid: Array | None = None
) -> tuple[Array, Array, Array]:
    r"""Compute aperture and stop coherence curves for a given preset.

    Returns the amplitude coherence $\gamma_\mathrm{ap}$ (Hill Eq. 53) and
    intensity coherence $|\gamma_\mathrm{stop}|^2$ (Eq. 56, squared per
    Eq. 17) on a pitch grid.  For physically motivated presets these are the
    two terms in the decomposition of the total mutual intensity (Eq. 51).

    Args:
        preset: name of a preset ("bolocalc", "trunc_gauss", "he11")
            or a dict with beam model parameters.
        p_grid: pitch grid in $F\lambda$ units. Defaults to
            linspace(0, 5, 100).

    Returns:
        ``(p_grid, gamma_apert, gamma_stop)`` arrays.
    """
    if p_grid is None:
        p_grid = jnp.linspace(0, 5, 100)

    params: BeamParams = PRESETS[preset] if isinstance(preset, str) else preset

    model = params["model"]

    R_ap: float = params.get("R_ap", 1.0)  # type: ignore[assignment]

    if model == "poly_taper":
        a1: float = params["a1"]  # type: ignore[assignment]
        a2: float = params["a2"]  # type: ignore[assignment]
        n: float = params["n"]  # type: ignore[assignment]
        R_zero: float = params.get("R_zero", 1.0 / a1 if abs(a2) < 1e-10 else 1.0)  # type: ignore[assignment]
        beam_func = lambda r: poly_taper(r, a1, a2, n)  # noqa: E731
        gamma_apert = beam_coherence(p_grid, beam_func, r_max=R_zero)

        if preset == "bolocalc":
            # Stop correlation loaded from array
            p_stored, c_stored = load_bolocalc_stop()
            gamma_stop = jnp.interp(p_grid, p_stored, c_stored)
        else:
            gamma_stop_signed = beam_coherence(
                p_grid, beam_func, r_min=R_ap, r_max=R_zero + 0.1
            )
            gamma_stop = gamma_stop_signed**2

    elif model == "trunc_gauss":
        sigma: float = params["sigma"]  # type: ignore[assignment]
        R: float = params["R"]  # type: ignore[assignment]
        beam_func = lambda r: trunc_gauss(r, sigma, R)  # noqa: E731
        gamma_apert = beam_coherence(p_grid, beam_func, r_max=R_ap)
        gamma_stop_signed = beam_coherence(p_grid, beam_func, r_min=R_ap, r_max=R + 0.1)
        gamma_stop = gamma_stop_signed**2

    elif model == "he11":
        R: float = params["R"]  # type: ignore[assignment]
        R_taper: float = params["R_taper"]  # type: ignore[assignment]
        beam_func = lambda r: he11(r, R, R_taper)  # noqa: E731
        gamma_apert = beam_coherence(p_grid, beam_func, r_max=R_ap)
        gamma_stop_signed = beam_coherence(
            p_grid, beam_func, r_min=R_ap, r_max=R_taper + 0.1
        )
        gamma_stop = gamma_stop_signed**2

    else:
        msg = f"Unknown beam model: {model}"
        raise ValueError(msg)

    return p_grid, gamma_apert, gamma_stop
