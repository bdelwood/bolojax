# Beam correlation

bolojax computes photon noise correlations between detector pixels in densely
packed focal plane arrays. The original BoloCalc included pre-computed amplitude
coherence curves based on an undocumented beam model. This implementation
follows the formalism of Hill & Kusaka (2024), "Photon noise correlations in
millimeter-wave telescopes," Applied Optics 63(6), 1654.
[arXiv:2309.01153](https://arxiv.org/abs/2309.01153).

## Physical picture

When detector pixels are packed with pitch $p_\text{pix} \lesssim 1.2 F\lambda$,
neighboring detectors sample overlapping spatial modes of the incoming radiation
field. The Bose (wave noise) term of the photon noise then correlates between
pixels through the Hanbury Brown & Twiss (HBT) effect. As a result,
array-averaged sensitivity scales less favorably than the uncorrelated
$N_\text{det}^{-1/2}$ limit.

This matters for mm-wave CMB experiments where the mean photon occupation number
$\bar{n} \sim 1$ places the detector in the crossover regime between
shot-noise-dominated (optical, $\bar{n} \ll 1$) and wave-noise-dominated (radio,
$\bar{n} \gg 1$).

## Theory

### Mutual intensity and coherence

The mutual intensity between detectors $i$ and $j$ is (Hill Eq 6):

$$B_{ij}(\nu) = \sum_k S_{ik}^*(\nu) S_{jk}(\nu) n(T_k, \nu) \label{eq:mutual-intensity}$$

where $S_{ik}$ is the scattering matrix coupling source mode $k$ to detector
$i$, and $n(T,\nu) = [\exp(h\nu/k_BT) - 1]^{-1}$ is the Bose-Einstein occupation
number.

The **amplitude coherence** ("van Cittert-Zernike Thoerem" (VCZT) coefficient)
is the normalized mutual intensity (Hill Eq 10):

$$\gamma_{ij}(\nu) = \frac{B_{ij}(\nu)}{\sqrt{B_{ii}(\nu) B_{jj}(\nu)}} \label{eq:amplitude-coherence}$$

The **intensity coherence** (HBT coefficient) that enters the noise covariance
is its squared magnitude (Hill Eq 17):

$$\gamma_{ij}^{(2)}(\nu) = |\gamma_{ij}(\nu)|^2 \label{eq:intensity-coherence}$$

### Aperture and stop decomposition

For a simplified telescope model (Hill Sec 3A), the discrete sum over source
modes in $\eqref{eq:mutual-intensity}$ becomes a continuous integral over the
aperture plane. Two steps are involved:

First, the scattering matrix $S_{ik}$ reduces to the pupil function $G_i(u, v)$:
detector $i$'s beam back-propagated to coordinates $(u, v)$ on the aperture
plane. This is the van Cittert-Zernike theorem: for spatially incoherent thermal
sources (coherence length $\sim \lambda \ll D_\text{ap}$), the coupling between
a source at position $(u, v)$ and detector $i$ is simply $G_i$ evaluated there.

Second, the occupation number $n(T_k)$ factors out of the integral. In
$\eqref{eq:mutual-intensity}$, each source mode $k$ has its own temperature
$T_k$. But within the aperture region, all sources are treated as having a
single effective occupation number $n(T_\text{ap})$ (defined by the weighted sum
over optical elements in Hill Eq 46). Since $n$ is constant across the
integration domain, it becomes a prefactor. The integral becomes just the
overlap of the beam patterns between dector $i$ and $j$:

$$B_{\text{ap},ij}(\nu) \propto n(T_\text{ap}) \iint G_i^*(u,v) G_j(u,v) e^{2\pi i (u x_{ji} + v y_{ji}) / D_\text{ap} F\lambda} \mathrm{d}u \mathrm{d}v \label{eq:general-mutual}$$

where $(x_{ji}, y_{ji})$ is the displacement between detectors $j$ and $i$ in
the focal plane. Since a phase tilt in the pupil plane corresponds to a pointing
offset in the focal plane, the phase factor encodes the detector separation
$p_{ij} = \sqrt{x_{ji}^2 + y_{ji}^2}$. The factored-out $n(T_\text{ap})$
reappears later as the occupation number weight in $\eqref{eq:combined-hbt}$.

Under the assumption that all detectors share the same beam pattern
($G_i \simeq G_j \equiv G$; Hill Eq 52 $\to$ 53), the cross-correlation
$G_i^* G_j$ simplifies to the power spectrum $|G|^2$, and the coherence becomes
a function of $p_{ij}$ alone.

The aperture stop divides the pupil plane into two regions: radiation passing
through the aperture (sky signal, as seen through the optical chain before the
stop, at effective temperature $T_\text{ap}$) and radiation from the stop itself
(thermal emission at physical temperature $T_\text{stop}$). Since each region
has a uniform occupation number, the mutual intensity separates into two
independent integrals (Hill Eq 51), with $n(T_\text{ap})$ factoring out of the
aperture term and $n(T_\text{stop})$ out of the stop term.

Working in normalized coordinates where $\rho = 2\sqrt{u^2 + v^2}/D_\text{ap}$
(so $\rho = 1$ at the aperture edge):

**Aperture radiation**: sky-side sources, $\rho \in [0, 1]$ (Hill Eq 53):

$$\gamma_{\text{ap},ij}(\nu) = \frac{1}{\eta_\text{ap}} \iint_{\rho \in [0, 1]} |G(u, v)|^2 e^{2\pi i p_{ij} u / F\lambda} \mathrm{d}u \mathrm{d}v \label{eq:aperture-coherence}$$

**Stop radiation**: thermal emission from beyond the aperture edge,
$\rho \in [1, \infty)$ (Hill Eq 56):

$$\gamma_{\text{stop},ij}(\nu) = \frac{1}{1-\eta_\text{ap}} \iint_{\rho \in [1, \infty)} |G(u, v)|^2 e^{2\pi i p_{ij} u / F\lambda} \mathrm{d}u \mathrm{d}v \label{eq:stop-coherence}$$

where $\eta_\text{ap} = \iint_{\rho \leq 1} |G(u,v)|^2 \mathrm{d}u \mathrm{d}v$
is the aperture spillover efficiency. The frequency dependence enters through
$F\lambda = Fc/\nu$ in the phase factor. For identical beams, $\gamma_{ij}$
depends on the detector pair only through their separation $p_{ij}$.

For a circularly symmetric pupil $G(u, v) = G(\rho)$, the 2D Fourier transforms
reduce to 1D Hankel transforms. Writing $p = p_{ij}/F\lambda$ as the separation
in diffraction units:

$$\gamma(p) = \frac{1}{\eta} \int_{\rho_{\mathrm{min}}}^{\rho_{\mathrm{max}}} |G(\rho)|^2 J_0(2\pi p \rho) 2\pi \rho \mathrm{d}\rho \label{eq:hankel}$$

with limits $[0, 1]$ for the aperture and $[1, \infty)$ for the stop. This is
the form implemented in by the preset models in the `beam_coherence` module.

### Combined HBT coefficient

Because the aperture and stop radiate at different temperatures, their
contributions to the wave noise must be weighted by their respective photon
occupation numbers $n(T, \nu) = [\exp(h\nu/k_BT) - 1]^{-1}$. The combined
intensity coherence is (Hill Eq 59):

$$|\gamma_{ij}^{\text{np}}|^2 = \frac{n(T_{\text{ap},i})}{n(T_{(i)})} \eta_{\text{ap},i} |\gamma_\text{ap}|^2 + \frac{n(T_\text{stop})}{n(T_{(i)})} (1-\eta_{\text{ap},i}) |\gamma_\text{stop}|^2 \label{eq:combined-hbt}$$

Each weight is the fraction of photons that sources deliver to detector $i$:

- $\eta_{\text{ap},i}$ is the aperture spillover efficiency of detector $i$,
  i.e. the fraction of $G_i$'s beam power passing through the aperture.
- $n(T_{\text{ap},i})$ is the effective photon occupation number of all
  radiation entering through the aperture as seen by detector $i$ (Hill Eq 46):
  $n(T_{\text{ap},i}, \nu) = \sum_\rho \epsilon_\rho H_\rho(\nu) n(T_\rho, \nu)$,
  where the sum runs over every optical element on the sky side of the aperture
  stop (CMB, atmosphere, mirrors, windows, filters, etc), weighted by their
  emissivities $\epsilon_\rho$ and cumulative throughput $H_\rho$.
- $T_\text{stop}$ is the physical temperature of the cold stop.
- Thus,
  $n(T_{(i)}) = \eta_{\text{ap},i} n(T_{\text{ap},i}) +
  (1-\eta_{\text{ap},i}) n(T_\text{stop})$
  is the total photon occupation number at detector $i$ from both sources (Hill
  Eq 60).

Under the identical-beam assumption ($G_i = G$), all per-detector quantities
collapse: $\eta_{\text{ap},i} \to \eta_\text{ap}$,
$T_{\text{ap},i} \to T_\text{ap}$, and $T_{(i)} \to T$.

### Impact on array sensitivity

The array noise variance including correlations is (Hill Eq 68):

$$\sigma_\text{arr}^2 = \frac{\sigma_\text{shot}^2 + (1 + \gamma^{(2)}) \sigma_\text{wave}^2 + \sigma_\text{int}^2}{N_\text{det}} \label{eq:array-noise}$$

where
$\gamma^{(2)} = N_\text{det}^{-1} \sum_{ij}
(1 - \delta_{ij}) \gamma_{ij}^{(2)}$
is the array-averaged HBT coefficient (Hill Eq 67). In `compute/noise.py`, this
enters as a per-element correlation factor that modifies the wave-noise
contribution to the photon NEP.

## Implementation

### Beam models

bolojax implements three beam illumination models in the
`compute.beam_correlation` module:

**Polynomial taper** (`poly_taper`): $[\max(1 - a_1 r - a_2 r^2, 0)]^n$

**Truncated Gaussian** (`trunc_gauss`): $\exp(-2\sigma^2 r^2) \Pi(r / R)$, a
Gaussian illumination truncated at radius $R$. Generalization of Hill Eq 54.

**HE11 mode** (`he11`): $J_0(2.405 r/R)^2 \Pi(r / R_\text{taper})$, the
fundamental corrugated horn mode truncated at $R_\text{taper}$.

$\Pi(x)$ is a rect/top-hat function, implemented as a smooth sigmoid for
numerical stability/differentiability.

The Hankel transform $\eqref{eq:hankel}$ is evaluated numerically by
`beam_coherence`, with integration limits selecting aperture or stop.

### Presets

Three presets bundle default beam parameters:

| Preset        | Model       | Parameters                                   |
| ------------- | ----------- | -------------------------------------------- |
| `bolocalc`    | poly_taper  | a1=1.0825, a2=-0.0413, n=1.300, R_zero=0.961 |
| `trunc_gauss` | trunc_gauss | sigma=1.33, R=1.0                            |
| `he11`        | he11        | R=1.05, R_taper=1.05                         |

The `bolocalc` preset is the default and reproduces original BoloCalc behavior.
For the aperture coherence, it uses the polynomial taper beam model (fit using
the original coherent aperture correlation curves). For the stop coherence, it
loads the original BoloCalc curve directly from a stored array. For the
physically motivated presets (`trunc_gauss`, `he11`), both aperture and stop
coherence derive from the same beam model with different integration limits.

The beam model is configured with the `beam_model` field on the camera. To use a
preset by name:

```yaml
camera:
  beam_model: trunc_gauss
```

To set custom beam parameters, pass a dict:

```yaml
camera:
  beam_model:
    model: trunc_gauss
    sigma: 1.5
    R: 0.9
```

Any parameter not specified falls back to the preset default. The available
parameters for each model are listed in the table above.

### Correlation factor computation

`Noise.corr_facts` computes per-element correlation factors by:

1. Computing aperture and stop coherence curves $\gamma(p)$ on a pitch grid from
   0 to 5 $F\lambda$.
2. Evaluating $\gamma$ at each shell separation allowed by a hex grid:
   $p_s \in \{p, \sqrt{3}p, 2p, 2\sqrt{3}p, \ldots\}$ out to a cutoff (default 3
   $F\lambda$).
3. Summing $\sum_s n_s \gamma(p_s)$ with $n_s = 6$ at every shell.
4. Returning $\sqrt{1 + \sum_s n_s \gamma(p_s)}$ as the per-element factor.

Each optical element uses either the aperture or stop coherence curve: elements
on the sky side of the aperture stop use $\gamma_\text{ap}$, the aperture stop
itself uses $\gamma_\text{stop}$, and elements on the detector side have a
factor of 1 (no correlation).

**Assumptions baked in:**

- **Identical beams**: a single coherence curve is used for all detector pairs
  at a given separation.
- **Hex-packed focal plane**: the array-averaged HBT coefficient
  $\gamma^{(2)} = N_\text{det}^{-1} \sum_{j \neq i} \gamma^{(2)}(p_{ij})$
  requires summing over all detector pairs. Since $\gamma^{(2)}$ depends only on
  separation, this reduces to $\sum_s n_s \gamma^{(2)}(p_s)$, where $n_s$ is the
  number of detectors at distance $p_s$ from a given detector. The code assumes
  $n_s = 6$ at every shell (exact for nearest neighbors in a hex lattice,
  approximate for farther shells).
- **Temperature-independent weighting**: the implementation currently assumes
  temperature independence. In other words, setting
  $T_\text{ap} = T_\text{stop} \equiv T$ causes the occupation number ratios to
  drop out of $\eqref{eq:combined-hbt}$,

  $$|\gamma_{ij}|^2 = \eta_\text{ap} |\gamma_\text{ap}|^2 + (1-\eta_\text{ap}) |\gamma_\text{stop}|^2$$

  In practice $T_\text{ap} \gt T_\text{stop}$, so this approximation overstates
  the relative stop contribution.

- **Unpolarized**: Hill Eq 62 gives the Stokes Q intensity coherence as
  $\gamma_{ij}^{Q,(2)} = \cos[2(\psi_i - \psi_j)] |\gamma_{ij}|^2$, where
  $\psi_i$ is the polarization angle of detector $i$. The current implementation
  ignores this modulation.
- **Truncated at 3 $F\lambda$**: correlations beyond `flamb_max` are ignored.
