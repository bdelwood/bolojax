# Atmospheric model

bolojax computes atmospheric brightness temperature and transmission using
[am](https://doi.org/10.5281/zenodo.640645) (Scott Paine's atmospheric radiative
transfer code). Internally, it uses
[am-python](https://github.com/bdelwood/am-python) to dispatch `am` calls.

## How it works

An `am` configuration file (`.amc`) defines the atmospheric layer structure,
molecular abundances, and spectral line parameters for a given site and season.
Runtime parameters are passed to `am` through positional placeholders (`%1`,
`%2`, ...) in the `.amc` file.

Rather than calling `am` for every parameter combination at runtime, bolojax
pre-computes a grid of atmosphere profiles across the Cartesian product of the
varying parameters. The grid is cached to disk, so subsequent runs load
instantly.

At evaluation time, each sampled parameter combination is matched to the nearest
grid point, and the brightness temperature and transmission spectra are
interpolated onto the channel's frequency grid.

## Configuration

The atmosphere is configured in the `universe.atmosphere` section of the YAML
config:

```yaml
universe:
  atmosphere:
    amc_file: SPole_annual_median_MERRA2_2007-2016.amc
    amc_args: [0, 300, zenith, pwv_scale]
    profile_pwv_mm: 0.425
```

### Fields

- **`amc_file`**: Path to an `am` configuration file defining the atmospheric
  layer structure and spectral properties for a given site.

- **`amc_args`**: The argument list passed to `am`, corresponding positionally
  to the `%1`, `%2`, ... placeholders in the `.amc` file. Numeric entries are
  static. The keywords `zenith` and `pwv_scale` are resolved at runtime from the
  instrument's elevation and PWV fields.

- **`profile_pwv_mm`**: The reference PWV (in mm) baked into the `.amc` profile.
  Used by the `pwv_scale` derived parameter to compute the water vapor scaling
  factor.

### Observational parameters

Dynamic atmosphere parameters are configured on the instrument (not the
atmosphere) because they are observational quantities that may be sampled from
distributions:

```yaml
instrument:
  elevation:
    var_type: pdf
    fname: Pdf/elevation_pole.txt
  pwv:
    var_type: pdf
    fname: Pdf/pwv_pole.txt
```

For deterministic evaluation (for example, Fisher analysis), use fixed scalars:

```yaml
instrument:
  elevation: 60.0
  pwv: 0.6
```

## Parameter resolution

The two dynamic parameters, **elevation** and **PWV**, are resolved from
`amc_args` string entries via built-in transforms:

| Keyword     | Source      | Transform              | Grid step |
| ----------- | ----------- | ---------------------- | --------- |
| `zenith`    | `elevation` | `90 - elevation`       | 1°        |
| `pwv_scale` | `pwv`       | `pwv / profile_pwv_mm` | 0.1 mm    |

Numeric entries in `amc_args` are passed to `am` as-is. For
`amc_args: [0, 300, zenith, pwv_scale]`:

- `0` and `300` set the frequency range (static)
- `zenith` is computed from the instrument's `elevation` field
- `pwv_scale` is computed from the instrument's `pwv` field

## Grid

Grid extents are inferred lazily from the sampled parameter values; no manual
grid specification is needed. On first evaluation, a regular grid is built
covering the sampled range at each parameter's grid step.

For a typical South Pole configuration (PWV 0.2–1.6 mm at 0.1 mm step, elevation
55–65° at 1° step), this produces ~170 profiles. Subsequent runs load from
cache.

## Backends

### `AmAtm` (primary)

Computes atmosphere profiles from an `.amc` file using `am-python`. Builds a 2D
grid over elevation and PWV, caches to disk, and performs nearest-neighbor
lookup at evaluation time.

### `AtmProfile` (fallback)

Loads a single fixed atmosphere profile from a text file (three columns:
frequency in GHz, brightness temperature in K, transmission). Ignores all
runtime parameters. Activated by `custom_atm_file` on the instrument config:

```yaml
instrument:
  custom_atm_file: Bands/atacama_atm.txt
```

### No atmosphere

If neither `amc_file` nor `custom_atm_file` is configured, the atmosphere model
returns `None` and no atmospheric effects are applied.

## Shipped profiles

bolojax ships one `.amc` profile:

- **`SPole_annual_median_MERRA2_2007-2016.amc`** — South Pole annual median
  atmosphere derived from MERRA-2 reanalysis (2007–2016),
  `profile_pwv_mm = 0.425`.

Custom `.amc` files for other sites (e.g. Atacama, balloon altitudes) can be
used by pointing `amc_file` to the file and adjusting `amc_args` to match the
file's placeholder conventions.
