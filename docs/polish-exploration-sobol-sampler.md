# Polish exploration — Sobol perturbation sampler (finish LHS deferral)

Status: implementation. Lands `Suite.sampling: sobol` as a working
sampler, replacing the `NotImplementedError` placeholder shipped in
[the LHS PR](polish-exploration-lhs-sampling.md).

## Why this matters

LHS gives **perfect marginal stratification** — every axis covered at
exactly `n_samples` strata — but the *joint* distribution is
randomised by per-axis permutations. For low-discrepancy goals
(e.g. covering 2D failure-mode pockets in `(lighting, camera_offset)`
jointly), Sobol does strictly better: a Sobol sequence at `N = 64`
points has lower star-discrepancy than any random or LHS draw of the
same size.

Concretely, for a 5-axis suite at `n_samples = 64` the 1D marginal
histogram (10 bins) std is:

| Sampler | Std of per-axis 10-bin histogram |
|---------|----------------------------------|
| Uniform random | ~2.4 |
| LHS | ~0.4 (perfect for `n_samples` bins) |
| Sobol (this PR) | ~0.5-0.7 |

LHS wins on marginal histograms at exact `n_samples`-bin granularity
(it's its design guarantee). Sobol wins on **2D and higher-dimensional
joint coverage** — Sobol projections onto any pair of axes are also
quasi-uniform, while LHS projections onto axis pairs are essentially
random. For users running a 5-axis suite to find the
`(lighting=dim, distractor=many)` joint failure pocket, Sobol gets you
there with fewer samples than LHS.

## Algorithm — Joe-Kuo Sobol

We use the **Joe-Kuo 6.21201** direction-number table (public domain,
http://web.maths.unsw.edu.au/~fkuo/sobol/), the modern reference
shipped with most QMC libraries.

The first 20 dimensions of direction numbers (taken verbatim from the
upstream table — `d`, `s`, `a`, `m_i`) are embedded inline in
`src/gauntlet/suite/sobol.py`. With dimension 0 hardcoded as the van
der Corput sequence (which the table omits because it is parameter-free),
the inline table covers 21 dimensions total — the harness's perturbation
grids rarely exceed 10 axes, so this is comfortable headroom.

If a Suite requests more dimensions than embedded, the sampler raises
a `ValueError` with explicit guidance:

```
Sobol sampler ships direction numbers for {N} dimensions; suite uses
{M}; extend the table or use LHS
```

### Generator shape

```python
def sobol_unit_cube(n_samples: int, n_dims: int, *, skip: int = 1) -> NDArray[np.float64]:
    """Generate ``n_samples`` Sobol points in [0, 1)^n_dims."""
```

Uses 32-bit direction numbers and Gray-code iteration (the standard
Bratley-Fox 1988 form). Returns floats divided by `2**32`, so values
are always in `[0, 1)` — never exactly 1.0.

### `skip` semantics

`skip` defaults to **1** (drop the leading origin `(0, 0, ..., 0)`).
This is the Owen 2003 recommendation for small `n_samples`: the
Sobol sequence with the leading origin retained has worse uniformity
for low budgets. Other libraries default to `skip = 0` or
`skip = n_samples - 1`; we pin `skip = 1` because:

* `skip = 0` lets a user accidentally hit `(low, low, ..., low)` on
  every axis — degenerate.
* `skip = n_samples - 1` aligns power-of-2 counts perfectly but means
  the same physical Sobol draw differs by `n_samples`, a violation of
  the principle of least surprise for users who change `n_samples`
  at the YAML level.
* `skip = 1` is small, predictable, and improves marginal coverage
  for `n_samples` from 8 up to a few hundred.

Users can override `skip` at the `SobolSampler(skip=N)` construction
layer if they need bit-for-bit reproducibility against a third-party
reference.

## Mapping unit-cube samples back onto AxisSpec values

Identical to the LHS sampler — see `src/gauntlet/suite/lhs.py`'s
`_axis_value_from_unit`. Continuous axes use the affine map
`low + u * (high - low)`; categorical axes use
`values[min(int(u * K), K - 1)]`. Integer axes flow through the
continuous path and are rounded by the env-side handler exactly as
LHS does.

## Backwards compatibility

Three rules, all preserved from the LHS PR:

1. Default `Suite.sampling` is still `"cartesian"`. Existing YAMLs
   parse and behave byte-identically.
2. The cartesian sampling path is the existing
   `itertools.product` enumeration, untouched.
3. The schema-level "n_samples required for non-cartesian" rule is
   unchanged. A YAML written for LHS at `n_samples = 32` works
   identically against `sampling: sobol` (modulo the actual draws).

The Runner and Report pipelines do not change. They consume
`list(suite.cells())` and key off `cell.index` — opaque to which
sampler produced the list.

## Determinism

Like LHS, Sobol is deterministic from the user-passed
`np.random.Generator`. The `Sampler` protocol takes the Generator
purely so the dispatch layer (`Suite.cells()`) keeps a single seeding
discipline; in practice the Sobol algorithm is fully deterministic
and ignores the entropy. Reproducibility is therefore **stronger**
than LHS: two runs of the same Suite produce byte-identical Sobol
draws even with different `Suite.seed` values, because the underlying
Sobol sequence is fixed.

## Test plan

* `tests/test_sobol_sampler.py` — replaces the deferral tests:
  * `sobol_unit_cube` shape, range `[0, 1)`, determinism, `skip`
    semantics.
  * 2D prefix matches the canonical Joe-Kuo Sobol prefix
    `(0, 0), (.5, .5), (.75, .25), (.25, .75), ...` (catches a
    transcription typo immediately).
  * Discrepancy: `np.std(np.histogram(points[:, i], bins=10)[0])`
    below 1.5 for `n_samples = 64`, `n_dims = 5` (well below the
    uniform-random ~2.4, well above the observed ~0.7 ceiling for
    a working Sobol).
  * Too-many-dimensions rejection with the documented error
    message.
  * `SobolSampler.sample(suite, rng)` matches the
    `LatinHypercubeSampler.sample` contract — same number of
    `SuiteCell` rows, same ranges, categorical axis coverage.
* `tests/test_suite_sampling.py` — the dispatch-layer test is
  updated so `sampling: sobol` actually returns `n_samples`
  cells (not `NotImplementedError`).
