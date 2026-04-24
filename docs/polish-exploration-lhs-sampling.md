# Polish exploration — Latin-hypercube + Sobol perturbation sampling

Status: exploration. Targets `Suite.sampling: cartesian | latin_hypercube |
sobol`. Scope: ship LHS, defer Sobol.

## Why this matters

The current `Suite` schema enumerates a Cartesian grid: `steps_per_axis ^
n_axes` cells. Five axes at five steps each = 3,125 cells; ten episodes
each = 31,250 rollouts before a single bug surfaces. That's the
combinatorial wall any robotics-eval harness hits as soon as users add a
fifth interesting axis.

Latin Hypercube Sampling (McKay/Beckman/Conover, 1979) and Sobol
sequences (low-discrepancy quasi-random) cover the same parameter
hypercube with **dramatically fewer samples and provably better marginal
coverage**. The empirical headline:

* 5 axes x 5 steps cartesian = 3,125 cells.
* LHS at `n_samples = 64` covers every axis at 64 distinct strata —
  marginal coverage strictly *higher* than the cartesian grid (5
  strata per axis), with `3125 / 64 = 49x` fewer rollouts.

Adding `sampling: latin_hypercube` as opt-in lets users explore failure
modes that a 5x5x5x5x5 grid would never reach in a sitting. The default
remains `cartesian` so every existing suite YAML (and every test that
pins on cell ordering) keeps its current behaviour byte-for-byte.

## Algorithm sketch (LHS, McKay 1979)

For `N` samples across `D` axes, each axis gets the unit interval
`[0, 1]` divided into `N` equal-width strata. Per axis:

1. Draw one uniform sample inside each stratum:
   `u_i = (i + rng.uniform()) / N` for `i in range(N)`.
2. Randomly permute the `N` samples (independently per axis).
3. The result is `N` points in `[0, 1]^D` such that *every axis* has
   exactly one point per stratum — perfect marginal stratification.

In numpy:

```python
def lhs_unit(n_samples: int, n_axes: int, rng: np.random.Generator) -> np.ndarray:
    cuts = (np.arange(n_samples) + rng.uniform(size=(n_samples, n_axes))) / n_samples
    for axis in range(n_axes):
        rng.shuffle(cuts[:, axis])
    return cuts  # shape (n_samples, n_axes), values in [0, 1)
```

That's the entire algorithm — six lines. Reproducibility is one-seed:
the same `np.random.Generator` produces the same matrix.

## Mapping unit-cube samples back onto AxisSpec values

Each axis value `u in [0, 1)` maps onto the axis spec by shape:

* **Continuous `{low, high}`**: `low + u * (high - low)`. Inclusive
  lower, exclusive upper — matches `np.random.Generator.uniform(low,
  high)` semantics. Acceptable: LHS strata are never *at* the bounds.
* **Integer (`distractor_count`)**: same affine map, then
  `int(round(value))`. Equivalently `low + int(u * (high - low + 1))` so
  every integer in `[low, high]` gets equal-width strata. We pick the
  affine-then-round form to keep code simple; the `round` collapses
  ties at half-integers but that's OK at our cardinalities.
* **Categorical `{values: [...]}`**: `values[min(int(u * K), K - 1)]`
  where `K = len(values)`. Each category gets one `1/K`-wide stratum.
  With LHS at `N >= K` every category is hit exactly `N // K` times
  (modulo permutation), so a 2-value categorical still has full marginal
  coverage.

The `AxisSpec` shape is the only source of truth — we don't need to
consult `gauntlet.env.perturbation` for kind information. Continuous vs
categorical is recoverable from `(low/high/steps) is None` vs
`values is None`.

## Sobol — defer

Sobol sequences (low-discrepancy quasi-random) would give *even better*
star-discrepancy than LHS at the same `n_samples`. The pure-numpy
implementation requires Joe-Kuo direction numbers (a public-domain table
of ~21,000 integers for the first 1,000 dimensions) plus the Gray-code
sequence generator and a Brownian shuffle scrambler.

That's roughly 200-400 lines of new code, plus tests for orthogonality,
balanced subsets, and direction-number table integrity. The risk of
shipping a subtly wrong Sobol implementation (where every test passes
but the discrepancy guarantee is broken) is real and not worth the
schedule risk for this PR.

`scipy.stats.qmc.Sobol` would be a one-liner *if* scipy were a core
dependency — it isn't, and adding scipy purely for Sobol would balloon
the install size by an order of magnitude.

**Decision**: this PR ships LHS only. `sampling: sobol` is accepted by
the schema (so the YAML grammar is forward-compatible) and raises
`NotImplementedError("Sobol sampler is planned for a follow-up PR; LHS
is supported")` when constructed. A follow-up PR can land Sobol without
any schema migration.

## Backwards compatibility

Three rules:

1. Default value of `Suite.sampling` is `"cartesian"`. Existing YAMLs
   parse and behave identically.
2. The cartesian sampling path is the existing
   `itertools.product(*per_axis_enumerations)` enumeration, untouched.
   A regression test pins the exact `(index, dict(values))` sequence
   produced by the shipped `examples/suites/tabletop-basic-v1.yaml`.
3. `AxisSpec` validation is loosened in one place only: a continuous
   axis (`{low, high}`) may omit `steps` *if* the parent Suite uses a
   non-cartesian sampling mode. Cartesian suites still require all
   three. The Suite-level `model_validator` enforces this, so the error
   reads "axis foo: steps is required when sampling=cartesian" rather
   than "axis spec malformed".

The Runner does not change. It calls `list(suite.cells())` and the
sampler decides what that list contains. Per-cell seed derivation,
ordering, and the WorkItem shape are untouched.

## Sampler protocol

```python
class Sampler(Protocol):
    def sample(self, suite: Suite, rng: np.random.Generator) -> list[SuiteCell]: ...
```

Three concrete implementations:

* `CartesianSampler` — wraps the existing `itertools.product` logic.
* `LatinHypercubeSampler` — implements the algorithm above. Emits
  `n_samples` cells, each a `SuiteCell(index=i, values={...})`.
* `SobolSampler` — raises `NotImplementedError` until the follow-up.

Dispatch lives in `Suite.cells()` keyed off `self.sampling`. The RNG is
seeded from `suite.seed` (existing field; `None` means OS entropy via
`np.random.SeedSequence`).

## Open questions

* **Should LHS pick `n_samples` automatically when omitted?** No — the
  whole point of opt-in non-cartesian sampling is that the user picks
  the budget. Validator rejects `sampling != "cartesian"` without
  `n_samples`.
* **Should non-cartesian suites accept `steps` and ignore it?** No.
  Reject explicitly. A user who writes `steps: 5` under
  `sampling: latin_hypercube` is confused; the loader should say so.
  *(Implementation note: this is enforced in the same Suite-level
  validator that requires `steps` for cartesian.)*
* **Does the Runner need to know which sampling mode produced a cell?**
  No. The `SuiteCell` API is `(index, values)`; the sampler is opaque
  past the suite layer. Per-cell trajectory dumps and reports key off
  `cell.index` exactly as before.

## Test plan

1. `tests/test_suite_sampling.py` — `CartesianSampler` produces
   byte-identical cells to the current `Suite.cells()` for the shipped
   smoke YAML (regression pin).
2. `tests/test_lhs_sampler.py` — determinism, stratum coverage, range
   bounds, categorical/integer handling.
3. `tests/test_sobol_sampler.py` — `sampling: sobol` raises a clear
   `NotImplementedError`.
4. Loader regression: existing `examples/suites/*.yaml` still load and
   their cell sequences match a pre-recorded golden.
