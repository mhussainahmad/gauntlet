"""Sim-vs-real correlation report (B-28) — SureSim / SIMPLER inspired.

Consumes a directory of paired ``(sim_episode, real_episode)`` runs that
match on ``(suite_hash, cell_index, episode_index)`` and emits a per-axis
sim-real correlation table. Lets the user spot axes that "matter in sim
but not in real" (low |correlation|, divergent means) at a glance — the
gold-standard sim-to-real validity signal per SureSim (arxiv 2510.04354,
Oct 2025) and SIMPLER (arxiv 2405.05941, CoRL 2024).

Public surface:

* :class:`AxisTransfer` — one axis-row of the report.
* :class:`SimRealReport` — top-level meta-report.
* :func:`compute_sim_real_correlation` — pure function that loads two
  directories of episodes.json files, pairs them, and returns the report.

Pairing contract:

* The match key is ``(suite_hash, cell_index, episode_index)``.
* Episodes whose ``suite_hash`` is ``None`` on either side cannot be
  matched and are skipped. Their count is surfaced as
  :attr:`SimRealReport.n_unmatched_sim` /
  :attr:`SimRealReport.n_unmatched_real` so the caller knows the report
  did not silently drop data.
* Episodes from the real side are expected to carry
  ``Episode.source == "real"`` (and sim-side ``"sim"`` or ``None`` for
  legacy runs); the field is not validated here — the directory layout
  is the real contract — but the field exists for downstream consumers
  who want to keep both halves in one place.

Per-axis correlation:

* For each axis (key in ``perturbation_config``) and each cell, compute
  the cell's sim success rate and real success rate. The Pearson
  correlation across cells is the axis-level transferability signal.
* ``transferability_score`` is ``correlation ** 2`` (R^2 of the linear
  fit) when correlation is finite, else NaN.
* Degenerate inputs (n < 2 cells, zero variance on either side) return
  NaN correlation rather than crashing — the caller surfaces NaN as a
  dash in the rendered table.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gauntlet.runner.episode import Episode

__all__ = [
    "AxisTransfer",
    "SimRealReport",
    "compute_sim_real_correlation",
    "load_episodes_dir",
    "pair_episodes",
]


# ---------------------------------------------------------------------------
# Schema.
# ---------------------------------------------------------------------------


class AxisTransfer(BaseModel):  # type: ignore[explicit-any]
    """Per-axis sim-vs-real transferability summary.

    ``correlation`` is the Pearson correlation of (sim_rate, real_rate)
    across cells that vary this axis. ``sim_mean`` / ``real_mean`` are
    the mean per-cell success rates on each side — close means with low
    correlation is the SureSim "matters in sim, not in real" pattern.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    axis: str
    correlation: float
    transferability_score: float
    sim_mean: float
    real_mean: float
    n_paired_episodes: int


class SimRealReport(BaseModel):  # type: ignore[explicit-any]
    """Top-level sim-vs-real correlation report.

    Field order is signal-first (per spec §6 "never aggregate away
    failures"): ``per_axis`` is the breakdown surface above the scalar
    means and the unmatched-episode counters.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    per_axis: dict[str, AxisTransfer] = Field(default_factory=dict)
    overall_correlation: float
    n_paired_total: int
    n_unmatched_sim: int = 0
    n_unmatched_real: int = 0
    sim_runs_dir: str
    real_runs_dir: str


# ---------------------------------------------------------------------------
# Episode I/O.
# ---------------------------------------------------------------------------


def _discover_episode_files(directory: Path) -> list[Path]:
    """Recursively find files literally named ``episodes.json``."""
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    return sorted(directory.rglob("episodes.json"))


def load_episodes_dir(directory: Path) -> list[Episode]:
    """Load every ``episodes.json`` under *directory* into one flat list.

    Order is the path-sorted concatenation of files. Malformed files
    raise :class:`ValueError` with the relative path in the message.
    """
    files = _discover_episode_files(directory)
    if not files:
        raise ValueError(f"no episodes.json files found under {directory}")

    out: list[Episode] = []
    for path in files:
        rel = path.relative_to(directory).as_posix()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{rel}: invalid JSON ({exc.msg} at line {exc.lineno})") from exc
        if not isinstance(raw, list):
            raise ValueError(f"{rel}: expected top-level JSON list of episodes")
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"{rel}: episode {i} is not an object")
            try:
                out.append(Episode.model_validate(item))
            except ValidationError as exc:
                raise ValueError(f"{rel}: episode {i} failed validation: {exc}") from exc
    return out


# ---------------------------------------------------------------------------
# Pairing.
# ---------------------------------------------------------------------------


_MatchKey = tuple[str, int, int]


def _episode_key(ep: Episode) -> _MatchKey | None:
    """Return the ``(suite_hash, cell_index, episode_index)`` match key.

    ``None`` when ``suite_hash`` is missing — those episodes cannot be
    paired and are surfaced via the unmatched counters in the report.
    """
    if ep.suite_hash is None:
        return None
    return (ep.suite_hash, ep.cell_index, ep.episode_index)


def pair_episodes(
    sim_episodes: Iterable[Episode],
    real_episodes: Iterable[Episode],
) -> tuple[list[tuple[Episode, Episode]], int, int]:
    """Match sim/real episodes by ``(suite_hash, cell_index, episode_index)``.

    Returns ``(pairs, n_unmatched_sim, n_unmatched_real)``. Episodes
    with a ``None`` suite_hash on either side count as unmatched. When
    the same key appears on both sides multiple times (shouldn't, but
    is defended against) only the first occurrence on each side pairs;
    surplus copies count as unmatched.
    """
    sim_index: dict[_MatchKey, Episode] = {}
    sim_unmatched = 0
    for ep in sim_episodes:
        key = _episode_key(ep)
        if key is None or key in sim_index:
            sim_unmatched += 1
            continue
        sim_index[key] = ep

    pairs: list[tuple[Episode, Episode]] = []
    real_seen: set[_MatchKey] = set()
    real_unmatched = 0
    for ep in real_episodes:
        key = _episode_key(ep)
        if key is None or key in real_seen:
            real_unmatched += 1
            continue
        real_seen.add(key)
        sim_match = sim_index.pop(key, None)
        if sim_match is None:
            real_unmatched += 1
            continue
        pairs.append((sim_match, ep))

    sim_unmatched += len(sim_index)
    return pairs, sim_unmatched, real_unmatched


# ---------------------------------------------------------------------------
# Correlation primitives.
# ---------------------------------------------------------------------------


def _safe_pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation that returns NaN on degenerate input.

    ``statistics.correlation`` raises ``StatisticsError`` for n<2 or
    zero-variance inputs. Sim-vs-real reports have to render gracefully
    in those cases (a single-cell suite or a constant-success-rate axis
    is not a bug; it's just uninformative) so we trap and return NaN.
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return float("nan")


def _cell_success_rates(
    pairs: list[tuple[Episode, Episode]],
) -> tuple[list[float], list[float]]:
    """Reduce per-episode pairs to per-cell success rates on each side.

    Cells are keyed by ``(suite_hash, cell_index)`` — the same axis
    bucket on both sides by construction (pairs are matched on that
    prefix). The output lists are aligned: ``sim_rates[i]`` and
    ``real_rates[i]`` belong to the same cell.
    """
    by_cell: dict[tuple[str, int], list[tuple[bool, bool]]] = defaultdict(list)
    for sim_ep, real_ep in pairs:
        # ``suite_hash`` is non-None by construction (pair_episodes
        # would have rejected the pair otherwise) — assert for mypy.
        assert sim_ep.suite_hash is not None
        cell_key = (sim_ep.suite_hash, sim_ep.cell_index)
        by_cell[cell_key].append((sim_ep.success, real_ep.success))

    sim_rates: list[float] = []
    real_rates: list[float] = []
    for cell_key in sorted(by_cell.keys()):
        outcomes = by_cell[cell_key]
        n = len(outcomes)
        sim_rates.append(sum(1 for s, _ in outcomes if s) / n)
        real_rates.append(sum(1 for _, r in outcomes if r) / n)
    return sim_rates, real_rates


def _per_axis_transfer(
    pairs: list[tuple[Episode, Episode]],
) -> dict[str, AxisTransfer]:
    """Compute one :class:`AxisTransfer` per ``perturbation_config`` key.

    Per axis: for each *value bucket* of that axis (e.g.
    ``lighting_intensity=0.3``) compute the sim and real success rate
    across episodes that share that value. Pearson across the buckets
    is the axis-level correlation.
    """
    # Discover axis names from the sim side; both sides should agree
    # because they were paired on the same suite_hash, but the sim side
    # is the canonical reference. Use insertion order for determinism.
    axis_names: dict[str, None] = {}
    for sim_ep, _ in pairs:
        for k in sim_ep.perturbation_config:
            if k not in axis_names:
                axis_names[k] = None

    out: dict[str, AxisTransfer] = {}
    for axis in axis_names:
        # bucket episodes by the axis value rounded through the same
        # _norm semantics as the report layer (avoid 1e-15 jitter).
        buckets: dict[float, list[tuple[bool, bool]]] = defaultdict(list)
        for sim_ep, real_ep in pairs:
            value = sim_ep.perturbation_config.get(axis)
            if value is None:
                continue
            buckets[round(value, 12)].append((sim_ep.success, real_ep.success))

        sim_rates: list[float] = []
        real_rates: list[float] = []
        n_eps = 0
        for value in sorted(buckets.keys()):
            outcomes = buckets[value]
            n = len(outcomes)
            n_eps += n
            sim_rates.append(sum(1 for s, _ in outcomes if s) / n)
            real_rates.append(sum(1 for _, r in outcomes if r) / n)

        corr = _safe_pearson(sim_rates, real_rates)
        score = corr * corr if math.isfinite(corr) else float("nan")
        sim_mean = statistics.fmean(sim_rates) if sim_rates else float("nan")
        real_mean = statistics.fmean(real_rates) if real_rates else float("nan")
        out[axis] = AxisTransfer(
            axis=axis,
            correlation=corr,
            transferability_score=score,
            sim_mean=sim_mean,
            real_mean=real_mean,
            n_paired_episodes=n_eps,
        )
    return out


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def compute_sim_real_correlation(
    sim_runs_dir: Path,
    real_runs_dir: Path,
) -> SimRealReport:
    """Compute the sim-vs-real correlation report for two run directories.

    Discovers ``episodes.json`` recursively under each directory, pairs
    by ``(suite_hash, cell_index, episode_index)``, and reduces per
    axis. Episodes missing ``suite_hash`` are skipped (and counted in
    the unmatched totals on the report).

    Args:
        sim_runs_dir: directory of sim-side ``episodes.json`` files.
        real_runs_dir: directory of real-side ``episodes.json`` files.

    Returns:
        A fully populated :class:`SimRealReport`. ``overall_correlation``
        is NaN when fewer than two cells pair successfully.

    Raises:
        FileNotFoundError: if either directory does not exist.
        ValueError: if either directory contains no ``episodes.json``
            or any file is malformed.
    """
    sim_eps = load_episodes_dir(sim_runs_dir)
    real_eps = load_episodes_dir(real_runs_dir)

    pairs, n_unmatched_sim, n_unmatched_real = pair_episodes(sim_eps, real_eps)

    sim_cell_rates, real_cell_rates = _cell_success_rates(pairs)
    overall = _safe_pearson(sim_cell_rates, real_cell_rates)
    per_axis = _per_axis_transfer(pairs)

    return SimRealReport(
        per_axis=per_axis,
        overall_correlation=overall,
        n_paired_total=len(pairs),
        n_unmatched_sim=n_unmatched_sim,
        n_unmatched_real=n_unmatched_real,
        sim_runs_dir=str(sim_runs_dir),
        real_runs_dir=str(real_runs_dir),
    )
