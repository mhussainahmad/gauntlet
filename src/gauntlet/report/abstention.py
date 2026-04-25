"""Calibration-aware abstention scoring (B-04).

Selective-prediction metrics over a list of :class:`Episode` whose B-01
conformal failure detector has already populated
:attr:`Episode.failure_score` and :attr:`Episode.failure_alarm`. The
single public entry point is :func:`compute_abstention_metrics`.

Convention notes — read these before changing the AURC computation:

* The literature's standard AURC (Geifman & El-Yaniv 2017) integrates
  *risk* (= 1 - accuracy) over coverage; perfect detectors trend to
  ``0`` and useless detectors trend to the dataset error rate. The
  spec under which this module ships (``docs/backlog.md`` B-04) calls
  for the *complementary* convention — area under the
  coverage-accuracy curve — so a perfect detector reads HIGH (≈ 1)
  and a useless detector reads ≈ overall success rate (≈ 0.5 on a
  balanced split). This is the FIPER (arxiv 2510.09459) framing and
  matches the report's existing "higher == better" colour vocabulary.
  ``aurc`` here is therefore ``1 - standard_AURC``; do not "fix" it
  back to the textbook definition without also flipping the HTML
  colour palette.
* Episodes with ``failure_score is None`` (greedy-policy episodes —
  see the B-18 asymmetry on :class:`gauntlet.runner.Episode`) are
  dropped from BOTH numerator and denominator so a partial-coverage
  dataset (mixed sampleable + greedy policies) is not biased toward
  zero. Returns ``None`` only when NO episode in the dataset carried
  a score — the "B-01 never ran" gate.
* When every episode triggered ``failure_alarm`` (the policy
  abstained on its entire input),
  ``abstention_corrected_success_rate`` is ``None`` rather than a
  divide-by-zero or a misleading ``0.0`` — the kept set is empty and
  the metric is genuinely undefined.

Refs: FIPER (arxiv 2510.09459, NeurIPS 2025), FAIL-Detect (arxiv
2503.08558), LangForce (arxiv 2601.15197), surgical UQ (arxiv
2501.10561).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from gauntlet.report.schema import AbstentionMetrics
from gauntlet.runner.episode import Episode

__all__ = ["compute_abstention_metrics"]


def compute_abstention_metrics(
    episodes: Iterable[Episode],
) -> AbstentionMetrics | None:
    """Compute selective-prediction metrics over a list of episodes.

    Returns ``None`` when no episode in ``episodes`` carries a
    populated ``failure_score`` — i.e. the B-01 conformal detector
    never ran on this dataset. This is the gate documented in
    ``docs/backlog.md`` B-04: the metric is only meaningful when
    upstream calibration has run AND we trust its calibration set.

    Episodes with ``failure_score is None`` are dropped from the
    score-derived metrics (AURC); they still count toward the
    abstention-corrected rate's denominator only if they carry a
    ``failure_alarm`` flag, which by construction they do not (B-01
    sets both fields together) — so they end up dropped from both,
    matching the partial-coverage handling on the rest of the report.

    Args:
        episodes: Iterable of :class:`Episode`. May mix scored and
            un-scored episodes (greedy policies leave both fields
            ``None``).

    Returns:
        :class:`AbstentionMetrics` when at least one episode is
        scored; ``None`` otherwise.
    """
    episodes = list(episodes)
    scored = [ep for ep in episodes if ep.failure_score is not None]
    if not scored:
        return None

    n_with_score = len(scored)

    # ── AURC (coverage-accuracy convention) ────────────────────────
    # Sort scored episodes ascending by score (lowest = most
    # confident "in distribution"). At each prefix length k we keep
    # the k most-confident episodes and measure their accuracy. We
    # then trapezoidal-integrate accuracy over coverage = k / N.
    scored_sorted = sorted(scored, key=lambda ep: float(ep.failure_score or 0.0))
    successes = np.array([1.0 if ep.success else 0.0 for ep in scored_sorted], dtype=np.float64)
    cum_successes = np.cumsum(successes)
    k = np.arange(1, n_with_score + 1, dtype=np.float64)
    accuracy_at_k = cum_successes / k
    coverage = k / n_with_score
    # Trapezoidal integration over coverage ∈ (0, 1]. We anchor the
    # left edge at coverage=0 with the same accuracy as the first
    # kept episode so a degenerate single-episode dataset still
    # integrates to a defined value (the single accuracy itself).
    coverage_axis = np.concatenate([[0.0], coverage])
    accuracy_axis = np.concatenate([[float(accuracy_at_k[0])], accuracy_at_k])
    aurc = float(np.trapezoid(accuracy_axis, coverage_axis))

    # ── Abstention-corrected success rate ─────────────────────────
    # The "what the policy delivers when it knows when to shut up"
    # number. Episodes with ``failure_alarm is True`` are the ones
    # the policy refused to commit to; success rate is computed over
    # the rest. ``None`` when every episode triggered the alarm
    # (kept set empty — the metric is undefined, not zero).
    kept = [ep for ep in scored if ep.failure_alarm is not True]
    abstention_corrected: float | None
    if kept:
        n_kept_success = sum(1 for ep in kept if ep.success)
        abstention_corrected = n_kept_success / len(kept)
    else:
        abstention_corrected = None

    # ── Confusion-style abstention counts ────────────────────────
    # Selective-prediction's true/false positive split, where the
    # "positive" event is "abstained from this episode". A correctly
    # abstained episode is one that would have failed; a falsely
    # abstained one is one that would have succeeded — the policy
    # gave up on a good rollout. Both counts are over the scored
    # subset; alarm without a score is not a real B-01 output.
    n_correctly_abstained = sum(1 for ep in scored if ep.failure_alarm is True and not ep.success)
    n_falsely_abstained = sum(1 for ep in scored if ep.failure_alarm is True and ep.success)

    return AbstentionMetrics(
        n_with_score=n_with_score,
        aurc=aurc,
        abstention_corrected_success_rate=abstention_corrected,
        n_correctly_abstained=n_correctly_abstained,
        n_falsely_abstained=n_falsely_abstained,
    )
