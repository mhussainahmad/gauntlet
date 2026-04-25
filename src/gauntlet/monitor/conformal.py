"""Conformal-calibrated failure-prediction detector — backlog B-01.

Drop-in companion to the AE drift detector that already lives in
:mod:`gauntlet.monitor`. The detector calibrates a per-policy threshold
on the :attr:`gauntlet.runner.episode.Episode.action_variance` field
shipped by B-18 (PR #70) and surfaces, per candidate episode:

* :attr:`gauntlet.runner.episode.Episode.failure_score` — the candidate
  episode's ``action_variance`` divided by the calibration quantile.
  Greater than 1.0 means "more uncertain than the calibration set was at
  this confidence level"; less than 1.0 means "in distribution".
* :attr:`gauntlet.runner.episode.Episode.failure_alarm` — the boolean
  flag, ``True`` iff ``failure_score > 1.0``.

The statistical machinery is *split conformal prediction* (Vovk,
Gammerman, Shafer 2005) applied to a single non-conformity score: the
per-episode action variance. Given a confidence level ``1 - alpha`` and
``n`` calibration episodes the threshold is the
``ceil((n + 1) * (1 - alpha)) / n``-quantile of the calibration scores
— this is the Romano-style finite-sample correction used by FIPER
(arxiv 2510.09459, NeurIPS 2025) and FAIL-Detect (arxiv 2503.08558),
and it is what gives the false-alarm-rate guarantee under the i.i.d.
exchangeability assumption.

Asymmetry (anti-feature, documented in ``docs/backlog.md`` B-01): the
detector is only meaningful for policies that implement the
``SamplablePolicy`` Protocol (B-18). Greedy / open-loop policies leave
``Episode.action_variance`` ``None``, in which case both
:meth:`ConformalFailureDetector.fit` and
:meth:`ConformalFailureDetector.score` skip / propagate the ``None``
rather than fabricate a 0.0.

Pure numpy. No torch, no scipy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gauntlet.runner.episode import Episode

__all__ = ["ConformalFailureDetector"]


# Schema version stamped into the persisted detector JSON. Bump when the
# on-disk shape changes incompatibly so a stale file fails fast on load.
_SCHEMA_VERSION: int = 1


@dataclass
class ConformalFailureDetector:
    """Split-conformal failure detector keyed off ``Episode.action_variance``.

    Two-step lifecycle:

    1. :meth:`fit` consumes a calibration set of *successful* Episodes
       (the assumption being that successful rollouts define the
       in-distribution non-conformity score under the policy / env
       pair). It computes the conformal quantile threshold and stores
       it on ``self``.
    2. :meth:`score` consumes one candidate Episode and returns the
       ``(failure_score, failure_alarm)`` pair. The score is the
       candidate's ``action_variance`` divided by the threshold; the
       alarm is ``failure_score > 1.0``.

    Both ``alpha`` (target false-positive rate) and ``threshold`` are
    serialised by :meth:`to_dict` so the detector round-trips through
    JSON without losing the calibration set itself — only the threshold
    matters at score time.
    """

    alpha: float
    threshold: float
    n_calibration: int

    # ------------------------------------------------------------------
    # Construction.
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        calibration_episodes: list[Episode],
        alpha: float = 0.05,
    ) -> ConformalFailureDetector:
        """Fit a detector on the calibration set.

        Episodes whose ``action_variance is None`` are skipped (they
        come from greedy policies that did not opt into the B-18
        sampler) — see the asymmetry note in the module docstring.

        ``alpha`` is the target false-positive rate; the threshold is
        the ``ceil((n + 1) * (1 - alpha)) / n``-quantile of the
        remaining ``action_variance`` values, computed with
        ``method="higher"`` so the cut-off lands on an actual
        calibration sample (the conservative choice on small ``n``).

        Raises:
            ValueError: ``alpha`` outside ``(0, 1)``, no usable
                calibration episodes, or the implied conformal quantile
                level exceeds 1 (``n`` too small for the requested
                ``alpha``).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")

        scores = [
            float(ep.action_variance)
            for ep in calibration_episodes
            if ep.action_variance is not None
        ]
        n = len(scores)
        if n == 0:
            raise ValueError(
                "calibration set has no episodes with action_variance — "
                "fit a stochastic / chunked policy (SamplablePolicy) and "
                "set Runner(measure_action_consistency=True)",
            )

        # Romano-style finite-sample correction: the conformal quantile
        # level is ``ceil((n + 1) * (1 - alpha)) / n``. Clamp to <= 1.0;
        # if the level lands above 1.0 we cannot give a finite-sample
        # guarantee at the requested ``alpha`` and ``n``.
        rank = math.ceil((n + 1) * (1.0 - alpha))
        if rank > n:
            raise ValueError(
                f"calibration set too small (n={n}) for alpha={alpha}; "
                f"need n >= ceil(1/alpha) - 1 = {math.ceil(1.0 / alpha) - 1}",
            )
        quantile_level = rank / n
        threshold = float(
            np.quantile(np.asarray(scores, dtype=np.float64), quantile_level, method="higher"),
        )

        return cls(alpha=alpha, threshold=threshold, n_calibration=n)

    # ------------------------------------------------------------------
    # Scoring.
    # ------------------------------------------------------------------

    def score(self, episode: Episode) -> tuple[float | None, bool | None]:
        """Score a single candidate episode.

        Returns ``(failure_score, failure_alarm)``:

        * ``failure_score = action_variance / threshold`` — values > 1.0
          mean "more uncertain than the calibration set at this
          confidence", values < 1.0 mean "in distribution".
        * ``failure_alarm = failure_score > 1.0`` — the boolean flag.

        Both halves are ``None`` when the candidate's
        ``action_variance`` is ``None`` (greedy-policy asymmetry; see
        module docstring). The CLI persists the ``None`` straight onto
        the Episode rather than fabricating a 0.0 / False that would
        look like a measurement.

        A zero ``threshold`` (calibration set fully collapsed onto a
        single mode) collapses the score to ``+inf`` for any positive
        candidate variance and ``0.0`` for an exactly-zero candidate;
        the alarm fires in the former case, which matches the intent
        ("threshold of zero ⇒ everything above zero is anomalous").
        """
        var = episode.action_variance
        if var is None:
            return (None, None)

        if self.threshold == 0.0:
            if var == 0.0:
                return (0.0, False)
            return (math.inf, True)

        score = float(var) / self.threshold
        alarm = score > 1.0
        return (score, alarm)

    # ------------------------------------------------------------------
    # Persistence.
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, float | int]:
        """Serialise to a JSON-safe dict.

        Only the threshold and metadata round-trip — the calibration
        set itself is intentionally not retained, matching the conformal
        contract that the threshold is the only sufficient statistic.
        """
        return {
            "schema_version": _SCHEMA_VERSION,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "n_calibration": self.n_calibration,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float | int]) -> ConformalFailureDetector:
        """Inverse of :meth:`to_dict`. Validates the schema version."""
        version = int(payload.get("schema_version", 0))
        if version != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported detector schema_version={version} (expected {_SCHEMA_VERSION})",
            )
        return cls(
            alpha=float(payload["alpha"]),
            threshold=float(payload["threshold"]),
            n_calibration=int(payload["n_calibration"]),
        )

    def save(self, path: Path) -> None:
        """Write :meth:`to_dict` to ``path`` as UTF-8 JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> ConformalFailureDetector:
        """Inverse of :meth:`save`."""
        if not path.is_file():
            raise FileNotFoundError(f"detector file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: top-level JSON must be an object")
        return cls.from_dict(payload)
