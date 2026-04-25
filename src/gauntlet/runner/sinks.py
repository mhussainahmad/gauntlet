"""Optional W&B / MLflow per-Episode mirror sinks (backlog B-27).

This module exists ONLY so users who already live inside Weights & Biases
or MLflow can mirror per-Episode results into their existing dashboards.
**Both sinks are off by default** (the corresponding ``--wandb`` /
``--mlflow`` CLI flags default to ``False``); a default ``gauntlet run``
invocation never imports either backend, never touches the network, and
never references either package.

Anti-feature warning (PRODUCT.md, "no cloud, no telemetry, no internet
round-trip required at view time"). When opted in:

* :class:`WandbSink` exfiltrates per-Episode metrics to ``wandb.ai``
  unless the user has explicitly redirected via ``WANDB_BASE_URL``
  (self-hosted W&B) or ``WANDB_MODE=offline`` (local-only).
* :class:`MlflowSink` writes to a local ``./mlruns/`` directory by
  default (no network) — but **becomes a remote sink the moment the
  user sets the standard ``MLFLOW_TRACKING_URI`` env var** to a remote
  HTTP(S) endpoint. Of the two sinks, MLflow is the local-first fit;
  W&B has no offline-by-default mode.

Both sinks lazy-import their backend inside ``__init__`` so a user who
does not pass the corresponding flag never pays the import cost (wandb
ships a transitive ``gql`` graph + a SQLite client; mlflow ships
sqlalchemy + scikit-learn-style numpy machinery).

Sink failures are warn-and-continue rather than fatal: a network blip on
``log_episode`` should never lose the run's local ``episodes.json`` /
``report.html``. The runner is the source of truth; the sink is the
mirror.

Common interface:

* ``__init__(*, run_name, suite_name, config)`` — opens the backend run.
* ``log_episode(episode)`` — mirrors the per-Episode metrics. Best-
  effort; warns on failure.
* ``close()`` — finishes the backend run. Best-effort; warns on
  failure.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Protocol, runtime_checkable

from gauntlet.runner.episode import Episode

__all__ = ["EpisodeSink", "MlflowSink", "WandbSink"]


@runtime_checkable
class EpisodeSink(Protocol):
    """Common interface every per-Episode sink satisfies.

    The runner does not care which backend is wired — both sinks emit
    the same per-Episode metric dict from
    :func:`_episode_metrics`.
    """

    def log_episode(self, episode: Episode) -> None:
        """Mirror one :class:`Episode`'s metrics to the backend."""
        ...

    def close(self) -> None:
        """Finalise the backend run; release any open file handles."""
        ...


def _episode_metrics(episode: Episode) -> dict[str, Any]:
    """Flatten an :class:`Episode` into a backend-agnostic metric dict.

    Picks only the fields that are meaningful as time-series scalars
    (success, reward, steps) plus the per-axis perturbation config
    flattened to ``axis.<name>`` keys so a per-cell scatter is plottable
    in either UI without server-side joining. The full Episode is the
    durable artifact in ``episodes.json``; the sink is a mirror, not a
    source of truth.
    """
    metrics: dict[str, Any] = {
        "cell_index": episode.cell_index,
        "episode_index": episode.episode_index,
        "seed": episode.seed,
        "success": int(episode.success),
        "total_reward": episode.total_reward,
        "step_count": episode.step_count,
    }
    for axis_name, axis_value in episode.perturbation_config.items():
        metrics[f"axis.{axis_name}"] = axis_value
    return metrics


class WandbSink:
    """Mirror per-Episode metrics to Weights & Biases (``wandb.ai``).

    WARNING: opting in to this sink exfiltrates per-Episode results to
    ``wandb.ai`` over the network unless the user has set
    ``WANDB_BASE_URL`` to a self-hosted W&B endpoint or
    ``WANDB_MODE=offline``. This directly contradicts PRODUCT.md's
    local-first / no-telemetry contract; the sink only exists so users
    who already live in W&B can mirror gauntlet results into their
    existing dashboards. **The sink is off by default**; the user opts
    in explicitly via ``gauntlet run --wandb``.

    Lazy-imports :mod:`wandb` inside ``__init__`` so the default
    extras-free install never touches the package.
    """

    def __init__(
        self,
        *,
        run_name: str,
        suite_name: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - defensive guard
            raise ImportError(
                "WandbSink requires the optional [wandb] extra. "
                'Install with `pip install "gauntlet[wandb]"`. '
                "WARNING: this sink exfiltrates per-Episode results to "
                "wandb.ai unless WANDB_BASE_URL is set."
            ) from exc

        self._wandb = wandb
        self._run = wandb.init(
            project="gauntlet",
            name=run_name,
            group=suite_name,
            config=config or {},
            reinit=True,
        )

    def log_episode(self, episode: Episode) -> None:
        """Mirror one Episode's metrics. Failures warn-and-continue."""
        try:
            self._wandb.log(_episode_metrics(episode))
        except Exception as exc:
            warnings.warn(
                f"WandbSink.log_episode failed (continuing run): {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )

    def close(self) -> None:
        """Finish the W&B run. Failures warn-and-continue."""
        try:
            if self._run is not None:
                self._run.finish()
        except Exception as exc:
            warnings.warn(
                f"WandbSink.close failed (continuing): {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )


class MlflowSink:
    """Mirror per-Episode metrics to MLflow.

    Uses MLflow's local file backend by default — writes to
    ``./mlruns/`` next to the current working directory, no network
    round-trip. **Becomes a remote sink the moment the user exports
    ``MLFLOW_TRACKING_URI`` pointing at an HTTP(S) endpoint** (an
    upstream MLflow tracking server, Databricks, etc.). The sink does
    NOT inspect the URI — MLflow's own client honours it transparently
    — but PRODUCT.md's local-first contract requires we name the
    behaviour explicitly: opting in to ``--mlflow`` plus a remote
    ``MLFLOW_TRACKING_URI`` exfiltrates per-Episode results.

    Lazy-imports :mod:`mlflow` inside ``__init__`` so the default
    extras-free install never touches the package.
    """

    def __init__(
        self,
        *,
        run_name: str,
        suite_name: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:  # pragma: no cover - defensive guard
            raise ImportError(
                "MlflowSink requires the optional [mlflow] extra. "
                'Install with `pip install "gauntlet[mlflow]"`. '
                "Local-by-default: writes to ./mlruns/ unless "
                "MLFLOW_TRACKING_URI is set to a remote endpoint, in "
                "which case per-Episode results leave the machine."
            ) from exc

        self._mlflow = mlflow
        # Surface the resolved tracking URI in the warning stream so a
        # user who set MLFLOW_TRACKING_URI to a remote endpoint sees
        # the destination on stderr before any data leaves the box.
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "<local ./mlruns>")
        if tracking_uri.startswith(("http://", "https://")):
            warnings.warn(
                f"MlflowSink: MLFLOW_TRACKING_URI={tracking_uri!r} is "
                "remote — per-Episode results will leave this machine.",
                RuntimeWarning,
                stacklevel=2,
            )
        mlflow.set_experiment(suite_name)
        self._run = mlflow.start_run(run_name=run_name)
        if config:
            try:
                mlflow.log_params({str(k): v for k, v in config.items()})
            except Exception as exc:
                warnings.warn(
                    f"MlflowSink.log_params failed (continuing run): {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def log_episode(self, episode: Episode) -> None:
        """Mirror one Episode's metrics. Failures warn-and-continue."""
        # Use a monotonically increasing step id so MLflow's history
        # plot stays in episode order across re-runs.
        step = episode.cell_index * 10_000 + episode.episode_index
        try:
            metrics = _episode_metrics(episode)
            # MLflow's ``log_metrics`` only accepts numeric values; the
            # ``axis.*`` floats and integer ``cell_index`` / ``seed``
            # already satisfy that, so the dict goes through unchanged.
            self._mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items()},
                step=step,
            )
        except Exception as exc:
            warnings.warn(
                f"MlflowSink.log_episode failed (continuing run): {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )

    def close(self) -> None:
        """End the MLflow run. Failures warn-and-continue."""
        try:
            self._mlflow.end_run()
        except Exception as exc:
            warnings.warn(
                f"MlflowSink.close failed (continuing): {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
