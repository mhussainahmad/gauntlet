"""DriftReport / PerEpisodeDrift Pydantic schemas.

Torch-free. These models describe the sidecar ``drift.json`` the
``gauntlet monitor score`` command writes next to the
Phase-1 ``report.json``; the HTML template also imports them on the
torch-free install path to decide whether a drift panel can be
rendered. RFC §10 is the source of truth.

Public surface:

* :class:`PerEpisodeDrift` — one row per candidate episode.
* :class:`DriftReport` — the whole sidecar: suite identity + reference
  baselines + per-episode rows + top-k OOD pointer list.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = ["DriftReport", "PerEpisodeDrift"]


class PerEpisodeDrift(BaseModel):
    """Drift metrics for a single candidate-sweep episode.

    Field order mirrors RFC §10. The autoencoder error fields describe
    the observation side; ``action_std_per_dim`` + ``action_entropy``
    describe the action side.

    Attributes:
        cell_index: Suite cell index (echoed from Episode.cell_index
            for cross-referencing).
        episode_index: Zero-based index within the cell.
        seed: Env seed (echoed from Episode.seed).
        perturbation_config: Perturbation values for this cell.
        n_steps: Episode length in env steps. Can be zero for very
            early-terminated rollouts.
        reconstruction_error_mean: Mean per-step L2 reconstruction
            error. The primary OOD signal.
        reconstruction_error_max: Per-step max over the same trajectory
            — surfaces spike-shaped anomalies aggregated-mean hides.
        action_std_per_dim: Length-7 list of per-dim action stds (see
            RFC §5). Population std (``ddof=0``).
        action_entropy: Mean of :attr:`action_std_per_dim`; the scalar
            plotted on the HTML panel.
    """

    model_config = ConfigDict(extra="forbid")

    cell_index: int
    episode_index: int
    seed: int
    perturbation_config: dict[str, float]
    n_steps: int

    reconstruction_error_mean: float
    reconstruction_error_max: float

    action_std_per_dim: list[float]
    action_entropy: float


class DriftReport(BaseModel):
    """Sidecar ``drift.json`` — one analysis of one candidate sweep.

    Two groups of fields drive the HTML panel:

    * The reference baselines (``reference_reconstruction_error_mean``
      and ``reference_reconstruction_error_p95``) are copied from the
      AE checkpoint's ``normalization.json`` so the renderer can draw
      the p95 threshold line without re-loading the AE.
    * The candidate aggregates echo the summary across all
      ``per_episode`` rows.

    ``top_ood_episodes`` holds the indices (into ``per_episode``) of the
    top-k most-OOD episodes in descending error order — the table the
    HTML panel renders directly. The default top_k lives in
    :mod:`gauntlet.monitor.score`.
    """

    model_config = ConfigDict(extra="forbid")

    suite_name: str
    n_episodes: int
    ae_mode: Literal["state", "image"]
    ae_latent_dim: int
    ae_reference_suite: str | None

    # Baselines from the AE checkpoint — echoed here so the HTML panel
    # has a single file to read.
    reference_reconstruction_error_mean: float
    reference_reconstruction_error_p95: float

    # Candidate-sweep aggregates.
    candidate_reconstruction_error_mean: float
    candidate_reconstruction_error_p95: float
    candidate_action_entropy_mean: float

    per_episode: list[PerEpisodeDrift]
    top_ood_episodes: list[int]
