"""Score candidate trajectories against a trained AE.

Torch-only. Consumes:

* A JSON :class:`~gauntlet.runner.Episode` list (``episodes.json``).
* A candidate trajectory directory (NPZ sidecars from
  ``gauntlet run --record-trajectories``).
* A trained AE checkpoint directory (from
  :func:`gauntlet.monitor.train.train_ae`).

Emits a :class:`~gauntlet.monitor.schema.DriftReport`. The orchestrator
delegates heavy lifting: the AE per-row scoring lives on
:class:`StateAutoencoder`, action entropy lives in
:mod:`gauntlet.monitor.entropy`, JSON I/O is stdlib.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

_MONITOR_INSTALL_HINT = (
    "gauntlet.monitor.score requires the 'monitor' extra. Install with:\n"
    "    uv sync --extra monitor\n"
    "or, for a plain pip env:\n"
    "    pip install 'gauntlet[monitor]'"
)

try:
    import torch  # noqa: F401  # import here so missing extras fail early
except ImportError as exc:  # pragma: no cover
    raise ImportError(_MONITOR_INSTALL_HINT) from exc

from gauntlet.monitor.ae import (
    STATE_OBS_KEYS,
    LoadedAutoencoder,
    load_state_autoencoder,
)
from gauntlet.monitor.entropy import action_entropy
from gauntlet.monitor.schema import DriftReport, PerEpisodeDrift
from gauntlet.monitor.train import flatten_obs_arrays
from gauntlet.runner import Episode
from gauntlet.runner.worker import trajectory_path_for

__all__ = ["score_drift"]


def _load_episodes(episodes_path: Path) -> list[Episode]:
    """Read and validate ``episodes.json``; raise with a clean message on error."""
    if not episodes_path.is_file():
        raise FileNotFoundError(f"episodes file not found: {episodes_path}")
    raw = json.loads(episodes_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"{episodes_path}: top-level JSON must be a list of episodes; got {type(raw).__name__}",
        )
    try:
        return [Episode.model_validate(row) for row in raw]
    except ValidationError as exc:
        raise ValueError(f"{episodes_path}: invalid episodes.json: {exc}") from exc


def _score_one_episode(
    loaded: LoadedAutoencoder,
    obs_arrays: dict[str, NDArray[np.float64]],
    actions: NDArray[np.float64],
) -> tuple[float, float, NDArray[np.float64], float]:
    """Return ``(recon_mean, recon_max, per_dim_std, action_entropy)`` for one episode.

    Separated from the orchestrator so the test suite can drive it
    directly with hand-built obs/action arrays.
    """
    flat = flatten_obs_arrays(obs_arrays)
    if flat.shape[0] == 0:
        # Zero-step rollout (very early termination) — per RFC §10 both
        # reconstruction stats are zero and action entropy collapses to
        # zero as well. We guard the zero-step case explicitly so
        # action_entropy's "T >= 1" precondition is respected.
        per_dim_std = np.zeros(7, dtype=np.float64)
        return 0.0, 0.0, per_dim_std, 0.0
    normalised = (flat - loaded.normalization.mean) / loaded.normalization.std
    per_row_error = loaded.ae.per_sample_score(normalised)
    recon_mean = float(per_row_error.mean())
    recon_max = float(per_row_error.max())
    # Action entropy runs on the raw action trajectory — the RFC §5
    # metric is defined in the action space, not post-normalisation.
    if actions.shape[0] == 0:
        per_dim_std = np.zeros(actions.shape[1] if actions.ndim == 2 else 7, dtype=np.float64)
        action_scalar = 0.0
    else:
        stats = action_entropy(actions.astype(np.float64, copy=False))
        per_dim_std = stats.per_dim_std
        action_scalar = stats.scalar
    return recon_mean, recon_max, per_dim_std, action_scalar


def score_drift(
    episodes_path: Path,
    trajectory_dir: Path,
    ae_dir: Path,
    *,
    top_k: int = 10,
) -> DriftReport:
    """Analyse a candidate sweep and return a :class:`DriftReport`.

    Args:
        episodes_path: Path to ``episodes.json`` (the Phase-1 artefact).
        trajectory_dir: Directory with the candidate sweep's per-episode
            NPZ sidecars. Must contain one NPZ per Episode — matched by
            the deterministic filename scheme
            :func:`gauntlet.runner.worker.trajectory_path_for`.
        ae_dir: Trained AE checkpoint (see
            :func:`gauntlet.monitor.train.train_ae`).
        top_k: How many "most OOD" episode indices to return on
            :attr:`DriftReport.top_ood_episodes`.

    Raises:
        FileNotFoundError: If ``episodes.json``, any per-episode NPZ,
            or any AE artefact is missing.
        ValueError: If an NPZ's identity scalars disagree with its
            filename (the defensive cross-check from RFC §6).
    """
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0; got {top_k}")

    episodes = _load_episodes(episodes_path)
    loaded = load_state_autoencoder(ae_dir)

    per_episode: list[PerEpisodeDrift] = []
    recon_means: list[float] = []
    action_scalars: list[float] = []

    for ep in episodes:
        npz_path = trajectory_path_for(trajectory_dir, ep.cell_index, ep.episode_index)
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"missing trajectory NPZ for cell={ep.cell_index} "
                f"ep={ep.episode_index}: {npz_path}",
            )
        with np.load(npz_path, allow_pickle=False) as npz:
            stored_cell = int(npz["cell_index"])
            stored_ep = int(npz["episode_index"])
            if stored_cell != ep.cell_index or stored_ep != ep.episode_index:
                raise ValueError(
                    f"{npz_path}: identity mismatch "
                    f"(file says cell={stored_cell} ep={stored_ep}, "
                    f"episodes.json says cell={ep.cell_index} ep={ep.episode_index})",
                )
            obs = {k: np.asarray(npz[f"obs_{k}"]) for k in STATE_OBS_KEYS}
            actions = np.asarray(npz["action"], dtype=np.float64)
        recon_mean, recon_max, per_dim_std, action_scalar = _score_one_episode(loaded, obs, actions)
        per_episode.append(
            PerEpisodeDrift(
                cell_index=ep.cell_index,
                episode_index=ep.episode_index,
                seed=ep.seed,
                perturbation_config=dict(ep.perturbation_config),
                n_steps=ep.step_count,
                reconstruction_error_mean=recon_mean,
                reconstruction_error_max=recon_max,
                action_std_per_dim=[float(v) for v in per_dim_std.tolist()],
                action_entropy=action_scalar,
            )
        )
        recon_means.append(recon_mean)
        action_scalars.append(action_scalar)

    # Aggregates + top-k ordering. ``argsort(-x)`` gives descending
    # order; ``[:top_k]`` clamps at the end. If ``top_k`` exceeds the
    # episode count we just return every index.
    recon_arr = np.asarray(recon_means, dtype=np.float64)
    action_arr = np.asarray(action_scalars, dtype=np.float64)
    candidate_mean = float(recon_arr.mean()) if recon_arr.size > 0 else 0.0
    candidate_p95 = float(np.percentile(recon_arr, 95)) if recon_arr.size > 0 else 0.0
    candidate_action_mean = float(action_arr.mean()) if action_arr.size > 0 else 0.0
    k = min(top_k, len(per_episode))
    top_indices = np.argsort(-recon_arr, kind="stable")[:k].tolist() if k > 0 else []

    suite_name = episodes[0].suite_name if episodes else ""

    return DriftReport(
        suite_name=suite_name,
        n_episodes=len(per_episode),
        ae_mode="state",
        ae_latent_dim=loaded.ae.latent_dim,
        ae_reference_suite=loaded.reference_suite,
        reference_reconstruction_error_mean=loaded.reference_reconstruction_error_mean,
        reference_reconstruction_error_p95=loaded.reference_reconstruction_error_p95,
        candidate_reconstruction_error_mean=candidate_mean,
        candidate_reconstruction_error_p95=candidate_p95,
        candidate_action_entropy_mean=candidate_action_mean,
        per_episode=per_episode,
        top_ood_episodes=[int(i) for i in top_indices],
    )
