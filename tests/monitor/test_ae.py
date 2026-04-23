"""Autoencoder / train / score smoke tests — require the [monitor] extra.

Marked ``@pytest.mark.monitor`` so the default (torch-free) pytest
invocation skips them. The dedicated ``monitor-tests`` CI job syncs the
extra and runs this file.

Coverage:

* AE loss decreases on a synthetic reference batch.
* Two fresh training runs with the same ``seed`` produce bit-identical
  weights (determinism is non-negotiable per spec §6).
* Reconstruction error is materially larger on a shifted distribution.
* End-to-end ``score_drift`` on a hand-built tiny fixture produces a
  well-formed :class:`DriftReport`.
* ``load_state_autoencoder`` round-trips the full checkpoint.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.monitor

# Torch + the torch-backed monitor modules are only importable in the
# ``monitor-tests`` CI job. The default pytest invocation uses
# ``-m 'not ... and not monitor'`` so these tests are deselected, but
# collection still walks the module body — ``pytest.importorskip``
# short-circuits the file when torch is missing without failing the
# whole run.
torch = pytest.importorskip("torch")

from gauntlet.monitor.ae import (  # noqa: E402
    INPUT_DIM,
    StateAutoencoder,
    load_state_autoencoder,
)
from gauntlet.monitor.schema import DriftReport, PerEpisodeDrift  # noqa: E402
from gauntlet.monitor.score import score_drift  # noqa: E402
from gauntlet.monitor.train import train_ae  # noqa: E402
from gauntlet.runner.worker import trajectory_path_for, write_trajectory_npz  # noqa: E402

# ----------------------------------------------------------------------------
# Tiny synthetic-data helpers.
# ----------------------------------------------------------------------------


def _write_synthetic_trajectories(
    traj_dir: Path,
    *,
    n_episodes: int,
    steps_per_episode: int,
    obs_mean: float = 0.0,
    obs_scale: float = 1.0,
    seed: int,
) -> None:
    """Build ``n_episodes`` NPZs under ``traj_dir`` with obs drawn from N(mean, scale)."""
    traj_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for ep in range(n_episodes):
        # Build one obs dict matching the TabletopEnv key schema. Shape
        # sums to INPUT_DIM = 14.
        obs = {
            "cube_pos": rng.normal(obs_mean, obs_scale, size=(steps_per_episode, 3)),
            "cube_quat": rng.normal(obs_mean, obs_scale, size=(steps_per_episode, 4)),
            "ee_pos": rng.normal(obs_mean, obs_scale, size=(steps_per_episode, 3)),
            "gripper": rng.normal(obs_mean, obs_scale, size=(steps_per_episode, 1)),
            "target_pos": rng.normal(obs_mean, obs_scale, size=(steps_per_episode, 3)),
        }
        # Actions: modest per-dim variance so action_entropy has signal.
        actions = rng.normal(0.0, 0.3, size=(steps_per_episode, 7))
        write_trajectory_npz(
            trajectory_path_for(traj_dir, cell_index=0, episode_index=ep),
            obs_arrays={k: v.astype(np.float64) for k, v in obs.items()},
            actions=actions.astype(np.float64),
            seed=1000 + ep,
            cell_index=0,
            episode_index=ep,
        )


def _write_lowrank_trajectories(
    traj_dir: Path,
    *,
    n_episodes: int,
    steps_per_episode: int,
    rank: int = 4,
    noise: float = 0.05,
    seed: int,
) -> None:
    """NPZ trajectories whose obs lie on a low-rank linear manifold.

    Samples are drawn as ``latent @ A + eps`` where ``latent`` is
    ``(T, rank)`` gaussian, ``A`` is a fixed ``(rank, INPUT_DIM)``
    projection, and ``eps`` is small gaussian noise. A bottleneck AE
    with latent_dim >= rank can reconstruct these to within the noise
    floor, giving the fit-loop assertion a sane convergence target.
    """
    traj_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # One projection matrix shared across episodes -> all the data
    # lives on one common manifold.
    A = rng.normal(size=(rank, 14)).astype(np.float64)
    for ep in range(n_episodes):
        latent = rng.normal(size=(steps_per_episode, rank)).astype(np.float64)
        base = latent @ A
        perturbed = base + rng.normal(scale=noise, size=base.shape)
        # Slice into the TabletopEnv obs key layout.
        obs = {
            "cube_pos": perturbed[:, 0:3].copy(),
            "cube_quat": perturbed[:, 3:7].copy(),
            "ee_pos": perturbed[:, 7:10].copy(),
            "gripper": perturbed[:, 10:11].copy(),
            "target_pos": perturbed[:, 11:14].copy(),
        }
        actions = rng.normal(0.0, 0.3, size=(steps_per_episode, 7)).astype(np.float64)
        write_trajectory_npz(
            trajectory_path_for(traj_dir, cell_index=0, episode_index=ep),
            obs_arrays=obs,
            actions=actions,
            seed=1000 + ep,
            cell_index=0,
            episode_index=ep,
        )


def _episodes_json_for(traj_dir: Path, n_episodes: int, out_path: Path) -> None:
    """Write a minimal ``episodes.json`` matching the synthetic trajectories."""
    episodes = [
        {
            "suite_name": "synthetic",
            "cell_index": 0,
            "episode_index": ep,
            "seed": 1000 + ep,
            "perturbation_config": {},
            "success": False,
            "terminated": False,
            "truncated": True,
            "step_count": 50,
            "total_reward": 0.0,
            "metadata": {"master_seed": 0},
        }
        for ep in range(n_episodes)
    ]
    out_path.write_text(json.dumps(episodes, indent=2) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# Core tests.
# ----------------------------------------------------------------------------


def test_state_autoencoder_shapes() -> None:
    """Forward + reconstruct + per-sample score return the expected shapes."""
    ae = StateAutoencoder(input_dim=INPUT_DIM, latent_dim=8)
    x = torch.zeros((4, INPUT_DIM), dtype=torch.float32)
    out = ae(x)
    assert out.shape == x.shape

    x_np = np.zeros((4, INPUT_DIM), dtype=np.float64)
    recon = ae.reconstruct(x_np)
    assert recon.shape == (4, INPUT_DIM)
    per_row = ae.per_sample_score(x_np)
    assert per_row.shape == (4,)


def test_ae_training_reduces_loss(tmp_path: Path) -> None:
    """Training must drop training-set MSE materially vs a cold init.

    A bottleneck AE only learns well on data with structure; pure
    isotropic noise forces the AE to memorise. We build a low-rank
    synthetic set (points living on a 4-D linear manifold embedded in
    14-D) so the 8-D latent can comfortably reconstruct. The real
    discriminative property — OOD shift -> bigger error — is locked
    in by :func:`test_reconstruction_error_increases_on_ood_samples`;
    this test just defends that the fit loop converges at all.
    """
    traj_dir = tmp_path / "ref"
    # Low-rank synthetic set so the AE can learn a meaningful
    # compression with small latent_dim.
    ae_dir = tmp_path / "ae"
    _write_lowrank_trajectories(traj_dir, n_episodes=16, steps_per_episode=200, seed=0)

    from gauntlet.monitor.train import (  # local import to avoid top-level torch.fn
        _compute_normalization,
        _normalize,
        load_reference_matrix,
    )

    X = load_reference_matrix(traj_dir)
    stats = _compute_normalization(X)
    X_norm = _normalize(X, stats).astype(np.float32)
    untrained_ae = StateAutoencoder(input_dim=INPUT_DIM, latent_dim=8)
    untrained_score = float(untrained_ae.per_sample_score(X_norm.astype(np.float64)).mean())

    train_ae(traj_dir, out_dir=ae_dir, epochs=30, batch_size=64, seed=0)

    loaded = load_state_autoencoder(ae_dir)
    # Conservative 2x budget — on low-rank structured data the AE
    # typically gets a 10x+ reduction but tight CI timings make the
    # looser bar the defensible assertion.
    assert loaded.reference_reconstruction_error_mean < 0.5 * untrained_score, (
        f"AE did not reduce reconstruction error: "
        f"untrained={untrained_score:.4f}, "
        f"trained={loaded.reference_reconstruction_error_mean:.4f}"
    )


def test_ae_training_is_deterministic(tmp_path: Path) -> None:
    """Two train_ae runs with the same seed -> bit-identical state_dict."""
    traj_dir = tmp_path / "ref"
    _write_synthetic_trajectories(traj_dir, n_episodes=4, steps_per_episode=40, seed=1)

    ae_dir_a = tmp_path / "ae_a"
    ae_dir_b = tmp_path / "ae_b"
    train_ae(traj_dir, out_dir=ae_dir_a, epochs=3, batch_size=32, seed=7)
    train_ae(traj_dir, out_dir=ae_dir_b, epochs=3, batch_size=32, seed=7)

    state_a = torch.load(ae_dir_a / "weights.pt", weights_only=True, map_location="cpu")
    state_b = torch.load(ae_dir_b / "weights.pt", weights_only=True, map_location="cpu")
    assert state_a.keys() == state_b.keys()
    for k, v_a in state_a.items():
        assert torch.equal(v_a, state_b[k]), f"param {k!r} differs across seed=7 reruns"


def test_reconstruction_error_increases_on_ood_samples(tmp_path: Path) -> None:
    """Train on N(0,1); score N(5,1) -> much larger per-row error."""
    traj_dir = tmp_path / "ref"
    _write_synthetic_trajectories(
        traj_dir,
        n_episodes=8,
        steps_per_episode=80,
        obs_mean=0.0,
        obs_scale=1.0,
        seed=3,
    )
    ae_dir = tmp_path / "ae"
    train_ae(traj_dir, out_dir=ae_dir, epochs=5, batch_size=64, seed=3)
    loaded = load_state_autoencoder(ae_dir)

    # Build in-distribution vs OOD evaluation batches.
    rng = np.random.default_rng(99)
    in_dist = rng.normal(0.0, 1.0, size=(200, INPUT_DIM))
    ood = rng.normal(5.0, 1.0, size=(200, INPUT_DIM))
    in_norm = (in_dist - loaded.normalization.mean) / loaded.normalization.std
    ood_norm = (ood - loaded.normalization.mean) / loaded.normalization.std
    in_score = float(loaded.ae.per_sample_score(in_norm).mean())
    ood_score = float(loaded.ae.per_sample_score(ood_norm).mean())

    # Very conservative multiplier; the real margin on N(0,1) vs N(5,1)
    # in a 14-D space is ~20x or more.
    assert ood_score > 3.0 * in_score, (
        f"OOD not materially higher: in={in_score:.4f}, ood={ood_score:.4f}"
    )


def test_load_state_autoencoder_round_trips(tmp_path: Path) -> None:
    """Save + load yields an AE that reproduces the same scores."""
    traj_dir = tmp_path / "ref"
    _write_synthetic_trajectories(traj_dir, n_episodes=4, steps_per_episode=40, seed=11)
    ae_dir = tmp_path / "ae"
    train_ae(traj_dir, out_dir=ae_dir, epochs=2, batch_size=32, seed=11)

    loaded_a = load_state_autoencoder(ae_dir)
    loaded_b = load_state_autoencoder(ae_dir)

    x = np.ones((5, INPUT_DIM), dtype=np.float64)
    np.testing.assert_allclose(loaded_a.ae.reconstruct(x), loaded_b.ae.reconstruct(x))


def test_load_state_autoencoder_rejects_missing_artefacts(tmp_path: Path) -> None:
    """Half-a-checkpoint must not silently load."""
    traj_dir = tmp_path / "ref"
    _write_synthetic_trajectories(traj_dir, n_episodes=2, steps_per_episode=20, seed=2)
    ae_dir = tmp_path / "ae"
    train_ae(traj_dir, out_dir=ae_dir, epochs=1, batch_size=16, seed=2)

    # Remove a required file.
    (ae_dir / "normalization.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_state_autoencoder(ae_dir)


def test_score_drift_end_to_end(tmp_path: Path) -> None:
    """Train on N(0,1); score N(3,1) candidate -> DriftReport shape correct."""
    # Reference sweep.
    ref_traj = tmp_path / "ref"
    _write_synthetic_trajectories(ref_traj, n_episodes=6, steps_per_episode=50, seed=20)
    ae_dir = tmp_path / "ae"
    train_ae(
        ref_traj,
        out_dir=ae_dir,
        epochs=5,
        batch_size=64,
        reference_suite="reference-suite",
        seed=20,
    )

    # Candidate sweep: same schema, shifted mean so reconstruction
    # error blows up and the top-k picks real outliers.
    cand_traj = tmp_path / "cand"
    _write_synthetic_trajectories(
        cand_traj, n_episodes=4, steps_per_episode=50, obs_mean=3.0, seed=21
    )
    episodes_path = tmp_path / "episodes.json"
    _episodes_json_for(cand_traj, n_episodes=4, out_path=episodes_path)

    drift = score_drift(
        episodes_path=episodes_path,
        trajectory_dir=cand_traj,
        ae_dir=ae_dir,
        top_k=2,
    )

    assert isinstance(drift, DriftReport)
    assert drift.n_episodes == 4
    assert drift.suite_name == "synthetic"
    assert drift.ae_mode == "state"
    assert drift.ae_latent_dim == 8
    assert drift.ae_reference_suite == "reference-suite"
    assert len(drift.per_episode) == 4
    assert all(isinstance(row, PerEpisodeDrift) for row in drift.per_episode)
    # Top-k clamped to the 2 we asked for, descending error order.
    assert len(drift.top_ood_episodes) == 2
    errors = [drift.per_episode[i].reconstruction_error_mean for i in drift.top_ood_episodes]
    assert errors == sorted(errors, reverse=True)
    # Candidate is materially more OOD than reference.
    assert (
        drift.candidate_reconstruction_error_mean > 1.5 * drift.reference_reconstruction_error_mean
    )


def test_score_drift_identity_mismatch_raises(tmp_path: Path) -> None:
    """If NPZ scalars disagree with the filename cell/episode, the scorer bails.

    Covers RFC §6's defensive cross-check. We swap filenames to induce
    a mismatch (the NPZ's stored cell/ep will not match the new path's).
    """
    # Minimal reference + train so we have a real AE to hand to score.
    ref_traj = tmp_path / "ref"
    _write_synthetic_trajectories(ref_traj, n_episodes=4, steps_per_episode=20, seed=5)
    ae_dir = tmp_path / "ae"
    train_ae(ref_traj, out_dir=ae_dir, epochs=1, batch_size=16, seed=5)

    # Candidate: one episode. Shuffle the NPZ onto a different path so
    # the identity scalar won't match the filename.
    cand_traj = tmp_path / "cand"
    _write_synthetic_trajectories(cand_traj, n_episodes=1, steps_per_episode=20, seed=5)
    wrong_path = trajectory_path_for(cand_traj, cell_index=7, episode_index=7)
    shutil.move(trajectory_path_for(cand_traj, 0, 0), wrong_path)

    episodes_path = tmp_path / "episodes.json"
    # Episode says cell=7, ep=7 — matches the filename but NOT the
    # stored scalars (which are 0/0 from the write).
    episodes = [
        {
            "suite_name": "synthetic",
            "cell_index": 7,
            "episode_index": 7,
            "seed": 0,
            "perturbation_config": {},
            "success": False,
            "terminated": False,
            "truncated": True,
            "step_count": 20,
            "total_reward": 0.0,
            "metadata": {"master_seed": 0},
        }
    ]
    episodes_path.write_text(json.dumps(episodes) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identity mismatch"):
        score_drift(
            episodes_path=episodes_path,
            trajectory_dir=cand_traj,
            ae_dir=ae_dir,
            top_k=1,
        )
