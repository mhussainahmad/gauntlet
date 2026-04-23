"""Fit a :class:`StateAutoencoder` on a reference trajectory dump.

Torch-only. Import path raises the shared install-hint
:class:`ImportError` when the extra is missing.

The training pipeline is deliberately minimal — RFC §6 "small deps"
forbids anything heavier than torch. `torch.optim.Adam`,
`torch.utils.data.DataLoader`, and `F.mse_loss` carry the whole thing.

Determinism: :func:`train_ae` seeds ``torch.manual_seed`` and the
DataLoader's generator at entry. Two calls with the same ``seed``
produce bit-identical ``state_dict`` tensors — the property test
:func:`train_ae_two_runs_bitwise_match` in the monitor-extra test suite
enforces this.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

_MONITOR_INSTALL_HINT = (
    "gauntlet.monitor.train requires the 'monitor' extra. Install with:\n"
    "    uv sync --extra monitor\n"
    "or, for a plain pip env:\n"
    "    pip install 'gauntlet[monitor]'"
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(_MONITOR_INSTALL_HINT) from exc

if TYPE_CHECKING:
    import torch as _torch  # noqa: F401

from gauntlet.monitor.ae import (
    INPUT_DIM,
    STATE_OBS_DIMS,
    STATE_OBS_KEYS,
    NormalizationStats,
    StateAutoencoder,
    save_state_autoencoder,
)

__all__ = ["flatten_obs_arrays", "load_reference_matrix", "train_ae"]


# ----------------------------------------------------------------------------
# Data assembly.
# ----------------------------------------------------------------------------


def flatten_obs_arrays(obs: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
    """Stack the per-key obs arrays into a dense ``(T, INPUT_DIM)`` matrix.

    ``STATE_OBS_KEYS`` defines the order so the trainer and scorer
    always disagree or agree together — a missing key raises
    ``KeyError`` with the missing name, a shape mismatch raises
    ``ValueError`` pointing at the offending key.
    """
    pieces: list[NDArray[np.float64]] = []
    for key, expected_dim in zip(STATE_OBS_KEYS, STATE_OBS_DIMS, strict=True):
        if key not in obs:
            raise KeyError(f"trajectory missing obs key {key!r}; have {sorted(obs.keys())}")
        arr = np.asarray(obs[key], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != expected_dim:
            raise ValueError(f"obs[{key!r}] must have shape (T, {expected_dim}); got {arr.shape}")
        pieces.append(arr)
    return np.concatenate(pieces, axis=1).astype(np.float64, copy=False)


def load_reference_matrix(trajectory_dir: Path) -> NDArray[np.float64]:
    """Gather every NPZ sidecar in ``trajectory_dir`` into one ``(N, D)`` matrix.

    Walks the directory in sorted filename order (so the matrix ordering
    is deterministic), loads each NPZ, flattens via
    :func:`flatten_obs_arrays`, and concatenates. An empty directory is
    an error — there is nothing useful to train on — and a zero-row
    matrix after concatenation is treated the same way.
    """
    paths = sorted(trajectory_dir.glob("*.npz"))
    if not paths:
        raise ValueError(
            f"no NPZ trajectories found under {trajectory_dir}; "
            "did you forget --record-trajectories?"
        )
    rows: list[NDArray[np.float64]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as npz:
            obs = {k: np.asarray(npz[f"obs_{k}"]) for k in STATE_OBS_KEYS}
        flat = flatten_obs_arrays(obs)
        if flat.shape[0] > 0:
            rows.append(flat)
    if not rows:
        raise ValueError(
            f"trajectory dir {trajectory_dir} has only zero-length episodes",
        )
    return np.concatenate(rows, axis=0).astype(np.float64, copy=False)


# ----------------------------------------------------------------------------
# Fit loop.
# ----------------------------------------------------------------------------


def _compute_normalization(X: NDArray[np.float64]) -> NormalizationStats:
    """Per-dim mean + std over the reference set, clamped so std >= 1e-6.

    The clamp avoids a divide-by-zero in the scorer when a column is
    constant across the reference set (e.g. ``gripper`` during a
    scripted-trajectory reference sweep).
    """
    mean = X.mean(axis=0).astype(np.float64, copy=False)
    std = X.std(axis=0, ddof=0).astype(np.float64, copy=False)
    std = np.where(std < 1e-6, 1.0, std)
    return NormalizationStats(mean=mean, std=std)


def _normalize(X: NDArray[np.float64], stats: NormalizationStats) -> NDArray[np.float64]:
    """Apply ``(X - mean) / std`` using ``stats``; float32 output for torch."""
    return cast(
        NDArray[np.float64],
        ((X - stats.mean) / stats.std).astype(np.float64, copy=False),
    )


def train_ae(
    trajectory_dir: Path,
    *,
    out_dir: Path,
    reference_suite: str | None = None,
    latent_dim: int = 8,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> None:
    """Train a state autoencoder on a reference sweep and persist it.

    The reference trajectories are expected under ``trajectory_dir`` as
    produced by ``gauntlet run --record-trajectories``; every NPZ's
    ``obs_*`` arrays are flattened and concatenated.

    Args:
        trajectory_dir: Directory with ``cell_*_ep_*.npz`` files.
        out_dir: Checkpoint directory — receives ``weights.pt``,
            ``normalization.json``, ``config.json``.
        reference_suite: Optional human-readable suite identifier; echoed
            into ``config.json`` so ``monitor score`` can surface
            cross-suite warnings.
        latent_dim: AE bottleneck size (RFC §7 default: 8).
        epochs: Number of passes over the reference set. RFC §7 default
            is 50; tests override with small values.
        batch_size: Minibatch size. Default 256 — the reference sweeps
            are tiny (order 10k rows), so this is almost always
            ``len(X)`` in practice.
        lr: Adam learning rate. RFC §7 default is 1e-3.
        seed: Torch / DataLoader seed for bit-identical reruns.

    The function returns ``None`` — every output is a file under
    ``out_dir``.
    """
    # Full deterministic context. ``torch.manual_seed`` seeds the CPU +
    # default CUDA RNG, and the DataLoader's generator is seeded so the
    # shuffled epoch order is reproducible.
    torch.manual_seed(seed)

    X = load_reference_matrix(trajectory_dir)
    stats = _compute_normalization(X)
    X_norm = _normalize(X, stats).astype(np.float32, copy=False)
    input_dim = X_norm.shape[1]
    if input_dim != INPUT_DIM:
        # Defensive: STATE_OBS_DIMS sums to INPUT_DIM.
        raise ValueError(
            f"reference matrix has dim={input_dim}; expected {INPUT_DIM}",
        )

    tensor = torch.as_tensor(X_norm, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    generator = torch.Generator().manual_seed(seed)
    # ``num_workers=0`` keeps shuffle determinism — a worker pool
    # reintroduces nondeterminism through process-scheduling order.
    loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    ae = StateAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    optimiser = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ae.train()
    for _epoch in range(epochs):
        for (batch,) in loader:
            optimiser.zero_grad(set_to_none=True)
            recon = ae(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimiser.step()

    # Reference baselines — computed on the full reference set so the
    # percentile is from the same distribution the AE saw. No held-out
    # split: for the small reference sweeps this RFC targets (~10k rows)
    # the AE's training-set error IS a good proxy for "in-distribution
    # error". A held-out split is a follow-up (RFC §12).
    ae.eval()
    with torch.no_grad():
        reference_per_row = np.asarray(ae.per_sample_score(X_norm.astype(np.float64)))
    ref_mean = float(reference_per_row.mean())
    ref_p95 = float(np.percentile(reference_per_row, 95))

    save_state_autoencoder(
        ae,
        stats,
        ae_dir=out_dir,
        reference_suite=reference_suite,
        reference_reconstruction_error_mean=ref_mean,
        reference_reconstruction_error_p95=ref_p95,
        train_seed=seed,
    )
