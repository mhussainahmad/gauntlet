"""State-observation autoencoder — torch-only.

14→64→32→8→32→64→14 MLP per RFC §7. Kept deliberately small: the whole
checkpoint is ~10 KB on disk and a CPU forward on a batch of 1 is
sub-millisecond. Training a new AE over a 50-episode reference sweep
fits inside a few seconds on ubuntu-latest.

Importing this module requires the ``[monitor]`` extra. The failure
mode is a clean :class:`ImportError` with an install hint — no mystery
``ModuleNotFoundError`` from deep inside a third call-site.

The public surface:

* :class:`StateAutoencoder` — the ``nn.Module`` subclass that owns the
  weights.
* :func:`save_state_autoencoder` / :func:`load_state_autoencoder` —
  checkpoint serialiser and loader. ``weights_only=True`` on load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

_MONITOR_INSTALL_HINT = (
    "gauntlet.monitor.ae requires the 'monitor' extra. Install with:\n"
    "    uv sync --extra monitor\n"
    "or, for a plain pip env:\n"
    "    pip install 'gauntlet[monitor]'"
)

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(_MONITOR_INSTALL_HINT) from exc

if TYPE_CHECKING:
    import torch as _torch  # noqa: F401

__all__ = [
    "INPUT_DIM",
    "NormalizationStats",
    "StateAutoencoder",
    "load_state_autoencoder",
    "save_state_autoencoder",
]


# TabletopEnv flattens to 14-D proprio (cube_pos 3 + cube_quat 4 +
# ee_pos 3 + gripper 1 + target_pos 3). This is the AE's input size
# on the state path.
INPUT_DIM: int = 14

# Canonical order the obs dict is flattened in. Kept as a module-level
# constant so the trainer and scorer never disagree about offsets.
STATE_OBS_KEYS: tuple[str, ...] = (
    "cube_pos",
    "cube_quat",
    "ee_pos",
    "gripper",
    "target_pos",
)

# Per-key slice lengths, matching ``STATE_OBS_KEYS``.
STATE_OBS_DIMS: tuple[int, ...] = (3, 4, 3, 1, 3)


@dataclass(frozen=True)
class NormalizationStats:
    """Per-input-dim mean/std used to standardise the AE's inputs.

    Computed at train time over the reference sweep and saved alongside
    the weights. The scorer applies the same transform so a candidate
    sample that is bit-identical-to-training scores near zero.
    """

    mean: NDArray[np.float64]  # shape (INPUT_DIM,)
    std: NDArray[np.float64]  # shape (INPUT_DIM,)

    def __post_init__(self) -> None:
        if self.mean.shape != (INPUT_DIM,):
            raise ValueError(f"mean shape must be ({INPUT_DIM},); got {self.mean.shape}")
        if self.std.shape != (INPUT_DIM,):
            raise ValueError(f"std shape must be ({INPUT_DIM},); got {self.std.shape}")


class StateAutoencoder(nn.Module):
    """Small MLP autoencoder over the flattened 14-D proprio vector.

    Architecture (RFC §7):
    ``14 → 64 → 32 → 8 → 32 → 64 → 14``. ReLU on every hidden layer,
    no activation on the latent or the output. Loss is
    :func:`torch.nn.functional.mse_loss` over the normalised input.

    The ``latent_dim`` is configurable (default 8) so tests and
    follow-ups can shrink the representation without forking the
    module.
    """

    def __init__(self, *, input_dim: int = INPUT_DIM, latent_dim: int = 8) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive; got {input_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive; got {latent_dim}")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode + decode; returns the reconstruction."""
        return cast(torch.Tensor, self.decoder(self.encoder(x)))

    def reconstruct(self, x_np: NDArray[np.float64]) -> NDArray[np.float64]:
        """NumPy helper: reconstruct a (N, D) batch under ``torch.no_grad``.

        The scorer path never needs autograd; we flip it off explicitly
        so the grad graph does not silently inflate memory on a long
        candidate sweep.
        """
        self.eval()
        with torch.no_grad():
            t_in = torch.as_tensor(x_np, dtype=torch.float32)
            t_out = self.forward(t_in)
            return cast(NDArray[np.float64], t_out.detach().cpu().numpy().astype(np.float64))

    def per_sample_score(self, x_np: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-row L2 reconstruction error — the primary OOD signal.

        Shape: ``(N,)`` float64. One value per input row. The trainer's
        reference p95 lives on this distribution; the scorer reports
        ``mean(per_sample_score(...))`` per candidate episode.
        """
        recon = self.reconstruct(x_np)
        diff = (x_np - recon).astype(np.float64, copy=False)
        # L2 norm per row — ``linalg.norm(..., axis=1)`` is the canonical
        # vectorised form.
        return cast(NDArray[np.float64], np.linalg.norm(diff, axis=1).astype(np.float64))


def save_state_autoencoder(
    ae: StateAutoencoder,
    normalization: NormalizationStats,
    *,
    ae_dir: Path,
    reference_suite: str | None,
    reference_reconstruction_error_mean: float,
    reference_reconstruction_error_p95: float,
    train_seed: int,
) -> None:
    """Persist an AE checkpoint to ``ae_dir``.

    Writes three files:

    * ``weights.pt`` — ``torch.save(state_dict)``.
    * ``normalization.json`` — per-input-dim mean/std (list form for JSON).
    * ``config.json`` — mode, input/latent dims, reference metadata,
      echoed reference baselines (mean + p95 reconstruction error), and
      the train seed for reproducibility.

    ``ae_dir`` is created if missing. Callers can resume the scoring
    side with just the three files; no extra state is implicit.
    """
    ae_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ae.state_dict(), ae_dir / "weights.pt")
    (ae_dir / "normalization.json").write_text(
        json.dumps(
            {
                "mean": normalization.mean.tolist(),
                "std": normalization.std.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (ae_dir / "config.json").write_text(
        json.dumps(
            {
                "mode": "state",
                "input_dim": ae.input_dim,
                "latent_dim": ae.latent_dim,
                "reference_suite": reference_suite,
                "reference_reconstruction_error_mean": reference_reconstruction_error_mean,
                "reference_reconstruction_error_p95": reference_reconstruction_error_p95,
                "train_seed": train_seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class LoadedAutoencoder:
    """Bundle returned by :func:`load_state_autoencoder`.

    Grouping these together means the scorer only does one load call;
    passing the three items around separately is more error-prone.
    """

    ae: StateAutoencoder
    normalization: NormalizationStats
    reference_suite: str | None
    reference_reconstruction_error_mean: float
    reference_reconstruction_error_p95: float


def load_state_autoencoder(ae_dir: Path) -> LoadedAutoencoder:
    """Inverse of :func:`save_state_autoencoder`.

    Loads the weights under ``weights_only=True`` (torch's recommended
    safe-load path since 2.4; we require ``torch>=2.2`` but the flag is
    accepted from 2.0 onwards). Re-reads the normalisation + config
    JSON and populates a fresh :class:`StateAutoencoder` before
    returning.
    """
    config_path = ae_dir / "config.json"
    normalization_path = ae_dir / "normalization.json"
    weights_path = ae_dir / "weights.pt"
    for p in (config_path, normalization_path, weights_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing AE artefact: {p}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "state":
        raise ValueError(
            f"AE at {ae_dir} has mode={config.get('mode')!r}; only 'state' is supported",
        )
    input_dim = int(config["input_dim"])
    latent_dim = int(config["latent_dim"])

    ae = StateAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    state = torch.load(weights_path, weights_only=True, map_location="cpu")
    ae.load_state_dict(state)
    ae.eval()

    norm_raw = json.loads(normalization_path.read_text(encoding="utf-8"))
    mean = np.asarray(norm_raw["mean"], dtype=np.float64)
    std = np.asarray(norm_raw["std"], dtype=np.float64)
    normalization = NormalizationStats(mean=mean, std=std)

    return LoadedAutoencoder(
        ae=ae,
        normalization=normalization,
        reference_suite=config.get("reference_suite"),
        reference_reconstruction_error_mean=float(config["reference_reconstruction_error_mean"]),
        reference_reconstruction_error_p95=float(config["reference_reconstruction_error_p95"]),
    )
