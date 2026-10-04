"""A small CNN row-guidance policy for ``env: crop-row`` (numpy inference).

The network regresses the vehicle's lateral and heading error from the
camera image; :func:`gauntlet.policy.crop_row.row_steering` turns the
estimate into a steering command (the same controller the classical
policy uses, so the two differ only in perception).

It was trained by ``train_cnn.py`` on **clean scenes only**: no dust,
glare, blur, weeds, curvature or lighting change. Evaluating it with the
``crop-row-field`` suite measures how that training distribution holds
up under field conditions.

Inference is plain numpy, so running the policy needs no torch::

    PYTHONPATH=examples/crop_row uv run gauntlet run \\
        examples/suites/crop-row-field.yaml --policy cnn_policy:make_policy --out out/cnn
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from gauntlet.policy.crop_row import row_steering

WEIGHTS = Path(__file__).with_name("cnn_weights.npz")
LABEL_SCALE = 0.1  # labels are (lateral_error, heading_error) / LABEL_SCALE
# (out_channels, kernel, stride) for the three conv layers.
CONV_LAYERS = ((16, 5, 2), (32, 3, 2), (32, 3, 2))


def preprocess(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """``[H, W, 3]`` uint8 -> ``[3, H/2, W/2]`` float in [0, 1] (2x2 average pool)."""
    x = image.astype(np.float32) / 255.0
    h, w = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    x = x[:h, :w].reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _conv(
    x: NDArray[np.float32], w: NDArray[np.float32], b: NDArray[np.float32], stride: int
) -> NDArray[np.float32]:
    k = w.shape[-1]
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(xp, (k, k), axis=(1, 2))[:, ::stride, ::stride]
    out = np.einsum("chwij,ocij->ohw", win, w, optimize=True) + b[:, None, None]
    return np.maximum(out, 0.0).astype(np.float32)


class CropRowCNNPolicy:
    """Image -> (lateral, heading) error estimate -> steering."""

    def __init__(self, weights: Path | str = WEIGHTS) -> None:
        z = np.load(weights)
        self._p: dict[str, NDArray[np.float32]] = {k: z[k].astype(np.float32) for k in z.files}

    def estimate(self, image: NDArray[np.uint8]) -> tuple[float, float]:
        x = preprocess(image)
        for i, (_, _, stride) in enumerate(CONV_LAYERS):
            x = _conv(x, self._p[f"conv{i}.weight"], self._p[f"conv{i}.bias"], stride)
        h = np.maximum(self._p["fc0.weight"] @ x.ravel() + self._p["fc0.bias"], 0.0)
        y = self._p["fc1.weight"] @ h + self._p["fc1.bias"]
        return float(y[0]) * LABEL_SCALE, float(y[1]) * LABEL_SCALE

    def act(self, obs: Any) -> NDArray[np.float64]:
        return row_steering(*self.estimate(np.asarray(obs["image"], dtype=np.uint8)))


def make_policy() -> CropRowCNNPolicy:
    return CropRowCNNPolicy()
