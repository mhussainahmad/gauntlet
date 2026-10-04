"""Train the crop-row CNN on clean scenes and export numpy weights.

Collects labelled frames from ``CropRowEnv`` with **no perturbations**
(the vehicle is driven by a noisy ground-truth controller so the frames
cover a spread of lateral and heading errors), trains a three-layer CNN
to regress both errors, and writes ``cnn_weights.npz`` for
``cnn_policy.py``. Needs torch (``uv sync --extra monitor``)::

    uv run python examples/crop_row/train_cnn.py

CPU-only and seeded; takes about a minute.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from cnn_policy import CONV_LAYERS, LABEL_SCALE, WEIGHTS, preprocess

from gauntlet.env.crop_row import CropRowEnv
from gauntlet.policy.crop_row import row_steering


def collect(n_episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    env = CropRowEnv(max_steps=100)
    xs, ys = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 10_000 + ep)
        noise = rng.uniform(0.2, 0.7)
        done = False
        while not done:
            xs.append(preprocess(obs["image"]))
            ys.append([info["lateral_error"] / LABEL_SCALE, info["heading_error"] / LABEL_SCALE])
            a = row_steering(info["lateral_error"], info["heading_error"])
            a = np.clip(a + rng.normal(0.0, noise, size=1), -1.0, 1.0)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def build() -> nn.Sequential:
    layers: list[nn.Module] = []
    c_in = 3
    for c_out, k, s in CONV_LAYERS:
        layers += [nn.Conv2d(c_in, c_out, k, stride=s, padding=k // 2), nn.ReLU()]
        c_in = c_out
    layers += [nn.Flatten(), nn.LazyLinear(32), nn.ReLU(), nn.Linear(32, 2)]
    return nn.Sequential(*layers)


def export(model: nn.Sequential, path: Path) -> None:
    convs = [m for m in model if isinstance(m, nn.Conv2d)]
    fcs = [m for m in model if isinstance(m, nn.Linear)]
    arrays = {}
    for i, m in enumerate(convs):
        arrays[f"conv{i}.weight"] = m.weight.detach().numpy()
        arrays[f"conv{i}.bias"] = m.bias.detach().numpy()
    for i, m in enumerate(fcs):
        arrays[f"fc{i}.weight"] = m.weight.detach().numpy()
        arrays[f"fc{i}.bias"] = m.bias.detach().numpy()
    np.savez_compressed(path, **arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=WEIGHTS)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    x, y = collect(args.episodes, args.seed)
    n_val = len(x) // 10
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    tr_x, tr_y, va_x, va_y = xt[n_val:], yt[n_val:], xt[:n_val], yt[:n_val]
    model = build()
    model(tr_x[:1])  # materialise the lazy layer
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr_x), generator=gen)
        for i in range(0, len(perm), 128):
            idx = perm[i : i + 128]
            loss = nn.functional.smooth_l1_loss(model(tr_x[idx]), tr_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            err = (model(va_x) - va_y).abs().mean(dim=0) * LABEL_SCALE
        print(
            f"epoch {epoch + 1:2d}  val |lateral| {err[0]:.4f} m  |heading| {err[1]:.4f} rad",
            flush=True,
        )
    export(model, args.out)
    print(f"{len(x)} frames; wrote {args.out}")


if __name__ == "__main__":
    main()
