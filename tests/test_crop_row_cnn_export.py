"""The crop-row CNN example: torch training/export vs numpy inference parity.

Needs torch, so it runs in the ``monitor`` CI job.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.monitor

_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "crop_row"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _EXAMPLE_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # train_cnn imports cnn_policy by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example_modules() -> Iterator[tuple[Any, Any]]:
    sys.path.insert(0, str(_EXAMPLE_DIR))
    try:
        yield _load("cnn_policy"), _load("train_cnn")
    finally:
        sys.path.remove(str(_EXAMPLE_DIR))
        sys.modules.pop("cnn_policy", None)
        sys.modules.pop("train_cnn", None)


def test_numpy_inference_matches_torch(example_modules: tuple[Any, Any]) -> None:
    cnn_policy, train_cnn = example_modules
    from gauntlet.env.crop_row import CropRowEnv

    weights = np.load(cnn_policy.WEIGHTS)
    model = train_cnn.build()
    env = CropRowEnv()
    frames = [env.reset(seed=s)[0]["image"] for s in range(3)]
    x = torch.from_numpy(np.stack([cnn_policy.preprocess(f) for f in frames]))
    model(x[:1])
    convs = [m for m in model if isinstance(m, torch.nn.Conv2d)]
    fcs = [m for m in model if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for prefix, mods in (("conv", convs), ("fc", fcs)):
            for i, m in enumerate(mods):
                m.weight.copy_(torch.from_numpy(weights[f"{prefix}{i}.weight"]))
                m.bias.copy_(torch.from_numpy(weights[f"{prefix}{i}.bias"]))
        ref = model(x).numpy() * cnn_policy.LABEL_SCALE
    policy = cnn_policy.make_policy()
    got = np.array([policy.estimate(f) for f in frames])
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_training_script_writes_loadable_weights(tmp_path: Path) -> None:
    out = tmp_path / "w.npz"
    subprocess.run(
        [
            sys.executable,
            str(_EXAMPLE_DIR / "train_cnn.py"),
            "--episodes",
            "2",
            "--epochs",
            "1",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        timeout=300,
    )
    from gauntlet.env.crop_row import CropRowEnv

    cnn_policy = _load("cnn_policy")
    sys.modules.pop("cnn_policy", None)
    obs, _ = CropRowEnv().reset(seed=0)
    action = cnn_policy.CropRowCNNPolicy(out).act(obs)
    assert action.shape == (1,) and np.isfinite(action).all()
