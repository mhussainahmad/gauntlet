"""B-27 — optional W&B / MLflow per-Episode mirror sink tests.

Marked ``@pytest.mark.wandb`` / ``@pytest.mark.mlflow`` per the marker
contract in ``pyproject.toml`` (deselected by the default test job so
the suite stays extras-free). Each test mocks the backend module by
injecting a stub into ``sys.modules`` BEFORE the sink class is
constructed — the lazy-import inside ``__init__`` then picks up the
stub instead of the real package, so the tests run cleanly even when
neither extra is installed.

Anti-feature reminder (PRODUCT.md): both sinks are off by default;
opting in exfiltrates per-Episode results to the chosen backend. The
tests verify the off-by-default surface (no import on the default
code path) AND the opt-in surface (correct field forwarding, close
called, sink failure non-fatal).
"""

from __future__ import annotations

import sys
import warnings
from typing import Any
from unittest.mock import MagicMock

import pytest

from gauntlet.runner.episode import Episode


def _ep(
    *,
    cell_index: int = 0,
    episode_index: int = 0,
    success: bool = True,
    seed: int = 42,
    config: dict[str, float] | None = None,
    suite_name: str = "tiny",
) -> Episode:
    return Episode(
        suite_name=suite_name,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=seed,
        perturbation_config=dict(config or {"lighting": 0.5}),
        success=success,
        terminated=success,
        truncated=False,
        step_count=10,
        total_reward=1.0 if success else 0.0,
    )


# ──────────────────────────────────────────────────────────────────────
# WandbSink — wandb-marked tests.
# ──────────────────────────────────────────────────────────────────────


def _install_fake_wandb(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake ``wandb`` module into sys.modules and return it."""
    fake = MagicMock(name="wandb")
    fake.init.return_value = MagicMock(name="wandb_run")
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.mark.wandb
def test_wandb_sink_init_calls_wandb_init(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_wandb(monkeypatch)
    from gauntlet.runner.sinks import WandbSink

    sink = WandbSink(run_name="run-a", suite_name="tiny", config={"k": 1})

    fake.init.assert_called_once()
    kwargs = fake.init.call_args.kwargs
    assert kwargs["project"] == "gauntlet"
    assert kwargs["name"] == "run-a"
    assert kwargs["group"] == "tiny"
    assert kwargs["config"] == {"k": 1}
    sink.close()


@pytest.mark.wandb
def test_wandb_sink_log_episode_forwards_metric_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_wandb(monkeypatch)
    from gauntlet.runner.sinks import WandbSink

    sink = WandbSink(run_name="run-a", suite_name="tiny")
    sink.log_episode(_ep(cell_index=2, episode_index=3, success=True, seed=99))

    fake.log.assert_called_once()
    payload: dict[str, Any] = fake.log.call_args.args[0]
    assert payload["cell_index"] == 2
    assert payload["episode_index"] == 3
    assert payload["seed"] == 99
    assert payload["success"] == 1
    assert payload["axis.lighting"] == 0.5


@pytest.mark.wandb
def test_wandb_sink_close_finishes_run(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_wandb(monkeypatch)
    from gauntlet.runner.sinks import WandbSink

    sink = WandbSink(run_name="run-a", suite_name="tiny")
    sink.close()

    fake.init.return_value.finish.assert_called_once()


@pytest.mark.wandb
def test_wandb_sink_log_failure_is_warn_and_continue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_wandb(monkeypatch)
    fake.log.side_effect = RuntimeError("network down")
    from gauntlet.runner.sinks import WandbSink

    sink = WandbSink(run_name="run-a", suite_name="tiny")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sink.log_episode(_ep())
    assert any("WandbSink.log_episode failed" in str(w.message) for w in caught)


@pytest.mark.wandb
def test_wandb_missing_extra_raises_clean_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the [wandb] extra installed AND no stub injected, the
    lazy import inside __init__ raises a clean ImportError carrying the
    install hint and the telemetry warning."""
    monkeypatch.setitem(sys.modules, "wandb", None)
    # Drop any cached sinks module so the sink picks up the missing wandb.
    monkeypatch.delitem(sys.modules, "gauntlet.runner.sinks", raising=False)
    from gauntlet.runner.sinks import WandbSink

    with pytest.raises(ImportError, match=r"gauntlet\[wandb\]"):
        WandbSink(run_name="run-a", suite_name="tiny")


# ──────────────────────────────────────────────────────────────────────
# MlflowSink — mlflow-marked tests.
# ──────────────────────────────────────────────────────────────────────


def _install_fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake ``mlflow`` module into sys.modules and return it."""
    fake = MagicMock(name="mlflow")
    fake.start_run.return_value = MagicMock(name="mlflow_run")
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    # Force a local default URI so the remote-sink warning never fires
    # in tests unless a test explicitly opts in.
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return fake


@pytest.mark.mlflow
def test_mlflow_sink_init_starts_local_run(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_mlflow(monkeypatch)
    from gauntlet.runner.sinks import MlflowSink

    sink = MlflowSink(run_name="run-a", suite_name="tiny", config={"k": "v"})

    fake.set_experiment.assert_called_once_with("tiny")
    fake.start_run.assert_called_once_with(run_name="run-a")
    fake.log_params.assert_called_once_with({"k": "v"})
    sink.close()


@pytest.mark.mlflow
def test_mlflow_sink_log_episode_forwards_numeric_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_mlflow(monkeypatch)
    from gauntlet.runner.sinks import MlflowSink

    sink = MlflowSink(run_name="run-a", suite_name="tiny")
    sink.log_episode(_ep(cell_index=4, episode_index=1, success=False, seed=7))

    fake.log_metrics.assert_called_once()
    metrics: dict[str, float] = fake.log_metrics.call_args.args[0]
    step = fake.log_metrics.call_args.kwargs["step"]
    assert step == 4 * 10_000 + 1
    assert metrics["cell_index"] == 4.0
    assert metrics["episode_index"] == 1.0
    assert metrics["success"] == 0.0
    assert metrics["axis.lighting"] == 0.5


@pytest.mark.mlflow
def test_mlflow_sink_close_calls_end_run(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_mlflow(monkeypatch)
    from gauntlet.runner.sinks import MlflowSink

    sink = MlflowSink(run_name="run-a", suite_name="tiny")
    sink.close()
    fake.end_run.assert_called_once()


@pytest.mark.mlflow
def test_mlflow_sink_log_failure_is_warn_and_continue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_mlflow(monkeypatch)
    fake.log_metrics.side_effect = RuntimeError("tracking server 500")
    from gauntlet.runner.sinks import MlflowSink

    sink = MlflowSink(run_name="run-a", suite_name="tiny")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sink.log_episode(_ep())
    assert any("MlflowSink.log_episode failed" in str(w.message) for w in caught)


@pytest.mark.mlflow
def test_mlflow_remote_tracking_uri_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """When MLFLOW_TRACKING_URI is HTTP(S), the sink warns at init time
    so the user sees the remote destination before any data leaves."""
    fake = _install_fake_mlflow(monkeypatch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")

    from gauntlet.runner.sinks import MlflowSink

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sink = MlflowSink(run_name="run-a", suite_name="tiny")
    assert any(
        "remote" in str(w.message) and "mlflow.example.com" in str(w.message) for w in caught
    )
    sink.close()
    # Reach into the fake to suppress unused-var lint.
    assert fake.start_run.called


@pytest.mark.mlflow
def test_mlflow_missing_extra_raises_clean_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    monkeypatch.delitem(sys.modules, "gauntlet.runner.sinks", raising=False)
    from gauntlet.runner.sinks import MlflowSink

    with pytest.raises(ImportError, match=r"gauntlet\[mlflow\]"):
        MlflowSink(run_name="run-a", suite_name="tiny")
