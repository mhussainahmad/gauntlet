"""DriftReport / PerEpisodeDrift schema — torch-free round-trip checks.

``ConfigDict(extra="forbid")`` + ``model_dump`` + ``model_validate`` is
the contract callers (HTML template, JSON consumers) rely on. The
round-trip test locks in the payload shape so future additions are
either additive or visibly breaking.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from gauntlet.monitor import DriftReport, PerEpisodeDrift


def _sample_per_episode(i: int = 0) -> PerEpisodeDrift:
    """Hand-built :class:`PerEpisodeDrift` for reuse across tests."""
    return PerEpisodeDrift(
        cell_index=i,
        episode_index=0,
        seed=42 + i,
        perturbation_config={"lighting_intensity": 0.8},
        n_steps=20,
        reconstruction_error_mean=0.1 + 0.01 * i,
        reconstruction_error_max=0.5,
        action_std_per_dim=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        action_entropy=0.4,
    )


def _sample_report() -> DriftReport:
    per_episode = [_sample_per_episode(i) for i in range(3)]
    return DriftReport(
        suite_name="drift-smoke",
        n_episodes=len(per_episode),
        ae_mode="state",
        ae_latent_dim=8,
        ae_reference_suite="reference-smoke",
        reference_reconstruction_error_mean=0.05,
        reference_reconstruction_error_p95=0.15,
        candidate_reconstruction_error_mean=0.2,
        candidate_reconstruction_error_p95=0.6,
        candidate_action_entropy_mean=0.4,
        per_episode=per_episode,
        top_ood_episodes=[2, 1, 0],
    )


def test_drift_report_round_trips_through_json() -> None:
    """DriftReport -> JSON string -> DriftReport is an identity."""
    original = _sample_report()
    payload = json.loads(original.model_dump_json())
    restored = DriftReport.model_validate(payload)
    assert restored.model_dump() == original.model_dump()


def test_per_episode_drift_round_trips() -> None:
    original = _sample_per_episode(5)
    payload = json.loads(original.model_dump_json())
    restored = PerEpisodeDrift.model_validate(payload)
    assert restored.model_dump() == original.model_dump()


def test_drift_report_rejects_unknown_fields() -> None:
    """``extra="forbid"`` catches silent field additions by downstream callers."""
    payload = _sample_report().model_dump(mode="json")
    payload["drift_score"] = 0.42  # not on the schema
    with pytest.raises(ValidationError):
        DriftReport.model_validate(payload)


def test_drift_report_rejects_wrong_ae_mode() -> None:
    payload = _sample_report().model_dump(mode="json")
    payload["ae_mode"] = "nonsense"
    with pytest.raises(ValidationError):
        DriftReport.model_validate(payload)


def test_ae_reference_suite_is_optional() -> None:
    """None is a valid ``ae_reference_suite`` (RFC §12 default)."""
    payload = _sample_report().model_dump(mode="json")
    payload["ae_reference_suite"] = None
    report = DriftReport.model_validate(payload)
    assert report.ae_reference_suite is None
