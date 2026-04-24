"""Episode result schema.

See :mod:`gauntlet.runner.runner` for the producer side.

The schema is deliberately minimal — Phase 1 task 7 (``Report``) extends it
with reporting helpers but must NOT rename or drop any existing field.
``ConfigDict(extra="forbid")`` is the contract that lets downstream code
key off the exact field names without worrying about silent additions.

Field invariants:

* ``suite_name`` — ``Suite.name`` of the originating suite.
* ``cell_index`` — :attr:`gauntlet.suite.SuiteCell.index` of the cell that
  produced this episode (zero-based ordinal in
  :meth:`gauntlet.suite.Suite.cells` enumeration order).
* ``episode_index`` — zero-based index within ``cell_index``; in
  ``[0, episodes_per_cell)``.
* ``seed`` — the integer seed handed to :meth:`TabletopEnv.reset`. This is
  the bit that bit-reproduces the rollout. Derived deterministically from
  the master seed (see :class:`gauntlet.runner.runner.Runner` docstring).
* ``perturbation_config`` — a copy of :attr:`SuiteCell.values`.

Extensibility:

* ``metadata`` accepts arbitrary scalar values (the pydantic ``int`` type
  is unbounded so it round-trips Python's ``SeedSequence.entropy``, which
  can exceed 64 bits). The Runner uses it to record ``master_seed`` for
  reproducibility of None-seed runs.
* ``video_path`` (Polish "rollout MP4 video recording") is an optional
  string carrying the relative path of an MP4 dumped by the runner when
  the user opts in via ``Runner(record_video=True)``. The default is
  ``None`` so old JSONs (pre-PR Episode dicts) load cleanly under the
  field default. A *new* Episode always emits ``"video_path": null``
  when no video is recorded, which is semantically inert; an older
  ``gauntlet`` reader that lacks the field WILL reject the JSON because
  of ``extra="forbid"`` — that is the standard schema-addition trade-off
  documented here so it does not surprise downstream consumers.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Episode"]


class Episode(BaseModel):
    """Result of one rollout. Pure data; no behaviour."""

    # ``ser_json_inf_nan="strings"`` round-trips ``float('nan')`` /
    # ``float('inf')`` through JSON as ``"NaN"`` / ``"Infinity"`` /
    # ``"-Infinity"`` instead of pydantic's default ``null`` (which then
    # fails revalidation as ``float``). Live Runner-emitted Episodes do
    # not currently carry non-finite floats, but a third-party env that
    # returns NaN reward (broken policy, NaN obs from a learned model)
    # would otherwise produce an unreplayable episodes.json.
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    # Identity / reproducibility.
    suite_name: str
    cell_index: int
    episode_index: int
    seed: int
    perturbation_config: dict[str, float]

    # Outcome.
    success: bool
    terminated: bool
    truncated: bool
    step_count: int
    total_reward: float

    # Extensible bag for future fields (trajectory summary, timing, master
    # seed echo, etc.). Task 7 will add reporting helpers that read keys
    # from here without changing the schema above.
    metadata: dict[str, float | int | str | bool] = Field(default_factory=dict)

    # Optional path to an MP4 recording of this episode. Populated only
    # when the user opts in via ``Runner(record_video=True)``. Stored as
    # a string (not :class:`pathlib.Path`) so the schema round-trips
    # losslessly through pydantic's JSON mode. The path is always
    # *relative* — typically ``"videos/episode_cell{NNNN}_ep{NNNN}_seed
    # {S}.mp4"`` relative to the run output directory — so the HTML
    # report can embed it via ``<video src="{video_path}">`` without a
    # web server. Default ``None`` keeps the field semantically inert
    # for the byte-identical opt-out path.
    video_path: str | None = None
