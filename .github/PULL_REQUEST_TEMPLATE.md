## Summary

<!-- One paragraph. What does this PR change and why? -->

## Type of change

- [ ] Bug fix (non-breaking, fixes an issue).
- [ ] New feature (non-breaking, adds public surface).
- [ ] Breaking change — public-API symbol removed / renamed, or schema
      field semantics changed. Requires deprecation per
      `docs/stability.md`.
- [ ] Documentation only.
- [ ] Internal refactor (no public API change).

## Linked issue / RFC / backlog item

<!-- Fixes #123, implements RFC-005, lands B-37. -->

## Pre-merge checklist

- [ ] `uv run ruff check .` clean.
- [ ] `uv run mypy src/gauntlet` clean.
- [ ] `uv run pytest -m "<your scope>"` clean. Full CI matrix runs on
      PR push.
- [ ] If public API changed: `CHANGELOG.md` entry under
      `## [Unreleased]`.
- [ ] If on-disk schema changed: additive, or deprecation note added.
- [ ] If a new optional extra: added to `pyproject.toml` and documented
      in README.
- [ ] If a new RFC: file at `docs/phaseN-rfc-NNN-<slug>.md` and linked
      from README.

## Reproducibility

- [ ] Every new rollout reproducible from `(suite_name, axis_config,
      seed)`. If determinism is not yet pinned for this code path,
      called out explicitly in the description.
- [ ] If the PR touches Episode / Report serialisation, tests use
      `gauntlet.runner.episode_deterministic_dump` (timing fields are
      not part of the determinism contract).

## Risks / follow-ups

<!--
What could break? What did we knowingly not do? What's the next PR
that builds on this?
-->

