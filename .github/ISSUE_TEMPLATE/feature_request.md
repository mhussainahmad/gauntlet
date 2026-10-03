---
name: Feature request
about: Propose a new perturbation axis, metric, sampler, env, sink, or CLI surface.
title: "feat: "
labels: enhancement
assignees: ''
---

## Motivation

<!--
Which audience from PRODUCT.md does this serve — the VLA researcher
or the fleet operator? What concrete failure-mode analysis does
Gauntlet currently make awkward or impossible? Be specific. Cite a
paper / benchmark / production incident if applicable.
-->

## Proposal

<!--
The shape of the change. If you have a sketch of the public API
(new function signature / new CLI flag / new Suite YAML key), include
it here. Bonus points for a worked example end-to-end.
-->

## Alternatives considered

<!--
What else could solve this? Why is the proposal the right shape?
The §6 hard rules in GAUNTLET_SPEC.md are non-negotiable; a
proposal that fights them needs to explain why.
-->

## Impact on the public API

- [ ] Adds a new public symbol (semver: minor).
- [ ] Changes an existing public symbol (semver: minor → major;
      requires a deprecation cycle per `docs/stability.md`).
- [ ] Adds a new on-disk schema field (additive, non-breaking).
- [ ] Adds a new optional extra in `pyproject.toml`.
- [ ] Pure internal refactor (no API change).

## Anti-feature check

<!--
Every backlog entry in docs/backlog.md ends with an "Anti-feature?"
section asking what could go wrong. Apply that lens here. If the
feature is asymmetric across backends, dead code on baselines, or
creates a misuse surface, name it now.
-->
