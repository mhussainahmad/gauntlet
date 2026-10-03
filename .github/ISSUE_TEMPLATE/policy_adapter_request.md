---
name: Policy adapter request
about: Ask for a new VLA / diffusion / RL policy adapter.
title: "policy-adapter: "
labels: ["enhancement", "policy-adapter"]
assignees: ''
---

## Policy

- Name + paper / repo URL:
- License:
- Embodiment the checkpoint targets (action-space dim, observation shape, instruction modality):
- Public checkpoint(s) available? (HF Hub repo IDs, file sizes):

## Embodiment fit

Gauntlet's `TabletopEnv` is **7-D EE-twist + gripper** with a
configurable image observation. Does this policy's pretrained
action space match? If not (eg. 6-D joint position vs 7-D EE
twist), how do you propose bridging — `action_remap`, a fine-tune
pointer, or a new env variant?

If the answer is "the user must fine-tune," that is fine — Gauntlet
ships adapters whose zero-shot success is ~0% (see the SmolVLA
example) — but the adapter MUST print a runtime warning banner
before downloading weights so a new user knows what to expect.

## Adapter sketch

```python
# The ≤ 20-line factory you'd write. See
# examples/evaluate_openvla.py and examples/evaluate_smolvla.py
# for the reference shape.
```

## Optional-extra footprint

What pip packages does the adapter pull in? Gauntlet's core stays
torch-free; any torch-heavy adapter lands behind an optional extra
in `pyproject.toml`.

## Test plan

How will we cover the adapter without downloading 3+ GB of weights in
CI? The existing pattern (`tests/lerobot/`, `tests/hf/`) is mock-driven
modules that exercise the wiring; live runs are out-of-CI examples.
