---
name: Bug report
about: Report a defect — incorrect behaviour, crash, regression, doc bug.
title: "bug: "
labels: bug
assignees: ''
---

## Summary

<!-- One sentence. What did Gauntlet do that it should not have done? -->

## Reproducer

```bash
# Minimal CLI / Python invocation that surfaces the bug. Ideally a
# < 10-line example that another developer can paste into a fresh
# venv and reproduce.
```

If the bug needs a custom Suite YAML or policy adapter, paste it inline
or attach the file.

## Expected behaviour

<!-- What should have happened instead? -->

## Observed behaviour

<!-- What actually happened. Include exact stderr / traceback / report
output. Quote error messages verbatim — do not paraphrase. -->

## Environment

- Gauntlet version (`python -c "import gauntlet; print(gauntlet.__version__)"`):
- Python version:
- OS / distribution:
- Install method (`pip install gauntlet-robotics`, `uv sync`, source checkout):
- Which extras are installed (`hf`, `lerobot`, `pybullet`, `genesis`, `isaac`, `monitor`, `ros2`, …):

## Determinism

If this is a determinism / reproducibility bug, please include:

- [ ] The exact `seed` value(s) used.
- [ ] Whether the bug reproduces with `n_workers=1` (serial) as well as parallel.
- [ ] The `episode_hash` of the divergent rollouts, if you have them.

## Additional context

<!-- Logs, screenshots, related issues, anything else. -->
