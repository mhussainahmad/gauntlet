"""Instruction-paraphrase wrapper for the ``instruction_paraphrase`` axis (B-05).

Backend-agnostic language-perturbation wrapper. Wraps any
:class:`gauntlet.env.base.GauntletEnv` and injects a single string under
``obs["instruction"]`` on every :meth:`reset` / :meth:`step` so VLA
policies (OpenVLA, SmolVLA, pi0 family) can read the active task
phrasing. The wrapper does not enter the inner sim — it operates purely
on the observation dict the env returns.

Why a wrapper rather than a per-backend axis? The active instruction is
a *language* perturbation, not a physics one. Every backend already
returns the same observation surface (a dict); injecting one extra key
post-step keeps the four physics backends (MuJoCo, PyBullet, Genesis,
Isaac) untouched and lets the wrapper own the entire paraphrase
registry. The trade-off is honest: scripted / random policies that
ignore ``obs["instruction"]`` see no behavioural change — exactly the
"VLA-only signal" anti-feature note in backlog B-05. That's the point:
B-05's value is the *measurement gap* it surfaces between policies that
read language and those that don't.

Encoding of the axis value
--------------------------
The axis is **categorical, integer-coded** — the suite YAML's
``values:`` shape carries a string list (the paraphrases) and the
schema layer enumerates them to ``(0.0, 1.0, ..., len-1)`` indices.
:meth:`set_perturbation` rounds the float index back to an int and
selects the matching string from the registry the wrapper was
constructed with.

Determinism
-----------
The wrapper holds no RNG. The injected ``obs["instruction"]`` value is
a pure function of ``(paraphrases, current_index)`` — deterministic by
construction. ``reset(seed=...)`` and ``step(action)`` simply forward
to the inner env and overlay the current paraphrase on the returned
obs.

Integration
-----------
Currently a *building block*. The runner does not yet auto-instantiate
the wrapper when a suite declares an ``instruction_paraphrase`` axis —
that wiring (extracting the paraphrase list from the YAML via
:meth:`gauntlet.suite.schema.AxisSpec.paraphrases`, building the
wrapper, threading the per-cell index through) is a follow-up. Today,
callers that want language perturbations construct
``InstructionWrapper(inner_env, paraphrases=("...", "...", ...))``
themselves and hand the result to the runner via ``env_factory``.

Backward compatibility
----------------------
The wrapper is opt-in. Without it, ``obs["instruction"]`` simply does
not appear and policies that read it MUST use ``.get("instruction",
"")`` — exactly the same contract any optional obs key follows.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym

from gauntlet.env.base import Action, GauntletEnv, Observation

__all__ = [
    "AXIS_NAME",
    "DEFAULT_INSTRUCTION_KEY",
    "InstructionWrapper",
]


# Canonical name of the axis; mirrors :func:`gauntlet.env.image_attack`'s
# practice of pinning the name in one place so suite loaders + tests can
# import it instead of re-typing the string literal.
AXIS_NAME: str = "instruction_paraphrase"

# Key the wrapper writes into the observation dict. Pinned here so VLA
# policies and tests share the same constant.
DEFAULT_INSTRUCTION_KEY: str = "instruction"


def _index_from_value(value: float, n_paraphrases: int) -> int:
    """Round a float axis value to the nearest paraphrase index.

    Raises :class:`ValueError` when the rounded index lies outside
    ``[0, n_paraphrases)``. The axis value travels as a float through
    the categorical sampler; the wrapper converts it back to an int
    here. Mirrors the helper layout in
    :mod:`gauntlet.env.image_attack` (``_attack_id_from_value``).
    """
    idx = round(float(value))
    if idx < 0 or idx >= n_paraphrases:
        raise ValueError(
            f"instruction_paraphrase: index {idx} (rounded from {value!r}) "
            f"out of range [0, {n_paraphrases}); the wrapper was constructed "
            f"with {n_paraphrases} paraphrase(s)."
        )
    return idx


class InstructionWrapper:
    """Wrap a :class:`GauntletEnv` and inject ``obs["instruction"]``.

    Structurally satisfies :class:`GauntletEnv`: forwards every Protocol
    method to the inner env, augmenting :attr:`AXIS_NAMES` with
    ``"instruction_paraphrase"`` and intercepting that axis in
    :meth:`set_perturbation`. Perturbations the inner env handles flow
    through unchanged.

    Attribute access for non-Protocol methods (e.g. backend-specific
    helpers like
    :meth:`gauntlet.env.tabletop.TabletopEnv.set_initial_state_ood_prior`)
    is proxied via :meth:`__getattr__` so callers can keep using the
    inner surface without unwrapping.

    Parameters
    ----------
    env:
        Inner :class:`GauntletEnv` to wrap. Must expose the standard
        ``reset`` / ``step`` / ``set_perturbation`` /
        ``restore_baseline`` / ``close`` surface.
    paraphrases:
        Tuple of natural-language strings — one per categorical axis
        index. Index 0 is the baseline phrasing (e.g.
        ``"pick up the red cube"``); higher indices are the OOD
        paraphrases (``"grab the crimson block"``,
        ``"move the scarlet box"``). Must be non-empty.
    instruction_key:
        Optional override for the obs key. Defaults to
        :data:`DEFAULT_INSTRUCTION_KEY` (``"instruction"``) which is
        what the OpenVLA / SmolVLA / pi0 policy adapters in
        :mod:`gauntlet.policy` consume.
    """

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset()  # populated per-instance below

    def __init__(
        self,
        env: GauntletEnv,
        paraphrases: tuple[str, ...],
        *,
        instruction_key: str = DEFAULT_INSTRUCTION_KEY,
    ) -> None:
        if len(paraphrases) == 0:
            raise ValueError(
                "InstructionWrapper requires at least one paraphrase string; got an empty tuple"
            )
        self._inner: GauntletEnv = env
        self._paraphrases: tuple[str, ...] = tuple(paraphrases)
        self._instruction_key: str = instruction_key
        # Per-instance AXIS_NAMES override — append the instruction axis
        # to the inner backend's set. Mirrors the ImageAttackWrapper
        # pattern so the GauntletEnv structural check on the wrapper
        # still passes.
        inner_axes = type(env).AXIS_NAMES
        self.AXIS_NAMES = frozenset(inner_axes | {AXIS_NAME})  # type: ignore[misc]
        # Default to index 0 (baseline phrasing). The runner sets the
        # per-cell index via ``set_perturbation`` before each
        # ``reset(seed=...)`` call.
        self._current_index: int = 0

    # ----- gym.Space surface --------------------------------------------------
    @property
    def observation_space(self) -> gym.spaces.Space[Any]:
        return self._inner.observation_space

    @property
    def action_space(self) -> gym.spaces.Space[Any]:
        return self._inner.action_space

    # ----- read-only inspection ----------------------------------------------
    @property
    def paraphrases(self) -> tuple[str, ...]:
        """Return the paraphrase registry the wrapper was constructed with."""
        return self._paraphrases

    @property
    def current_instruction(self) -> str:
        """Return the string that will be injected on the next obs."""
        return self._paraphrases[self._current_index]

    # ----- GauntletEnv.set_perturbation --------------------------------------
    def set_perturbation(self, name: str, value: float) -> None:
        """Queue a perturbation; instruction_paraphrase stays here, others delegate.

        ``instruction_paraphrase`` is intercepted and translated into a
        zero-based index into the paraphrase registry; every other axis
        is forwarded to the inner env. Raises :class:`ValueError` for
        unknown names — same contract as the underlying
        :meth:`GauntletEnv.set_perturbation`.
        """
        if name == AXIS_NAME:
            self._current_index = _index_from_value(value, len(self._paraphrases))
            return
        # Defensive: catch unknowns at the wrapper boundary even though
        # the inner env will also reject them, for a clearer error path.
        if name not in self.AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        self._inner.set_perturbation(name, value)

    def restore_baseline(self) -> None:
        """Reset the active paraphrase to index 0 + delegate to the inner env."""
        self._current_index = 0
        self._inner.restore_baseline()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset inner env, then overlay ``obs[instruction_key]``.

        The wrapper holds no RNG of its own — the injected paraphrase is
        a pure function of the current index, which the runner sets via
        :meth:`set_perturbation` before this call.
        """
        obs, info = self._inner.reset(seed=seed, options=options)
        return self._apply_to_obs(obs), info

    def step(
        self,
        action: Action,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Step inner env, then overlay ``obs[instruction_key]``."""
        obs, reward, terminated, truncated, info = self._inner.step(action)
        return self._apply_to_obs(obs), reward, terminated, truncated, info

    def close(self) -> None:
        """Release the inner env's resources. Idempotent."""
        self._inner.close()

    # ----- attribute proxy ---------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access for backend-specific methods.

        :meth:`__getattr__` is only invoked when normal attribute lookup
        fails, so the wrapper's own methods always win. Callers that
        rely on backend extras (e.g. ``set_initial_state_ood_prior``)
        keep working without unwrapping. The proxy intentionally does
        NOT proxy ``_inner`` itself — that would loop.
        """
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    # ----- internals ---------------------------------------------------------
    def _apply_to_obs(self, obs: Observation) -> Observation:
        """Inject the active paraphrase under ``obs[instruction_key]``.

        Mutates the obs dict in place (matching the
        :class:`gauntlet.env.image_attack.ImageAttackWrapper` style) and
        returns it for caller convenience. The value type is ``str``;
        callers that pre-typed ``obs`` to ``dict[str, NDArray]`` should
        treat the instruction key specially.
        """
        # ``Observation`` is typed as ``dict[str, NDArray]`` for the
        # array-shaped keys; the instruction key carries a plain string
        # which downstream VLA adapters consume verbatim. The cast is
        # intentional — we deliberately step outside the array-only
        # invariant here.
        obs[self._instruction_key] = self._paraphrases[self._current_index]  # type: ignore[assignment]
        return obs
