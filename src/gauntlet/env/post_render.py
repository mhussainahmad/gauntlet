"""Runner wiring for the wrapper-implemented axes.

Three axes are not backend perturbations but env wrappers:

* ``color_shift_synthetic`` — :class:`gauntlet.env.color_attack.ColorShiftWrapper`
* ``image_attack`` — :class:`gauntlet.env.image_attack.ImageAttackWrapper`
* ``instruction_paraphrase`` — :class:`gauntlet.env.instruction.InstructionWrapper`

:func:`wrap_env_factory` looks at a :class:`~gauntlet.suite.Suite`'s axes
and returns an env factory that applies exactly the wrappers the suite
needs, so a suite declaring any of these axes runs through
``gauntlet run`` / :class:`~gauntlet.runner.Runner` without the caller
building wrappers by hand.

Wrapper order (outermost first)::

    InstructionWrapper(ImageAttackWrapper(ColorShiftWrapper(backend)))

Colour shift sits closest to the renderer because it models a colour
cast in the camera / ISP; image attacks (noise, JPEG, occlusion, camera
dropout) model what happens to the frame after that. The instruction
overlay touches no pixels. The order is part of the reproducibility
contract — changing it changes attacked frames.

Image axes need images. When the factory comes from the env registry
and an image axis is present, the backend is built with
``render_in_obs=True``. A caller-supplied factory is used as-is; if it
emits no image while an attack is active, the wrapper raises on the
first observation rather than producing a sweep where the axis did
nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final, cast

from gauntlet.env.base import GauntletEnv
from gauntlet.env.color_attack import ColorShiftWrapper
from gauntlet.env.image_attack import ImageAttackWrapper
from gauntlet.env.instruction import InstructionWrapper
from gauntlet.env.registry import get_env_factory

if TYPE_CHECKING:  # pragma: no cover -- typing-only import
    from gauntlet.suite.schema import Suite

__all__ = [
    "IMAGE_AXES",
    "WRAPPER_AXES",
    "WrappedEnvFactory",
    "suite_needs_render",
    "wrap_env_factory",
]

IMAGE_AXES: Final[frozenset[str]] = frozenset({"image_attack", "color_shift_synthetic"})
WRAPPER_AXES: Final[frozenset[str]] = IMAGE_AXES | {"instruction_paraphrase"}


@dataclass(frozen=True)
class WrappedEnvFactory:
    """Picklable zero-arg factory: build ``base()`` and apply wrappers.

    A frozen dataclass rather than a closure so spawned Runner workers
    can unpickle it.
    """

    base: Callable[[], GauntletEnv]
    image_attack: bool = False
    color_shift: bool = False
    paraphrases: tuple[str, ...] | None = None

    def __call__(self) -> GauntletEnv:
        # The wrappers satisfy GauntletEnv structurally at runtime, but
        # their per-instance AXIS_NAMES (shadowing the ClassVar) hides
        # that from mypy; the casts document the widening.
        env: GauntletEnv = self.base()
        if self.color_shift:
            env = cast(GauntletEnv, ColorShiftWrapper(env))
        if self.image_attack:
            env = cast(GauntletEnv, ImageAttackWrapper(env))
        if self.paraphrases is not None:
            env = cast(GauntletEnv, InstructionWrapper(env, self.paraphrases))
        return env


def suite_needs_render(suite: Suite) -> bool:
    """True iff the suite declares an axis that operates on rendered images."""
    return any(name in IMAGE_AXES for name in suite.axes)


def wrap_env_factory(
    suite: Suite,
    env_factory: Callable[[], GauntletEnv] | None,
) -> Callable[[], GauntletEnv]:
    """Return an env factory that honours the suite's wrapper axes.

    Args:
        suite: the suite being run.
        env_factory: caller-supplied factory, or ``None`` to dispatch on
            ``suite.env`` through the env registry.

    Returns:
        The input factory (or the registry factory) unchanged when the
        suite uses none of :data:`WRAPPER_AXES`; otherwise a
        :class:`WrappedEnvFactory` around it.
    """
    axes = set(suite.axes)
    if env_factory is None:
        backend = get_env_factory(suite.env)
        base: Callable[[], GauntletEnv] = (
            partial(backend, render_in_obs=True) if axes & IMAGE_AXES else backend
        )
    else:
        base = env_factory
    if not axes & WRAPPER_AXES:
        return base

    paraphrases: tuple[str, ...] | None = None
    spec = suite.axes.get("instruction_paraphrase")
    if spec is not None:
        paraphrases = spec.paraphrases()
        if paraphrases is None:
            raise ValueError(
                "instruction_paraphrase axis needs a string list under 'values' "
                "(one natural-language instruction per index)"
            )
    return WrappedEnvFactory(
        base=base,
        image_attack="image_attack" in axes,
        color_shift="color_shift_synthetic" in axes,
        paraphrases=paraphrases,
    )
