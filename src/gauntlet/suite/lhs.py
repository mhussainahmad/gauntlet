"""Latin Hypercube Sampling (LHS) sampler — McKay/Beckman/Conover (1979).

The real algorithm + per-axis range mapping lands in a later commit in
this same Polish task. This file is the placeholder surface so the
:func:`gauntlet.suite.sampling.build_sampler` dispatch table can name
:class:`LatinHypercubeSampler` from the very first commit that adds the
:class:`Sampler` protocol.

When the next commit lands, the body of :meth:`LatinHypercubeSampler.sample`
is replaced with the real numpy implementation; the public surface
(class name, method signature, return shape) does not change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gauntlet.suite.schema import Suite, SuiteCell

__all__ = ["LatinHypercubeSampler"]


class LatinHypercubeSampler:
    """Stratified-sampling sampler — placeholder for the real implementation.

    The follow-up commit replaces :meth:`sample` with the McKay 1979
    algorithm: divide ``[0, 1]`` into ``n_samples`` equal-width strata
    per axis, draw one sample per stratum, randomly permute the
    per-axis stratum index, then map each unit-cube point onto the
    axis's declared range / value list.
    """

    def sample(self, suite: Suite, rng: np.random.Generator) -> list[SuiteCell]:
        del suite, rng
        raise NotImplementedError(
            "LatinHypercubeSampler.sample is implemented in a later commit "
            "in the same Polish task series."
        )
