"""Security regression: Runner uses ``spawn`` start method exclusively.

Phase 2.5 Task 16 — pins the existing safety property of
:class:`gauntlet.runner.runner.Runner`. The Runner refuses every
multiprocessing start method except ``"spawn"``, raising
:class:`ValueError` at construction time.

Threat model:

* The Runner serialises ``WorkItem`` objects across the worker
  boundary via Python's standard pickle protocol. The objects pickled
  are: a string suite name, ints, a dict of axis-name -> float, a
  :class:`numpy.random.SeedSequence` (immutable, integer state only),
  and the ``policy_factory`` / ``env_factory`` callables provided by
  the caller.

* ``spawn`` start method launches a fresh interpreter for each worker,
  so the worker never inherits the parent's open file descriptors,
  threads, or in-memory state. ``fork`` would inherit MuJoCo's GL
  context (broken — see runner.py:135-141) AND any other process
  state, including mutable globals an attacker who controls the
  policy module could pre-populate. ``forkserver`` is intermediate
  but still inherits a frozen import state from the forkserver
  bootstrap.

* The pickle deserialisation IS itself a risk surface — an attacker
  who can substitute the pickled payload (e.g. via shared memory in
  a sandbox) can trigger arbitrary code execution. We do not defend
  against that here (the worker boundary is intra-process by design),
  but we DO pin that the start method is ``spawn`` so:
    1. each worker starts with no inherited mutable state, and
    2. the WorkerInitArgs (``env_factory``, ``policy_factory``) are
       both module-level callables — picklable, but not arbitrary
       code injected at runtime.

This is documented as a security model rather than a defence: gauntlet
trusts the policy / env modules the user passes on the CLI. The pin
only checks that the multiprocessing start method does not silently
shift to ``fork`` (which would break the contract).

References:
* https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
* https://docs.python.org/3/library/pickle.html — § "Restricting Globals"
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

from gauntlet.runner.runner import Runner

# ----- Runner refuses non-spawn start methods -------------------------------


@pytest.mark.parametrize("bad_method", ["fork", "forkserver", "thread", "", "Spawn", "SPAWN"])
def test_runner_rejects_non_spawn_start_method(bad_method: str) -> None:
    """``Runner(start_method=...)`` MUST raise ``ValueError`` for anything
    except the literal string ``"spawn"`` (case-sensitive).

    The MuJoCo renderer holds GL state that ``fork`` will not safely
    duplicate (see runner.py:136-141), and any non-spawn method would
    also break the security model in this file's docstring.
    """
    with pytest.raises(ValueError, match="spawn"):
        Runner(n_workers=2, start_method=bad_method)


def test_runner_default_start_method_is_spawn() -> None:
    """Construction with no override must land on ``spawn``.

    Pin against a future refactor that quietly changes the default
    (e.g. "use fork on Linux for performance"). The Runner's docstring
    explicitly forbids that; this test makes the contract executable.
    """
    runner = Runner(n_workers=2)
    # Runner stores the start_method on a private attribute; the safer
    # contract test is "constructing with start_method='spawn' explicit
    # also works", since both should produce the same Runner.
    explicit = Runner(n_workers=2, start_method="spawn")
    assert runner._start_method == explicit._start_method == "spawn"


def test_spawn_is_an_available_start_method_on_this_platform() -> None:
    """Sanity check: the ``spawn`` start method MUST be supported.

    Some exotic CI environments (musl-libc + missing /proc) can fail
    to provide it. If that ever happens we want a clear failure here
    rather than a confusing pool-creation error in production.
    """
    available = mp.get_all_start_methods()
    assert "spawn" in available, (
        f"spawn start method not available on this platform; got {available}"
    )


# ----- Runner does not accept inherently-untrusted callables ---------------


def test_runner_n_workers_must_be_positive() -> None:
    """Defensive: ``n_workers < 1`` is rejected.

    A ``n_workers=0`` Runner would silently skip every WorkItem; a
    negative value would break ``mp.Pool`` with an obscure error.
    Asserting the bound here catches future refactors that drop the
    explicit guard in ``Runner.__init__``.
    """
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        Runner(n_workers=0)
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        Runner(n_workers=-1)
