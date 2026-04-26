# Extension points

Gauntlet exposes six `[project.entry-points]` groups so third-party packages
can extend the harness with a drop-in `pip install` — no fork required.
Each group has a discovery helper in `gauntlet.plugins` (cached via
`functools.lru_cache`) and is wired into the matching consumer module.

| Group              | Discovery helper            | Consumer                           |
|--------------------|------------------------------|------------------------------------|
| `gauntlet.policies`| `discover_policy_plugins`    | `gauntlet.policy.registry`         |
| `gauntlet.envs`    | `discover_env_plugins`       | `gauntlet.env.registry`            |
| `gauntlet.axes`    | `discover_axis_plugins`      | `axis_for` + Suite schema          |
| `gauntlet.samplers`| `discover_sampler_plugins`   | `build_sampler` + Suite schema     |
| `gauntlet.sinks`   | `discover_sink_plugins`      | `gauntlet run --sink <name>`       |
| `gauntlet.cli`     | `discover_cli_plugins`       | `gauntlet ext <name>`              |

## Universal precedence

Built-ins always win on identity collision. A plugin entry point that
loads the same class / factory object the built-in registry already
holds is silent (the dogfood case); a real third-party shadow loads a
different object and triggers a `RuntimeWarning` naming the plugin's
distribution. Same precedent across all six groups.

## `gauntlet.axes`

Contract: zero-arg callable returning a
`gauntlet.env.perturbation.PerturbationAxis`. The returned axis's
`.name` MUST match the entry-point key.

```toml
# pyproject.toml
[project.entry-points."gauntlet.axes"]
my_axis = "my_pkg.axes:make_my_axis"
```

```python
# my_pkg/axes.py
from gauntlet.env.perturbation import (
    AXIS_KIND_CONTINUOUS,
    PerturbationAxis,
    make_continuous_sampler,
)

def make_my_axis() -> PerturbationAxis:
    return PerturbationAxis(
        name="my_axis",
        kind=AXIS_KIND_CONTINUOUS,
        sampler=make_continuous_sampler(0.0, 1.0),
        low=0.0,
        high=1.0,
    )
```

**Caveat:** `set_perturbation` is hardcoded per backend env, so a plugin
axis is only useful when paired with a custom env (via `gauntlet.envs`)
whose `set_perturbation` recognises the axis name. Built-in tabletop
backends gate on a backend-private `AXIS_NAMES` ClassVar set and
reject unknown names.

## `gauntlet.samplers`

Contract: zero-arg constructible class implementing the
`gauntlet.suite.sampling.Sampler` Protocol — a single
`sample(suite, rng) -> list[SuiteCell]` method.

```toml
[project.entry-points."gauntlet.samplers"]
my_sampler = "my_pkg.samplers:MySampler"
```

```python
# my_pkg/samplers.py
from gauntlet.suite.schema import Suite, SuiteCell
import numpy as np

class MySampler:
    def sample(self, suite: Suite, rng: np.random.Generator) -> list[SuiteCell]:
        # Emit cells however you like.
        return [SuiteCell(index=0, values={k: 0.0 for k in suite.axes})]
```

Reference the sampler from a suite YAML's `sampling:` field. The
Suite schema's `sampling` validator consults the plugin registry as a
fallback after the four built-ins (`cartesian`, `latin_hypercube`,
`sobol`, `adversarial`); `n_samples` is required for plugin samplers
just like for LHS / Sobol.

## `gauntlet.sinks`

Contract: class implementing the
`gauntlet.runner.sinks.EpisodeSink` Protocol, with a constructor
accepting keyword-only `run_name: str`, `suite_name: str`, and
`config: Mapping[str, ...] | None` — same shape as the built-in
`WandbSink` / `MlflowSink`.

```toml
[project.entry-points."gauntlet.sinks"]
my_sink = "my_pkg.sinks:MySink"
```

```python
# my_pkg/sinks.py
from gauntlet.runner import Episode

class MySink:
    def __init__(self, *, run_name: str, suite_name: str, config: object) -> None:
        self.run_name = run_name

    def log_episode(self, episode: Episode) -> None:
        ...   # mirror the episode wherever you like

    def close(self) -> None:
        ...   # flush / finish
```

Activate with the new repeatable `--sink <name>` flag:

```bash
gauntlet run my_suite.yaml --policy random --out runs/ --sink my_sink
```

`--wandb` and `--mlflow` keep working (back-compat); `--sink` is
purely additive and can be combined with either or both.

## `gauntlet.cli`

Contract: a `typer.Typer` sub-app, a `click.Command`, or a
`click.Group`. Plugin commands appear at runtime under
`gauntlet ext <name>` so they cannot shadow first-party commands.

```toml
[project.entry-points."gauntlet.cli"]
mycommand = "my_pkg.cli:my_command"
```

```python
# my_pkg/cli.py
import typer

my_command = typer.Typer(name="mycommand", help="My plugin command.")

@my_command.command("ping")
def ping() -> None:
    typer.echo("pong")
```

Invoke as `gauntlet ext mycommand ping`. Click commands are wrapped
in a passthrough Typer command at registration time so flag parsing
flows through the click command's own option parser unchanged.

## Discovery + caching

Every helper is `@lru_cache(maxsize=1)`. Tests that mock
`importlib.metadata.entry_points` must call `cache_clear()` on each
helper they touch in fixture teardown (see
`tests/test_extension_points.py` for the canonical pattern).

Failed `ep.load()` calls are wrapped in a `RuntimeWarning` and the
offending entry point is dropped from the returned dict — a broken
plugin never crashes the rest of gauntlet. Duplicate entry-point
names within one group emit a `RuntimeWarning` naming both
distributions; first-seen wins.

## Dogfood

`pyproject.toml` registers gauntlet's own torch-free, dep-free
built-ins through these groups so the discovery path is exercised on
every install:

- `gauntlet.policies` — `random`, `scripted`, plus the optional `huggingface` / `lerobot` extras.
- `gauntlet.envs` — `tabletop` plus the optional pybullet / genesis / isaac extras.
- `gauntlet.axes` — `lighting_intensity`, `distractor_count`.
- `gauntlet.samplers` — `cartesian`.
- `gauntlet.sinks`, `gauntlet.cli` — empty by design (the built-in
  sinks need optional extras at construction time; there is no
  first-party `ext` command to dogfood).
