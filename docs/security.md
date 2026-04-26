# Gauntlet security model

Phase 2.5 Task 16. This document records the threat model the harness
defends against, the regression tests that pin the safe behaviour, and
the CI gates that catch fresh CVEs and accidentally-committed secrets.

## Trust boundaries

Gauntlet is a developer tool. It runs on the user's workstation or in a
CI runner the user controls; it does not face the public internet. The
honest trust model is:

* **Trusted**: the gauntlet source tree, the policy / env modules the
  user passes on the CLI, the suite YAML the user writes, the
  `pyproject.toml` / `uv.lock` resolved dependency tree.
* **Partially trusted (input boundary)**: the suite YAML and the
  episodes / report JSON files passed via CLI arguments. These may
  come from a shared CI artefact, a teammate's branch, or a fuzzer.
  The harness validates them before any side effect.
* **Out of scope**: protecting against an attacker who has already
  achieved arbitrary-code execution in the user's Python interpreter
  (e.g. via a malicious policy module). Such an attacker can do
  anything the user can do.

## Defended attack classes

Each row maps to one or more regression tests in `tests/`. The "safe
because" column states the actual mechanism — these tests pin the
existing behaviour, they do not add new defences.

| Class | OWASP / CWE | Test module | Safe because |
|-------|-------------|-------------|--------------|
| Unsafe YAML deserialisation | CVE-2017-18342 class | `tests/test_security_yaml.py`, `tests/test_security_yaml_guard.py` | Every YAML read in `src/gauntlet` flows through `gauntlet.security.safe_yaml_load`, a thin wrapper around `yaml.safe_load`. The grep gate in `tests/test_security_yaml_guard.py::test_no_unsafe_yaml_load_in_src` asserts no `yaml.load(` occurrence (without `safe_`) appears under `src/gauntlet` outside the wrapper module — closing the door on a future refactor that re-introduces the unsafe loader. |
| Path traversal at sink boundaries | OWASP A01 / CWE-22 | `tests/test_security_paths.py` | The CLI section pins typer-boundary behaviour for traversal-shaped inputs (`../etc/passwd`, NUL injection, symlink escape) — every subcommand surfaces a clean `typer.Exit(1)`. The helper section pins `gauntlet.security.safe_join`, which resolves `base + parts` and rejects parent-traversal, absolute-path injection, and (in `follow_symlinks=False` mode) symlink escape. `safe_join` is wired into `gauntlet.runner.parquet.parquet_path_for` as defence-in-depth; on every legitimate input the result equals the bare `/`-join, but a future refactor that threads a user-controlled string into the filename cannot bypass the check without a visible diff. |
| HTML XSS in report | OWASP A03 / CWE-79 | `tests/test_security_html_report.py` | `gauntlet.report.html` builds its Jinja env with `select_autoescape(("html","xml","jinja"))` and the template is `report.html.jinja`. Every user-controlled interpolation (`suite_name`, axis names, cluster keys) is HTML-escaped. The embedded `<script id="report-data">` JSON block uses `|tojson` which escapes `</script>` as `<\/script>`. |
| Override-string command injection | OWASP A03 / CWE-78 | `tests/test_security_overrides.py` | `gauntlet.replay.parse_override` is pure string slicing + `float()`. There is no shell invocation, `eval`, or template expansion. The test sniffs `sys.modules` to catch a future refactor that lazily imports `subprocess`. |
| Multiprocessing fork-time state inheritance | n/a | `tests/test_security_pickle.py` | `gauntlet.runner.runner.Runner` raises `ValueError` for any `start_method` except `"spawn"` (runner.py:135-141). Every worker starts with a fresh interpreter and no inherited mutable globals. |

## Centralised helpers (`gauntlet.security`)

Phase 2.5 Task 16 collected the input-boundary helpers under one
package so the audit surface is one ripgrep away:

* `gauntlet.security.safe_join(base, *parts, follow_symlinks=True)` —
  sandboxed path join. Raises `PathTraversalError` (a `ValueError`
  subclass) on `..` escape, absolute-path injection, or — when
  `follow_symlinks=False` — a symlink whose target leaves `base`.
  Use at boundaries where a filename component is joined under a
  trusted root.
* `gauntlet.security.safe_yaml_load(stream)` — canonical YAML entry
  point. Thin alias of `yaml.safe_load`; exception envelope is
  unchanged. The CI grep gate asserts no `yaml.load(` (without
  `safe_`) occurrence appears under `src/gauntlet` outside this
  wrapper.
* `gauntlet.security.PathTraversalError`,
  `gauntlet.security.YamlSecurityError` — `ValueError` subclasses
  reserved for the helpers above. `YamlSecurityError` is forward-
  compatible (the current wrapper does not raise it; a strict-mode
  policy layered later would).

### Boundaries intentionally NOT hardened

The honest scope of `safe_join`. These boundaries accept cross-dir
or arbitrary user-supplied paths by design; sandboxing them would
regress legitimate use:

* `gauntlet.suite.loader._resolve_and_check_pilot_report` — pilot-
  report paths are intentionally resolved relative to the suite YAML's
  parent and may legitimately reference a sibling directory (`../
  pilots/foo.json`).
* `Runner.trajectory_dir` / `Runner.video_dir` / `Runner.cache_dir` —
  user-supplied output roots. The user is the trust root; there is
  no base to sandbox against.
* `gauntlet.plugins` entry-point discovery — relies on the package-
  resource path materialised by `importlib.metadata`. The plugin
  contract trusts the resolved entry-point target.

If a future audit decides any of these should be hardened, the change
should land as a follow-up task with an `xfail(strict=True, reason=
"follow-up")` regression test added here first.

## Known follow-up gaps

None at the time of writing. The boundary tests above all pass against
the existing `src/` code; if a future audit pass uncovers a real
vulnerability, the discovering PR should add an `xfail(strict=True,
reason="follow-up")` test here and open a separate task to fix the
underlying issue.

## CI gates

Two GitHub Actions jobs in `.github/workflows/ci.yml` run on every
push and PR:

* `security-audit` — runs `uvx pip-audit --strict --vulnerability-
  service osv .` against the resolved core dependency tree. OSV
  aggregates GHSA + NVD + OSV-direct. This is a HARD GATE: any new CVE
  in a pinned dep fails the build. To accept a deferral, document the
  reasoning in the PR body and pass `--ignore-vuln <ID>`.
* `secret-scan` — runs `detect-secrets-hook --baseline
  .secrets.baseline` over every git-tracked file. Any finding NOT
  already in `.secrets.baseline` fails the build. To accept a future
  false positive, re-run `uvx detect-secrets scan > .secrets.baseline`
  from a clean tree and commit the diff with a justification.

The audit toolchain itself lives in the `[security]` dependency-group
in `pyproject.toml`; nothing in `src/` or the test suite imports it.
