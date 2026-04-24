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
| Unsafe YAML deserialisation | CVE-2017-18342 class | `tests/test_security_yaml.py` | `gauntlet.suite.loader` uses `yaml.safe_load`, which rejects `!!python/object/apply`, `!!python/object/new`, and `!!python/name`. |
| HTML XSS in report | OWASP A03 / CWE-79 | `tests/test_security_html_report.py` | `gauntlet.report.html` builds its Jinja env with `select_autoescape(("html","xml","jinja"))` and the template is `report.html.jinja`. Every user-controlled interpolation (`suite_name`, axis names, cluster keys) is HTML-escaped. The embedded `<script id="report-data">` JSON block uses `|tojson` which escapes `</script>` as `<\/script>`. |
| CLI path traversal / NUL injection | OWASP A01 / CWE-22 | `tests/test_security_paths.py` | All CLI subcommands gate file inputs through `Path.is_file()` before any open attempt and surface failures as clean `typer.Exit(1)`. Traversal-shaped paths fail with `not found` or with `yaml.YAMLError` / `ValidationError`; no payload is ever silently accepted as a valid suite. |
| Override-string command injection | OWASP A03 / CWE-78 | `tests/test_security_overrides.py` | `gauntlet.replay.parse_override` is pure string slicing + `float()`. There is no shell invocation, `eval`, or template expansion. The test sniffs `sys.modules` to catch a future refactor that lazily imports `subprocess`. |
| Multiprocessing fork-time state inheritance | n/a | `tests/test_security_pickle.py` | `gauntlet.runner.runner.Runner` raises `ValueError` for any `start_method` except `"spawn"` (runner.py:135-141). Every worker starts with a fresh interpreter and no inherited mutable globals. |

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
