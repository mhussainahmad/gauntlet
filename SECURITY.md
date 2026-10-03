# Security policy

## Supported versions

Gauntlet follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from `0.2.0` onward (see [`docs/stability.md`](./docs/stability.md)).
Security fixes are issued for the latest `0.x` minor line. Pre-1.0,
the previous minor line stops receiving fixes when the next minor
ships.

| Version | Status               |
|---------|----------------------|
| `0.2.x` | Supported.           |
| `< 0.2` | Not supported.       |

## Reporting a vulnerability

**Do not open a public GitHub issue for security reports.**

If you discover a security issue in Gauntlet, please report it via
GitHub's private vulnerability reporting:

1. Go to the **Security** tab of the repo.
2. Click **Report a vulnerability**.
3. Fill in the form. We will acknowledge within **3 business days**.

If GitHub private vulnerability reporting is not available to you,
email the maintainer with the subject `[gauntlet-security]`.

Please include, when possible:

- Affected version(s) and commit SHA.
- A minimal reproducer (Suite YAML / policy spec / CLI invocation).
- The observed impact (RCE, path traversal, denial of service, etc.).
- Your proposed fix or workaround, if you have one.

## What counts as a vulnerability

In scope:

- **Code execution from untrusted input** — Suite YAML, policy spec
  strings, Episode JSON, scene-input manifests, override CLI args.
  `gauntlet.security.safe_yaml_load` is the canonical YAML entry
  point; bypassing it is a defect.
- **Path traversal** — anywhere Gauntlet writes to a user-supplied
  directory (`--out`, `--record-trajectories`, replay overrides,
  realsim manifests).
- **Pickle / arbitrary deserialisation** — Gauntlet does not use
  pickle on user input by design (see
  `docs/phase2-rfc-002-lerobot-smolvla.md` §6 on the spawn-pool
  pickle contract). Any pickle-load on untrusted data is a defect.
- **HTML injection** in the report renderer. The Jinja templates
  use `autoescape=True`; any escape-bypass is a defect.
- **Denial of service** that a malicious Suite YAML or scene-input
  manifest can trigger in seconds without resource exhaustion being
  the obvious cause (eg. quadratic parsing). Linear-in-input
  memory pressure from a deliberately-huge `episodes_per_cell` is
  out of scope — that is the user telling Gauntlet to do the work.

Out of scope:

- Vulnerabilities in optional-extra dependencies (`torch`,
  `transformers`, `lerobot`, `mujoco`, `pybullet`, `genesis`,
  `isaacsim`, `rclpy`, `imageio[ffmpeg]`, `wandb`, `mlflow`).
  Please report those upstream and we'll bump the dependency pin
  in a patch release.
- Vulnerabilities that require running an attacker-controlled
  policy adapter or env factory. Both are arbitrary code by
  design — wrapping malicious Python in a `Policy` is functionally
  equivalent to running malicious Python directly.
- Vulnerabilities in the reference benchmark (`scripts/`) — that
  script is dev tooling, not the library surface.

## Disclosure

We aim to ship a fix within **14 days** of acknowledging a valid
report. Once a fix is available:

- Patch release on PyPI with the fix.
- `CHANGELOG.md` entry under a `Security` section, with the
  reporter's preferred attribution.
- GitHub Security Advisory published with a CVE if applicable.

We do not currently run a bug-bounty programme.

## Hardening notes for users

If you are running Gauntlet against untrusted policy checkpoints or
Suite YAMLs (eg. a hosted CI service that accepts user-submitted
suites):

- Run in a process namespace / container with no network access and
  a temp-dir-only output root.
- Pin `gauntlet>=0.2,<0.3` and update on every patch release.
- Treat the `policy:` spec as code — `module.path:attr` resolves
  via `importlib`, and the resolved factory runs in your process.
- The HTML report ships with Chart.js loaded from a CDN. Self-host
  the JS if your environment forbids third-party CDNs.
