"""Generate the public reference benchmark artefacts.

Runs two policies on the same Suite, writes their reports, and emits the
structured regression delta between them. The bundled output is what we
publish to GitHub Pages so an external buyer can see — before installing
anything — what a Gauntlet report actually looks like and how a
regression surfaces.

Two policies:

* :class:`ClosedLoopReachPolicy` — closed-loop, reads ``obs["cube_pos"]``,
  ``obs["ee_pos"]``, ``obs["target_pos"]``; succeeds on the unperturbed
  baseline.
* :class:`DegradedReachPolicy` — same closed-loop logic with
  injected proprioceptive noise (σ=0.06 m on cube_pos / ee_pos /
  target_pos). Models a fine-tune that regressed against the previous
  checkpoint — the control law is unchanged but its observed-state
  estimate is dirtier.

Outputs under ``<out>/`` (default ``benchmarks/v0.2.0/``):

* ``baseline/`` — report from the good policy.
* ``regressed/`` — report from the noisy policy.
* ``compare.json`` — `gauntlet compare` verdict between them.
* ``diff.txt`` / ``diff.json`` — human + machine-readable per-cell deltas.
* ``index.html`` — landing page linking the two reports + the diff.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from gauntlet.policy.base import Action, Observation

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Final[Path] = _REPO_ROOT / "examples" / "suites" / "tabletop-smoke.yaml"
_DEFAULT_OUT: Final[Path] = _REPO_ROOT / "benchmarks" / "v0.2.0"


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

# Action layout (per env.tabletop): [dx, dy, dz, drx, dry, drz, gripper]
# Magnitudes clipped to [-1, 1]; the env scales linear ones by 0.05 m.
_GRIPPER_OPEN: Final[float] = 1.0
_GRIPPER_CLOSED: Final[float] = -1.0


def _unit_step(delta_xyz: NDArray[np.float64], *, scale: float = 5.0) -> NDArray[np.float64]:
    """Scale a metric XYZ delta into the env's normalised action range.

    Multiplier chosen so a 0.10 m gap saturates the unit cube — keeps the
    closed loop crisp without overshoot for the ~0.20 m workspace.
    """
    return np.clip(delta_xyz * scale, -1.0, 1.0).astype(np.float64)


class ClosedLoopReachPolicy:
    """Reach the cube, grasp, lift, translate, release.

    Reads ``obs["cube_pos"]``, ``obs["ee_pos"]`` and ``obs["target_pos"]``;
    derives the action one step at a time. No internal state beyond the
    phase machine.
    """

    _LIFT_STEPS: Final[int] = 3

    def __init__(self, *, lift_height: float = 0.10) -> None:
        self._lift_height = float(lift_height)
        self._phase: int = 0  # 0=descend, 1=grasp, 2=lift, 3=translate, 4=hold
        self._lift_counter: int = 0

    def reset(self, rng: np.random.Generator | None = None) -> None:
        # Accept the rng kwarg so the policy satisfies
        # :class:`~gauntlet.policy.ResettablePolicy`. The deterministic
        # baseline does not consume it.
        del rng
        self._phase = 0
        self._lift_counter = 0

    def act(self, obs: Observation) -> Action:
        ee = np.asarray(obs["ee_pos"], dtype=np.float64).reshape(3)
        cube = np.asarray(obs["cube_pos"], dtype=np.float64).reshape(3)
        target = np.asarray(obs["target_pos"], dtype=np.float64).reshape(3)

        action = np.zeros(7, dtype=np.float64)
        action[6] = _GRIPPER_OPEN

        if self._phase == 0:
            # Descend onto the cube XY.
            xy_gap = float(np.linalg.norm(cube[:2] - ee[:2]))
            if xy_gap < 0.02 and ee[2] - cube[2] < 0.03:
                self._phase = 1
            else:
                delta = np.array([cube[0] - ee[0], cube[1] - ee[1], cube[2] - ee[2]])
                action[0:3] = _unit_step(delta)
        elif self._phase == 1:
            # Close on the cube.
            action[6] = _GRIPPER_CLOSED
            self._phase = 2
        elif self._phase == 2:
            # Lift straight up for a fixed number of steps — the env snaps
            # the cube to the EE while grasped, so a height-based exit
            # condition would never fire.
            action[6] = _GRIPPER_CLOSED
            action[2] = 1.0
            self._lift_counter += 1
            if self._lift_counter >= self._LIFT_STEPS:
                self._phase = 3
        elif self._phase == 3:
            # Translate to target XY.
            action[6] = _GRIPPER_CLOSED
            delta_xy = np.array([target[0] - cube[0], target[1] - cube[1], 0.0])
            if float(np.linalg.norm(delta_xy[:2])) < 0.01:
                self._phase = 4
            else:
                action[0:3] = _unit_step(delta_xy)
        else:
            # Hold position with the gripper closed.
            action[6] = _GRIPPER_CLOSED

        return action.astype(np.float64)


class DegradedReachPolicy(ClosedLoopReachPolicy):
    """Same closed-loop policy with proprioceptive noise.

    Models a fine-tune that regressed: identical control law on paper,
    but its observed-state estimate is corrupted with Gaussian noise. On
    perturbed cells this is enough to miss the grasp window and slip the
    cube, surfacing as a per-cell flip + a failure cluster.
    """

    def __init__(self, *, noise_std: float = 0.060, seed: int = 7) -> None:
        super().__init__()
        self._noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        # The Runner-supplied rng is per-episode; reusing it for the
        # proprioceptive noise stream means the degraded policy is
        # bit-reproducible from the suite seed exactly like every other
        # rollout.
        super().reset(rng)
        if rng is not None:
            self._rng = rng

    def act(self, obs: Observation) -> Action:
        noisy = dict(obs)
        for key in ("cube_pos", "ee_pos", "target_pos"):
            base = np.asarray(obs[key], dtype=np.float64)
            noisy[key] = base + self._rng.normal(0.0, self._noise_std, size=base.shape)
        return super().act(noisy)


def _baseline_factory() -> ClosedLoopReachPolicy:
    return ClosedLoopReachPolicy()


def _regressed_factory() -> DegradedReachPolicy:
    return DegradedReachPolicy()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run_one(suite_path: Path, policy_spec: str, out_dir: Path, n_workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "gauntlet.cli",
        "run",
        str(suite_path),
        "--policy",
        policy_spec,
        "--out",
        str(out_dir),
        "--n-workers",
        str(n_workers),
    ]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)


def _gauntlet_compare(report_a: Path, report_b: Path, out_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "gauntlet.cli",
        "compare",
        str(report_a),
        str(report_b),
        "--out",
        str(out_path),
    ]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)


def _gauntlet_diff_json(report_a: Path, report_b: Path, out_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "gauntlet.cli",
        "diff",
        str(report_a),
        str(report_b),
        "--json",
    ]
    print("$ " + " ".join(cmd))
    res = subprocess.run(cmd, check=True, cwd=_REPO_ROOT, capture_output=True, text=True)
    out_path.write_text(res.stdout)


def _gauntlet_diff_text(report_a: Path, report_b: Path, out_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "gauntlet.cli",
        "diff",
        str(report_a),
        str(report_b),
    ]
    print("$ " + " ".join(cmd))
    res = subprocess.run(cmd, check=True, cwd=_REPO_ROOT, capture_output=True, text=True)
    out_path.write_text(res.stdout)


def _write_landing_page(out: Path, summary: dict[str, object]) -> None:
    baseline = float(cast(float, summary["baseline_success_rate"]))
    regressed = float(cast(float, summary["regressed_success_rate"]))
    verdict = str(summary["compare_verdict"])
    suite_name = str(summary["suite_name"])
    n_episodes = int(cast(int, summary["n_episodes"]))
    gauntlet_version = str(summary["gauntlet_version"])
    delta = regressed - baseline
    delta_pct = delta * 100.0

    css = """
    body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #111; }
    h1 { margin-bottom: 0.25rem; }
    .lede { color: #555; margin-top: 0; }
    .verdict { padding: 1rem 1.25rem; border-radius: 6px; margin: 1.5rem 0; font-variant-numeric: tabular-nums; }
    .verdict.regressed { background: #fdecea; border-left: 4px solid #c62828; }
    .verdict.stable { background: #e7f4ec; border-left: 4px solid #2e7d32; }
    .verdict .num { font-size: 1.6rem; font-weight: 600; }
    table { width: 100%; border-collapse: collapse; margin: 1.25rem 0; font-variant-numeric: tabular-nums; }
    th, td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #e5e5e5; text-align: left; }
    th { background: #fafafa; }
    a { color: #0b65c2; text-decoration: none; border-bottom: 1px solid currentColor; }
    code { background: #f3f3f3; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.95em; }
    footer { color: #888; font-size: 0.9rem; margin-top: 3rem; }
    """

    regressed_cls = "regressed" if verdict == "regressed" else "stable"

    html = textwrap.dedent(
        f"""\
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <title>Gauntlet — Reference benchmark (v{gauntlet_version})</title>
          <style>{css}</style>
        </head>
        <body>
          <h1>Gauntlet — Reference benchmark</h1>
          <p class="lede">
            Two policies. Same Suite. One regressed against the other.
            This is what <code>gauntlet compare</code> + <code>gauntlet diff</code>
            surface on a real run.
          </p>

          <div class="verdict {regressed_cls}">
            <div><strong>Verdict:</strong> {verdict.upper()}</div>
            <div class="num">{baseline*100:.1f}% &rarr; {regressed*100:.1f}%
              ({delta_pct:+.1f} pp)</div>
            <div>Suite: <code>{suite_name}</code> · Rollouts per side: {n_episodes}</div>
          </div>

          <table>
            <thead>
              <tr><th>Run</th><th>Policy</th><th>Success rate</th><th>Report</th></tr>
            </thead>
            <tbody>
              <tr>
                <td>baseline</td>
                <td><code>ClosedLoopReachPolicy</code></td>
                <td>{baseline*100:.1f}%</td>
                <td><a href="baseline/report.html">report.html</a> &middot; <a href="baseline/report.json">report.json</a></td>
              </tr>
              <tr>
                <td>regressed</td>
                <td><code>DegradedReachPolicy</code> (proprio noise σ=0.06 m)</td>
                <td>{regressed*100:.1f}%</td>
                <td><a href="regressed/report.html">report.html</a> &middot; <a href="regressed/report.json">report.json</a></td>
              </tr>
            </tbody>
          </table>

          <h2>Structured delta</h2>
          <ul>
            <li><a href="compare.json"><code>compare.json</code></a> — verdict + headline regression detected by <code>gauntlet compare</code>.</li>
            <li><a href="diff.txt"><code>diff.txt</code></a> — human-readable per-cell + per-axis delta from <code>gauntlet diff</code>.</li>
            <li><a href="diff.json"><code>diff.json</code></a> — machine-readable <code>ReportDiff</code> for CI integration.</li>
          </ul>

          <h2>Reproduce locally</h2>
          <pre><code>pip install gauntlet
git clone https://github.com/gauntlet-eval/gauntlet
cd gauntlet
python scripts/generate_reference_benchmark.py --out ./benchmarks/v{gauntlet_version}/</code></pre>

          <footer>
            Generated by <code>scripts/generate_reference_benchmark.py</code>
            from Gauntlet {gauntlet_version}.
          </footer>
        </body>
        </html>
        """
    )
    (out / "index.html").write_text(html)


def _summarise(out: Path) -> dict[str, object]:
    baseline_report = json.loads((out / "baseline" / "report.json").read_text())
    compare = json.loads((out / "compare.json").read_text())

    import gauntlet

    delta = float(compare["delta_success_rate"])
    threshold = float(compare.get("threshold", 0.1))
    verdict = (
        "regressed"
        if delta <= -abs(threshold)
        else "improved"
        if delta >= abs(threshold)
        else "stable"
    )

    return {
        "gauntlet_version": gauntlet.__version__,
        "suite_name": baseline_report.get("suite_name")
        or compare["a"].get("name", "unknown"),
        "n_episodes": int(compare["a"].get("n_episodes", 0)),
        "baseline_success_rate": float(compare["a"]["overall_success_rate"]),
        "regressed_success_rate": float(compare["b"]["overall_success_rate"]),
        "compare_verdict": verdict,
        "compare_delta": delta,
        "compare_threshold": threshold,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--clean", action="store_true", help="Wipe --out before regenerating.")
    args = parser.parse_args(argv)

    out: Path = args.out.resolve()
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    baseline_spec = "scripts.generate_reference_benchmark:_baseline_factory"
    regressed_spec = "scripts.generate_reference_benchmark:_regressed_factory"

    # Make `scripts.` importable from cwd.
    init = _REPO_ROOT / "scripts" / "__init__.py"
    if not init.exists():
        init.write_text("# Auto-created so the reference-benchmark factories are importable.\n")

    _run_one(args.suite, baseline_spec, out / "baseline", args.n_workers)
    _run_one(args.suite, regressed_spec, out / "regressed", args.n_workers)

    _gauntlet_compare(
        out / "baseline" / "report.json",
        out / "regressed" / "report.json",
        out / "compare.json",
    )
    _gauntlet_diff_text(
        out / "baseline" / "report.json",
        out / "regressed" / "report.json",
        out / "diff.txt",
    )
    _gauntlet_diff_json(
        out / "baseline" / "report.json",
        out / "regressed" / "report.json",
        out / "diff.json",
    )

    summary = _summarise(out)
    _write_landing_page(out, summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    baseline_pct = float(cast(float, summary["baseline_success_rate"])) * 100.0
    regressed_pct = float(cast(float, summary["regressed_success_rate"])) * 100.0
    print(f"\nBenchmark artefacts written to: {out}")
    print(f"  baseline:  {baseline_pct:.1f}%")
    print(f"  regressed: {regressed_pct:.1f}%")
    print(f"  verdict:   {summary['compare_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
