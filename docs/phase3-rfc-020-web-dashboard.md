# Phase 3 RFC 020 — Self-contained web dashboard

Status: Accepted
Phase: 3
Task: 20
Spec reference: `GAUNTLET_SPEC.md` §7 ("Hosted service / web dashboard").

## 1. Why this exists

`gauntlet` already emits two static HTML artefacts: a per-run
`report.html` (Phase 1) and a fleet-aggregate `fleet_report.html`
(Phase 3 Task 19, RFC 019). Both are great for one specific question
("how does *this* run / fleet-aggregate look?") but neither answers the
team-scale workflow:

* Iterating on a checkpoint and asking "how did the success rate of
  this suite move across the last 30 runs?"
* Slicing a directory of runs by `policy_label`, `env`, or
  `suite_name` and seeing only the matching subset.
* Linking from the index back into the original per-run `report.html`
  for a deep dive.

A *dashboard* — one HTML page that indexes every `report.json` under a
directory and lets the user filter / chart across the fleet — fills
that gap. `gauntlet dashboard build <dir> --out <out>` materialises a
self-contained static SPA from a directory of run artefacts; the user
opens `<out>/index.html` in a browser and explores.

## 2. Non-goals

* No backend server. No Flask, no WebSocket, no auth. Self-hosted means
  "drop the output dir into S3 / GitHub Pages / `python -m http.server`
  and you're done", but the file is also openable via `file://`.
* No new external Python dependencies. Reuses `pydantic`, `jinja2`,
  `typer`, and `Chart.js` (CDN) — same allow-list as Task 19.
* No mutation surface. The dashboard is read-only — there is no way
  to delete / re-run / re-label runs from the UI. Mutations happen
  through the existing `gauntlet` CLI; the dashboard re-builds against
  the latest snapshot when the user re-runs `gauntlet dashboard build`.
* No live polling or streaming. The embedded JSON is a snapshot at
  build time. Re-run the builder to refresh.

## 3. Self-contained vs hosted decision

We pick **self-contained** for three reasons:

1. **Distribution.** A directory of three files (`index.html`,
   `dashboard.js`, `dashboard.css`) zips up trivially and works on a
   teammate's laptop, in a S3 bucket, or attached to a Slack thread.
   A hosted service would force users to provision a long-lived
   container and a TLS cert just to look at offline data.
2. **Portability with `file://`.** `gauntlet`'s existing report HTMLs
   open from a `file://` path with no web server (and no CORS
   exception). The dashboard preserves that property — see §4.
3. **Backwards-compat.** Adding a server changes the operational
   contract. Self-contained means existing `gauntlet run` users get
   the dashboard for free without changing their deployment.

## 4. Embedded-JSON vs separate-fetch decision

The dashboard needs to ship the per-run index + per-axis aggregates to
the browser. Two options:

* **Separate fetch.** Builder writes `reports.json` next to
  `index.html`; the SPA does `fetch('reports.json')` on load.
* **Embedded inline JSON.** Builder serialises the index into a
  `<script type="application/json" id="dashboard-data">…</script>`
  block in `index.html` itself.

We pick **embedded inline JSON** because `fetch('reports.json')`
breaks under `file://` in Chromium (CORS rejects same-origin file
fetches by default; the user has to launch Chrome with
`--allow-file-access-from-files`, which is a non-starter for a "just
double-click the HTML" workflow). The same `<script
type="application/json">` pattern is what
`src/gauntlet/aggregate/templates/fleet_report.html.jinja` already
uses for `fleet-data`, so we mirror it (one less idiom for readers to
learn).

The trade-off is that `index.html` grows linearly with the number of
runs. At 1 000 runs × ~5 KB/run-summary the file is ~5 MB — well
within browser tolerance, smaller than a single screenshot. If a
team's fleet outgrows that, the natural next step is a backend (out of
scope per §2).

## 5. Page layout

The SPA renders these sections, top-to-bottom:

1. **Index card.** Total runs, total episodes, mean ± std success
   rate. Same metric vocabulary as the fleet report's summary card.
2. **Filter UI.** Three `<select>` boxes: `env`, `suite_name`,
   `policy_label`. Each starts with an `(all)` option; selecting any
   value re-renders the table and charts against the matching subset.
   Filtering is purely client-side, runs entirely off the embedded
   JSON literal — no extra HTTP, no rebuild required.
3. **Time-series chart.** X-axis = report.json mtime (ISO-formatted in
   the JS), Y-axis = `overall_success_rate`. Chart.js line. Re-renders
   when the filters change. Uses `path.stat().st_mtime` captured at
   build time (the `Report` schema has no timestamp field — see §6).
4. **Per-axis aggregate chart.** Bar chart per axis, aggregated across
   the *currently filtered* runs. Same heat-colour palette as the
   fleet report.
5. **Per-run table.** One row per run: `run_id`, `policy_label`,
   `suite_name`, `env`, `n_episodes`, `success_rate`, plus a `report`
   link. The link is populated only when a sibling `report.html`
   exists next to the `report.json` (the `gauntlet run --out` default
   layout); when absent the cell is empty. Filterable + sortable
   client-side.

## 6. mtime as time-series x-axis

`gauntlet.report.schema.Report` has no `timestamp` field — adding one
would be a schema change, which §2 forbids. Instead the builder
captures `path.stat().st_mtime` for each discovered `report.json` at
discovery time and stores it on the per-run dict in the embedded JSON
under the key `mtime`. The JS converts to a JavaScript `Date` for
display.

This is "good enough" for the dashboard's time-series view: the mtime
of `report.json` is set when `gauntlet run` writes the file, so it
reflects when the run *finished*. The exception is files copied
between machines (e.g. via `scp`): mtime is preserved by `scp -p` but
not by plain `scp`. Users who want deterministic timestamps can
supply the `-p` flag or call `touch` on the files post-transfer.

## 7. Open questions

* **Sortable columns.** First-pass per-run table is row-rendered in
  discovery order (sorted by source path). Adding column sort is a
  trivial follow-up but adds JS budget; deferred.
* **Multi-fleet comparison.** Two dashboards in two browser tabs
  cover the simple case. A built-in "compare" mode (overlay two
  series on the same chart) is a follow-up RFC.
* **Auto-refresh.** Polling the JSON for changes would require either
  a server or a file watcher. Deferred — re-run the builder.
