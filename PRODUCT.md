# Product

## Register

product

## Users

Two equally-weighted audiences, both working at a Linux workstation in deep-focus mode:

- **The VLA researcher.** Iterating on a single policy or checkpoint. Just retrained, wants to know *what regressed and why*. Reads the per-run `report.html` after every sweep. Cares about which axis combinations made things worse, not the headline success number.
- **The fleet operator.** Watching dozens of policy / checkpoint / suite combinations across a team. Reads the dashboard. Cares about persistent failure clusters that survive across runs, not one-off variance.

Neither user is in a browser by accident. Gauntlet ships as a downloadable Linux artifact (CLI + self-contained HTML reports). No cloud, no auth, no telemetry, no account, no internet round-trip required at view time. The reports open from the local filesystem.

The user's prior context when opening a Gauntlet artifact is almost always *frustration* — a checkpoint they trusted just regressed, and they don't yet know why. Design for someone who is annoyed, focused, and capable.

## Product Purpose

Answer one question for any learned robot policy:

> *"How does this policy fail, and has the latest checkpoint regressed against the last one?"*

The surrounding category — robotics evaluation — habitually averages this answer away. A 78% mean success rate hides that the policy fails 100% of the time when the cube texture is wood and the lighting drops below 0.6. Gauntlet exists to refuse that compression. Every breakdown leads with the failure axis combination, not the aggregate.

Success looks like: an engineer opens a `report.html`, the failure-cluster table is the first thing they see, and within ten seconds they know which axis combination broke and by how much.

## Brand Personality

**Rigorous · candid · unflinching.**

- *Rigorous* — numbers are exact, never decorative. Tabular numerals everywhere. Never a chart where a table would be more honest.
- *Candid* — the failure rate is the headline, not the success rate. The pass/fail dot is the loudest element on the summary card. We do not soften.
- *Unflinching* — when the policy is bad, the report says so plainly. No "needs improvement" hedging. No green checkmarks for 60% success. Failure clusters are red and they lead.

The voice is closer to a senior engineer's bug report than a marketing dashboard. Direct, specific, no warmth that would dilute the signal.

## Anti-references

Avoid these templates, palettes, and rhetorical patterns:

- **SaaS observability marketing** — gradient hero blocks, big-number-with-tiny-label cards, "All systems operational" green-bias. Datadog and New Relic landing-page aesthetics are the wrong lane entirely.
- **Hero-metric reflex.** Never lead with one giant success-rate number framed in a gradient card. The whole point is that the aggregate hides the answer.
- **The training-data robotics-eval look** — TensorBoard chrome, default Matplotlib colors, charts with no opinion, walls of `metric_name_v2_FINAL_v3` plot cards.
- **Decorative motion** — no spring physics, no morphing shapes, no scroll-triggered reveals. Charts can transition values; layout cannot bounce.
- **Color-blind-hostile pass/fail** — never use red/green as the only signal. Always pair with shape, position, label, or icon.
- **Glassmorphism, gradient text, side-stripe accents on cards.** Cliché.
- **Onboarding modals, tooltips that explain obvious things, empty-state hand-holding.** The user is an expert. Trust them.

## Design Principles

Five principles, derived from Gauntlet's existing spec (§6) and the conversation that produced this file. Use them to make any design judgment call:

1. **Failures over averages.** The failure rate, the failure clusters, the per-axis breakdown — these are always above the success rate, always larger, always sharper. If a layout pulls the eye to the aggregate, the layout is wrong.
2. **Local-first, no chrome.** Reports are static HTML opened from disk. No login state, no cloud spinner, no telemetry banner, no "share this report" affordance. Every UI element either helps the engineer find a failure or is removed.
3. **One screen, one decision.** The summary card answers "did this regress?" The clusters table answers "where?" The per-axis charts answer "how badly?" Each section earns its place by answering exactly one question. No section restates another.
4. **Expert confidence.** No tooltips defining what "lift" or "axis" means. No empty-state copy explaining what a report would look like if there were data. The audience is engineers shipping policies; explain nothing they already know.
5. **Numbers are exact, ranks are loud.** Tabular numerals for every metric. Worst-failing cluster typographically larger than the rest of the table — rank order is the most important signal on the page.

## Accessibility & Inclusion

- **WCAG 2.1 AA** as the floor for body text, controls, headings, and the pass/fail/warn dot. One acknowledged exception: the diverging green→red heatmap surface trades AA text contrast at *mid-range* cells for an unbroken diverging signal, paired with a mandatory screen-reader text alternative on every cell so the data is lossless regardless. Documented in DESIGN.md §4 as the *Heatmap Contrast Exception*.
- **Pass/fail never relies on color alone.** Always paired with shape (the summary dot has a thick border that reads as a different shape per state), position (worst-failing rows top of the table), or text. Color blindness is statistically common in engineering audiences and trivially fixable here.
- **Keyboard navigation** for every interactive element on the dashboard (filter selects, sortable columns, expandable details). No mouse-only affordances.
- **Reduced-motion respect.** Any chart transitions disabled when `prefers-reduced-motion: reduce`. The dashboard is functional with all motion off.
- **Screen-reader semantics.** Tables are real `<table>` with proper `<th scope>`. Charts include a sibling text summary or an accessible description so the failure-cluster signal is not gated on visual chart-reading.
