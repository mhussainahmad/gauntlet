---
name: Gauntlet
description: Failure-first evaluation reports for learned robot policies
colors:
  ink-black: "#1a1a1a"
  gravel: "#555555"
  paper: "#fafafa"
  page-white: "#ffffff"
  hairline: "#e1e1e1"
  header-wash: "#f0f0f0"
  pass-green: "#2ca02c"
  fail-red: "#d62728"
  caution-amber: "#f0ad4e"
  no-data-gray: "#d0d0d0"
  trace-blue: "#1f77b4"
typography:
  display:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "1.6rem"
    fontWeight: 600
    lineHeight: "1.2"
  headline:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 600
    lineHeight: "1.3"
  title:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "1rem"
    fontWeight: 600
    lineHeight: "1.4"
  metric-value:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 600
    lineHeight: "1.2"
    fontFeature: "tnum"
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: "1.4"
  subtle:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "0.9rem"
    fontWeight: 400
    lineHeight: "1.4"
  label:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 600
    letterSpacing: "0.04em"
  code:
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    fontSize: "0.85rem"
    fontWeight: 400
  heatmap-cell:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 600
    fontFeature: "tnum"
rounded:
  sm: "4px"
  md: "8px"
  dot: "50%"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  card:
    backgroundColor: "{colors.page-white}"
    rounded: "{rounded.md}"
    padding: "16px 20px"
  chart-card:
    backgroundColor: "{colors.page-white}"
    rounded: "{rounded.md}"
    padding: "12px"
    height: "220px"
  pass-dot:
    backgroundColor: "{colors.pass-green}"
    rounded: "{rounded.dot}"
    size: "32px"
  fail-dot:
    backgroundColor: "{colors.fail-red}"
    rounded: "{rounded.dot}"
    size: "32px"
  warn-dot:
    backgroundColor: "{colors.caution-amber}"
    rounded: "{rounded.dot}"
    size: "32px"
  metric-label:
    textColor: "{colors.gravel}"
    typography: "{typography.label}"
  metric-value:
    textColor: "{colors.ink-black}"
    typography: "{typography.metric-value}"
  heatmap-cell:
    typography: "{typography.heatmap-cell}"
    textColor: "{colors.page-white}"
    width: "56px"
    height: "32px"
  select-filter:
    backgroundColor: "{colors.page-white}"
    textColor: "{colors.ink-black}"
    rounded: "{rounded.sm}"
    padding: "6px 10px"
---

# Design System: Gauntlet

## 1. Overview

**Creative North Star: "The Bug Report"**

Gauntlet's design takes its cues from the rhetorical mode of a senior engineer's bug report, not a SaaS dashboard. A bug report does not decorate. It lists the failure first, names the reproduction conditions exactly, ranks severity by position, and trusts the reader to know the vocabulary. Every visual choice in Gauntlet is derived from that posture: failures lead, numbers are exact, and the page contains nothing that does not contribute to a decision.

The system is deliberately **flat, single-column, and almost monochrome**, so the moments when color *does* appear — the pass / fail / warn dot, a red row at the top of the failure cluster table, a saturated cell in a heatmap — carry the weight they should. Color is not theme; it is signal. The chrome stays out of the way, in low-contrast neutrals on the same near-white paper an engineer would print this report onto. There is no dark mode and no marketing chrome and no celebratory empty-state. The user already knows what success looks like; we are here for the failures.

This system explicitly rejects: SaaS observability marketing aesthetics (Datadog / New Relic landing pages), the hero-metric template, gradient-clipped text, glassmorphism, decorative motion, and any pass/fail signaling that depends on color alone.

**Key Characteristics:**
- Failure-first hierarchy — the *failure rate*, not success, is the headline metric.
- Flat, hairline-bordered cards on near-white paper. No shadows.
- System font stack — fast, native, never needs a font-face download.
- Tabular numerals everywhere. Rank order is the loudest signal on the page.
- Two color systems running in parallel: low-chroma chrome and high-chroma semantic state.
- Static HTML, opens from disk, no network round-trip required at view time.

## 2. Colors

A near-monochrome chrome layered with a deliberately small set of saturated semantic colors borrowed from the Matplotlib `tab10` palette every robotics ML engineer's eye is already calibrated to.

### Primary

The primary palette is the *semantic state* set. These three colors are reserved for pass/fail/warn signaling — never used as decoration, never used to brand a button or a banner.

- **Pass Green** (`#2ca02c`, ≈ `oklch(63% 0.18 142)`): The pass dot, the upper end of the heatmap diverging scale, success-rate axes when above-target. Borrowed from Matplotlib `tab10[2]` — the green every Python user has stared at for a decade.
- **Fail Red** (`#d62728`, ≈ `oklch(58% 0.21 28)`): The fail dot, the lower end of the heatmap diverging scale, the failure-rate metric on the summary card. From `tab10[3]`. Loud on purpose. The whole product exists to surface this color.
- **Caution Amber** (`#f0ad4e`, ≈ `oklch(78% 0.14 70)`): The warn dot — partial regression, statistical edge cases, "passes but barely". Used sparingly; if everything is amber, nothing is.

### Secondary

One single neutral-information accent, used for non-state-bearing data series.

- **Trace Blue** (`#1f77b4`, ≈ `oklch(52% 0.13 248)`): The time-series line on the dashboard's success-over-time chart. From `tab10[0]`. This is the only place blue appears in the system, and it is deliberately not a brand color — it is a chart line.

### Neutral

The chrome. Six low-chroma steps from ink to page. Used for text, surfaces, borders, and table chrome. Nothing else.

- **Ink Black** (`#1a1a1a`): Primary body text and headings. A near-black, not pure black.
- **Gravel** (`#555555`): Secondary text — labels, captions, the "subtle" class. WCAG AA contrast on Paper.
- **Hairline** (`#e1e1e1`): Card borders, table row dividers. The single most-used border value.
- **Header Wash** (`#f0f0f0`): Table header backgrounds. The lightest distinguishable surface above Paper.
- **Paper** (`#fafafa`): The page background. A very slightly off-white that lets a Page-White card sit above it visually.
- **Page White** (`#ffffff`): Card and chart-card backgrounds. The brightest surface in the system; everything important lives on it. *Grandfathered as pure `#ffffff` for byte-compat with the already-shipped `report.html`. New surfaces and any redesign should use a tinted alternative like `oklch(99% 0.003 25)` per the "no `#000`/`#fff`" Don't below.*
- **No-Data Gray** (`#d0d0d0`): The third heatmap state — neither pass nor fail, just absent. Also used for the empty heatmap-cell text.

### Named Rules

**The Tab10 Pact.** The pass / fail / warn / trace colors are taken directly from Matplotlib's `tab10` qualitative palette, on purpose. Every robotics engineer reading a Gauntlet report has been calibrated by Matplotlib for years; picking different greens or reds would be cosmetic vanity at the cost of legibility. Do not "improve" them.

**The Two-System Rule.** Chrome colors and semantic colors are separate vocabularies that never blend. A button is never `pass-green`. A page background is never `caution-amber`. If a non-state element wants to be saturated, the design is wrong, not the token.

**The 5% Color Budget.** Across any single screen, the *combined* surface area of saturated color (pass/fail/warn/trace) should not exceed roughly 5%. Failure clusters are red; everything else is grayscale. The rarity is what makes red mean *failure*.

## 3. Typography

**Display & Body Font:** System UI stack — `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif`.
**Mono Font:** System mono stack — `ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`.

**Character:** Native system fonts only. The reports open from disk in a browser that already has these installed; no `@font-face` download, no FOIT, no flicker. This is also part of the unflinching personality — a custom display face would feel like dressing up a bug report in a serif.

### Hierarchy

- **Display** (1.6rem, weight 600): The single page-level `<h1>` — "Gauntlet report — `<suite_name>`". One per page, never repeated.
- **Headline** (1.25rem, weight 600, margin-top 32px): Section headings — "Where it failed", "Per-axis success rate", "Axis-pair heatmaps". Generous space above to separate sections optically.
- **Title** (1rem, weight 600): Sub-section headings inside chart cards (axis names, heatmap pair names).
- **Metric Value** (1.5rem, weight 600, `font-variant-numeric: tabular-nums`): The big numbers in the summary card — failure rate, episode count. Tabular nums so digits align column-perfectly across rows.
- **Body** (1rem, weight 400, line-height 1.4): Paragraph text and table cell content.
- **Subtle** (0.9rem, weight 400, color Gravel): Captions, "n/24 episodes", mtimes, secondary metadata.
- **Label** (0.75rem, weight 600, `letter-spacing: 0.04em`, **UPPERCASE**, color Gravel): Metric labels — "FAILURE RATE", "EPISODES", "SUITE". Small, wide-tracked, the chrome label that names a number.
- **Code** (0.85rem, mono stack): Axis values, suite names, run IDs, perturbation configurations. Anything an engineer might `grep` for.
- **Heatmap Cell** (0.75rem, weight 600, tabular nums, color Page White, text-shadow `0 1px 1px rgba(0,0,0,0.25)`): Percentage labels inside diverging-color heatmap cells. The text-shadow keeps them legible against any cell color.

### Named Rules

**The Tabular Numerals Rule.** Every numeric value uses `font-variant-numeric: tabular-nums`. Always. This is non-negotiable: numbers in a Gauntlet report exist to be compared by eye in a column, and proportional digits make that impossible. Apply via `td.num`, `th.num`, every metric value, every heatmap cell.

**The One H1 Rule.** Each report page has exactly one `<h1>`, and it identifies the suite. The H1 is never used decoratively or repeated.

**The Don't-Explain Rule.** No paragraph of body copy explains what the page is for. The H1 names the report; the section headings name the question; the data answers it. There is no introduction.

## 4. Elevation

**Flat by doctrine.** Gauntlet uses no `box-shadow` anywhere. Depth comes from a single 1px hairline border (`#e1e1e1`) plus a subtle background contrast: Page White (`#ffffff`) cards floating on Paper (`#fafafa`) — a 2-step lightness delta that reads as a layer without lifting.

This is the right choice for an evaluation report. Shadows imply UI affordance ("I can be lifted, dragged, dismissed"). A failure-cluster card cannot be dismissed; the failure is the point. The hairline communicates "this is a contained surface" without inviting interaction it does not afford.

### Shadow Vocabulary

There is no shadow vocabulary. If a future component needs a shadow, the question to ask first is "would a thicker hairline or a darker background do this job instead?"

The one allowed exception is **text shadow on heatmap cells** — `0 1px 1px rgba(0,0,0,0.25)` — used purely to keep white percentage text legible across the full pass/fail color range underneath it. This is functional, not decorative.

### Named Rules

**The Flat-By-Default Rule.** Surfaces are flat at rest. There is no hover-lift, no card shadow, no menu shadow. The only "lift" in the system is the 2-step Paper → Page White contrast.

**The No Drop Shadows Rule.** `box-shadow` is forbidden on cards, buttons, modals, and tooltips. If a designer reaches for one, the structure is wrong — return to the hairline.

**The Heatmap Contrast Exception.** The diverging green→red heatmap trades WCAG AA text contrast at *mid-range* cells (rate ≈ 0.4–0.6 lands on a brownish-olive ~3.4:1 against white text) for the value of an unbroken diverging signal. This is an acknowledged exception, not an oversight. To stay accessible: every heatmap cell must carry a screen-reader text alternative (`<td title="...">` or `aria-label`) with the full label (`"texture=wood × lighting=0.5: 47% success, n=24"`), so the *information* is lossless to SR users even when the *visual* contrast at midpoints dips. Empty cells (No-Data Gray + Gravel em-dash, ~3.4:1) carry the same SR alternative for the same reason. The pass/fail/warn dot, all body text, and all metric values remain at AA or above; the exception is scoped strictly to the heatmap surface.

## 5. Components

The system is small. Five primitives carry almost the entire surface area: card, chart-card, dot, metric tile, heatmap cell. Two more — select filter and table — are documented because the dashboard relies on them.

### Cards / Containers

**Character:** A flat, hairline-bordered surface that contains one answer to one question.

- **Corner Style:** `border-radius: 8px` — gently rounded, not pill-shaped, not square.
- **Background:** Page White (`#ffffff`) on a Paper (`#fafafa`) page background.
- **Border:** `1px solid` Hairline (`#e1e1e1`).
- **Internal Padding:** `16px 20px` for content cards, `12px` for chart cards (slightly tighter so the chart breathes within the canvas).
- **Shadow Strategy:** None. See Elevation §4.

### Chart Cards

A specialized card that wraps a single Chart.js `<canvas>`. Identical chrome to a card, but with `max-height: 220px !important` on the canvas so the per-axis chart row stays scannable in a single eyeful. Title (`<h3>`, Title typography) sits above the chart inside the same card.

### Pass / Fail / Warn Dot

**Character:** The single loudest visual primitive in the system. A 32px filled circle that headlines the summary card and signals run state at a glance.

- **Size:** 32×32 px.
- **Shape:** Perfect circle (`border-radius: 50%`).
- **Border:** `2px solid rgba(0, 0, 0, 0.15)` — a subtle ring that gives the dot dimension on any background and (importantly) renders distinguishable to fully-monochrome viewers via the border weight.
- **Variants:** `pass-dot` Pass Green, `fail-dot` Fail Red, `warn-dot` Caution Amber.
- **Pairing:** The dot is *always* paired with a text metric (the failure-rate value). The dot alone never carries the signal — it amplifies a number that is also written.

### Metric Tile (Label + Value)

**Character:** A two-row vertical pair — a small uppercase label and a large tabular-num value. Used in the summary card and across the dashboard.

- **Label:** Label typography (0.75rem, 600, 0.04em letter-spacing, UPPERCASE, Gravel).
- **Value:** Metric Value typography (1.5rem, 600, Ink Black). Variant `metric-value.fail` swaps text color to Fail Red — used for the failure-rate metric specifically, never for arbitrary "this is bad" emphasis.
- **Layout:** Tiles sit in a horizontal flex row inside their parent card, separated by `gap: 20px 32px`.

### Heatmap Cell

**Character:** The grid primitive of the per-cell axis-pair heatmaps. A diverging green→red gradient maps `success_rate` linearly from 1.0 (green) to 0.0 (red), with No-Data Gray for absent cells.

- **Size:** 56×32 px.
- **Background:** Computed via the Tab10-derived diverging interpolation — `r: 214 → 44, g: 39 → 160, b: 40 → 44` as rate goes 0 → 1. Identical formula in both report.html and dashboard.js (the `heatColor()` function); they must stay in lockstep.
- **Text:** Heatmap-cell typography in Page White, with a `0 1px 1px rgba(0,0,0,0.25)` text shadow for legibility against any cell color.
- **Empty Variant:** No-Data Gray background, Gravel em-dash text, no shadow. Visually distinct from any rate so "missing" is never confused with "low".

### Tables

**Character:** The system's primary data surface. Plain, dense, scannable. The failure-cluster table is the most important table in the product and gets the page's center of gravity.

- **Border model:** `border-collapse: collapse` with `border-bottom: 1px solid` Hairline on every cell.
- **Header:** Header Wash (`#f0f0f0`) background, font-weight 600, left-aligned by default.
- **Numeric columns:** `text-align: right` and `font-variant-numeric: tabular-nums`. Apply via `.num` class.
- **Row hover:** `background: #fafafa` — Paper, the same color as the page. The hover is barely-there on purpose; the table is for reading, not selecting.
- **Rank-loud variant (failure-clusters table):** The first row (worst-failing cluster) renders one weight step heavier (`font-weight: 600` vs body 400) and Fail Red on the failure-rate cell. Rank order is the most important signal on the page.

### Select Filter (Dashboard)

The dashboard's filter bar uses native `<select>` elements. Today they ship browser-default; the canonical styling, when implemented, follows:

- **Style:** Page White background, 1px Hairline border, `border-radius: 4px`, `padding: 6px 10px`.
- **Typography:** Body typography.
- **Focus:** Border shifts to Trace Blue (`#1f77b4`), 2px outline ring at `rgba(31, 119, 180, 0.25)` — the only place Trace Blue appears outside a chart.
- **Disabled:** Opacity 0.5, cursor `not-allowed`.

### Details / Summary

The per-cell breakdown table (the "wall of numbers" §6 of `GAUNTLET_SPEC` insists on hiding) is wrapped in `<details>`.

- **Summary:** `cursor: pointer`, font-weight 600, `padding: 4px 0`. No chevron decoration — the browser-default disclosure triangle is sufficient.
- **Open content:** Standard card with `margin-top: 8px`.

### Named Rules

**The Cards-Only-When-Necessary Rule.** A card exists to contain one answer to one question. If a section's answer is a single number, it goes in a metric tile inside an existing card — not in its own card. Nested cards are forbidden.

**The No Side-Stripe Rule.** Never use a `border-left` greater than 1px as a colored accent on a card, list item, or alert. If something needs to read as "important" or "failing", use a metric value color or row weight, never a side stripe.

## 6. Do's and Don'ts

### Do:

- **Do** lead the page with the failure rate, not the success rate. The summary card's headline metric is *failure*, every time. (Quoted from PRODUCT.md: *"Failures over averages."*)
- **Do** apply `font-variant-numeric: tabular-nums` to every numeric value — metric values, table cells, heatmap cells, axis ticks. Without exception.
- **Do** pair every pass/fail color signal with a non-color signal — shape, position, label, or icon. The summary dot has a 2px border ring for monochrome viewers; the failure-cluster row sits at the top *and* renders heavier *and* uses Fail Red.
- **Do** use the system font stack. Reports open from disk; a custom face would FOIT or fail.
- **Do** keep the chrome low-chroma so saturated color reads as signal. The 5% Color Budget rule is the test.
- **Do** match the `heatColor()` interpolation byte-for-byte between `report.html.jinja` and `dashboard.js`. Diverging green→red, `r: 214 → 44, g: 39 → 160, b: 40 → 44`.
- **Do** add a screen-reader text alternative to every heatmap cell — `<td title="texture=wood × lighting=0.5: 47% success, n=24">` or equivalent `aria-label`. The visual percentage label loses AA contrast at midpoints; the SR alternative makes the data lossless regardless. (See the Heatmap Contrast Exception in §4.)
- **Do** respect `prefers-reduced-motion: reduce` — disable Chart.js animations when set.
- **Do** keep cards flat with a 1px Hairline border on Page White over Paper.

### Don't:

- **Don't** ever lead with one giant success-rate number framed in a gradient card. (PRODUCT.md anti-reference: *"Hero-metric reflex."*)
- **Don't** use SaaS observability marketing aesthetics — gradient hero blocks, big-number-with-tiny-label cards, "All systems operational" green-bias. (PRODUCT.md anti-reference: *"SaaS observability marketing — Datadog and New Relic landing-page aesthetics."*)
- **Don't** use `background-clip: text` with a gradient. Gradient text is decorative, never meaningful, and forbidden by the shared design laws.
- **Don't** use glassmorphism, backdrop-filter blurs, or "frosted" surfaces. Forbidden.
- **Don't** add `box-shadow` to cards, buttons, modals, or tooltips. The system is flat. (See Elevation §4.)
- **Don't** use `border-left` greater than 1px as a colored stripe to mark a "failing" or "important" element. Use row weight, metric color, or position instead.
- **Don't** rely on red/green color alone for pass/fail signaling. The 2px dot border, the rank-loud row weight, the metric-value Fail Red — color is always paired with another channel.
- **Don't** invent new accent colors. The semantic palette is fixed at three (pass/fail/warn) plus one neutral-info chart line (Trace Blue). Do not add a fourth.
- **Don't** wrap tooltips around terms an engineer already knows ("lift", "axis", "cluster"). (PRODUCT.md principle: *"Expert confidence."*)
- **Don't** add empty-state hand-holding ("No reports yet — generate one with `gauntlet run`!"). The empty state is silent; the user knows what to do.
- **Don't** use `#000` or `#fff` directly as foreground or background outside the existing `--card-bg: #ffffff` token — for any *new* token, prefer slightly tinted neutrals (e.g. `oklch(20% 0.005 25)` for an even-deeper ink that still reads as black).
- **Don't** decorate motion. Layout never bounces, slides, or springs. Chart values may transition; structure does not.
- **Don't** use TensorBoard chrome, default Matplotlib chart frames, or "wall of `metric_v2_FINAL` plot cards" layouts. (PRODUCT.md anti-reference.)
- **Don't** add a "share this report" button, a sign-in modal, a telemetry consent banner, or any other cloud-app chrome. Reports are local files; there is nothing to share, no one to sign in as.
