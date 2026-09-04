# Design

<!-- impeccable:design-schema 1 -->

Visual direction for the Fraud-Spike Review Queue dashboard. Product truth lives in [PRODUCT.md](PRODUCT.md); this file owns the look, and it is the durable record of the direction — not a one-session brief.

## Surface and mode

One surface: the reviewer's console at `/`. Mode is **Operate**. The visitor completes a task — clear a queue of flagged risk briefs and record a decision against each. Scanability, density, and consistency outrank expression. Brand lives in precise details, never in decoration.

Secondary audience: the Razorpay AI Buildathon panel, evaluating in a short window. That does not change the mode. A console that is genuinely good to work in is what reads as credible to people who build design systems; a console dressed up as a pitch page reads as neither.

## Direction: a developer-first financial canvas

Inspired by **Blade**, Razorpay's open-source design system (MIT, `github.com/razorpay/blade`). Blade's stated philosophy is a *Developer-First Financial Canvas*: precise, data-dense, documentation-like, with sharp-yet-subtly-rounded corners and thin lines communicating efficiency.

Tokens below were read from `packages/blade/src/tokens/global/` at `master` and converted from Blade's `hsla()` definitions to hex. They are real token values, not approximations.

### Intellectual-property boundary

Take the design language — palette, spacing discipline, density, typographic scale, component styling. **Do not** reproduce Razorpay's logo, the razor-blade "r" symbol, or the wordmark. Do not imply this project is an official Razorpay product or affiliated with them. This is a hackathon submission *to* them, not a product *by* them. A header that visually rhymes with their dashboard is good; a copied mark is a problem.

Do not import Blade. It is a React component library; this is a server-rendered Flask app. Take the tokens and the principles, not the dependency.

## Anti-references

These are the specific tells of default AI-generated UI. The previous revision of this dashboard had all of them, and none may return:

- `#0f172a` slate-navy page background (the default dark-mode-app ground)
- blue-to-purple gradient text (`#60a5fa → #a78bfa`) — or any gradient, anywhere
- glassmorphism: `backdrop-filter: blur()` over translucent cards
- 12–14px border radii
- emoji anywhere in the interface
- Inter from Google Fonts as the *only* typographic decision
- stock Tailwind semantic colors (`#ef4444`, `#10b981`, `#f59e0b`)

## Principles

1. **Light canvas, not dark.** Razorpay's dashboard is a light, information-dense financial surface. Inverting to light breaks the generic-dark-AI-app read and matches the real product family.
2. **4px radii.** Blade's `border.radius.xsmall`. Sharp and engineered, not soft and consumer.
3. **Flat surfaces, hairline borders.** 1px rules (Blade `border.width.thin`) and honest separation instead of blur, glow, and translucency. Depth comes from spacing and hierarchy, never from effects. No `box-shadow` used as decoration.
4. **Density is a feature.** A reviewer's working console, not a marketing page. Tight rows, more information per screen, tabular alignment.
5. **Single-hue discipline.** One accent doing real work. Color is reserved for status and never spent on decoration.
6. **Restrained semantics.** Risk and resolution states need distinguishable colors, pulled from Blade's own semantic scales rather than stock palettes.
7. **The interface states its own limits.** Defense-only is a design element: the console says plainly what it cannot do.

## Tokens

### Color

Anchors given by the brief:

| Token | Hex | Role |
|---|---|---|
| `--c-prussian` | `#012652` | Deep header band, institutional foundation |
| `--c-navy` | `#0C2651` | Enterprise-gravity base, header gradient-free depth |
| `--c-accent` | `#0D94FB` | Primary action, active state |

Blade `azure` (chromatic primary), converted from `hsla`:

`50 #F5F9FF` · `100 #D6E5FF` · `200 #A8C8FF` · `300 #75AAFF` · `400 #4287FF` · `500 #1364F1` · `600 #0E54CD` · `700 #0A44A9` · `800 #073688` · `900 #052761` · `1000 #021331`

Blade `ashGrayLight` (neutral canvas and text):

`0 #FFFFFF` · `50 #F9F9FA` · `100 #F4F5F6` · `200 #EFF0F1` · `300 #E2E3E4` · `400 #CBCED2` · `500 #ABAFB5` · `600 #93989F` · `700 #858B93` · `800 #6B717B` · `900 #545A64` · `1000 #3F4550` · `1100 #272D35` · `1200 #1C2026` · `1300 #0C0F13`

Blade semantic scales — these replace the Tailwind defaults entirely:

| Meaning | Blade scale | Value used |
|---|---|---|
| Negative / risk-increasing / confirmed fraud | `crimson` | `700 #AA180E` on light, `50 #FDF3F2` as wash |
| Positive / risk-decreasing / dismissed | `emerald` | `700 #00753B`, wash `50 #E6F4ED` |
| Notice / escalated | `cider` | `700 #C75300`, wash `50 #FFF6F0` |
| Information | `sapphire` | `700 #0070A8`, wash `50 #E7F7FD` |

**Rule:** color appears only on status, risk direction, and the single primary action. Never on containers, headings, or dividers.

### Typography

Three decisions, not one:

- **Text:** `Inter` — Blade's real `fontFamily.text` token — with a full system fallback stack so a cold start or a blocked font request degrades to a near-identical rendering rather than to Times.
- **Numerals:** `font-variant-numeric: tabular-nums` wherever numbers are compared down a column — risk scores, costs, counts, timestamps. Columns must line up; this is the single most legible signal that the surface is a financial instrument.
- **Code:** Blade's `fontFamily.code` token (`Menlo, ui-monospace, …`) for entity IDs, model feature names, and raw scores. These are identifiers, and identifiers are set in mono.

Scale is Blade's `typography.onDesktop.fonts.size` (desktop): `10 · 11 · 12 · 14 · 16 · 18 · 20 · 24 · 32 · 40`. Weights are Blade's four: `400 / 500 / 600 / 700`. Line heights from Blade's `lineHeights`: `16 · 17 · 20 · 24 · 26 · 32`.

Uppercase micro-labels at 11px with positive tracking carry section and column headers — documentation-like, which is the register Blade names.

### Spacing

Blade's `spacing` scale verbatim: `0 · 2 · 4 · 8 · 12 · 16 · 20 · 24 · 32 · 40 · 48 · 56`. No value outside this scale. The 2px and 4px steps are what make density possible without crowding.

### Border and radius

- Radius: `4px` (`border.radius.xsmall`) on every container. `2px` (`2xsmall`) on tags and inline chips. Nothing else.
- Width: `1px` (`border.width.thin`) hairlines only. Never thicker, and never a coloured side-rule to encode status — a thick tinted `border-left` is itself one of the most recognisable generated-UI tells. Status is carried by a tag, not by a stripe. The single exception is the 2px underline on the active view tab, which marks selection rather than decorating a container.

### Motion

Blade's `motion` tokens: duration `quick 200ms` / `moderate 280ms`, easing `standard cubic-bezier(0.3, 0, 0.2, 1)`. Motion is confined to state feedback — a row resolving, a panel switching. Nothing decorative, nothing on load, and everything inside `prefers-reduced-motion`.

## Layout

- **Header band** in `--c-prussian`, full-bleed, holding the product name, the defense-only statement, and instance meta. This is the one dark element on the page; it is what rhymes with the Razorpay dashboard.
- **Metric strip** directly beneath: queue depth, mean risk score, aggregate FP cost at risk, resolved count. Hairline-separated cells, tabular figures, no cards.
- **View switch**: `Pending review` / `Audit trail`. The audit trail is a named requirement of the competition track and the strongest part of the architecture, so it is a peer view, not a detail hidden behind a row.
- **Queue items** as dense bordered blocks: identity and score on one line, brief beneath, contributing factors as a real table, actions in a footer bar with an inline note field.

## Copy

Precise and institutional. No exclamation marks, no encouragement, no product-marketing verbs. The reviewer is a professional doing a job; the interface reports and gets out of the way.

**Defense-only statement** is substantive, not a badge: state plainly that no control in this interface can block, hold, or cancel a transaction, and that every decision is written to an append-only log. Never as an emoji chip.

Forbidden in copy as well as in code: any verb implying the system acts on money.

## Non-negotiables inherited from the product

- No framework, no build step, no npm. Plain HTML/CSS/JS served by Flask.
- No number rendered that does not trace to a committed artifact.
- Must be legible in a 1080p screen recording — which sets a floor on type size and contrast that density is not allowed to breach.

## Decisions taken during the build

Recorded because each one resolves a real conflict, and a later session should not have to rediscover it.

**Dodger Blue is the active accent, not the button fill.** The brief names `#0D94FB` as the primary action colour. White text on it measures 3.16:1 — under the 4.5:1 floor for body-size text. The filled control therefore uses Blade azure 600 `#0E54CD` (6.6:1 with white), and Dodger carries every state where it sits behind no text: focus ring, active tab rule, input focus border. The brief's intent is kept; the contrast floor is not breached to keep its literal hex.

**No kicker above the heading.** The masthead first read `Fraud-Spike Detector` as a small tracked uppercase label above `Review Queue`. That is the eyebrow pattern, banned outright — the heading carries its own weight. It is now one heading, `Fraud-Spike Review Queue`, with a standfirst that says what the surface is for.

**The loading state has no animation.** A queue read is milliseconds. A looping bar or spinner would be perpetual motion demanding attention the wait has not earned, so the loading panel is a line of text and nothing else.

**The audit table carries its own border, not a wrapper.** The scroll container is unstyled; the boundary sits on the table. The header row and row hover are meant to run the full width of the box, so the inset comes from cell padding rather than from padding on a container around it.

**Inter is kept, deliberately, against the general-purpose warning.** Inter is Blade's real `fontFamily.text` token, so dropping it would move the surface away from the system it is meant to rhyme with. The tell the brief actually names is *Inter as the only typographic decision* — that is addressed by three decisions rather than one: Inter for text, tabular lining figures wherever numbers are compared down a column, and Blade's `fontFamily.code` (Menlo) for identifiers, feature names, and scores. The mechanical detector still flags the face by name; that finding is accepted rather than suppressed.

## Verified

Measured on the built result, against the demo snapshot, not asserted as intent:

- 36 distinct text-on-background pairs across both views, zero below WCAG AA (4.5:1 body, 3:1 large). Placeholder text 4.91:1.
- Mechanical detector clean on the fully rendered DOM apart from the accepted `overused-font` finding above; zero findings on the CSS and JS sources.
- Renders at 1440×900 and 375×812; no console errors; keyboard focus ring visible and correct.
