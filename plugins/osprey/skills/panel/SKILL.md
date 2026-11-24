---
name: panel
description: >
  Author a new OSPREY web-terminal panel — a self-contained, themed HTML
  mini-app the terminal mounts beside the chat. Use whenever someone wants to
  create, author, add, scaffold, or build an OSPREY panel, a design-system
  panel, or a standalone themed surface for the web terminal. Produces a
  token-only panel that passes the panel validator.
---

# Creating an OSPREY Panel

A **panel** is a directory bundling one HTML entry point plus a `manifest.json`.
It is a self-contained mini-app the OSPREY web terminal can mount alongside the
chat surface. A valid panel is **theme-aware** (it boots the shared theme before
first paint and honors `?theme=`) and **token-only** (every color, surface,
border, and font resolves through a `var(--…)` design token — never a raw hex
literal).

Follow these steps in order. The panel is done when the validator (Step 5)
raises nothing. Do not skip the self-check.

The canonical exemplar every panel is copied from ships with OSPREY at
`interfaces/design_system/panels/reference/` (`index.html` + `manifest.json`),
under the package root that
`python3 -c "import osprey, pathlib; print(pathlib.Path(osprey.__file__).parent)"`
prints. This skill inlines everything you need, but that reference is the source
of truth if anything here is ambiguous.

---

## Step 1 — Create the panel directory and copy the head verbatim

Create a new directory for the panel (e.g. `my-panel/`) with two files:
`index.html` and `manifest.json`.

Start `index.html` from the exact head below. The **order is load-bearing** and
the validator checks every item:

1. `theme-boot.js` **FIRST** in `<head>` — a plain (non-module) script that
   resolves and applies `data-theme` before first paint, so the panel never
   flashes the wrong theme. It reads the `?theme=` query param for free.
2. The page-ready writer by its own tag, directly after `theme-boot.js` and
   ahead of every other module or deferred script: a script of the panel that
   then fails to load marks the page `failed` instead of leaving it pending.
3. The three design-system sheets, in this order: `tokens.css` (the `var(--…)`
   custom properties everything styles against), `base.css`, and `primitives.css`
   (the shared `osprey-*` components — buttons, cards, chips, banners, fields;
   `interfaces/design_system/DESIGN.md` §9, §12 and §13 under the same package
   root).
4. The `fonts.css` stylesheet — the shared web fonts; without it the panel
   falls back to system fonts.
5. A `type="module"` script that calls `initTheme({ role: 'follower' })` (this
   surface is a follower, never the theme hub — the web terminal is the hub) and
   `applyEmbedded()` (sends the page, opened on its own, to the web terminal with
   the panel open, and marks a framed page as embedded), and calls
   `markPageReady()` once the panel's first state is drawn; `<html>` carries
   `data-page-ready="pending"` so the shared start-up leaves the mark to that
   call. Whatever drives the panel (a test, a recording, your own script) waits
   until `data-page-ready` is `true` or `failed`; `failed` means the page cannot
   be driven: stop there.

Copy this head exactly — do not reorder, and do not convert the boot script to a
module:

```html
<!DOCTYPE html>
<html lang="en" data-page-ready="pending">
<head>
<meta charset="UTF-8">
<!-- Pre-paint theme boot FIRST: resolves and applies data-theme before first
     paint (reads ?theme=), so the panel never flashes the wrong theme. -->
<script src="/design-system/js/theme-boot.js"></script>
<!-- The page-ready writer, ahead of every other module: a script of the panel
     that fails to load then marks the page failed. -->
<script type="module" src="/design-system/js/page-ready.js"></script>
<link rel="stylesheet" href="/design-system/css/tokens.css">
<link rel="stylesheet" href="/design-system/css/base.css">
<link rel="stylesheet" href="/design-system/css/primitives.css">
<link rel="stylesheet" href="/static/fonts/fonts.css">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>My Panel</title>
<script type="module">
  // Follower role: this standalone surface is never the theme hub. It applies
  // its own ?theme= when the host appends one, otherwise trusts what
  // theme-boot.js set pre-paint. applyEmbedded() sends a page opened on its
  // own to the web terminal with this panel open, and marks a framed page.
  import { initTheme } from '/design-system/js/theme-manager.js';
  import { applyEmbedded } from '/design-system/js/frame-params.js';
  import { markPageReady } from '/design-system/js/page-ready.js';
  initTheme({ role: 'follower' });
  applyEmbedded();
  // <html> declares the mark pending, so this call sets it. Call it once the
  // panel's first state is drawn; a panel that fetches its first state calls
  // it after the fetch, error or not.
  markPageReady();
</script>
<style>
  /* Every color/surface/border/font below resolves through a var(--…) token.
     ZERO raw hex anywhere in this file. */
</style>
</head>
<body>
  <!-- your panel content -->
</body>
</html>
```

---

## Step 2 — Style through tokens only (ZERO raw hex)

Every color, background, border, and font in your CSS **must** be a `var(--…)`
token from `tokens.css`. A raw hex color literal anywhere in the HTML, or in any
sibling `.css`/`.js` file, fails the validator (`RAW_HEX_COLOR`). This is what
lets the panel render correctly under every theme family, in both light and
dark, for free.

Use these real token names (verified against `tokens.css`):

| Purpose | Token |
|---|---|
| Page background | `var(--bg-primary)` |
| Elevated card/surface background | `var(--bg-elevated)` |
| Primary text | `var(--text-primary)` |
| Secondary text | `var(--text-secondary)` |
| Muted / caption text | `var(--text-muted)` |
| Default border | `var(--border-default)` |
| Accent border | `var(--border-accent)` |
| Accent / link / highlight color | `var(--color-accent-light)` |
| Success (semantic) | `var(--color-success)` + tint `var(--success-tint-08)` |
| Warning (semantic) | `var(--color-warning)` |
| Secondary accent / decorative highlight | `var(--color-accent-secondary-light)` + tint `var(--accent-secondary-tint-08)` |
| Error (semantic) | `var(--color-error)` + tint `var(--error-tint-08)` |
| Accent tint (subtle fill) | `var(--accent-tint-06)` |
| Neutral tint (subtle fill, e.g. `<code>`) | `var(--neutral-tint-08)` |
| Panel shadow | `var(--shadow-panel)` |
| Display / body font family | `var(--font-display)` |
| Monospace font family | `var(--font-mono)` |

If you need a color that isn't in this list, look it up in
`interfaces/design_system/static/css/tokens.css` under that same package root —
never invent a hex value. A `--…-tint-NN` token is a low-opacity fill of the base color (e.g.
`--success-tint-08`), useful for badge/pill backgrounds.

---

Reach for an `osprey-*` class from `primitives.css` before writing your own: a
button is `osprey-btn`, a card `osprey-card`, a status pill `osprey-chip`. Every
class the panel sets must be defined, in a `<style>` block of the page, the
panel's own `*.css`, or one of the three linked sheets.

---

## Step 3 — One header: the web terminal's

A panel draws no header, title bar or theme control of its own. In a tile and in
its window, the web terminal's tile bar is the panel's one header, so there is
nothing to hide when framed and no `body.embedded` rule. `applyEmbedded()` (Step
1) sends a panel page opened on its own to the web terminal with that panel
open, and marks a framed page.

- Controls a top bar would hold (a view switcher, an action, a search, menu
  entries) go to the tile bar through `contributeHeader([...])` from
  `/design-system/js/header-contrib.js`; their clicks come back through
  `onHeaderAction`.
- **`?theme=`** is applied pre-paint by `theme-boot.js` and thereafter by
  `initTheme`. You do nothing beyond styling with tokens.

---

## Step 4 — Write `manifest.json`

The manifest declares the panel's identity and entry point. Required fields:

- **`id`** — a lowercase kebab slug matching `^[a-z0-9][a-z0-9-]*$` (starts with
  a lowercase letter or digit, then lowercase letters/digits/single hyphens).
  Examples: `beam-status`, `rf-cavity`, `my-panel`. Not `My_Panel`, not
  `beam status`.
- **`label`** — human-readable display name (non-empty string).
- **`entry`** — the HTML entry filename, relative to the panel dir, and it must
  actually exist on disk. Use `index.html`.

Optional: **`version`** (integer, defaults to 1). Unknown keys are tolerated
(preserved for forward compatibility), so extra fields won't fail validation.

Exact shape:

```json
{
  "id": "my-panel",
  "label": "My Panel",
  "entry": "index.html",
  "version": 1
}
```

---

## Step 5 — Self-check with the validator (required)

Run the validator's command, replacing `PATH/TO/PANEL_DIR` with your panel
directory:

```bash
uv run python -m osprey.interfaces.design_system.panels.validator PATH/TO/PANEL_DIR
```

It prints every finding. A finding of a rule that refuses makes it print
`PANEL INVALID` and exit 1: the web terminal will not serve the panel. A finding
of a rule that only reports ends in `[<rule>: report]`; the panel is still
served. Fix each reported `PanelRule` and re-run:

| Rule | Consequence | Meaning | Fix |
|---|---|---|---|
| `MANIFEST_MISSING` | refuse | no `manifest.json` in the dir | add it (Step 4) |
| `MANIFEST_INVALID` | refuse | bad JSON, missing/empty/wrong-typed field, or `id` not a kebab slug | fix the manifest (Step 4) |
| `ENTRY_MISSING` | refuse | the `entry` file doesn't exist on disk | create it, or fix the `entry` value |
| `MISSING_DESIGN_SYSTEM_LINK` | refuse | entry HTML doesn't link all three sheets: `tokens.css`, `base.css`, `primitives.css` | add the missing `<link>` (Step 1) |
| `MISSING_THEME_BOOT` | refuse | entry HTML doesn't load `theme-boot.js` | add the `<script>`, first in head (Step 1) |
| `MISSING_FONT_LINK` | refuse | entry HTML doesn't link `/static/fonts/fonts.css` | add the `<link>` (Step 1) |
| `RAW_HEX_COLOR` | refuse | a raw `#rgb`/`#rrggbb`/… literal appears where a token belongs | replace it with a `var(--…)` token (Step 2) |
| `PAGE_READY` | refuse | the entry page never declares `data-page-ready="pending"`, does not load `page-ready.js` first, or no script it loads calls `markPageReady()` | add what is missing (Step 1) |
| `UNDEFINED_CLASS` | report | the panel's HTML or JS sets a class with no rule in a `<style>` block of the page, the panel's own `*.css` or a linked design-system sheet | add the rule, or drop the class; a state that carries no style belongs in a `data-` attribute |

Note the `RAW_HEX_COLOR` scan is a raw text scan: a URL fragment whose name is
all hex digits and exactly 3/4/6/8 long (e.g. `href="#abc"` or
`href="#deadbeef"`) is flagged as if it were a color. If that happens, rename
the fragment — never loosen the token-only rule.

The panel is complete when the command prints `PANEL OK` and no line.

A panel that belongs to an application (`apps/<name>/panel/`) is judged by
`osprey app check`, which runs these same rules on `panel/index.html` (an app
panel has no manifest): a rule that refuses fails the check, a rule that only
reports is printed. Run it from the deployment and fix until it passes.

---

## Step 6 — Re-theme reality (know this, don't fight it)

The web terminal serves its panels through the panel proxy, so panel and host
share one origin — there is no cross-origin surface, and the interface apps
register no CORS middleware. The proxy is strict in both directions: it strips
`cookie`, `authorization` and the operator-secret header from what it forwards
to your backend, and strips `Set-Cookie` and `Access-Control-*` from what your
backend returns, so a panel cannot set a cookie on the terminal's origin or open
itself to another one. Re-theme by having the host append an updated `?theme=`
and reloading — that path always works and is the one to rely on.
`postMessage`-based live re-theming is same-origin only; since panel and host
share an origin it is available, but the `?theme=` boot (Step 1) plus reload
remains the robust path and is what you should build against.

---

## Anti-Patterns

Do NOT:

- Reorder the head — `theme-boot.js` must be first, then the page-ready
  writer, then `tokens.css`, `base.css`, `primitives.css`.
- Set a class no stylesheet defines — the validator reports it.
- Convert the boot script to `type="module"` — it must run synchronously,
  pre-paint.
- Use any raw hex color literal, anywhere, in HTML/CSS/JS — always a `var(--…)`.
- Invent token names or hex values — use the table in Step 2 or look them up in
  `tokens.css`.
- Give the manifest `id` uppercase, spaces, or underscores — kebab slug only.
- Point `entry` at a file that doesn't exist.
- Skip the Step 5 validator self-check — a panel is not done until it prints
  `PANEL OK` and no line.
- Expect live cross-origin `postMessage` re-theming (Step 6).
- Draw a header of the panel's own, or write a `body.embedded` rule: the tile
  bar is the one header (Step 3).
