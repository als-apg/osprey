---
name: creating-an-osprey-panel
description: >
  Author a new OSPREY web-terminal panel — a self-contained, themed HTML
  mini-app the terminal mounts beside the chat. Use whenever someone wants to
  create, author, add, scaffold, or build an OSPREY panel, a design-system
  panel, or a standalone themed surface for the web terminal. Produces a
  token-only panel that passes the panel validator.
summary: Author a themed, token-only OSPREY web-terminal panel that passes the validator
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

The canonical exemplar every panel is copied from lives in the OSPREY source at
`src/osprey/interfaces/design_system/panels/reference/` (`index.html` +
`manifest.json`). This skill inlines everything you need, but that reference is
the source of truth if anything here is ambiguous.

---

## Step 1 — Create the panel directory and copy the head verbatim

Create a new directory for the panel (e.g. `my-panel/`) with two files:
`index.html` and `manifest.json`.

Start `index.html` from the exact head below. The **order is load-bearing** and
the validator checks two of these lines:

1. `theme-boot.js` **FIRST** in `<head>` — a plain (non-module) script that
   resolves and applies `data-theme` before first paint, so the panel never
   flashes the wrong theme. It reads the `?theme=` query param for free.
2. The `tokens.css` stylesheet — the `var(--…)` custom properties everything
   styles against.
3. A `type="module"` script that calls `initTheme({ role: 'follower' })` (this
   surface is a follower, never the theme hub — the web terminal is the hub) and
   `applyEmbedded()` (reads `?embedded=true` and hides standalone chrome).

Copy this head exactly — do not reorder, and do not convert the boot script to a
module:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<!-- Pre-paint theme boot FIRST: resolves and applies data-theme before first
     paint (reads ?theme=), so the panel never flashes the wrong theme. -->
<script src="/design-system/js/theme-boot.js"></script>
<link rel="stylesheet" href="/design-system/css/tokens.css">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>My Panel</title>
<script type="module">
  // Follower role: this standalone surface is never the theme hub. It applies
  // its own ?theme= when the host appends one, otherwise trusts what
  // theme-boot.js set pre-paint. applyEmbedded() reads ?embedded=true and
  // hides the standalone chrome so the panel sits flush in its host frame.
  import { initTheme } from '/design-system/js/theme-manager.js';
  import { applyEmbedded } from '/design-system/js/frame-params.js';
  initTheme({ role: 'follower' });
  applyEmbedded();
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
| Warning / amber (semantic) | `var(--color-amber)` + tint `var(--amber-tint-08)` |
| Error (semantic) | `var(--color-error)` + tint `var(--error-tint-08)` |
| Accent tint (subtle fill) | `var(--accent-tint-06)` |
| Neutral tint (subtle fill, e.g. `<code>`) | `var(--neutral-tint-08)` |
| Panel shadow | `var(--shadow-panel)` |
| Display / body font family | `var(--font-display)` |
| Monospace font family | `var(--font-mono)` |

If you need a color that isn't in this list, look it up in
`src/osprey/interfaces/design_system/static/css/tokens.css` — never invent a hex
value. A `--…-tint-NN` token is a low-opacity fill of the base color (e.g.
`--success-tint-08`), useful for badge/pill backgrounds.

---

## Step 3 — Honor `?theme=` and `?embedded=`

Both are handled for free by the head in Step 1 — you only wire up the CSS:

- **`?theme=`** is applied pre-paint by `theme-boot.js` and thereafter by
  `initTheme`. You do nothing beyond styling with tokens.
- **`?embedded=true`** makes `applyEmbedded()` add `body.embedded`. Put any
  standalone-only chrome (a page header/title bar that only makes sense when the
  panel is opened on its own) behind a rule that hides it when embedded:

  ```css
  .panel-chrome { /* header shown only standalone */ }
  body.embedded .panel-chrome { display: none; }
  ```

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

There is no CLI wrapper. Run the validator with this one-liner, replacing
`PATH/TO/PANEL_DIR` with your panel directory. It prints nothing and exits 0
when the panel is valid; it raises `PanelValidationError` (listing every
failure) when it is not:

```bash
uv run python -c "from osprey.interfaces.design_system.panels.validator import assert_valid_panel; assert_valid_panel('PATH/TO/PANEL_DIR')"
```

If it raises, fix each reported `PanelRule` and re-run until it is silent:

| Rule | Meaning | Fix |
|---|---|---|
| `MANIFEST_MISSING` | no `manifest.json` in the dir | add it (Step 4) |
| `MANIFEST_INVALID` | bad JSON, missing/empty/wrong-typed field, or `id` not a kebab slug | fix the manifest (Step 4) |
| `ENTRY_MISSING` | the `entry` file doesn't exist on disk | create it, or fix the `entry` value |
| `MISSING_DESIGN_SYSTEM_LINK` | entry HTML doesn't link `tokens.css` | add the `<link>` (Step 1) |
| `MISSING_THEME_BOOT` | entry HTML doesn't load `theme-boot.js` | add the `<script>`, first in head (Step 1) |
| `RAW_HEX_COLOR` | a raw `#rgb`/`#rrggbb`/… literal appears where a token belongs | replace it with a `var(--…)` token (Step 2) |

Note the `RAW_HEX_COLOR` scan is a raw text scan: a URL fragment whose name is
all hex digits and exactly 3/4/6/8 long (e.g. `href="#abc"` or
`href="#deadbeef"`) is flagged as if it were a color. If that happens, rename
the fragment — never loosen the token-only rule.

The panel is complete when the one-liner raises nothing.

---

## Step 6 — Cross-origin re-theme reality (know this, don't fight it)

The web terminal and its panels are served from different origins. A panel
re-themes on **reload** when the host appends an updated `?theme=` — that path
works. But `postMessage`-based live re-theming is **same-origin only**, so do
**not** build a panel that expects to receive live theme broadcasts from a
cross-origin host without a reload. Rely on the pre-paint `?theme=` boot (Step 1)
plus reload, not on live cross-origin postMessage.

---

## Anti-Patterns

Do NOT:

- Reorder the head — `theme-boot.js` must be first, before `tokens.css`.
- Convert the boot script to `type="module"` — it must run synchronously,
  pre-paint.
- Use any raw hex color literal, anywhere, in HTML/CSS/JS — always a `var(--…)`.
- Invent token names or hex values — use the table in Step 2 or look them up in
  `tokens.css`.
- Give the manifest `id` uppercase, spaces, or underscores — kebab slug only.
- Point `entry` at a file that doesn't exist.
- Skip the Step 5 validator self-check — a panel is not done until it is silent.
- Expect live cross-origin `postMessage` re-theming (Step 6).
