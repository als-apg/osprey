// @ts-check
/**
 * OSPREY Web Terminal — the control-target bar on an embedded JupyterLab page.
 *
 * A notebook cell writes to the machine, so the question the hub's header chip
 * answers — *if this writes now, where does it land, and will it be refused?* —
 * has to be answerable without leaving the notebook. This module is the whole
 * of that: it is the entry module the panel proxy injects into JupyterLab's
 * `<head>` (`_inject_control_target_bar` in `routes/proxy.py`), and everything
 * below it is the hub's own chip, unchanged.
 *
 * **One picker per window.** Inside the hub the JUPYTER tab is an iframe under
 * the header, and the header chip is the picker; a second one in the frame
 * would be the same switch drawn twice. So the bar mounts only when the Lab
 * page is its own top-level document — popped out of the hub, or opened at
 * its own address — and does nothing at all when framed: no host, no
 * stylesheets, no stream. The proxy injects the tag either way, because only
 * the page can tell which it is ({@link isEmbedded}).
 *
 * It owns three things and nothing else:
 *
 * - **a stable host.** One fixed-position element on `document.body`, created
 *   once and never re-rendered. JupyterLab owns `#main`'s layout and reflows
 *   what it finds there, so the bar lives outside it; and the chip's docs are
 *   explicit that a host re-rendered away leaves the chip's anchor detached
 *   with both its timers stopping themselves on the next tick.
 * - **the stylesheets the chip is written against.** The hub's design tokens
 *   and `terminal.css`, both fetched back out of the hub's own tree through
 *   the panel-scoped static routes. Nothing here restates a chip rule: a copy
 *   would drift from the hub's within a release, and the two surfaces showing
 *   the same machine in different colours is the failure this whole feature
 *   exists to prevent.
 * - **the theme.** JupyterLab publishes its own light/dark choice on
 *   `<body data-jp-theme-light>`; the tokens key off `data-theme`. The bar
 *   translates one into the other, and follows a change.
 *
 * What it deliberately does NOT do: position itself relative to any JupyterLab
 * element, take focus, bind a key, or add a word of its own copy. The chip is
 * the whole surface.
 *
 * Two effects it DOES have on a document it does not own, both named here
 * because neither is visible from the bar's own markup:
 *
 * - **`color-scheme`.** `tokens.css` declares it on `:root` (dark) and on each
 *   `[data-theme]`, and JupyterLab core declares none of its own, so the bar
 *   decides the Lab document's native scrollbar and form-control scheme. It
 *   follows Lab's own light/dark through `applyTheme`, so the result agrees
 *   with the page — but it is a global rule and not a bar-local one. Removing
 *   it would mean shipping a trimmed copy of the token file, which is the
 *   drift this module exists to avoid.
 * - **A reload on an expired session.** The chip's stream rides `api.js`'s
 *   shared EventSource, whose error path probes `/api/session` and reloads the
 *   PAGE on a definite 401 — and here the page is the operator's notebook.
 *   That is the right outcome (the same gate fronts the Lab page, so its own
 *   saves are 401ing too, and the reload lands on the login explanation), but
 *   anything unsaved since Lab's last autosave goes with it.
 *
 * The page has no `terminal.js`, no bar-item shell and no `.header-actions`,
 * which is exactly why `initControlTargetChip` takes a `host`.
 */

import { initControlTargetChip, teardownControlTargetChip } from './control-target-chip.js';
import {
  initControlTargetPopover,
  teardownControlTargetPopover,
} from './control-target-popover.js';

/** The bar itself. Also the chip's host, and the CSS hook below. */
const BAR_ID = 'osprey-control-target-bar';

/** The `<style>` this module injects: the bar's own layout, and nothing else. */
export const STYLE_ID = 'osprey-control-target-bar-style';

/** The two `<link>`s, by id, so a re-init cannot double them. */
export const TERMINAL_CSS_ID = 'osprey-control-target-bar-terminal-css';
export const TOKENS_CSS_ID = 'osprey-control-target-bar-tokens-css';

/**
 * The hub's `terminal.css`, addressed relative to this module.
 *
 * This file is served by `proxy_panel_terminal_static` at
 * `{prefix}/panel/{id}/terminal-static/js/`, so `../css/terminal.css` is the
 * same route's `css/` directory — the hub's own stylesheet, not the sidecar's.
 * Deriving it from `import.meta.url` rather than from an injected literal is
 * what makes the multi-user mount (`/u/<user>`) free: whatever prefix this
 * module was loaded under is the prefix its assets come from.
 */
export const TERMINAL_CSS_URL = new URL('../css/terminal.css', import.meta.url).href;

/**
 * The hub's design tokens, through the panel's design-system route — two
 * directories up from `terminal-static/js/` is the panel root.
 *
 * Needed because the chip's every colour, radius and duration is a token, and
 * a JupyterLab page loads none of them. This is the hub's copy for the same
 * reason every other embedded panel gets it (see `proxy_panel_design_system`):
 * the palette an operator sees must not depend on which release built the
 * sidecar.
 */
export const TOKENS_CSS_URL = new URL('../../design-system/css/tokens.css', import.meta.url).href;

/**
 * The bar's own layout. Everything the chip and its popover look like comes
 * from `terminal.css`; what is here is only where the bar sits.
 *
 * `top: 1px; right: 10px` parks it in the empty right end of JupyterLab's menu
 * bar, which is 28 px tall, exactly the chip's height, and left-aligned — so
 * the bar reads as part of the top chrome without covering a menu. It has to
 * be at the TOP: `.ctc-popover` opens `top: calc(100% + …)` under its trigger,
 * and a bar at the bottom of the viewport would drop its 420 px popover off
 * the screen.
 *
 * The z-index is a layer above JupyterLab's chrome and a layer BELOW the
 * confirm overlay (`--z-modal`, mounted on `document.body` by
 * `posture-confirm.js`), which is what keeps the hub's behaviour intact: the
 * confirm covers the popover that raised it.
 *
 * The `.ctc-anchor` rules are a floor under `terminal.css`, which carries the
 * same two: the popover is absolutely positioned against the anchor, so it
 * anchors correctly even in the moments before the stylesheet lands.
 */
const BAR_CSS = `
#${BAR_ID} {
  position: fixed;
  top: 1px;
  right: 10px;
  display: flex;
  align-items: center;
  z-index: var(--z-overlay, 300);
}

#${BAR_ID} .ctc-anchor {
  position: relative;
  display: flex;
}

#${BAR_ID} .ctc-anchor[hidden] {
  display: none;
}
`;

/**
 * JupyterLab's own light/dark publication, on `<body>`. Present once Lab has
 * booted; absent before that, and Lab's default theme is light.
 */
const JP_THEME_LIGHT_ATTR = 'data-jp-theme-light';

/**
 * Whether this document is framed by another — the hub's JUPYTER tab — rather
 * than being its own window.
 *
 * `self !== top` is the one test that needs nothing from the parent: reading
 * `top` itself never throws, only reaching into a cross-origin parent does,
 * and this never does. A page that cannot even read `top` is framed by
 * something stricter than the hub and is treated as framed.
 *
 * @param {{self?: unknown, top?: unknown}} [win] the window to ask; the global one by default
 * @returns {boolean}
 */
export function isEmbedded(win = /** @type {any} */ (globalThis)) {
  try {
    return win.self !== win.top;
  } catch {
    return true;
  }
}

/** @type {HTMLElement|null} */
let bar = null;

/** @type {MutationObserver|null} */
let themeObserver = null;

/**
 * Put the bar on the page and mount the chip into it.
 *
 * Idempotent: a second call re-uses the same bar element and re-enters the
 * chip's own idempotent init, which re-renders rather than mounting a second
 * chip. Answers `null` on a document with no `<body>` to mount into, and on
 * a document framed inside the hub, where the header chip is the picker.
 *
 * @param {{eventSourceFactory?: typeof import('./api.js').createEventSource, embedded?: boolean}} [opts]
 *   `eventSourceFactory` is passed straight through to the chip, which is how
 *   its own suite injects a stream; `undefined` takes the chip's own default.
 *   `embedded` overrides {@link isEmbedded} for a suite that cannot frame its
 *   document. Nothing else here is injectable.
 * @returns {HTMLElement|null} The bar, or null if it does not belong on this page.
 */
export function initControlTargetLabBar({ eventSourceFactory, embedded = isEmbedded() } = {}) {
  if (typeof document === 'undefined' || !document.body) return null;
  if (embedded) return null;

  ensureStyles();

  if (!bar || !bar.isConnected) {
    bar = document.getElementById(BAR_ID);
  }
  if (!bar) {
    bar = document.createElement('div');
    bar.id = BAR_ID;
  }
  if (bar.parentElement !== document.body) {
    // On `document.body`, never inside `#main`: JupyterLab lays out its own
    // dock area and a bar left in there is reflowed away.
    document.body.appendChild(bar);
  }

  applyTheme();
  watchTheme();

  initControlTargetChip({ host: bar, eventSourceFactory });
  // Only after the chip: the popover hangs off the chip's anchor and answers
  // null on a page where no chip has mounted.
  initControlTargetPopover();

  return bar;
}

/**
 * Take the bar back off the page and release everything it holds — the chip,
 * the popover, the theme observer, the styles and the node.
 */
export function teardownControlTargetLabBar() {
  teardownControlTargetPopover();
  teardownControlTargetChip();

  themeObserver?.disconnect();
  themeObserver = null;
  document.documentElement.removeAttribute('data-theme');

  bar?.remove();
  bar = null;

  for (const id of [STYLE_ID, TERMINAL_CSS_ID, TOKENS_CSS_ID]) {
    document.getElementById(id)?.remove();
  }
}

/**
 * The tokens, the hub's stylesheet and the bar's own layout, each once.
 *
 * Order matters between the two links only in that tokens define what
 * `terminal.css` reads; both are `<link>`s in `<head>`, so the cascade is
 * document order and the chip is styled the moment the second lands.
 */
function ensureStyles() {
  ensureStylesheet(TOKENS_CSS_ID, TOKENS_CSS_URL);
  ensureStylesheet(TERMINAL_CSS_ID, TERMINAL_CSS_URL);

  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = BAR_CSS;
    document.head.appendChild(style);
  }
}

/**
 * @param {string} id
 * @param {string} href
 */
function ensureStylesheet(id, href) {
  if (document.getElementById(id)) return;
  const link = document.createElement('link');
  link.id = id;
  link.rel = 'stylesheet';
  link.href = href;
  document.head.appendChild(link);
}

/**
 * Translate JupyterLab's theme onto the attribute the design tokens key off.
 *
 * `data-theme` goes on `<html>` rather than on the bar, because the confirm
 * dialog the popover raises mounts on `document.body` — a bar-scoped attribute
 * would leave that one surface reading the token file's default while
 * everything around it read Lab's choice.
 *
 * Absent means Lab has not booted yet, and Lab's default theme is light.
 */
function applyTheme() {
  const light = document.body?.getAttribute(JP_THEME_LIGHT_ATTR);
  document.documentElement.setAttribute('data-theme', light === 'false' ? 'dark' : 'light');
}

/** Follow a theme the operator changes from inside JupyterLab. */
function watchTheme() {
  if (themeObserver || typeof MutationObserver === 'undefined') return;
  themeObserver = new MutationObserver(applyTheme);
  themeObserver.observe(document.body, {
    attributes: true,
    attributeFilter: [JP_THEME_LIGHT_ATTR],
  });
}

/**
 * This module is an entry point — the page that loads it has no app shell to
 * call `init` from — so it starts itself.
 *
 * Inside a `try`, because the one thing worse than a missing chip is a broken
 * notebook: nothing this module can throw is worth taking JupyterLab down
 * with, and the console message is what says which it was.
 *
 * A module script runs after the document is parsed, so `document.body` is
 * there; the listener is for the case where something loads this module
 * earlier than the tag the proxy injects does.
 */
function boot() {
  try {
    initControlTargetLabBar();
  } catch (err) {
    console.error('OSPREY control-target bar failed to start:', err);
  }
}

if (typeof document !== 'undefined') {
  if (document.body) {
    boot();
  } else {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  }
}
