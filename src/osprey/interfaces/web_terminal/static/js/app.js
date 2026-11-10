/* OSPREY Web Terminal — Application Entry Point */

import {
  initTerminal,
  focusTerminal,
  pasteToTerminal,
  getCurrentSessionId,
  getTerminalInstance,
  startTerminalSurface,
  stopTerminal,
} from './terminal.js';
import { confirmSignOut } from './sign-out.js';
import {
  initPanelManager,
  navigateAndActivatePanel,
  getPanelStandaloneUrl,
  activateTab,
  isPanelHealthy,
  labelOf,
} from './panel-manager.js';
import { wireAgentLinks } from './agent-links.js';
import { wireArtifactLinks } from './artifact-links.js';
import { fetchJSON, withPrefix } from './api.js';
import { getPointer, setPointer } from './session-pointer.js';
import { subscribe as subscribeToSessions } from './session-model.js';
import '/design-system/js/components/osprey-drawer.js';
import { initSettings } from './settings.js';
import { initMemoryGallery } from './memory-gallery.js';
import { initScaffoldGallery } from './scaffold-gallery.js';
import { initHookDebug } from './hook-debug.js';
import { initSafetySwitches } from './safety-switches.js';
import { initCodeReviewTab } from './code-review-tab.js';
import { startNewSession } from './sessions.js';
import { attachSessionSwitcher } from './session-switcher.js';
import { initCommandPalette } from './palette-boot.js';
import { initMoreMenu } from './more-menu.js';
import { installPlaceMenus } from './place-menu.js';
import { initPaneWindow } from './pane-window.js';
import { initTheme } from '/design-system/js/theme-manager.js';
import { isMacPlatform } from '/design-system/js/platform.js';
import { isTerminalSurface, onSurfaceChange } from './agent-surface.js';
import { initChat, enterFromTerminal, renderChatBootFailure } from './chat.js';
import { initDockWorkspace, resetDockLayout } from './dock-workspace.js';
import { installDockMenus } from './dock-menus.js';
import { initSessionLayout } from './session-layout.js';
import { openDirectVisit, takeDirectVisit } from './direct-visit.js';
import { panelFocusAnchor } from './dock-iframe.js';
import { initHeaderContrib } from './tile-header-contrib.js';
import { initControlTargetChip } from './control-target-chip.js';
import { insertPrompt } from './first-contact.js';
import { initControlTargetPopover } from './control-target-popover.js';
import { initFeedback } from './feedback-boot.js';
import { hydrateIcons, initSidebar } from './sidebar.js';
import { initSidebarTarget } from './sidebar-target.js';
import {
  initSidebarArtifacts,
  revealOnLoad,
  teardownSidebarArtifacts,
} from './sidebar-artifacts.js';
import { whilePanelServed } from './panel-availability.js';
import { initTooltips } from '/design-system/js/tooltip.js';
import { queryAllDocuments } from './attached-window.js';
import { initJupyterChrome } from './jupyter-chrome.js';

document.addEventListener('DOMContentLoaded', async () => {
  initTheme({ role: 'hub' });
  // The server's last-active key wins over this browser's, before either
  // surface reads the pointer: both bind from it once, synchronously.
  await resumeLastActive();
  // A missing xterm.js asset is reported in the pane by initTerminal itself;
  // this guard keeps any other terminal init failure from killing the rest of
  // the boot.
  try {
    initTerminal('terminal-container');
    // A tab that opted into the terminal reopens straight onto its prompt.
    if (isTerminalSurface()) focusTerminal();
  } catch (err) {
    console.error('Failed to init terminal:', err);
  }
  // The operator chat. Guarded so a chat init failure can't break the rest of
  // the boot.
  try {
    initChat('operator-container');
  } catch (err) {
    console.error('Failed to init operator chat:', err);
    renderChatBootFailure('operator-container');
  }
  // Guarded the same way: a link failure must not break the rest of boot.
  try {
    initAgentLinks();
  } catch (err) {
    console.error('Failed to init agent answer links:', err);
  }
  try {
    initArtifactLinks();
  } catch (err) {
    console.error('Failed to init artifact links:', err);
  }
  // The sidebar shell — width toggle (Cmd/Ctrl+\), list disclosure, the Agent
  // row — before the panel manager renders its rows into it.
  initSidebar();
  // One tooltip layer for the whole page. A sidebar row whose label shows
  // needs only its explanation; every other tip carries label and detail.
  initTooltips(document.body, {
    // A session's title can be cut short in the row, so its tip keeps the label.
    mode: (el) =>
      el.closest('#sidebar') &&
      !el.matches('.osprey-tree-title') &&
      document.documentElement.getAttribute('data-sidebar') !== 'collapsed'
        ? 'detail'
        : 'label+detail',
  });
  // Before the panel manager creates any iframe, so no contribution can
  // arrive without a listener.
  initHeaderContrib();
  const panelsReady = initPanelManager('right-panel');
  // Before any panel's config read answers, so the notebook tile hears its start.
  initJupyterChrome();
  attachSessionSwitcher(document.getElementById('terminal-label'));
  // Dock the terminal + workspace panels into the dockview shell (replaces the
  // old fixed resize-handle split). Guarded so a dock init failure can't break
  // the rest of boot — the source subtrees stay in the page if it no-ops.
  try {
    initDockWorkspace();
  } catch (err) {
    console.error('Failed to init dock workspace:', err);
  }
  try {
    // Each session keeps its own layout from here on (after the dock exists).
    initSessionLayout();
  } catch (err) {
    console.error('Failed to init session layout:', err);
  }
  // A direct visit to a panel page landed here with that panel and page.
  void openDirectVisit(takeDirectVisit(), panelsReady);
  // The Agent row's session tree: session rows always; their groups and
  // counts only while the artifacts panel is served (Decision 14). The tree
  // follows every later sidebar render through `osprey-sidebar-rendered`.
  // Page load's artifact waits for both the listing and the served panel.
  const retryLoadReveal = initLoadReveal();
  initSidebarArtifacts({ served: false });
  whilePanelServed(
    'artifacts',
    () => {
      initSidebarArtifacts();
      retryLoadReveal();
    },
    teardownSidebarArtifacts
  );
  initKeyboardShortcuts();
  initCommandPalette();
  try {
    initMoreMenu({ getSessionId: getCurrentSessionId });
  } catch (err) {
    console.error('Failed to init the More menu:', err);
  }
  // The shell's places (chat, terminal, ...), here and in every pop-out
  // window, which pane-window wires as it dresses one.
  try {
    installPlaceMenus(document);
    initPaneWindow({ installPlaceMenus });
  } catch (err) {
    console.error('Failed to init the place menus:', err);
  }
  // The empty dock slot's menu. After the panel manager and the dock: it reads
  // the rail and the layout holder. panel-manager's verbs are injected, since
  // panel-menu-policy (imported by panel-manager) reads dock-menus.
  try {
    installDockMenus({
      activate: (id) => activateTab(id, { userInitiated: true }),
      isHealthy: isPanelHealthy,
      labelOf,
      reset: resetDockLayout,
    });
  } catch (err) {
    console.error('Failed to init the dock menus:', err);
  }
  initNewSessionButton();
  initSignOut();
  initSurfaceFollowUps();
  initControlSystemBlock();
  initFeedbackDialog();
  initDrawerTriggerHighlight();
  initSettings();
  initMemoryGallery();
  initScaffoldGallery();
  initHookDebug();
  initSafetySwitches();
  initCodeReviewTab();
  // Listen for paste requests from embedded iframes (gallery, ARIEL)
  initIframePasteBridge();
});

/* ---- Page-load resume ---- */

/** How long page load waits on the server's last-active key. */
export const LAST_ACTIVE_TIMEOUT_MS = 1500;

/**
 * Point the tab at the operator's last-active session, as the server knows
 * it, before either surface boots. One bounded GET: on no key, an error or a
 * timeout the pointer this browser holds stays, and with neither the surfaces
 * mint a key as they always have. It never throws.
 *
 * Exported for testability: the DOMContentLoaded bootstrap awaits it.
 * @param {{timeoutMs?: number}} [options]
 * @returns {Promise<void>}
 */
export async function resumeLastActive({ timeoutMs = LAST_ACTIVE_TIMEOUT_MS } = {}) {
  const controller = new AbortController();
  /** @type {ReturnType<typeof setTimeout> | undefined} */
  let timer;
  // The race bounds the wait even where an abort is not honoured.
  const expired = new Promise((_, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      reject(new Error(`no answer in ${timeoutMs} ms`));
    }, timeoutMs);
  });
  try {
    const body = await Promise.race([
      fetchJSON('/api/sessions/last-active', { signal: controller.signal }),
      expired,
    ]);
    const key = body?.session_key;
    if (typeof key === 'string' && key) setPointer(key);
  } catch (err) {
    console.warn('[app] last-active session unavailable:', /** @type {Error} */ (err).message);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Page load's artifact (OR-3): once the listing has the active row and the
 * artifacts panel is served, the tree reveals the row's last-open artifact
 * (sidebar-artifacts.js's `revealOnLoad`), never through the open rule. It
 * runs once; a key change before then (a switch, a new session) cancels it.
 *
 * Exported for testability: the DOMContentLoaded bootstrap calls it.
 * @returns {() => void} the retry, for when the artifacts panel is served
 */
export function initLoadReveal() {
  let pending = true;
  const attempt = () => {
    if (pending && revealOnLoad()) pending = false;
  };
  subscribeToSessions(({ reason }) => {
    if (reason === 'pointer' || reason === 'switched') pending = false;
    if (reason === 'rows') attempt();
  });
  return attempt;
}

/* ---- Control-system block ---- */

/**
 * The sidebar's control-system block. The chip — which machine this session
 * stands on, and whether a write there would land — mounts into
 * `#sidebar-target` and stays hidden until its first read answers; its
 * popover, the roster and every gesture that changes where this session
 * writes, mounts beside it. A deployment without the picker
 * (`html[data-control-target="off"]`, stamped by the server) mounts neither
 * and reads nothing from the posture route; the block then shows the health
 * dot alone. Guarded: a failure here costs the block, not the rest of boot.
 */
function initControlSystemBlock() {
  try {
    if (document.documentElement.dataset.controlTarget !== 'off') {
      initControlTargetChip({ host: document.getElementById('sidebar-target') });
      initControlTargetPopover();
    }
    initSidebarTarget();
  } catch (err) {
    console.error('Failed to init the control-system block:', err);
  }
}

/* ---- New Session Button ---- */

function initNewSessionButton() {
  const btn = /** @type {HTMLButtonElement} */ (document.getElementById('new-session-btn'));
  if (!btn) return;
  // Its `+` is a placeholder svg; the dock may hold the bar off the page when
  // the sidebar fills the page's icons.
  hydrateIcons(btn);

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    try {
      await startNewSession();
    } catch (err) {
      console.error('Failed to start new session:', err);
    } finally {
      btn.disabled = false;
    }
  });
}

/* ---- Sign out ---- */

/**
 * Sign out renders in the sidebar footer as `#logout-btn`, present only when
 * the server rendered a user and a non-empty `landing_url` (multi-user
 * deployments); the same condition stamps `data-landing-url` on <html>. With
 * it, the OSPREY mark (`a.sidebar-brand`) opens the same sign-out dialog
 * instead of following its home link. Plain `osprey web` emits neither, so
 * this is a no-op there and the mark stays a plain link. The dialog and the
 * flow are sign-out.js's and logout.js's, shared with the command palette.
 *
 * Exported for testability (see app-logout.test.mjs) — the module's
 * DOMContentLoaded bootstrap never fires the wiring on its own once that
 * event has already passed, e.g. in a test environment.
 */
export function initSignOut() {
  const btn = /** @type {HTMLButtonElement|null} */ (document.getElementById('logout-btn'));
  const rowLanding = btn?.dataset.landingUrl;
  if (btn && rowLanding) {
    btn.addEventListener('click', () => confirmSignOut({ landingUrl: rowLanding, button: btn }));
  }
  const landingUrl = document.documentElement.dataset.landingUrl;
  const brand = document.querySelector('a.sidebar-brand');
  if (brand && landingUrl) {
    brand.addEventListener('click', (event) => {
      event.preventDefault();
      confirmSignOut({ landingUrl });
    });
  }
}

/* ---- The Agent tile's surface ---- */

/**
 * Hand the session to whichever surface the Agent tile now shows. The tile's ⋯
 * menu and the hand-off's cancel paths own the change itself (agent-surface.js).
 *
 * Exported for testability (see app-surface-handoff.test.mjs) — the module's
 * DOMContentLoaded bootstrap never fires on its own once that event has
 * already passed, e.g. in a test environment.
 */
export function initSurfaceFollowUps() {
  onSurfaceChange(handOffSession);
}

/**
 * The hand-offs of successive surface changes, chained.
 *
 * Both surfaces are windows onto ONE session key, and the server lets exactly
 * one of them hold it — a second acquire arriving while the first is still
 * pending is refused outright. A change is one gesture, so an operator who
 * changes twice quickly would produce exactly that. Chaining makes the second
 * change wait for the first hand-off to settle instead, which is also why the
 * teardown belongs in here: dropping the terminal ahead of the queue would
 * strand a connection a queued change to the terminal is about to open.
 *
 * The chain covers surface changes and nothing else. The overlay's own retry
 * and the session picker take the key over through terminal.js directly, so a
 * change that lands between one of those and its answer is still two acquires;
 * both paths already show the server's refusal where the operator can see it.
 *
 * @type {Promise<void>}
 */
let sessionHandoff = Promise.resolve();

/**
 * Move the live agent to the surface now showing.
 *
 * Opening the terminal resumes the key the tab is pointed at over the terminal
 * socket, which shows its own overlay while the server tears the chat down.
 * Going back to chat drops that socket first — the server reads the closed
 * connection as the terminal letting go — and the console then asks for the
 * key and replays the transcript into itself. Neither direction reloads.
 *
 * Every step answers with a promise and every one of them is awaited. A change
 * queued behind a terminal acquire that is still in flight would otherwise drop
 * the socket before the server had seen it open; and the console would ask for
 * the key before the dropped socket's close had reached the server, which
 * reads that close as the terminal letting go.
 *
 * @returns {void}
 */
function handOffSession() {
  const toTerminal = isTerminalSurface();
  sessionHandoff = sessionHandoff
    .then(async () => {
      if (toTerminal) {
        await startTerminalSurface();
        focusTerminal();
        return;
      }
      await stopTerminal();
      await enterFromTerminal();
    })
    .catch((err) => {
      console.error('Failed to hand the session over:', err);
    });
}

/**
 * Wire the sidebar footer's Docs and Feedback rows and the feedback dialog —
 * see feedback-boot.js, which owns the whole arrangement.
 *
 * Returns as soon as the button is live; the deployment's own config arrives
 * over HTTP afterwards and fills itself in. Guarded like the other
 * network-dependent inits above, so a failure costs the two footer rows and
 * nothing more.
 *
 * Both terminal dependencies are injected from here rather than imported
 * inside the dialog: app.js is already the module that knows about the
 * terminal, and keeping the dependency pointing this way is what lets the
 * dialog and its transport be tested without one. They are passed as function
 * references, not values — the session id changes as the operator switches
 * sessions, and the terminal does not exist yet when a guarded `initTerminal()`
 * has failed.
 */
function initFeedbackDialog() {
  try {
    initFeedback({
      getSessionId: getCurrentSessionId,
      getTerminal: getTerminalInstance,
    });
  } catch (err) {
    console.error('Failed to init feedback:', err);
  }
}

/* ---- Drawer Trigger Highlight ---- */

/**
 * osprey-drawer doesn't manage its `[data-drawer]` trigger's `.active` state
 * itself (a page-level nicety, not part of the component's contract — see
 * its module docstring). Web_terminal owns its own triggers, so it wires
 * this via the `drawer:open`/`drawer:close` events the component dispatches
 * (bubbling) on the host, matching any trigger for that drawer id — either
 * the component's own `[data-drawer]` marker, or `[data-drawer-trigger]`,
 * web_terminal's convention for a trigger (like the More menu's Settings
 * row) that needs its own gating logic before opening and so must never
 * match the component's delegated `[data-drawer]` handler. Either way the
 * highlight stays in sync.
 */
function initDrawerTriggerHighlight() {
  const setActive = (/** @type {boolean} */ active) => (/** @type {Event} */ event) => {
    const drawer = event.target;
    if (!(drawer instanceof HTMLElement) || !drawer.id) return;
    document
      .querySelectorAll(`[data-drawer="${drawer.id}"], [data-drawer-trigger="${drawer.id}"]`)
      .forEach((btn) => btn.classList.toggle('active', active));
  };
  document.addEventListener('drawer:open', setActive(true));
  document.addEventListener('drawer:close', setActive(false));
}

/* ---- Iframe Paste Bridge ---- */

/**
 * Route clicks on links in the chat's agent answers: a link to a panel
 * this hub hosts opens it in the workspace, any other http(s) link opens in a
 * new tab, and the hub page itself never navigates away. Whether a panel is
 * hosted is read at click time, so a panel whose URL has not resolved opens in
 * a new tab instead.
 *
 * A loopback link to a panel backend's port (the address the agent's tools
 * name, e.g. the gallery's `http://127.0.0.1:<port>#focus`) is that panel too.
 * The ports come from `/api/panels`' `loopback_ports`, read once here; until
 * they arrive, or if the read fails, such a link opens in a new tab as before.
 *
 * Exported so the wiring can be driven directly in tests: the bootstrap above
 * runs on DOMContentLoaded, which has already fired by then.
 *
 * @returns {Promise<void>} settles once the loopback ports are read (or not)
 */
export function initAgentLinks() {
  const operatorRoot = document.getElementById('operator-container');
  if (!operatorRoot) return Promise.resolve();
  /** @type {ReadonlyMap<string, string>} backend port -> panel id */
  let loopbackPorts = new Map();
  wireAgentLinks(operatorRoot, {
    prefix: window.__OSPREY_PREFIX__ || '',
    isHostedPanel: (id) => getPanelStandaloneUrl(id) !== null,
    loopbackPorts: () => loopbackPorts,
    openPanel: navigateAndActivatePanel,
    openWindow: (url, target, features) => {
      window.open(url, target, features);
    },
  });
  return fetchJSON('/api/panels').then(
    (config) => {
      /** @type {Record<string, unknown>} */
      const ports = config?.loopback_ports ?? {};
      loopbackPorts = new Map(
        Object.entries(ports)
          .filter(([, port]) => Number.isInteger(port))
          .map(([id, port]) => [String(port), id])
      );
    },
    (err) => {
      console.warn('Agent links: panel ports unavailable:', err);
    }
  );
}

/** The panel that hosts the artifact gallery. */
const ARTIFACTS_PANEL_ID = 'artifacts';

// A panel handing focus back. The old name is read until canopy-screen-runtime
// 0.2.0, whose viewers stop sending it.
const FOCUS_EXIT_TYPES = new Set(['canopy-host-focus-exit', 'osprey-focus-exit']);

/**
 * Make an artifact id in a chat agent answer show that artifact: the
 * gallery is asked to focus it through the hub's panel proxy (the request the
 * agent's `artifact_focus` makes), and the artifacts panel opens once it has.
 * Focusing first means a gallery that loads only now reads the new focus on
 * its first `GET /api/focus`.
 *
 * Exported so the wiring can be driven directly in tests: the bootstrap above
 * runs on DOMContentLoaded, which has already fired by then.
 */
export function initArtifactLinks() {
  const operatorRoot = document.getElementById('operator-container');
  if (!operatorRoot) return;
  wireArtifactLinks(operatorRoot, {
    focusArtifact: async (id) => {
      let res;
      // The focus belongs to the session the operator is on (the gallery
      // resolves the key to its transcript); with none it is set in no session.
      const sessionKey = getPointer();
      const body = sessionKey ? { artifact_id: id, session_key: sessionKey } : { artifact_id: id };
      try {
        res = await fetch(withPrefix(`/panel/${ARTIFACTS_PANEL_ID}/api/focus`), {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(body),
        });
      } catch {
        return 'unreachable';
      }
      if (!res.ok) {
        if (res.status === 404) return 'missing';
        if (res.status === 401) return 'expired';
        if (res.status === 403) return 'refused';
        return 'unreachable';
      }
      navigateAndActivatePanel(ARTIFACTS_PANEL_ID, '');
      // The session does not change; the owner remembers it as last open.
      document.dispatchEvent(new CustomEvent('osprey-artifact-opened', { detail: { id } }));
      return 'shown';
    },
  });
}

/**
 * Let an embedded panel put text where the operator is composing.
 *
 * A panel cannot type for the operator, so it posts what it has and the hub
 * decides where that lands. `insertPrompt` is the one seam that knows where
 * that is on each surface — the chat textarea, or the terminal when the tab opted
 * into it — and it appends here
 * rather than replaces: what a panel contributes (the addresses just picked,
 * say) is an ingredient of a prompt the operator may already be writing, and
 * overwriting that would destroy the only copy of it. It is never sent; the
 * operator's next Enter is still their own.
 *
 * Exported so the bridge can be driven directly in tests: the bootstrap below
 * runs on DOMContentLoaded, which has already fired by then.
 */
export function initIframePasteBridge() {
  window.addEventListener('message', (e) => {
    if (e.origin !== window.location.origin) return;
    // Accept paste-to-terminal messages from embedded iframes
    if (e.data && e.data.type === 'osprey-paste-to-terminal' && e.data.text) {
      insertPrompt(e.data.text, { append: true });
    }
    // A panel asking its host to move THIS client to another panel — the
    // sender-local twin of the panel_focus SSE path (the gallery's logbook
    // submit is the first caller). Deliberately not a server broadcast: a
    // human gesture in one browser must not move anyone else's workspace,
    // and it gets a plain activation with no agent attribution.
    //
    // The url must be root-relative and NOT protocol-relative. The origin
    // check above is necessary but not sufficient: a same-origin sender can
    // still be an agent-authored artifact rendered in a sandboxed panel, and
    // this url reaches an iframe src via buildEmbedSrc, which preserves
    // whatever scheme it is handed. `javascript:alert(1)` survives it intact
    // and would execute in the HOST origin, and `//evil.example/x` resolves
    // to a cross-origin document — so a leading-slash test alone is a hole.
    // Every real panel url is root-relative and already server-prefixed, and
    // navigateAndActivatePanel then applies the guard's member rule: the url
    // must stay under that panel's own mount, or nothing moves.
    if (
      e.data &&
      e.data.type === 'osprey:navigate' &&
      typeof e.data.panel === 'string' &&
      typeof e.data.url === 'string' &&
      e.data.url.startsWith('/') &&
      !e.data.url.startsWith('//')
    ) {
      navigateAndActivatePanel(e.data.panel, e.data.url);
    }
    // A panel handing keyboard focus back: Tab (or Shift+Tab) ran off the end
    // of its own document, and a frame cannot move focus in its parent. Only
    // a panel frame may ask — the source check is what keeps any other
    // same-origin window from steering the operator's focus.
    if (
      e.data &&
      FOCUS_EXIT_TYPES.has(e.data.type) &&
      (e.data.direction === 'forward' || e.data.direction === 'backward')
    ) {
      moveFocusOutOfPanel(e.source, e.data.direction);
    }
    // ARIEL's "Use in Workspace" control asks only to activate the already
    // registered artifact gallery in THIS browser. Unlike the legacy navigate
    // message it carries no URL; bind it to the live ARIEL iframe identity so
    // another same-origin panel cannot activate arbitrary destinations.
    if (
      e.data &&
      e.data.type === 'osprey:activate' &&
      e.data.panel === 'artifacts' &&
      panelIdForMessageSource(e.source) === 'ariel'
    ) {
      navigateAndActivatePanel('artifacts', '');
    }
  });

  // Drop zone: accept dragged artifacts onto the terminal container. Only a
  // drag its source lets be copied is the terminal's; a dock drag (a header
  // tab, a tile or a sidebar row) allows only a move and belongs to the tile
  // beneath — pasting its key into the shell would be the wrong answer.
  const termContainer = document.getElementById('terminal-container');
  if (termContainer) {
    const COPYABLE = new Set(['copy', 'copyLink', 'copyMove', 'all', 'uninitialized']);
    /** @param {DragEvent} e */
    const isCopyDrag = (e) => !!e.dataTransfer && COPYABLE.has(e.dataTransfer.effectAllowed);
    termContainer.addEventListener('dragover', (e) => {
      if (!isCopyDrag(e)) return;
      e.preventDefault();
      /** @type {DataTransfer} */ (e.dataTransfer).dropEffect = 'copy';
    });
    termContainer.addEventListener('drop', (e) => {
      if (!isCopyDrag(e)) return;
      e.preventDefault();
      const text = /** @type {DataTransfer} */ (e.dataTransfer).getData('text/plain');
      if (text) {
        pasteToTerminal(text);
        focusTerminal();
      }
    });
  }
}

/** What the browser would stop on with Tab, before visibility is checked. */
const TABBABLE =
  'a[href], area[href], button, input, select, textarea, iframe, summary, ' +
  '[tabindex], [contenteditable]:not([contenteditable="false"])';

/**
 * Whether Tab could land on `el`: focusable, not taken out of the order, not
 * disabled, and neither it nor an ancestor hidden or inert.
 * @param {Element} el
 */
function isTabbable(el) {
  const h = /** @type {HTMLElement} */ (el);
  if (h.tabIndex < 0) return false;
  if (/** @type {HTMLInputElement} */ (el).disabled) return false;
  if (el instanceof HTMLInputElement && el.type === 'hidden') return false;
  if (el.closest('[hidden], [inert], fieldset[disabled]')) return false;
  for (let n = /** @type {Element | null} */ (el); n; n = n.parentElement) {
    const cs = getComputedStyle(n);
    if (cs.display === 'none') return false;
    if (n === el && cs.visibility === 'hidden') return false;
  }
  return true;
}

/**
 * Move focus to the host element before (`backward`) or after (`forward`) the
 * panel whose frame is `source`, in host document order relative to the
 * panel's place in the dock (see `panelFocusAnchor`). With nothing on that side
 * focus stays where it is, which is where the panel's own sentinel left it.
 *
 * Exported so it can be driven directly in tests.
 * @param {MessageEventSource | null} source
 * @param {'forward' | 'backward'} direction
 * @returns {boolean} whether focus moved
 */
export function moveFocusOutOfPanel(source, direction) {
  const found = panelFocusAnchor(source);
  if (!found) return false;
  const { anchor, skip } = found;
  const candidates = [...document.querySelectorAll(TABBABLE)].filter(
    (el) => !anchor.contains(el) && !skip(el) && isTabbable(el)
  );
  const FOLLOWING = Node.DOCUMENT_POSITION_FOLLOWING;
  const PRECEDING = Node.DOCUMENT_POSITION_PRECEDING;
  const before = candidates.filter((el) => anchor.compareDocumentPosition(el) & PRECEDING);
  const after = candidates.filter((el) => anchor.compareDocumentPosition(el) & FOLLOWING);
  const target = direction === 'backward' ? before[before.length - 1] : after[0];
  if (!target) return false;
  /** @type {HTMLElement} */ (target).focus();
  return true;
}

/**
 * Identify a live embedded panel by its window object, never by a value the
 * sender supplied in postMessage data.
 * @param {MessageEventSource | null} source
 * @returns {string|null}
 */
function panelIdForMessageSource(source) {
  if (!source) return null;
  for (const el of queryAllDocuments('iframe.panel-iframe')) {
    const iframe = /** @type {HTMLIFrameElement} */ (el);
    if (iframe.contentWindow === source && iframe.dataset.panelId) return iframe.dataset.panelId;
  }
  return null;
}

/* ---- Keyboard Shortcuts ---- */

/** The Settings row's trigger contract (settings-warning-gate.js binds it). */
const SETTINGS_TRIGGER = '[data-drawer-trigger="settings-drawer"]';

/**
 * Cmd+, (macOS) / Ctrl+, opens or closes the settings drawer through the same click the More menu's Settings row takes,
 * so the warning gate runs exactly as it does there. Cmd/Ctrl+\ (the
 * sidebar's width) and Cmd/Ctrl+K (the palette) are bound by their own
 * modules.
 */
function initKeyboardShortcuts() {
  const mac = isMacPlatform();
  document.addEventListener('keydown', (e) => {
    if (e.key !== ',' || !(mac ? e.metaKey : e.ctrlKey) || e.altKey || e.shiftKey) return;
    const trigger = /** @type {HTMLElement | null} */ (document.querySelector(SETTINGS_TRIGGER));
    if (!trigger) return;
    e.preventDefault();
    trigger.click();
  });
}
