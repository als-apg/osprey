/* OSPREY Web Terminal — Application Entry Point */

import {
  initTerminal,
  focusTerminal,
  pasteToTerminal,
  getCurrentSessionId,
  getTerminalInstance,
  startExpert,
  stopTerminal,
} from './terminal.js';
import { logout } from './logout.js';
import {
  initPanelManager,
  broadcastMode,
  handleUiModeFlip,
  navigateAndActivatePanel,
  getPanelStandaloneUrl,
} from './panel-manager.js';
import { wireAgentLinks } from './agent-links.js';
import { wireArtifactLinks } from './artifact-links.js';
import { fetchJSON, withPrefix } from './api.js';
import '/design-system/js/components/osprey-drawer.js';
import { initSettings } from './settings.js';
import { initMemoryGallery } from './memory-gallery.js';
import { initScaffoldGallery } from './scaffold-gallery.js';
import { initHookDebug } from './hook-debug.js';
import { initSessionSelector, startNewSession } from './sessions.js';
import { initCommandPalette } from './palette-boot.js';
import { initMoreMenu } from './more-menu.js';
import { initTheme } from '/design-system/js/theme-manager.js';
import { onModeChange } from '/design-system/js/frame-params.js';
import { isMacPlatform } from '/design-system/js/platform.js';
import { initChat, enterFromExpert, renderChatBootFailure } from './chat.js';
import { initDockWorkspace, applyDockMode } from './dock-workspace.js';
import { initHeaderContrib } from './tile-header-contrib.js';
import { initControlTargetChip } from './control-target-chip.js';
import { insertPrompt } from './first-contact.js';
import { initControlTargetPopover } from './control-target-popover.js';
import { initFeedback } from './feedback-boot.js';
import { initSidebar } from './sidebar.js';
import { initSidebarTarget } from './sidebar-target.js';
import { initSidebarArtifacts } from './sidebar-artifacts.js';

document.addEventListener('DOMContentLoaded', () => {
  initTheme({ role: 'hub' });
  // Guarded: xterm.js loads from a CDN by default (local only in
  // OSPREY_OFFLINE), so a network blip must degrade the terminal card, not
  // kill the whole boot.
  try {
    initTerminal('terminal-container');
    // The page opens straight onto the prompt, and nothing else claims focus
    // at boot — so boot hands it to the terminal.
    focusTerminal();
  } catch (err) {
    console.error('Failed to init terminal:', err);
  }
  // Simple-mode operator chat. Guarded so a chat init failure can't break the
  // rest of the boot (the expert terminal is already up at this point).
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
  // The sidebar shell — width toggle (Cmd/Ctrl+\), tooltips, list disclosure,
  // the Agent row — before the panel manager renders its rows into it.
  initSidebar();
  // Before the panel manager creates any iframe, so no contribution can
  // arrive without a listener.
  initHeaderContrib();
  initPanelManager('right-panel');
  initSessionSelector('session-selector');
  // Dock the terminal + workspace panels into the dockview shell (replaces the
  // old fixed resize-handle split). Guarded so a dock init failure can't break
  // the rest of boot — the source subtrees stay in the page if it no-ops.
  try {
    initDockWorkspace();
  } catch (err) {
    console.error('Failed to init dock workspace:', err);
  }
  // The sidebar's Artifacts and Script runs lists. Their rows are page-owned,
  // so they exist already; the lists follow every later sidebar render
  // through `osprey-sidebar-rendered`.
  try {
    initSidebarArtifacts();
  } catch (err) {
    console.error('Failed to init the sidebar artifact lists:', err);
  }
  initKeyboardShortcuts();
  initCommandPalette();
  try {
    initMoreMenu({ getSessionId: getCurrentSessionId });
  } catch (err) {
    console.error('Failed to init the More menu:', err);
  }
  initNewSessionButton();
  initLogoutButton();
  initUiModeFollowUps();
  initControlSystemBlock();
  initFeedbackDialog();
  initDrawerTriggerHighlight();
  initSettings();
  initMemoryGallery();
  initScaffoldGallery();
  initHookDebug();
  // Listen for paste requests from embedded iframes (gallery, ARIEL)
  initIframePasteBridge();
});

/* ---- Control-system block ---- */

/**
 * The sidebar's control-system block. The chip — which machine this session
 * stands on, and whether a write there would land — mounts into
 * `#sidebar-target` and stays hidden until its first read answers; its
 * popover, the roster and every gesture that changes where this session
 * writes, mounts beside it. A deployment without the picker
 * (`html[data-control-target="off"]`, stamped by the server) mounts neither
 * and reads nothing from the posture route; the block then shows the health
 * dot and the facility clock alone. Guarded: a failure here costs the block,
 * not the rest of boot.
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

/* ---- Logout Button ---- */

/**
 * Sign out renders in the More menu as `#logout-btn`, present only when the
 * server rendered a user and a non-empty `landing_url` (multi-user
 * deployments). Plain `osprey web` never emits it, so this is a no-op there.
 * The flow itself is logout.js's, shared with the command palette, which
 * offers Log out with no button on the page.
 *
 * Exported for testability (see app-logout.test.mjs) — the module's
 * DOMContentLoaded bootstrap never fires the button wiring on its own once
 * that event has already passed, e.g. in a test environment.
 */
export function initLogoutButton() {
  const btn = /** @type {HTMLButtonElement|null} */ (document.getElementById('logout-btn'));
  const landingUrl = btn?.dataset.landingUrl;
  if (!btn || !landingUrl) return;
  btn.addEventListener('click', () => logout(landingUrl, btn));
}

/* ---- UI Mode Toggle (Expert / Simple) ---- */

/**
 * The hub-only half of an Expert/Simple flip. The View rows of the More menu
 * (more-menu.js) and the command palette own the pick itself — through
 * frame-params.js's `pickUiMode()`, which persists the choice, drops a
 * leftover one-shot ?mode=, and posts the same-origin `osprey-mode-change`
 * message to this window — and frame-params.js's shared receive side stamps
 * html[data-ui-mode] before handing the mode here. What follows is what only
 * the hub has to do with it: broadcast to the panels, the dock and panel
 * follow-ups, then hand the session to the view the operator flipped to.
 * mode-boot.js already resolved the initial mode pre-paint, so this only ever
 * handles the runtime flip.
 *
 * Exported for testability (see app-mode-flip.test.mjs) — the module's
 * DOMContentLoaded bootstrap never fires on its own once that event has
 * already passed, e.g. in a test environment.
 */
export function initUiModeFollowUps() {
  onModeChange((mode) => {
    // Panels read the current mode off <html>, so broadcast only after the
    // swap (onModeChange stamped it before calling back).
    broadcastMode();
    // Dock half of the flip: stash the outgoing view's arrangement, then
    // restore the target view's own (or build its default). Runs after the
    // CSS/attribute swap so the dock reads the target mode; no-ops until the
    // workspace shell exists.
    applyDockMode(mode);
    // Panel half: a flip to expert ends the simple-UX chat-only suppression
    // and lets the default panel claim a still-empty workspace slot. Runs
    // after applyDockMode so the activation docks into the target layout.
    handleUiModeFlip(mode);
    // Session half, last: the target view's layout is on screen by now, so
    // whichever surface has to show a transitional state is the one the
    // operator is looking at.
    handOffSession(mode);
  });
}

/**
 * The hand-offs of successive flips, chained.
 *
 * Both views are windows onto ONE session key, and the server lets exactly one
 * of them hold it — a second acquire arriving while the first is still pending
 * is refused outright. A flip is one gesture, so an operator who flips twice
 * quickly would produce exactly that. Chaining makes the second flip wait for
 * the first hand-off to settle instead, which is also why the teardown belongs
 * in here: dropping the terminal ahead of the queue would strand a connection
 * a queued flip to Expert is about to open.
 *
 * The chain covers flips and nothing else. The overlay's own retry and the
 * session picker take the key over through terminal.js directly, so a flip
 * that lands between one of those and its answer is still two acquires; both
 * paths already show the server's refusal where the operator can see it.
 *
 * @type {Promise<void>}
 */
let sessionHandoff = Promise.resolve();

/**
 * Move the live agent to *mode*'s view.
 *
 * Flipping to Expert resumes the key the tab is pointed at over the terminal
 * socket, which shows its own overlay while the server tears the chat down.
 * Flipping to Simple drops that socket first — the server reads the closed
 * connection as the Expert view letting go — and the console then asks for the
 * key and replays the transcript into itself. Neither direction reloads.
 *
 * Every step answers with a promise and every one of them is awaited. A flip
 * queued behind an Expert acquire that is still in flight would otherwise drop
 * the socket before the server had seen it open; and the console would ask for
 * the key before the dropped socket's close had reached the server, which
 * reads that close as the Expert view letting go.
 *
 * @param {'expert'|'simple'} mode
 * @returns {void}
 */
function handOffSession(mode) {
  sessionHandoff = sessionHandoff
    .then(async () => {
      if (mode === 'expert') return startExpert();
      await stopTerminal();
      return enterFromExpert();
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
 * Route clicks on links in the Simple view's agent answers: a link to a panel
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

/**
 * Make an artifact id in a Simple-view agent answer show that artifact: the
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
      try {
        res = await fetch(withPrefix(`/panel/${ARTIFACTS_PANEL_ID}/api/focus`), {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ artifact_id: id }),
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
      return 'shown';
    },
  });
}

/**
 * Let an embedded panel put text where the operator is composing.
 *
 * A panel cannot type for the operator, so it posts what it has and the hub
 * decides where that lands. `insertPrompt` is the one seam that knows where
 * that is in each view — the terminal in expert, the chat textarea in Simple,
 * which the terminal branch alone would never reach — and it appends here
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
    // Every real panel url is root-relative and already server-prefixed.
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

  // Drop zone: accept dragged artifacts onto the terminal container
  const termContainer = document.getElementById('terminal-container');
  if (termContainer) {
    termContainer.addEventListener('dragover', (e) => {
      e.preventDefault();
      /** @type {DataTransfer} */ (e.dataTransfer).dropEffect = 'copy';
    });
    termContainer.addEventListener('drop', (e) => {
      e.preventDefault();
      const text = /** @type {DataTransfer} */ (e.dataTransfer).getData('text/plain');
      if (text) {
        pasteToTerminal(text);
        focusTerminal();
      }
    });
  }
}

/**
 * Identify a live embedded panel by its window object, never by a value the
 * sender supplied in postMessage data.
 * @param {MessageEventSource | null} source
 * @returns {string|null}
 */
function panelIdForMessageSource(source) {
  if (!source) return null;
  for (const el of document.querySelectorAll('iframe.panel-iframe')) {
    const iframe = /** @type {HTMLIFrameElement} */ (el);
    if (iframe.contentWindow === source && iframe.dataset.panelId) return iframe.dataset.panelId;
  }
  return null;
}

/* ---- Keyboard Shortcuts ---- */

/** The Settings row's trigger contract (settings-warning-gate.js binds it). */
const SETTINGS_TRIGGER = '[data-drawer-trigger="settings-drawer"]';

/**
 * Ctrl+` focuses the terminal; Cmd+, (macOS) / Ctrl+, opens or closes the
 * settings drawer through the same click the More menu's Settings row takes,
 * so the warning gate runs exactly as it does there. Cmd/Ctrl+\ (the
 * sidebar's width) and Cmd/Ctrl+K (the palette) are bound by their own
 * modules.
 */
function initKeyboardShortcuts() {
  const mac = isMacPlatform();
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === '`') {
      e.preventDefault();
      focusTerminal();
      return;
    }
    if (e.key !== ',' || !(mac ? e.metaKey : e.ctrlKey) || e.altKey || e.shiftKey) return;
    const trigger = /** @type {HTMLElement | null} */ (document.querySelector(SETTINGS_TRIGGER));
    if (!trigger) return;
    e.preventDefault();
    trigger.click();
  });
}
