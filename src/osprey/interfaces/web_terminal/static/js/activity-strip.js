// @ts-check
/* OSPREY Web Terminal — Activity Strip
 *
 * A slim, fixed zone in the host chrome showing the most recent agent action
 * as text ("agent wrote SR01:HCM1:SP"). Frames arrive through panel-manager's
 * setActivityStripHandler seam: every agent_activity frame the rail cannot
 * anchor (kinds 'channel'/'run'/'artifact', plus 'panel' frames whose id has
 * no rail entry) lands here verbatim.
 *
 * Model: a single visible entry, latest wins. Any newer frame — same target
 * or different — replaces the entry and resets the auto-clear timer, so rapid
 * event bursts coalesce into whatever happened last. The entry clears itself
 * after ACTIVITY_CLEAR_MS.
 *
 * Suppression: when the panel that self-signals this activity is already the
 * active panel, the strip stays silent (the gallery navigates on focus SSE,
 * the bluesky panel shows its launched banner, and the rail glow has already
 * fired in panel-manager). The kind→panel mapping is the SUPPRESSION table
 * below; suppression itself is a pure function (exported for tests).
 *
 * Three modules, one feature: this one owns the live line, activity-format.js
 * words every frame for both surfaces, and activity-history.js is the
 * click-to-expand popover over the server's ring (the strip mount is its
 * trigger, so the strip builds it and hands out its open/close).
 *
 * All agent-supplied strings (tool, detail, panel) are rendered as text nodes
 * only — createElement + textContent, never innerHTML.
 */

import { setActivityStripHandler, getActivePanel, labelOf } from './panel-manager.js';
import { ARRANGE_TOOL, formatActivity } from './activity-format.js';
import { createActivityHistory } from './activity-history.js';

/** @typedef {import('./panel-manager.js').AgentActivityEvent} AgentActivityFrame */
/** @typedef {AgentActivityFrame['target']} ActivityTarget */

/** How long an entry stays visible before auto-clearing (ms). */
export const ACTIVITY_CLEAR_MS = 6000;

// ---- Pure helpers (exported for tests) ----

/**
 * SUPPRESSION table: the panel id that self-signals a given activity kind.
 *
 *   artifact → 'artifacts'  (workspace gallery navigates on the focus SSE)
 *   run      → 'bluesky'    (the canonical scan panel shows its launched
 *                            banner in its Plans view; a config-renamed scan
 *                            panel falls outside this table and still gets
 *                            strip entries)
 *   panel    → the frame's own target.panel (that panel is the signal)
 *   channel  → none (channel writes have no self-signaling panel)
 *
 * @param {ActivityTarget} target
 * @returns {string | null} the panel id whose being active suppresses the
 *   entry, or null when the kind is never suppressed
 */
export function suppressionPanelFor(target) {
  switch (target.kind) {
    case 'artifact': return 'artifacts';
    case 'run': return 'bluesky';
    case 'panel': return target.panel ?? null;
    default: return null; // 'channel' — never suppressed
  }
}

/**
 * Whether a frame for `target` should be suppressed while `activePanel` is
 * the surfaced panel. Pure: feeds on getActivePanel() at call time.
 * @param {ActivityTarget} target
 * @param {string | null} activePanel
 * @returns {boolean}
 */
export function isSuppressed(target, activePanel) {
  const mapped = suppressionPanelFor(target);
  return mapped != null && mapped === activePanel;
}

// ---- Strip factory ----

/**
 * Build a strip bound to a mount element. Dependencies are injected so tests
 * drive it directly (frames via handleActivity, active panel via a stub,
 * history via a stub reader).
 * @param {{
 *   mount: HTMLElement,
 *   getActivePanel: () => string | null,
 *   clearMs?: number,
 *   fetchRecent?: (limit: number) => Promise<AgentActivityFrame[]>,
 *   labelOf?: (id: string) => string,
 * }} deps
 * @returns {{
 *   handleActivity: (frame: AgentActivityFrame) => void,
 *   clear: () => void,
 *   openHistory: () => Promise<void>,
 *   closeHistory: () => void,
 *   isHistoryOpen: () => boolean,
 * }}
 */
export function createActivityStrip({
  mount,
  getActivePanel,
  clearMs = ACTIVITY_CLEAR_MS,
  fetchRecent,
  labelOf: panelLabel,
}) {
  /** @type {ReturnType<typeof setTimeout> | null} */
  let timer = null;
  /** Arrange frames shown back-to-back in the current live window (see below). */
  let arrangeRun = 0;

  const history = createActivityHistory({ mount, fetchRecent, labelOf: panelLabel });

  function clear() {
    if (timer != null) { clearTimeout(timer); timer = null; }
    arrangeRun = 0;
    mount.textContent = '';
  }

  /** @param {AgentActivityFrame} frame */
  function handleActivity(frame) {
    const target = frame?.target;
    if (!target) return; // malformed frame — ignore

    // History records what the agent did, not what the strip chose to show,
    // so the live prepend happens BEFORE the suppression check — the server
    // ring keeps suppressed frames too, and the two must not disagree.
    if (history.isOpen()) history.prepend(frame);

    if (isSuppressed(target, getActivePanel())) return;

    // Arranging a workspace lands one frame per tile, and the single slot would
    // otherwise flicker through them. Same idiom as the latest-wins replacement
    // below — the run collapses into one line — except that consecutive arrange
    // frames count up instead of overwriting. Anything else ends the run.
    arrangeRun = frame.tool === ARRANGE_TOOL ? arrangeRun + 1 : 0;

    const { verb, subject } = formatActivity(frame, { labelOf: panelLabel, count: arrangeRun });

    // Text nodes only — agent-supplied strings must never reach innerHTML.
    const entry = document.createElement('span');
    entry.className = 'activity-strip-entry';
    const verbEl = document.createElement('span');
    verbEl.className = 'activity-strip-verb';
    verbEl.textContent = verb;
    entry.appendChild(verbEl);
    if (subject) {
      const subjectEl = document.createElement('span');
      subjectEl.className = 'activity-strip-subject';
      subjectEl.textContent = subject;
      entry.appendChild(subjectEl);
    }

    // Single slot, latest wins: replace whatever is showing and restart the clock.
    mount.textContent = '';
    mount.appendChild(entry);
    if (timer != null) clearTimeout(timer);
    timer = setTimeout(clear, clearMs);
  }

  return {
    handleActivity,
    clear,
    openHistory: history.open,
    closeHistory: history.close,
    isHistoryOpen: history.isOpen,
  };
}

// ---- Self-boot ----
//
// Wired like the other self-registering module scripts the templates load
// (cf. osprey-theme-switcher): the template provides the #activity-strip
// mount and this module registers itself on panel-manager's seam. Pages
// without the mount (or without a running panel-manager) no-op harmlessly.

/** @type {ReturnType<typeof createActivityStrip> | null} */
let bootedStrip = null;

/**
 * Boot the page's one strip on the template's #activity-strip mount and
 * register it on panel-manager's seam.
 *
 * Idempotent, and that is load-bearing: the session page (session.js) drives
 * the strip from its own SSE subscription because no panel-manager runs
 * there, so it calls this to reach the same instance the module's own boot
 * creates. A second strip on the shared mount would bind a second set of
 * click handlers and open a second history popover.
 *
 * @returns {ReturnType<typeof createActivityStrip> | null} null on a page with no mount
 */
export function bootActivityStrip() {
  if (bootedStrip) return bootedStrip;
  const mount = document.getElementById('activity-strip');
  if (!mount) return null;
  bootedStrip = createActivityStrip({ mount, getActivePanel, labelOf });
  setActivityStripHandler(bootedStrip.handleActivity);
  return bootedStrip;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootActivityStrip, { once: true });
} else {
  bootActivityStrip();
}
