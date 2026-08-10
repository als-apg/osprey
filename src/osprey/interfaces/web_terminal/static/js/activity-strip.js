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
 * the plan panel shows its launched banner, and the rail glow has already
 * fired in panel-manager). The kind→panel mapping is the SUPPRESSION table
 * below; suppression itself is a pure function (exported for tests).
 *
 * All agent-supplied strings (tool, detail, panel) are rendered as text nodes
 * only — createElement + textContent, never innerHTML.
 */

import { setActivityStripHandler, getActivePanel } from './panel-manager.js';

/** @typedef {import('./panel-manager.js').AgentActivityEvent} AgentActivityFrame */
/** @typedef {AgentActivityFrame['target']} ActivityTarget */

/** How long an entry stays visible before auto-clearing (ms). */
export const ACTIVITY_CLEAR_MS = 6000;

// ---- Pure helpers (exported for tests) ----

/**
 * SUPPRESSION table: the panel id that self-signals a given activity kind.
 *
 *   artifact → 'artifacts'  (workspace gallery navigates on the focus SSE)
 *   run      → 'plan'       (the canonical plans panel shows its launched
 *                            banner; a config-renamed plans panel falls
 *                            outside this table and still gets strip entries)
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
    case 'run': return 'plan';
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

/**
 * Human-readable two-part label for a frame. The subject carries the
 * agent-supplied string and is rendered as a text node by the caller.
 * @param {AgentActivityFrame} frame
 * @returns {{ verb: string, subject: string }}
 */
export function formatActivity(frame) {
  const t = frame.target;
  switch (t.kind) {
    case 'channel':
      return { verb: 'agent wrote', subject: t.detail || frame.tool };
    case 'run':
      return t.detail
        ? { verb: 'agent launched run', subject: t.detail }
        : { verb: 'agent launched a run', subject: '' };
    case 'artifact':
      return { verb: 'agent focused', subject: t.detail || 'an artifact' };
    case 'panel':
      // Generic fallback: a panel-kind frame whose id had no rail entry.
      return { verb: 'agent touched', subject: t.panel || frame.tool };
    default:
      // Unknown future kind from a newer server — still show something.
      return { verb: 'agent activity', subject: frame.tool };
  }
}

// ---- Strip factory ----

/**
 * Build a strip bound to a mount element. Dependencies are injected so tests
 * drive it directly (frames via handleActivity, active panel via a stub).
 * @param {{
 *   mount: HTMLElement,
 *   getActivePanel: () => string | null,
 *   clearMs?: number,
 * }} deps
 * @returns {{ handleActivity: (frame: AgentActivityFrame) => void, clear: () => void }}
 */
export function createActivityStrip({ mount, getActivePanel, clearMs = ACTIVITY_CLEAR_MS }) {
  /** @type {ReturnType<typeof setTimeout> | null} */
  let timer = null;

  function clear() {
    if (timer != null) { clearTimeout(timer); timer = null; }
    mount.textContent = '';
  }

  /** @param {AgentActivityFrame} frame */
  function handleActivity(frame) {
    const target = frame?.target;
    if (!target) return; // malformed frame — ignore
    if (isSuppressed(target, getActivePanel())) return;

    const { verb, subject } = formatActivity(frame);

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

  return { handleActivity, clear };
}

// ---- Self-boot ----
//
// Wired like the other self-registering module scripts the templates load
// (cf. osprey-theme-switcher): the template provides the #activity-strip
// mount and this module registers itself on panel-manager's seam. Pages
// without the mount (or without a running panel-manager) no-op harmlessly.

function boot() {
  const mount = document.getElementById('activity-strip');
  if (!mount) return;
  const strip = createActivityStrip({ mount, getActivePanel });
  setActivityStripHandler(strip.handleActivity);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}
