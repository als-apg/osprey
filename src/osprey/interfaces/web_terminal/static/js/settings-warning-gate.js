// @ts-check
/* OSPREY Web Terminal — Settings Drawer Warning Gate */

import { fetchJSON } from './api.js';
import { mountOverlay, fadeOutOverlay } from './modal-overlay.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';

// Per-session warning for the settings drawer (resets on server restart), and
// per RENDER: localStorage is origin-scoped, so on a multi-user mount a bare
// key would let whoever acknowledged the safety warning first wave it away for
// everyone else on the box. Resolved through scopedStorageKey() at each use —
// see storage-scope.js.
const SETTINGS_WARNING_KEY_BASE = 'osprey-settings-warning-ack';

// True from the moment a trigger click starts the gate (health check in
// flight, or the warning dialog itself is up) until it resolves one way or
// another. Guards against a rapid second click spawning a second dialog.
let warningGatePending = false;

/**
 * Gate opening the settings drawer behind a first-time-per-session warning
 * (these settings control agent behavior, safety hooks, and security
 * policies). The trigger is the ☰ More menu's Settings row (hub chrome's
 * only route into the drawer, present in both views). It intentionally
 * carries `data-drawer-trigger` rather than osprey-drawer's own
 * `[data-drawer]` marker, so the component's
 * delegated handler never matches it and never toggles the drawer directly —
 * this gate is the sole open path, and the click reaches every other
 * document-level listener (e.g. sessions.js's outside-click dropdown close)
 * completely unmodified, with no propagation tampering.
 *
 * @param {import('./settings.js').DrawerElement} drawer  the drawer the trigger opens
 */
export function initSettingsWarningGate(drawer) {
  const trigger = document.querySelector('[data-drawer-trigger="settings-drawer"]');
  if (!trigger) return;

  trigger.addEventListener('click', async () => {
    // Toggle-close stays FIRST, ahead of the gate: an already-open drawer
    // closes without re-running the warning. The gate never owns the toggle.
    if (drawer.hasAttribute('open')) {
      drawer.close();
      return;
    }
    if (await runWarningGate()) drawer.open();
  });
}

// Bounds how long a trigger click can leave `warningGatePending` true while
// waiting on /health -- a stalled backend or a dropped connection with no
// RST would otherwise never settle the fetch and permanently brick the
// trigger (every later click a no-op, only a reload recovering).
export const HEALTH_CHECK_TIMEOUT_MS = 4000;

/**
 * Read the server session from /health and whether its warning was already
 * acknowledged here. An unreachable or slow /health (past
 * HEALTH_CHECK_TIMEOUT_MS) reads as no session and no acknowledgment, so the
 * warning shows: the gate fails safe to the warning. Never rejects.
 * @returns {Promise<{ serverSession: string|null, acknowledged: boolean }>}
 */
async function readAcknowledgment() {
  let timeoutId;
  try {
    const timeout = new Promise((_, reject) => {
      timeoutId = setTimeout(
        () => reject(new Error('health check timed out')),
        HEALTH_CHECK_TIMEOUT_MS
      );
    });
    const healthCheck = fetchJSON('/health');
    // If the timeout wins the race, the still-pending fetch may reject later;
    // swallow that late rejection so it doesn't surface as an unhandled one.
    healthCheck.catch(() => {});
    const health = await Promise.race([healthCheck, timeout]);
    /** @type {string|null} */
    const serverSession = health.session_id ?? null;
    const acknowledged =
      serverSession !== null &&
      localStorage.getItem(scopedStorageKey(SETTINGS_WARNING_KEY_BASE)) === serverSession;
    return { serverSession, acknowledged };
  } catch {
    return { serverSession: null, acknowledged: false };
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Encapsulate ONLY the gate decision — the re-entrancy guard, the /health
 * check, and the first-per-session warning dialog. It never opens or closes
 * the drawer; callers do that on a true result. This is the sole gate path;
 * every gated seam (the menu trigger, openDrawerTab, revealSetting) routes
 * through it, and it alone sets and clears `warningGatePending`.
 *
 * Resolves true when the user may proceed: acknowledgment for this server
 * session is already stored, OR the user clicks Proceed. Resolves false on
 * Cancel, Escape, dialog-build failure, or a re-entrant call made while a
 * gate is already pending (the second caller does not spawn a second dialog).
 * @returns {Promise<boolean>}
 */
export async function runWarningGate() {
  if (warningGatePending) return false; // a check or the dialog is already in flight
  warningGatePending = true;
  try {
    const { serverSession, acknowledged } = await readAcknowledgment();
    if (acknowledged) return true;
    // Defense-in-depth: a dialog that cannot be built or shown declines the
    // gate rather than leaving it permanently pending.
    return await showSettingsWarning(serverSession);
  } catch (error) {
    console.error(
      'osprey web_terminal: failed to show the settings warning dialog; gate reset',
      error
    );
    return false;
  } finally {
    warningGatePending = false;
  }
}

/**
 * Build the first-time warning dialog. Persists acknowledgment per server
 * session so it reappears on restart. Resolves true if the user clicks
 * Proceed, false on Cancel or Escape; each dismissal path settles this
 * promise, and the caller decides what follows.
 * @param {string|null} serverSession
 * @returns {Promise<boolean>}
 */
function showSettingsWarning(serverSession) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'settings-warning-overlay';

    const dialog = document.createElement('div');
    dialog.className = 'settings-warning-dialog';

    dialog.innerHTML = `
      <div class="settings-warning-icon">⚠</div>
      <div class="settings-warning-title">Advanced Configuration Area</div>
      <div class="settings-warning-body">
        <p>These settings directly control <strong>agent behavior</strong>,
        <strong>safety hooks</strong>, and <strong>security policies</strong>.</p>
        <p>Incorrect changes can <strong>disable safety checks</strong>,
        bypass human approval requirements, or allow unvalidated writes
        to control system hardware.</p>
        <p>Only modify these settings if you understand the safety
        implications of each option.</p>
      </div>
      <div class="settings-warning-actions">
        <button class="settings-warning-cancel">Cancel</button>
        <button class="settings-warning-proceed">I Understand, Proceed</button>
      </div>
    `;

    overlay.appendChild(dialog);
    mountOverlay(overlay);

    // Unconditional on every dismissal path (Cancel, Proceed, Escape) — leaves
    // no stale `keydown` listener behind.
    const cleanup = () => {
      document.removeEventListener('keydown', onKey);
      fadeOutOverlay(overlay);
    };

    // Cancel/Escape decline the gate; Proceed acknowledges + accepts. The gate
    // decision is all this resolves — the caller owns opening the drawer.
    const cancel = () => {
      cleanup();
      resolve(false);
    };

    /** @type {HTMLElement} */ (dialog.querySelector('.settings-warning-cancel')).addEventListener(
      'click',
      cancel
    );

    /** @type {HTMLElement} */ (dialog.querySelector('.settings-warning-proceed')).addEventListener(
      'click',
      () => {
        if (serverSession) {
          localStorage.setItem(scopedStorageKey(SETTINGS_WARNING_KEY_BASE), serverSession);
        }
        cleanup();
        resolve(true);
      }
    );

    // Escape key cancels
    /** @param {KeyboardEvent} e */
    const onKey = (e) => {
      if (e.key === 'Escape') cancel();
    };
    document.addEventListener('keydown', onKey);
  });
}
