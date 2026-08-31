// @ts-check
/* OSPREY Web Terminal — One-Time Rail Hint
 *
 * A small callout beside the rail for operators arriving from the
 * pre-redesign UI: "Panel tabs now live here", with a one-click "Move to
 * top" escape hatch to the pre-redesign arrangement. Shown at most once EVER
 * (boolean localStorage flag, deliberately not keyed to the server session:
 * the hint is about a one-time transition, not a per-deployment notice).
 *
 * Self-registering module, wired like activity-strip.js: the template loads
 * it, it finds its own anchor (.panel-rail-region) and no-ops harmlessly on
 * pages without one.
 *
 * "Once EVER" means once per PERSONA. localStorage is origin-scoped, so on a
 * multi-user mount a bare flag would be one shared slot and the first operator
 * to dismiss the hint would silently swallow it for everyone else arriving
 * from the old UI — see storage-scope.js for the rule and why a scoped read
 * never falls back to the bare key.
 */

import { getRailPosition, setRailPosition } from './rail-position.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';

const FLAG_BASE = 'osprey-rail-hint-dismissed-v1';

/**
 * Show the hint if this browser has never seen it and the rail is still the
 * (new) left column. Exported for tests; the self-boot below is the runtime
 * caller.
 * @returns {boolean} whether the hint was shown
 */
export function maybeShowRailHint() {
  let seen = null;
  try {
    seen = localStorage.getItem(scopedStorageKey(FLAG_BASE));
  } catch {
    // Storage unavailable: never show rather than nag on every load.
    return false;
  }
  if (seen || getRailPosition() === 'top' || document.body.classList.contains('embedded')) {
    return false;
  }
  const region = document.querySelector('.panel-rail-region');
  if (!region) return false;

  const hint = document.createElement('div');
  hint.className = 'rail-hint';
  hint.setAttribute('role', 'note');

  const text = document.createElement('div');
  text.className = 'rail-hint-text';
  text.textContent =
    'Panel tabs now live here. Prefer them along the top, like before? '
    + 'You can switch any time from the ＋ menu.';
  hint.appendChild(text);

  const dismissHint = () => {
    try {
      localStorage.setItem(scopedStorageKey(FLAG_BASE), '1');
    } catch { /* storage blocked — the hint may reappear next load, harmless */ }
    hint.remove();
  };

  const actions = document.createElement('div');
  actions.className = 'rail-hint-actions';
  const moveBtn = document.createElement('button');
  moveBtn.type = 'button';
  moveBtn.className = 'rail-hint-move';
  moveBtn.textContent = 'Move to top';
  moveBtn.addEventListener('click', () => {
    setRailPosition('top');
    dismissHint();
  });
  const dismissBtn = document.createElement('button');
  dismissBtn.type = 'button';
  dismissBtn.className = 'rail-hint-dismiss';
  dismissBtn.textContent = 'Got it';
  dismissBtn.addEventListener('click', dismissHint);
  actions.appendChild(moveBtn);
  actions.appendChild(dismissBtn);
  hint.appendChild(actions);

  region.appendChild(hint);
  return true;
}

/* ---- Self-boot ---- */

function boot() {
  maybeShowRailHint();
}

// Skip the self-boot under vitest — tests drive maybeShowRailHint directly.
if (typeof window !== 'undefined' && !(/** @type {any} */ (window).__vitest_worker__)) {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  } else {
    boot();
  }
}
