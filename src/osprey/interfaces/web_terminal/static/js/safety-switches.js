/* OSPREY Web Terminal — runtime safety switches in the Settings drawer's Safety tab */

import { apiRequest, fetchJSON } from './api.js';
import { showUndoNotice } from './undo-notice.js';

/**
 * One row of `GET /api/safety-switches`.
 * @typedef {object} SafetySwitch
 * @property {string} name
 * @property {string} label
 * @property {boolean} on
 * @property {string | null} off_by
 * @property {number | null} off_at  epoch seconds
 */

/** The change notice the server broadcasts when a switch moves. */
export const SAFETY_SWITCH_FRAME = 'safety_switch';

/** How long the "turned off" notice stays up. */
const NOTICE_MS = 8000;

/** @type {Map<string, boolean>} last on-state seen per switch, to notice changes */
const lastSeen = new Map();

/** @type {HTMLElement | null} */
let section = null;

/**
 * Whether *data* is a safety-switch change notice.
 * @param {{type?: string}} data
 * @returns {boolean}
 */
export function isSafetySwitchFrame(data) {
  return data?.type === SAFETY_SWITCH_FRAME;
}

/**
 * Mount the switch rows in the Safety tab. Every operator sees which safety
 * features run; only a tier the server says may change them gets a live
 * toggle. The server is the gate: the toggle is a convenience, a refusal
 * reverts it.
 */
export function initSafetySwitches() {
  section = document.getElementById('safety-switch-section');
  const panel = document.getElementById('tab-safety');
  if (!section || !panel) return;
  panel.addEventListener('drawer:tab-activate', () => void refreshSafetySwitches());
}

/**
 * Re-read the switches, re-render the rows, and tell the operator about any
 * feature that is off and was not known to be off. Called on a change frame
 * and whenever the event stream (re)opens, since SSE has no replay.
 * @returns {Promise<void>}
 */
export async function refreshSafetySwitches() {
  /** @type {{switches: SafetySwitch[], can_change: boolean}} */
  let data;
  try {
    data = await fetchJSON('/api/safety-switches');
  } catch {
    return; // an older server, or a missed read: the next frame or open retries
  }
  const switches = Array.isArray(data?.switches) ? data.switches : [];
  for (const sw of switches) {
    const before = lastSeen.get(sw.name);
    if (!sw.on && before !== false) {
      showUndoNotice({ message: offSentence(sw), durationMs: NOTICE_MS });
    } else if (sw.on && before === false) {
      showUndoNotice({ message: `${sw.label} is back on.`, durationMs: NOTICE_MS });
    }
    lastSeen.set(sw.name, sw.on);
  }
  if (section) render(section, switches, data?.can_change === true);
}

/**
 * The sentence naming who turned a feature off and when.
 * @param {SafetySwitch} sw
 * @returns {string}
 */
export function offSentence(sw) {
  const by = sw.off_by ? ` by ${sw.off_by}` : '';
  const at = sw.off_at ? ` at ${new Date(sw.off_at * 1000).toLocaleTimeString()}` : '';
  return `${sw.label} is off${by}${at}. It returns at the next restart.`;
}

/**
 * @param {HTMLElement} container
 * @param {SafetySwitch[]} switches
 * @param {boolean} canChange
 */
function render(container, switches, canChange) {
  container.replaceChildren();
  container.hidden = switches.length === 0;
  for (const sw of switches) container.appendChild(row(sw, canChange));
}

/**
 * @param {SafetySwitch} sw
 * @param {boolean} canChange
 * @returns {HTMLElement}
 */
function row(sw, canChange) {
  const bar = document.createElement('div');
  bar.className = 'hook-debug-bar safety-switch-row';
  bar.dataset.switch = sw.name;

  const text = document.createElement('div');
  const label = document.createElement('span');
  label.className = 'hook-debug-label';
  label.textContent = sw.label;
  text.appendChild(label);
  const state = document.createElement('div');
  state.className = 'safety-switch-state' + (sw.on ? '' : ' off');
  state.textContent = sw.on ? 'On, as the profile sets it.' : offSentence(sw);
  text.appendChild(state);

  const toggleLabel = document.createElement('label');
  toggleLabel.className = 'toggle-switch';
  toggleLabel.title = canChange ? `Turn ${sw.label} ${sw.on ? 'off' : 'on'}` : 'Admin tier only';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = sw.on;
  input.disabled = !canChange;
  input.setAttribute('aria-label', sw.label);
  const slider = document.createElement('span');
  slider.className = 'toggle-slider';
  toggleLabel.append(input, slider);
  input.addEventListener('change', () => void move(sw, input));

  bar.append(text, toggleLabel);
  return bar;
}

/**
 * @param {SafetySwitch} sw
 * @param {HTMLInputElement} input
 */
async function move(sw, input) {
  const on = input.checked;
  if (!on && !confirm(`Turn ${sw.label} off for this terminal until it restarts?`)) {
    input.checked = true;
    return;
  }
  try {
    await apiRequest(`/api/safety-switches/${encodeURIComponent(sw.name)}`, {
      method: 'POST',
      json: { state: on ? 'on' : 'off' },
      errorPrefix: `Could not turn ${sw.label} ${on ? 'on' : 'off'}`,
    });
  } catch (err) {
    input.checked = !on;
    showUndoNotice({ message: /** @type {Error} */ (err).message, durationMs: NOTICE_MS });
    return;
  }
  // The frame re-renders every tab; this one re-reads at once.
  await refreshSafetySwitches();
}

/** Forget what was seen; for tests. */
export function _resetSafetySwitchesForTest() {
  lastSeen.clear();
  section = null;
}
