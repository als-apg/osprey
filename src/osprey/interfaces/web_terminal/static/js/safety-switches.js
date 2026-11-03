/* OSPREY Web Terminal — runtime safety switches in the Settings drawer's Safety tab
 *
 * The row is shared: the Review tab shows code review's switch with it, so a
 * switch looks and moves the same wherever it appears. */

import { formatFacilityTime } from '/design-system/js/facility-time.js';

import { apiRequest, fetchJSON } from './api.js';
import { postChatNotice } from './chat-events.js';
import { dismissConfirm, isConfirmUp, showConfirm } from './posture-confirm.js';
import { showUndoNotice } from './undo-notice.js';

/**
 * One row of `GET /api/safety-switches`.
 * @typedef {object} SafetySwitch
 * @property {string} name
 * @property {string} label
 * @property {string} [note]  what an off leaves running or stopped, for the dialog
 * @property {boolean} on
 * @property {string | null} off_by
 * @property {number | null} off_at  epoch seconds
 */

/** The change notice the server broadcasts when a switch moves. */
export const SAFETY_SWITCH_FRAME = 'safety_switch';

/** Beside a toggle the tier may not move. */
export const ADMIN_LOCK_TEXT = '🔒 Set by your deployment admin';

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
      announce(offSentence(sw));
    } else if (sw.on && before === false) {
      announce(`${sw.label} is back on.`);
    }
    lastSeen.set(sw.name, sw.on);
  }
  if (section) render(section, switches, data?.can_change === true);
}

/**
 * Tell the operator in the chat when the chat is on screen, else in a notice.
 * @param {string} message
 */
function announce(message) {
  if (!postChatNotice(message)) showUndoNotice({ message, durationMs: NOTICE_MS });
}

/**
 * The sentence naming who turned a feature off and when.
 * @param {SafetySwitch} sw
 * @returns {string}
 */
export function offSentence(sw) {
  const by = sw.off_by ? ` by ${sw.off_by}` : '';
  const time = sw.off_at
    ? formatFacilityTime(sw.off_at * 1000, { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' })
    : '';
  const at = time ? ` at ${time}` : '';
  return `${sw.label} is off in this terminal${by}${at}. It returns at the next restart.`;
}

/**
 * @param {HTMLElement} container
 * @param {SafetySwitch[]} switches
 * @param {boolean} canChange
 */
function render(container, switches, canChange) {
  container.replaceChildren();
  container.hidden = switches.length === 0;
  for (const sw of switches) container.appendChild(switchRow(sw, canChange));
}

/**
 * One switch: its label, its state, and a toggle that is live only for a tier
 * the server says may move it.
 * @param {SafetySwitch} sw
 * @param {boolean} canChange
 * @param {() => Promise<void>} [refresh]  re-reads the view the row sits in after a move
 * @returns {HTMLElement}
 */
export function switchRow(sw, canChange, refresh = refreshSafetySwitches) {
  const bar = document.createElement('div');
  bar.className = 'hook-debug-bar safety-switch-row';
  bar.dataset.switch = sw.name;

  const text = document.createElement('div');
  text.className = 'safety-switch-text';
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
  input.addEventListener('change', () => void move(sw, input, refresh));

  const right = document.createElement('span');
  right.className = 'safety-switch-lock';
  if (!canChange) {
    const lock = document.createElement('span');
    lock.textContent = ADMIN_LOCK_TEXT;
    right.appendChild(lock);
  }
  right.appendChild(toggleLabel);
  bar.append(text, right);
  return bar;
}

/**
 * The off dialog's body: how long the off lasts, and the switch's own note.
 * @param {SafetySwitch} sw
 * @returns {string}
 */
export function offDialogText(sw) {
  const note = sw.note ? ` ${sw.note}` : '';
  return (
    `It stays off for this terminal until it restarts.${note} ` +
    'The move is recorded in the audit trail with your name.'
  );
}

/**
 * @param {SafetySwitch} sw
 * @param {HTMLInputElement} input
 * @param {() => Promise<void>} refresh
 */
async function move(sw, input, refresh) {
  const on = input.checked;
  if (on) {
    try {
      await post(sw, true);
    } catch (err) {
      input.checked = false;
      showUndoNotice({ message: /** @type {Error} */ (err).message, durationMs: NOTICE_MS });
      return;
    }
    await refresh();
    return;
  }
  // Off waits for the dialog; until it is confirmed the toggle shows on.
  input.checked = true;
  /** @param {KeyboardEvent} event */
  const onKey = (event) => {
    if (event.key === 'Escape' && isConfirmUp()) {
      event.preventDefault();
      dismissConfirm();
    }
  };
  showConfirm({
    title: `Turn ${sw.label} off?`,
    body: [[offDialogText(sw)]],
    live: null,
    confirmLabel: 'Turn off',
    focus: 'cancel',
    onConfirm: (ui) => void confirmOff(sw, ui, refresh),
    onDismiss: () => document.removeEventListener('keydown', onKey, true),
  });
  document.addEventListener('keydown', onKey, true);
}

/**
 * Post a confirmed off. A refusal stays in the dialog, so the operator reads it
 * where they asked.
 * @param {SafetySwitch} sw
 * @param {import('./posture-confirm.js').ConfirmUi} ui
 * @param {() => Promise<void>} refresh
 */
async function confirmOff(sw, ui, refresh) {
  ui.confirm.disabled = ui.cancel.disabled = true;
  ui.error.hidden = true;
  try {
    await post(sw, false);
  } catch (err) {
    ui.error.textContent = /** @type {Error} */ (err).message;
    ui.error.hidden = false;
    ui.confirm.disabled = ui.cancel.disabled = false;
    return;
  }
  ui.done();
  // The frame re-renders every tab; this one re-reads at once.
  await refresh();
}

/**
 * @param {SafetySwitch} sw
 * @param {boolean} on
 */
async function post(sw, on) {
  await apiRequest(`/api/safety-switches/${encodeURIComponent(sw.name)}`, {
    method: 'POST',
    json: { state: on ? 'on' : 'off' },
    errorPrefix: `Could not turn ${sw.label} ${on ? 'on' : 'off'}`,
  });
}

/** Forget what was seen; for tests. */
export function _resetSafetySwitchesForTest() {
  lastSeen.clear();
  section = null;
}
