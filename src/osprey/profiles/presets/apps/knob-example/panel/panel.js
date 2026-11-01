// The panel: shows the backend's state of the one channel, refreshed on a timer.
// Its address is relative: the page is served under the terminal's /panel/<id>/.

const STATE_URL = 'api/state';
const REFRESH_MS = 2000;

/**
 * @typedef {{channel: string, target: string, readback: number}} KnobState
 */

/**
 * @param {Element} root
 * @param {string} name
 * @returns {HTMLElement}
 */
function field(root, name) {
  const element = root.querySelector(`[data-field="${name}"]`);
  if (!(element instanceof HTMLElement)) throw new Error(`panel has no ${name} field`);
  return element;
}

/**
 * Show one state.
 * @param {Element} root
 * @param {KnobState} state
 */
export function showState(root, state) {
  field(root, 'channel').textContent = state.channel;
  field(root, 'readback').textContent = String(state.readback);
  field(root, 'target').textContent = state.target;
  field(root, 'error').hidden = true;
}

/**
 * Show why there is no state, leaving the last one on view.
 * @param {Element} root
 * @param {string} message
 */
export function showError(root, message) {
  const error = field(root, 'error');
  error.textContent = message;
  error.hidden = false;
}

/**
 * Read the state once and show it.
 * @param {Element} root
 * @param {typeof fetch} [fetcher]
 */
export async function refresh(root, fetcher = fetch) {
  try {
    const response = await fetcher(STATE_URL, { headers: { Accept: 'application/json' } });
    if (!response.ok) throw new Error(`the backend answered ${response.status}`);
    showState(root, await response.json());
  } catch (error) {
    showError(root, error instanceof Error ? error.message : String(error));
  }
}

const panel = document.getElementById('knob-state');
if (panel) {
  void refresh(panel);
  setInterval(() => void refresh(panel), REFRESH_MS);
}
