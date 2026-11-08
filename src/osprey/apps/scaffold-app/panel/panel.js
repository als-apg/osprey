// The panel: shows the backend's state of the one channel, refreshed on a timer.
// Its address is relative: the page is served under the terminal's /panel/<id>/.

const STATE_URL = 'api/state';
const REFRESH_MS = 2000;

/**
 * The backend's state. With no channel configured, every value is null and `error`
 * names the config key to set.
 * @typedef {{
 *   channel: string | null,
 *   target: string | null,
 *   readback: number | null,
 *   error: string | null,
 * }} ScaffoldState
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
 * Show one state; a state that carries an error shows the error instead.
 * @param {Element} root
 * @param {ScaffoldState} state
 */
export function showState(root, state) {
  if (state.error) {
    showError(root, state.error);
    return;
  }
  field(root, 'channel').textContent = state.channel ?? '';
  field(root, 'readback').textContent = state.readback == null ? '' : String(state.readback);
  const target = field(root, 'target');
  target.textContent = state.target ?? '';
  target.hidden = !state.target;
  field(root, 'error').hidden = true;
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

const panel = document.getElementById('scaffold-state');
if (panel) {
  void refresh(panel);
  setInterval(() => void refresh(panel), REFRESH_MS);
}
