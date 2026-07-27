// @ts-check
/* OSPREY Web Terminal — Panel Health Polling
 *
 * The health-poll machinery behind the rail's LEDs, extracted from
 * panel-manager.js: fast startup retries (500ms, backing off toward 5s) that
 * switch to slow 10s maintenance polling once a panel first reports healthy.
 *
 * This module owns TIMING + FETCH + the state record's `healthy`/`polling`
 * flags only. Every consequence of a poll settling — rail LED, status bar,
 * enabling the entry, giving the empty slot away — belongs to the caller via
 * the `onSettled(panel, wasHealthy)` hook, so this module has no rail, dock,
 * or DOM knowledge.
 */

/** @typedef {import('./panel-manager.js').Panel} Panel */
/** @typedef {import('./panel-manager.js').PanelState} PanelState */

/** A poll-settle observer: called after every poll attempt (success or error)
 *  with the panel and its PREVIOUS healthy flag; read the new one off state.
 *  @typedef {(panel: Panel, wasHealthy: boolean) => void} OnSettled */

/**
 * Poll a panel's configured health endpoint once and record the result on its
 * state. `state.url` is already the server-prefixed root-relative
 * `<prefix>/panel/<id>` (routes/panels.py's compute_url_prefix()), so the
 * string concat is safe: fetch() resolves it against the current origin,
 * preserving the prefix as-is — never re-derive or re-prefix it here. Reentry
 * is guarded by `state.polling`; a missing url is a no-op (config not loaded).
 * @param {Panel} panel
 * @param {PanelState} state
 * @param {OnSettled} onSettled
 */
export async function pollHealth(panel, state, onSettled) {
  if (!state.url || state.polling) return;
  state.polling = true;

  const wasHealthy = state.healthy;
  try {
    const resp = await fetch(`${state.url}${panel.healthEndpoint || '/health'}`, {
      signal: AbortSignal.timeout(2000),
    });
    state.healthy = resp.ok;
  } catch {
    state.healthy = false;
  } finally {
    state.polling = false;
  }
  onSettled(panel, wasHealthy);
}

/**
 * Start the polling loop for a panel: an immediate probe, fast retries while
 * unhealthy (500ms growing 1.5× toward a 5s cap), then a 10s maintenance
 * interval once the first healthy settle lands. The live timer handle is kept
 * on `state.pollTimer`.
 * @param {Panel} panel
 * @param {PanelState} state
 * @param {OnSettled} onSettled
 */
export function startHealthPolling(panel, state, onSettled) {
  pollHealth(panel, state, onSettled);

  let delay = 500;
  function scheduleNext() {
    state.pollTimer = setTimeout(() => {
      pollHealth(panel, state, onSettled).then(() => {
        if (state.healthy) {
          // Switch to slow maintenance polling
          state.pollTimer = setInterval(() => pollHealth(panel, state, onSettled), 10000);
        } else {
          delay = Math.min(delay * 1.5, 5000);
          scheduleNext();
        }
      });
    }, delay);
  }
  scheduleNext();
}
