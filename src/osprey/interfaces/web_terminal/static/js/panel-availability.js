// @ts-check
/* OSPREY Web Terminal — Which panels are served
 *
 * A panel can be enabled and still have nothing behind it: the hub's proxy
 * answers 404 for any panel whose backend URL it never published. That URL is
 * the only input to availability, and the page holds it as `PanelState.url`.
 * The shell's own readers of a panel's routes (the sidebar's artifact lists and
 * health dot) follow it through this module, so they run exactly while the
 * panel is served and never ask a route that cannot answer.
 *
 * `served` is a projection of `PanelState.url`, written only by
 * panel-lifecycle.js's publishPanelUrl. This module is a leaf so a reader can
 * follow a panel without importing the panel stack.
 */

/** @typedef {{start: () => void, stop: () => void}} Follower */

/** @type {Set<string>} */
const served = new Set();

/** @type {Map<string, Set<Follower>>} */
const followers = new Map();

/**
 * Run a follower's half, keeping a failure to that follower: this runs inside
 * the panel lifecycle, which other panels depend on.
 * @param {string} panelId
 * @param {() => void} fn
 */
function run(panelId, fn) {
  try {
    fn();
  } catch (err) {
    console.error(`A follower of the ${panelId} panel failed:`, err);
  }
}

/**
 * Record whether a panel is served, telling its followers when that changed.
 * @param {string} panelId
 * @param {boolean} isServed
 */
export function setPanelServed(panelId, isServed) {
  if (served.has(panelId) === isServed) return;
  if (isServed) served.add(panelId);
  else served.delete(panelId);
  for (const follower of followers.get(panelId) ?? []) {
    run(panelId, isServed ? follower.start : follower.stop);
  }
}

/**
 * Call `start` whenever the panel becomes served, now included, and `stop`
 * whenever it stops being served.
 * @param {string} panelId
 * @param {() => void} start
 * @param {() => void} stop
 * @returns {() => void} stops following; calls neither half
 */
export function whilePanelServed(panelId, start, stop) {
  const follower = { start, stop };
  const own = followers.get(panelId) ?? new Set();
  followers.set(panelId, own);
  own.add(follower);
  if (served.has(panelId)) run(panelId, start);
  return () => own.delete(follower);
}
