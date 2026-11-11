// @ts-check
/**
 * The OPTIMIZER panel's polling cadence.
 *
 * One self-scheduling loop (never `setInterval`, so a slow tick can never
 * stack up behind itself): each tick reports the job status it saw, and the
 * next tick is scheduled at the delay that status calls for — about once a
 * second while a run is moving, every couple of seconds while a draft waits
 * (so an agent's edit shows up promptly), and slowly while idle or failing.
 *
 * @module poller
 */

/** Statuses of a job whose worker holds the machine (or is about to). */
export const ACTIVE_STATUSES = Object.freeze(['submitted', 'running', 'paused', 'cancelling']);

export const ACTIVE_MS = 1000;
export const DRAFT_MS = 2000;
export const IDLE_MS = 5000;
export const FAILED_MS = 5000;

/**
 * @param {string|null|undefined} status
 * @returns {boolean}
 */
export function isActive(status) {
  return !!status && ACTIVE_STATUSES.includes(status);
}

/**
 * Delay before the next tick.
 *
 * @param {{ status?: string|null, failed?: boolean }} seen
 * @returns {number}
 */
export function pollDelay({ status, failed }) {
  if (failed) return FAILED_MS;
  if (isActive(status)) return ACTIVE_MS;
  if (status === 'created') return DRAFT_MS;
  return IDLE_MS;
}

/**
 * @typedef {{ status?: string|null, failed?: boolean }} TickResult
 */

/**
 * @param {{
 *   tick: () => Promise<TickResult>,
 *   setTimer?: (fn: () => void, ms: number) => any,
 *   clearTimer?: (handle: any) => void,
 * }} options
 */
export function createPoller({ tick, setTimer = setTimeout, clearTimer = clearTimeout }) {
  /** @type {any} */
  let timer = null;
  let inFlight = false;
  let rerun = false;
  let stopped = true;

  async function run() {
    if (inFlight) {
      // A tick is already reading; ask it to go again as soon as it lands
      // rather than issuing a second concurrent read.
      rerun = true;
      return;
    }
    if (timer !== null) clearTimer(timer);
    timer = null;
    inFlight = true;
    /** @type {TickResult} */
    let seen = { failed: true };
    try {
      seen = await tick();
    } catch {
      seen = { failed: true };
    } finally {
      inFlight = false;
    }
    if (stopped) return;
    if (rerun) {
      rerun = false;
      void run();
      return;
    }
    timer = setTimer(() => {
      timer = null;
      void run();
    }, pollDelay(seen));
  }

  return {
    start() {
      if (!stopped) return;
      stopped = false;
      void run();
    },
    stop() {
      stopped = true;
      if (timer !== null) clearTimer(timer);
      timer = null;
    },
    /** Tick now (after an operator action), then resume the cadence. */
    now() {
      if (stopped) return Promise.resolve();
      return run();
    },
  };
}
