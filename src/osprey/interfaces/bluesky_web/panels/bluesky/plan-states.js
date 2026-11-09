// @ts-check
/**
 * BLUESKY panel — what the Plans view says when it has no plan to show.
 *
 * Two panes can be empty: the plan list in the sidebar, and the detail pane
 * beside it while no plan is selected. Neither is ever a blank pane
 * (DESIGN.md §12, "Panel chrome standard"): each shows one centred state —
 * an icon, one message, an optional second line, and at most one action —
 * and these pure functions decide which one, from what the view knows: how
 * far the plan catalog load got, how the filter fared, and what the queue
 * stream last said. The view renders the answer with the design system's
 * `renderState`. Technical detail (a status code, the bridge's own reason)
 * stays in the toolbar chip's popover, never in these sentences.
 *
 * @module plan-states
 */

/**
 * How far the plan catalog load got: still on its first read, read, or
 * failed (the list is empty because it could not be read).
 *
 * @typedef {'loading'|'ready'|'error'} PlansLoad
 */

/**
 * The queue as the Plans view sees it: no frame yet (or a dropped stream), a
 * manager the bridge could not read, or a readable queue.
 *
 * @typedef {'connecting'|'unreachable'|'ok'} QueueReach
 */

/**
 * One centred state. `retry` asks the view for its Retry action, the one
 * action these states offer.
 *
 * @typedef {object} PlanState
 * @property {string} icon An `icons.js` name.
 * @property {string} message
 * @property {string} [detail]
 * @property {boolean} [retry]
 */

/** @type {Readonly<PlanState>} */
const CONNECTING = Object.freeze({
  icon: 'radio',
  message: 'Connecting to the queue server…',
  detail: 'Stop and Abort are live while this connects.',
  retry: true,
});

/** @type {Readonly<PlanState>} */
const UNREACHABLE = Object.freeze({
  icon: 'triangle-alert',
  message: 'Queue server unreachable',
  detail: 'The queue manager could not be read. Stop and Abort are still sent when you click them.',
  retry: true,
});

/**
 * Read the queue view's last report (`onStatus`) as a reach: `null` is a
 * stream with no frame yet or a dropped one, and a summary the bridge could
 * not read is unreachable whatever else it carries.
 *
 * @param {import('./queue-client.js').QueueStatus|null} status
 * @returns {QueueReach}
 */
export function queueReach(status) {
  if (status === null) return 'connecting';
  return status.available === true ? 'ok' : 'unreachable';
}

/**
 * The plan list's state, or `null` when it has rows to show.
 *
 * A failed read says the same as an empty catalog: the list has nothing to
 * offer either way, and the detail pane beside it says why.
 *
 * @param {{load: PlansLoad, total: number, shown: number}} plans `total` is
 *   the catalog's size, `shown` how many rows pass the filter.
 * @returns {PlanState|null}
 */
export function planListState({ load, total, shown }) {
  if (load === 'loading') return { icon: 'refresh', message: 'Loading plans…' };
  if (load === 'error' || total === 0) return { icon: 'file-text', message: 'No plans available.' };
  if (shown === 0) return { icon: 'search', message: 'No plan matches the filter.' };
  return null;
}

/**
 * The detail pane's state while no plan is selected.
 *
 * A listed catalog only waits for a pick. Otherwise the queue speaks first,
 * because it is the bigger fact: an unreadable manager — or a failed plan
 * read while the queue stream is not up either, which is one unreachable
 * bridge — is "unreachable", and a stream still dialling is "connecting".
 * Both say that the two halts are live, and both offer Retry.
 *
 * @param {{load: PlansLoad, total: number, queue: QueueReach}} input
 * @returns {PlanState}
 */
export function planDetailState({ load, total, queue }) {
  if (load === 'ready' && total > 0) {
    return { icon: 'file-text', message: 'Select a plan to see its parameters.' };
  }
  if (queue === 'unreachable' || (load === 'error' && queue === 'connecting')) return UNREACHABLE;
  if (load === 'error') {
    return { icon: 'triangle-alert', message: 'Plans could not be loaded.', retry: true };
  }
  if (queue === 'connecting') return CONNECTING;
  if (load === 'loading') return { icon: 'refresh', message: 'Loading plans…' };
  return { icon: 'file-text', message: 'No plans available.' };
}
