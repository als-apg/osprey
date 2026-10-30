/**
 * The chat's lifecycle as events, for modules that react to it without a hook
 * in the renderer or the controller (the attention signal, screen-reader
 * announcements, export, the context meter, the sidebar's Agent row).
 *
 * One page holds one chat, so one bus serves it. A listener gets the detail
 * object. `replay` is true while history is redrawn, so a listener can tell a
 * past turn from a live one. A listener that throws is reported by the
 * browser and does not stop the renderer.
 *
 * @module chat-events
 */

/**
 * @typedef {object} ChatEventMap
 * @property {{ event: Record<string, any>, replay: boolean }} frame - every
 *   frame the renderer handled, after it was drawn
 * @property {{ text: string, replay: boolean }} user-message - an operator
 *   message added to the log
 * @property {{ markdown: string, replay: boolean }} agent-message - a whole
 *   agent message added at once (history without per-frame events)
 * @property {Record<string, never>} turn-end - a running turn ended (result,
 *   terminal error, or closed stream)
 * @property {{ busy: boolean }} busy - whether the console can start or leave
 *   a conversation: busy while a turn streams or a hand-off is in flight.
 *   Sent on every change of the console's controls, so a listener must be
 *   idempotent
 * @property {{ text: string }} notice - a line the console put in the log
 *   that is not part of the conversation (a transport failure, an export
 *   failure), for the screen-reader region
 * @property {{ waiting: number }} attention - how many approval, plan and
 *   question cards wait on the operator. Sent whenever that count changes,
 *   and with `0` at every reset and turn end, so a listener must be
 *   idempotent
 * @property {Record<string, never>} reset - the log was cleared
 */

const bus = new EventTarget();

/**
 * @template {keyof ChatEventMap} K
 * @param {K} name
 * @param {ChatEventMap[K]} detail
 */
export function emitChat(name, detail) {
  bus.dispatchEvent(new CustomEvent(name, { detail }));
}

/**
 * Listen for one chat event until the returned function is called.
 * @template {keyof ChatEventMap} K
 * @param {K} name
 * @param {(detail: ChatEventMap[K]) => void} listener
 * @returns {() => void} stop listening
 */
export function onChat(name, listener) {
  /** @param {Event} e */
  const handler = (e) => listener(/** @type {CustomEvent<ChatEventMap[K]>} */ (e).detail);
  bus.addEventListener(name, handler);
  return () => bus.removeEventListener(name, handler);
}
