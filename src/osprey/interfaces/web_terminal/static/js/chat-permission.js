// @ts-check
/**
 * OSPREY Web Terminal — operator chat: permission-card contract.
 *
 * The shapes the approval and question cards share: the body a card posts, the
 * hooks the renderer supplies, the handle a card returns, and how a request was
 * answered. Also `deliver`, the one path an answer takes to the renderer's hook.
 *
 * @module chat-permission
 */

/**
 * The body a card posts to `POST /api/chat/{chat_id}/permission`. The renderer
 * adds the request id.
 * @typedef {{ allow: boolean, answers?: Record<string, string>, message?: string }} PermissionBody
 */

/**
 * What a card needs from the renderer. `answer` delivers the body and rejects
 * with a `status` of 404 once the request is gone. `onSettled` tells the
 * renderer that the card's pending state changed, so the loader can return.
 * @typedef {{ answer: (body: PermissionBody) => Promise<unknown>, onSettled: () => void }} CardHooks
 */

/**
 * An approval or question card.
 * @typedef {object} PermissionCard
 * @property {HTMLElement} el
 * @property {boolean} pending - awaiting the operator, or mid-delivery
 * @property {() => void} cancel - make inert: the turn ended without an answer
 * @property {() => void} focus - move focus to the card's primary control
 * @property {(outcome: RemoteAnswer) => void} resolved - answered elsewhere:
 *   another tab, or the server's deadline
 */

/**
 * How a request was answered, as the `permission_resolved` frame reports it.
 * `expired` is set when nobody answered within the server's deadline: the
 * server declined it, and the card says so rather than "Denied by you".
 * @typedef {{ allow?: boolean, for_session?: boolean, answers?: Record<string, string> | null, expired?: boolean }} RemoteAnswer
 */

/**
 * Hand one answer to the renderer's hook and classify the outcome.
 * @param {CardHooks} hooks
 * @param {PermissionBody} body
 * @returns {Promise<'answered' | 'stale' | 'failed'>}
 */
export async function deliver(hooks, body) {
  try {
    await hooks.answer(body);
    return 'answered';
  } catch (err) {
    return /** @type {{ status?: number } | undefined} */ (err)?.status === 404
      ? 'stale'
      : 'failed';
  }
}
