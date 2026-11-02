// @ts-check
/**
 * OSPREY Web Terminal — operator chat: what a frame means for a turn.
 *
 * The controller draws every frame the transport decodes, but two questions
 * about a frame decide what it does with it: whether the frame closes the
 * running turn, and whether it is part of an agent's response rather than a
 * notice about one. Pure predicates over the decoded event; nothing here
 * touches the DOM or the network.
 *
 * @module chat-frames
 */

/** Frame types that belong to a response, as opposed to notices about one. */
const RESPONSE_FRAMES = new Set([
  'text',
  'delta',
  'thinking',
  'tool_use',
  'tool_result',
  'permission_request',
]);

/**
 * Whether the frame closes the running turn: its result, or an error other
 * than one the agent reports inside a message and keeps working past.
 * @param {import('./chat-client.js').ChatEvent} frame
 * @returns {boolean}
 */
export function endsTurn(frame) {
  return (
    frame.type === 'result' ||
    (frame.type === 'error' && frame.error_type !== 'AssistantMessageError')
  );
}

/**
 * Whether the frame is part of an agent's response (text, thinking, a tool
 * call or its result, a permission request) rather than a notice about one.
 * @param {import('./chat-client.js').ChatEvent} frame
 * @returns {boolean}
 */
export function isResponseFrame(frame) {
  return RESPONSE_FRAMES.has(frame.type);
}
