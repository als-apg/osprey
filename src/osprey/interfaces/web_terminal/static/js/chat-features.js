/**
 * The chat's feature modules, started once by initChat.
 *
 * A feature reads the chat's lifecycle from chat-events.js and reaches the
 * console through the context handed in here, so adding one needs no hook in
 * chat.js or chat-render.js. Register a feature by calling its init from
 * {@link initChatFeatures}.
 *
 * @module chat-features
 */
import { initChatAnnounce } from './chat-announce.js';
import { initChatAttachments } from './chat-attach.js';
import { initChatAttention } from './chat-attention.js';
import { initChatContext } from './chat-context.js';
import { initChatExport } from './chat-export.js';
import { initChatTasks } from './chat-tasks.js';

/**
 * @typedef {object} ChatFeatureContext
 * @property {HTMLElement} container - #operator-container
 * @property {HTMLElement} bar - the console's header bar
 * @property {HTMLElement} messages - the message log
 * @property {HTMLElement} inputArea - the composer
 * @property {HTMLTextAreaElement} textarea - the composer's text box
 * @property {ReturnType<typeof import('./chat-render.js').createChatRenderer>} renderer
 * @property {() => string} getKey - the session key the console shows now
 * @property {(transform: (prompt: string) => string) => void} onBeforeSend -
 *   register a transform applied, in registration order, to each prompt as it
 *   is sent. The operator's bubble shows the text as typed.
 */

/**
 * Start every feature module.
 * @param {ChatFeatureContext} ctx
 */
export function initChatFeatures(ctx) {
  // One call per feature module, each taking `ctx`.
  initChatAttention(ctx);
  initChatAnnounce(ctx);
  initChatExport(ctx);
  initChatContext(ctx);
  initChatAttachments(ctx);
  initChatTasks(ctx);
}
