/**
 * The chat's feature modules, started once by initChat.
 *
 * A feature reads the chat's lifecycle from chat-events.js and reaches the
 * console through the context handed in here, so adding one needs no hook in
 * chat.js or chat-render.js. Register a feature by calling its init from
 * {@link initChatFeatures}. The Agent bar holds only the session switcher and
 * `+`: a feature's action on the pane is a row in the pane's ⋯
 * (agent-menu.js setAgentMenuRow).
 *
 * @module chat-features
 */
import { initChatAnnounce } from './chat-announce.js';
import { initChatAttachments } from './chat-attach.js';
import { initChatAttention } from './chat-attention.js';
import { initChatContext } from './chat-context.js';
import { exportConversation, initChatExport } from './chat-export.js';
import { initChatMcp } from './chat-mcp.js';
import { installChatMenus } from './chat-menus.js';
import { initChatTasks } from './chat-tasks.js';

/**
 * @typedef {object} ChatFeatureContext
 * @property {HTMLElement} container - #operator-container
 * @property {HTMLElement} messages - the message log
 * @property {HTMLElement} inputArea - the composer
 * @property {HTMLTextAreaElement} textarea - the composer's text box
 * @property {ReturnType<typeof import('./chat-render.js').createChatRenderer>} renderer
 * @property {() => string} getKey - the session key the console shows now
 * @property {(transform: (prompt: string) => string) => void} onBeforeSend -
 *   register a transform applied, in registration order, to each prompt as it
 *   is sent. The operator's bubble shows the text as typed.
 * @property {(answer: (prompt: string) => boolean) => void} onLocalCommand -
 *   register a command the chat answers itself: *answer* takes the submitted
 *   prompt and returns true when it handled it, and the prompt is not sent.
 * @property {() => void} scrollToBottom - scrolls the log to its latest entry
 * @property {() => void} onNewSession - starts a conversation on a new session
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
  initChatMcp(ctx);
  // Message and log context menus: Copy, Quote in prompt; Scroll to latest,
  // Export conversation, New session.
  installChatMenus({ ...ctx, onExport: exportConversation });
}
