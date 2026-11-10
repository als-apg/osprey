// @ts-check
/**
 * OSPREY Web Terminal — operator chat: the `/mcp` command.
 *
 * The chat runs `/mcp` itself (slash_commands.py `BROWSER_COMMANDS`): the CLI's
 * own `/mcp` is an interactive screen the chat cannot show, so the prompt never
 * goes to the agent. The card lists each MCP server of the project as
 * `GET /api/mcp-servers` reports it: its name, its tool count, and OK or the
 * reason it could not be listed.
 *
 * @module chat-mcp
 */

import { fetchMcpServers } from './chat-client.js';
import { elem } from './chat-dom.js';
import { postChatNotice } from './chat-events.js';

/**
 * Whether *prompt* is the chat's `/mcp`, which the browser answers.
 * @param {string} prompt
 * @returns {boolean}
 */
const isMcpCommand = (prompt) => prompt.trim() === '/mcp';

/**
 * The state chip of one server: OK, its error, or that it was not asked.
 * @param {import('./chat-client.js').McpServer} server
 * @returns {HTMLElement}
 */
function stateChip(server) {
  const listed = Array.isArray(server.tools);
  const tone = listed ? 'ok' : server.error ? 'err' : 'info';
  const text = listed ? 'OK' : (server.error ?? `${server.transport ?? 'remote'}, not checked`);
  const chip = elem('span', `osprey-chip osprey-chip--${tone}`);
  chip.append(elem('span', 'osprey-chip-label', text));
  chip.title = text;
  return chip;
}

/**
 * The card `/mcp` puts into the log.
 * @param {import('./chat-client.js').McpServer[]} servers
 * @returns {HTMLElement}
 */
export function mcpServersCard(servers) {
  const card = elem('div', 'osprey-card op-mcp-card');
  card.append(elem('div', 'osprey-list-head', 'MCP servers'));
  if (servers.length === 0) {
    card.append(elem('div', 'osprey-list-empty', 'No MCP servers are configured.'));
    return card;
  }
  for (const server of servers) {
    const row = elem('div', 'op-mcp-row');
    row.append(elem('span', 'op-mcp-name', server.name));
    if (server.tool_count != null) {
      const count = elem('span', 'osprey-count', String(server.tool_count));
      count.title = `${server.tool_count} tool${server.tool_count === 1 ? '' : 's'}`;
      row.append(count);
    }
    row.append(stateChip(server));
    card.append(row);
  }
  return card;
}

/**
 * Answer `/mcp` in the chat: the card, or a notice when the listing fails.
 * @param {import('./chat-features.js').ChatFeatureContext} ctx
 */
export function initChatMcp(ctx) {
  ctx.onLocalCommand((prompt) => {
    if (!isMcpCommand(prompt)) return false;
    void fetchMcpServers().then(
      (servers) => {
        ctx.messages.append(mcpServersCard(servers));
        ctx.scrollToBottom();
      },
      (/** @type {Error} */ err) =>
        void postChatNotice(`Listing the MCP servers failed: ${err.message}`)
    );
    return true;
  });
}
