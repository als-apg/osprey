// @ts-check
/**
 * Unit tests for chat-mcp.js: the chat's own `/mcp` command. Run with:
 *   npx vitest run tests/interfaces/web_terminal/chat-mcp.test.mjs
 *
 * Pinned here: the card each server listing draws (name, tool count, and OK,
 * the error, or "not checked" for a server that was never asked), and the
 * command hook (it answers `/mcp` only, never a prompt that merely starts with
 * it, and a failed listing becomes a chat notice). The transport
 * (fetchMcpServers) is a double; the DOM helpers and the notice bus are real.
 * The wiring into a mounted chat is covered in chat.test.mjs.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

/** Transport double, re-armed per test. */
const transport = { fetchMcpServers: vi.fn() };

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat-client.js', () => ({
  fetchMcpServers: () => transport.fetchMcpServers(),
}));

const { mcpServersCard, initChatMcp } =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/chat-mcp.js');
const { setChatNoticeSink } =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js');

/** @type {string[]} */
let notices;

beforeEach(() => {
  transport.fetchMcpServers = vi.fn();
  notices = [];
  setChatNoticeSink((text) => {
    notices.push(text);
    return true;
  });
});

afterEach(() => {
  setChatNoticeSink(null);
  document.body.replaceChildren();
});

/** Let the listing and its follow-up settle. */
const settle = async () => {
  for (let i = 0; i < 5; i += 1) await Promise.resolve();
};

/** @param {HTMLElement} card */
const rowsOf = (card) =>
  [...card.querySelectorAll('.op-mcp-row')].map((row) => {
    const count = /** @type {HTMLElement | null} */ (row.querySelector('.osprey-count'));
    const chip = /** @type {HTMLElement} */ (row.querySelector('.osprey-chip'));
    return {
      name: row.querySelector('.op-mcp-name')?.textContent,
      tools: count?.textContent,
      toolsTitle: count?.title,
      state: chip.textContent,
      tone: [...chip.classList].find((c) => c.startsWith('osprey-chip--')),
    };
  });

describe('mcpServersCard', () => {
  test('one row per server: its name, tool count, and OK, its error, or not checked', () => {
    const card = mcpServersCard([
      { name: 'controls', transport: 'stdio', tool_count: 3, tools: [{}, {}, {}], error: null },
      { name: 'python', transport: 'stdio', tool_count: 1, tools: [{}], error: null },
      { name: 'archiver', transport: 'stdio', tool_count: null, tools: null, error: 'no answer' },
      { name: 'remote', transport: 'http', tool_count: null, tools: null, error: null },
    ]);

    expect(card.querySelector('.osprey-list-head')?.textContent).toBe('MCP servers');
    expect(rowsOf(card)).toEqual([
      { name: 'controls', tools: '3', toolsTitle: '3 tools', state: 'OK', tone: 'osprey-chip--ok' },
      { name: 'python', tools: '1', toolsTitle: '1 tool', state: 'OK', tone: 'osprey-chip--ok' },
      {
        name: 'archiver',
        tools: undefined,
        toolsTitle: undefined,
        state: 'no answer',
        tone: 'osprey-chip--err',
      },
      {
        name: 'remote',
        tools: undefined,
        toolsTitle: undefined,
        state: 'http, not checked',
        tone: 'osprey-chip--info',
      },
    ]);
  });

  test('no servers is one line saying so, and no rows', () => {
    const card = mcpServersCard([]);
    expect(card.querySelector('.osprey-list-empty')?.textContent).toBe(
      'No MCP servers are configured.'
    );
    expect(card.querySelectorAll('.op-mcp-row')).toHaveLength(0);
  });
});

describe('initChatMcp', () => {
  /** @returns {{ answer: (prompt: string) => boolean, messages: HTMLElement, scrollToBottom: import('vitest').Mock }} */
  function mount() {
    /** @type {((prompt: string) => boolean) | null} */
    let answer = null;
    const messages = document.createElement('div');
    document.body.append(messages);
    const scrollToBottom = vi.fn();
    initChatMcp(
      /** @type {any} */ ({
        messages,
        scrollToBottom,
        onLocalCommand: (/** @type {(prompt: string) => boolean} */ fn) => {
          answer = fn;
        },
      })
    );
    if (answer === null) throw new Error('initChatMcp registered no local command');
    return { answer, messages, scrollToBottom };
  }

  test('/mcp is answered in the chat: the card goes into the log, which scrolls to it', async () => {
    transport.fetchMcpServers.mockResolvedValue([
      { name: 'controls', transport: 'stdio', tool_count: 2, tools: [{}, {}], error: null },
    ]);
    const { answer, messages, scrollToBottom } = mount();

    expect(answer('  /mcp  ')).toBe(true);
    await settle();

    expect(transport.fetchMcpServers).toHaveBeenCalledTimes(1);
    expect(messages.querySelectorAll('.op-mcp-card')).toHaveLength(1);
    expect(scrollToBottom).toHaveBeenCalledTimes(1);
    expect(notices).toEqual([]);
  });

  test('a prompt that only starts with /mcp is not the command', () => {
    const { answer } = mount();
    expect(answer('/mcp-check SR01')).toBe(false);
    expect(answer('/mcp list')).toBe(false);
    expect(transport.fetchMcpServers).not.toHaveBeenCalled();
  });

  test('a failed listing is a chat notice, and no card', async () => {
    transport.fetchMcpServers.mockRejectedValue(new Error('HTTP 500: Server Error'));
    const { answer, messages } = mount();

    expect(answer('/mcp')).toBe(true);
    await settle();

    expect(notices).toEqual(['Listing the MCP servers failed: HTTP 500: Server Error']);
    expect(messages.querySelector('.op-mcp-card')).toBeNull();
  });
});
