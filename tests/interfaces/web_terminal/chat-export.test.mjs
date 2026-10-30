// @ts-check
/**
 * Unit tests for chat-export.js: the copy control on finished replies and the
 * conversation export. Run with:
 *   npx vitest run tests/interfaces/web_terminal/chat-export.test.mjs
 *
 * The renderer, the lifecycle bus and the export module are the real ones, so
 * a reply is drawn the way the live chat draws it and the bus carries what the
 * copy control reads. Only the transport (fetchHistory) and the clipboard are
 * doubles. The `marked` double renders a small subset of markdown, so the
 * rendered text differs from the source and a test can tell which one reached
 * the clipboard.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { stampFacilityZone } from '../_support/facility-zone.mjs';

const DIR = '../../../src/osprey/interfaces/web_terminal/static/js';
const KEY = 'abcdef12-3456-7890-aaaa-bbbbccccdddd';

/** Transport double, re-armed per test. */
const transport = { fetchHistory: vi.fn() };

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat-client.js', () => ({
  fetchHistory: (/** @type {any} */ key) => transport.fetchHistory(key),
}));

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/chat-render.js')} */
let render;
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js')} */
let bus;
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/chat-export.js')} */
let exportModule;

/** @type {HTMLElement} */
let messages;
/** @type {HTMLElement} */
let bar;
/** @type {ReturnType<typeof import('../../../src/osprey/interfaces/web_terminal/static/js/chat-render.js').createChatRenderer>} */
let renderer;
/** @type {{ writeText: import('vitest').Mock }} */
let clipboard;

/** Let the clipboard write and the history read settle. No fake time passes. */
const flush = () => vi.advanceTimersByTimeAsync(0);

/** The copy controls in the log, in order. */
const copyControls = () =>
  /** @type {HTMLButtonElement[]} */ ([...messages.querySelectorAll('.op-reply-copy')]);

/** Draw one agent turn from its live frames. */
/** @param {Array<{ type: string, [key: string]: any }>} frames */
function drawTurn(frames) {
  renderer.beginTurn();
  for (const event of frames) renderer.handleEvent(event);
}

beforeEach(async () => {
  vi.useFakeTimers();
  vi.stubGlobal('marked', {
    parse: (/** @type {string} */ text) =>
      text
        .replace(/^## (.*)$/gm, '<h2>$1</h2>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/`([^`]+)`/g, '<code>$1</code>'),
  });
  vi.stubGlobal('DOMPurify', { sanitize: (/** @type {string} */ html) => html });
  clipboard = { writeText: vi.fn().mockResolvedValue(undefined) };
  Object.defineProperty(navigator, 'clipboard', { value: clipboard, configurable: true });

  document.body.replaceChildren();
  messages = document.createElement('div');
  messages.className = 'op-messages';
  bar = document.createElement('div');
  bar.className = 'op-session-bar';
  document.body.append(bar, messages);

  vi.resetModules();
  render = await import(`${DIR}/chat-render.js`);
  bus = await import(`${DIR}/chat-events.js`);
  exportModule = await import(`${DIR}/chat-export.js`);
  renderer = render.createChatRenderer(messages);
  exportModule.initChatExport({ bar, messages, getKey: () => KEY });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  // The clipboard is a page-level property: put it back as the browser had it.
  delete (/** @type {any} */ (navigator).clipboard);
  delete (/** @type {any} */ (URL).createObjectURL);
  delete (/** @type {any} */ (URL).revokeObjectURL);
});

describe('copy: a reply is copied as its markdown source', () => {
  test.each([
    [
      'a streamed reply, built from deltas, copies the exact markdown',
      [
        { type: 'delta', kind: 'text', content: '## Status\n\n' },
        { type: 'delta', kind: 'text', content: 'The **pressure** is `nominal`.' },
        { type: 'result', is_error: false, duration_ms: 1200 },
      ],
      '## Status\n\nThe **pressure** is `nominal`.',
    ],
    [
      'a whole text block after streamed pieces replaces them in the copy',
      [
        { type: 'delta', kind: 'text', content: 'Hel' },
        { type: 'delta', kind: 'text', content: 'lo' },
        { type: 'text', content: 'Hello, **final**.' },
        { type: 'result', is_error: false },
      ],
      'Hello, **final**.',
    ],
    [
      'text on either side of a tool call is copied as two paragraphs',
      [
        { type: 'text', content: 'First.' },
        {
          type: 'tool_use',
          tool_use_id: 't1',
          tool_name: 'Read',
          tool_name_raw: 'Read',
          input: {},
        },
        { type: 'tool_result', tool_use_id: 't1', is_error: false, preview: 'file body' },
        { type: 'text', content: 'Second.' },
        { type: 'result', is_error: false },
      ],
      'First.\n\nSecond.',
    ],
  ])('%s', async (_name, frames, copied) => {
    renderer.addUserMessage('check the vacuum');
    drawTurn(frames);

    copyControls()[0].click();
    await flush();

    expect(clipboard.writeText).toHaveBeenCalledWith(copied);
  });

  test('a whole agent message from history copies its markdown', async () => {
    renderer.addAgentMessage('**Done.** See `log`.');

    copyControls()[0].click();
    await flush();

    expect(clipboard.writeText).toHaveBeenCalledWith('**Done.** See `log`.');
  });

  test('a replayed turn with events copies its text without the tool payloads', async () => {
    renderer.replay([
      { role: 'user', content: 'run it' },
      {
        role: 'assistant',
        content: 'Part one.\n\nPart two.',
        events: [
          { type: 'text', content: 'Part one.' },
          {
            type: 'tool_use',
            tool_use_id: 't1',
            tool_name: 'Bash',
            tool_name_raw: 'Bash',
            input: { command: 'ls' },
          },
          { type: 'tool_result', tool_use_id: 't1', is_error: false, preview: 'listing' },
          { type: 'text', content: 'Part two.' },
          { type: 'result', is_error: false },
        ],
      },
    ]);

    copyControls()[0].click();
    await flush();

    expect(clipboard.writeText).toHaveBeenCalledWith('Part one.\n\nPart two.');
  });

  test('a reply with no source falls back to its rendered text', async () => {
    const entry = document.createElement('div');
    entry.className = 'op-entry assistant';
    const body = document.createElement('div');
    body.className = 'op-entry-body';
    body.textContent = 'Rendered only';
    entry.append(body);
    messages.append(entry);

    bus.emitChat('turn-end', {});
    copyControls()[0].click();
    await flush();

    expect(clipboard.writeText).toHaveBeenCalledWith('Rendered only');
  });
});

describe('copy: the control', () => {
  test('is offered on a finished reply, and not on one still streaming', () => {
    renderer.beginTurn();
    renderer.handleEvent({ type: 'delta', kind: 'text', content: 'still writing' });
    expect(copyControls()).toHaveLength(0);

    renderer.handleEvent({ type: 'result', is_error: false });
    expect(copyControls()).toHaveLength(1);
  });

  test('gives each reply its own control, once', () => {
    drawTurn([
      { type: 'text', content: 'one' },
      { type: 'result', is_error: false },
    ]);
    drawTurn([
      { type: 'text', content: 'two' },
      { type: 'result', is_error: false },
    ]);
    expect(copyControls()).toHaveLength(2);
  });

  test('is not offered for a turn that said nothing', () => {
    drawTurn([
      { type: 'tool_use', tool_use_id: 't1', tool_name: 'Read', tool_name_raw: 'Read', input: {} },
      { type: 'tool_result', tool_use_id: 't1', is_error: false, preview: '' },
      { type: 'result', is_error: false },
    ]);
    expect(copyControls()).toHaveLength(0);
  });

  test('is a labelled button whose name says Copied briefly after a copy, then Copy again', async () => {
    drawTurn([
      { type: 'text', content: 'copy me' },
      { type: 'result', is_error: false },
    ]);

    const copy = copyControls()[0];
    // The visible label is the accessible name: no aria-label hides the state change.
    expect(copy.hasAttribute('aria-label')).toBe(false);
    expect(copy.tabIndex).toBe(0);
    expect(copy.textContent).toBe('Copy');

    copy.click();
    await flush();
    expect(copy.textContent).toBe('Copied');
    expect(copy.classList.contains('copied')).toBe(true);

    vi.advanceTimersByTime(1500);
    expect(copy.textContent).toBe('Copy');
    expect(copy.classList.contains('copied')).toBe(false);
  });

  test('a clipboard failure leaves the control as it was, without an error', async () => {
    clipboard.writeText.mockRejectedValueOnce(new Error('clipboard denied'));
    drawTurn([
      { type: 'text', content: 'no clipboard' },
      { type: 'result', is_error: false },
    ]);

    const copy = copyControls()[0];
    copy.click();
    await flush();

    expect(copy.textContent).toBe('Copy');
    expect(copy.classList.contains('copied')).toBe(false);
  });
});

describe('exportFilename', () => {
  afterEach(() => stampFacilityZone(null));

  test('is the date on the facility clock and the first eight characters of the key', () => {
    const when = new Date('2026-10-22T20:30:00Z');

    stampFacilityZone('Asia/Tokyo');
    expect(exportModule.exportFilename('0123456789abcdef', when)).toBe(
      'osprey-conversation-2026-10-23-01234567.md'
    );
    stampFacilityZone('UTC');
    expect(exportModule.exportFilename('0123456789abcdef', when)).toBe(
      'osprey-conversation-2026-10-22-01234567.md'
    );
  });
});

describe('conversationMarkdown', () => {
  const exportedAt = new Date('2026-10-22T15:00:00.000Z');

  test('writes operator and agent turns in order, with the transcript timestamps', () => {
    const markdown = exportModule.conversationMarkdown(
      [
        { role: 'user', content: 'Status of the RF?', timestamp: '2026-10-22T14:02:11Z' },
        { role: 'assistant', content: 'RF is nominal.', timestamp: '2026-10-22T14:02:19Z' },
      ],
      { key: KEY, exportedAt }
    );

    expect(markdown).toBe(
      [
        '# Osprey conversation',
        `Session: \`${KEY}\``,
        'Exported: 2026-10-22T15:00:00.000Z',
        '## Operator\n\n_2026-10-22T14:02:11Z_\n\nStatus of the RF?',
        '## Agent\n\n_2026-10-22T14:02:19Z_\n\nRF is nominal.',
      ].join('\n\n') + '\n'
    );
  });

  test('puts each tool call on one line with its decision and outcome, and no payloads', () => {
    const markdown = exportModule.conversationMarkdown(
      [
        {
          role: 'assistant',
          content: 'Checking. Done.',
          events: [
            { type: 'text', content: 'Checking.' },
            {
              type: 'tool_use',
              tool_use_id: 't1',
              tool_name: 'Bash',
              tool_name_raw: 'Bash',
              input: { command: 'SECRET-INPUT' },
            },
            {
              type: 'tool_result',
              tool_use_id: 't1',
              is_error: false,
              preview: 'SECRET-OUTPUT',
              decision: 'allowed',
            },
            {
              type: 'tool_use',
              tool_use_id: 't2',
              tool_name: 'Write',
              tool_name_raw: 'Write',
              input: {},
            },
            {
              type: 'tool_result',
              tool_use_id: 't2',
              is_error: true,
              preview: 'boom',
              decision: 'denied',
            },
            {
              type: 'tool_use',
              tool_use_id: 't3',
              tool_name: 'channel_read',
              tool_name_raw: 'mcp__osprey__channel_read',
              input: {},
            },
            { type: 'text', content: 'Done.' },
            { type: 'result', is_error: false },
          ],
        },
      ],
      { key: KEY, exportedAt }
    );

    expect(markdown).toContain(
      [
        '## Agent',
        'Checking.',
        '- Tool call: `Bash` · decision: allowed · outcome: ok',
        '- Tool call: `Write` · decision: denied · outcome: error',
        '- Tool call: `mcp__osprey__channel_read` · outcome: no result',
        'Done.',
      ].join('\n\n')
    );
    expect(markdown).not.toContain('SECRET-INPUT');
    expect(markdown).not.toContain('SECRET-OUTPUT');
  });

  test('an empty conversation says so under its header', () => {
    expect(exportModule.conversationMarkdown([], { key: KEY, exportedAt })).toBe(
      [
        '# Osprey conversation',
        `Session: \`${KEY}\``,
        'Exported: 2026-10-22T15:00:00.000Z',
        '_No messages in this conversation yet._',
      ].join('\n\n') + '\n'
    );
  });

  test('a turn with no text and no tool calls is left out', () => {
    const markdown = exportModule.conversationMarkdown(
      [
        { role: 'user', content: '   ' },
        { role: 'assistant', content: '', events: [] },
      ],
      { key: KEY, exportedAt }
    );
    expect(markdown).toContain('_No messages in this conversation yet._');
  });
});

describe('export button', () => {
  beforeEach(() => {
    transport.fetchHistory.mockReset();
    // Stub the download's three steps so the file's text can be read back.
    URL.createObjectURL = vi.fn(() => 'blob:osprey-test');
    URL.revokeObjectURL = vi.fn();
  });

  test('is an icon button in the Agent bar, named Export', () => {
    const button = /** @type {HTMLButtonElement} */ (bar.querySelector('.op-export'));
    expect(button.classList.contains('osprey-icon-btn')).toBe(true);
    expect(button.querySelector('svg.osprey-ic')).not.toBeNull();
    expect(button.getAttribute('aria-label')).toBe('Export');
    expect(button.title).toBeTruthy();
  });

  test('downloads the conversation read from the server, under the dated file name', async () => {
    vi.setSystemTime(new Date(2026, 9, 22, 9, 30));
    transport.fetchHistory.mockResolvedValue([
      { role: 'user', content: 'Ping' },
      { role: 'assistant', content: 'Pong' },
    ]);
    /** @type {string[]} */
    const names = [];
    /** @type {Blob[]} */
    const blobs = [];
    const createObjectURL = /** @type {import('vitest').Mock} */ (URL.createObjectURL);
    createObjectURL.mockImplementation((/** @type {Blob} */ blob) => {
      blobs.push(blob);
      return 'blob:osprey-test';
    });
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(
      /** @this {HTMLAnchorElement} */ function () {
        names.push(this.download);
      }
    );

    /** @type {HTMLButtonElement} */ (bar.querySelector('.op-export')).click();
    await flush();

    expect(transport.fetchHistory).toHaveBeenCalledWith(KEY);
    expect(names).toEqual(['osprey-conversation-2026-10-22-abcdef12.md']);
    const text = await blobs[0].text();
    expect(text).toContain('## Operator\n\nPing');
    expect(text).toContain('## Agent\n\nPong');
    click.mockRestore();
  });

  test('a failed export says so in the log, publishes the notice, and leaves the button usable', async () => {
    transport.fetchHistory.mockRejectedValue(new Error('network down'));
    const button = /** @type {HTMLButtonElement} */ (bar.querySelector('.op-export'));
    /** @type {string[]} */
    const published = [];
    bus.onChat('notice', ({ text }) => published.push(text));

    button.click();
    await flush();

    const notice = messages.querySelector('.op-system')?.textContent ?? '';
    expect(notice).toMatch(/export/);
    expect(published).toEqual([notice]);
    expect(button.disabled).toBe(false);
  });
});
