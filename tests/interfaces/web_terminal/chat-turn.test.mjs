/**
 * Unit tests for chat-turn.js -- the agent turn in flight.
 *
 * The turn places its blocks in order, closes one kind when another opens, and
 * ends with a footer that counts only what happened. The reasoning row is live
 * while it is open and settles when the turn moves on.
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-turn.test.mjs
 */

import { describe, test, expect, vi, afterEach } from 'vitest';

import { qs } from '../_support/dom.mjs';

import { createTurn } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-turn.js';

/** A turn whose text renders as plain text. */
function turn() {
  return createTurn({
    renderText: (el, text) => {
      el.textContent = text;
    },
  });
}

/**
 * @param {string} id
 * @param {string} raw
 */
const call = (id, raw) => ({
  type: 'tool_use',
  tool_name: raw,
  tool_name_raw: raw,
  tool_use_id: id,
  input: {},
});

/** The footer's parts as text, in order. */
const footerParts = (/** @type {HTMLElement} */ entry) =>
  [...entry.querySelectorAll('.op-footer-part')].map((part) => part.textContent);

/** The entry's blocks, in order, by what they are. */
const blocksOf = (/** @type {HTMLElement} */ entry) =>
  [...entry.children].map((el) => {
    if (el.classList.contains('op-tool')) return 'tool';
    if (el.classList.contains('op-entry-body')) return 'text';
    if (el.classList.contains('op-footer')) return 'footer';
    return el.className;
  });

afterEach(() => {
  vi.useRealTimers();
});

describe('the footer', () => {
  test('states the duration alone when the turn only wrote text', () => {
    const t = turn();
    t.text('Hello.');
    t.footer({ durationMs: 53_000, isError: false });

    expect(footerParts(t.entry)).toEqual(['53s']);
    expect(t.entry.querySelectorAll('.op-footer-sep')).toHaveLength(0);
  });

  test('counts the actions and the approvals, with the glyphs the rows wore', () => {
    const t = turn();
    const a = t.tool(call('t1', 'Read'));
    t.tool(call('t2', 'Read'));
    t.tool(call('t3', 'Bash'));
    t.agent({ type: 'tool_use', tool_name_raw: 'Agent', input: { subagent_type: 'scout' } });
    a.file(document.createElement('section'));
    t.countApproval();
    t.footer({ durationMs: 125_000, isError: false });

    expect(footerParts(t.entry)).toEqual(['2m 05s', '4 actions', '1 approval']);
    const footer = qs(t.entry, '.op-footer');
    expect(footer.querySelectorAll('.op-footer-sep')).toHaveLength(2);
    // The actions part wears one small icon per kind, in order of first use.
    expect(footer.querySelectorAll('.op-footer-stack > .op-icon')).toHaveLength(3);
    // Nothing in the footer is spoken apart from its text.
    for (const icon of footer.querySelectorAll('.op-icon')) {
      expect(icon.getAttribute('aria-hidden')).toBe('true');
    }
  });

  test('a lone action reads in the singular, and a failed turn says so last', () => {
    const t = turn();
    t.tool(call('t1', 'Read'));
    t.footer({ durationMs: 4_000, isError: true });

    expect(footerParts(t.entry)).toEqual(['4s', '1 action', 'ended with an error']);
    expect(qs(t.entry, '.op-footer-error').textContent).toBe('ended with an error');
  });

  test('the footer closes the turn: it comes last, after every row', () => {
    const t = turn();
    t.tool(call('t1', 'Read'));
    t.footer({ durationMs: 1_000, isError: false });
    t.tool(call('t2', 'Read'));

    expect(blocksOf(t.entry)).toEqual(['tool', 'footer', 'tool']);
  });
});

describe('the reasoning row', () => {
  test('a hidden thought cannot be opened while it runs, and one with text can', () => {
    const t = turn();
    t.think('', true);
    const row = qs(t.entry, '.op-thinking');
    const head = /** @type {HTMLButtonElement} */ (qs(row, '.op-row-head'));
    expect(row.dataset.state).toBe('running');
    expect(head.disabled).toBe(true);
    head.click();
    expect(qs(row, '.op-row-body').hidden).toBe(true);

    t.think('Checking the gaps.', true);
    expect(head.disabled).toBe(false);
    head.click();
    expect(qs(row, '.op-row-body').hidden).toBe(false);
  });

  test('is live and pulsing while open, then settles and says how long it took', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-10-25T10:00:00Z'));
    const t = turn();
    t.think('Let me ', true);
    t.think('see.', true);
    const row = qs(t.entry, '.op-thinking');
    expect(row.dataset.state).toBe('running');
    expect(row.classList.contains('pulse')).toBe(true);
    expect(qs(row, '.op-row-label').classList.contains('op-shimmer')).toBe(true);
    expect(qs(row, '.op-row-glyph .op-icon').getAttribute('aria-hidden')).toBe('true');

    vi.setSystemTime(new Date('2026-10-25T10:00:04Z'));
    t.text('Here is the answer.');
    expect(row.dataset.state).toBe('done');
    expect(row.classList.contains('pulse')).toBe(false);
    expect(qs(row, '.op-row-label').textContent).toBe('Thought for 4s');
    expect(qs(row, '.op-row-label').classList.contains('op-shimmer')).toBe(false);
    expect(qs(row, '.op-row-body').textContent).toBe('Let me see.');
    // The row still opens: the chevron at the right edge is its only status.
    expect(row.querySelector('.op-row-status .op-row-chevron')).not.toBeNull();
    expect(row.querySelector('.op-spin')).toBeNull();
  });

  test('live, the time runs from the frame before the reasoning, not its first words', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-10-25T10:00:03Z'));
    const t = turn();
    t.think('Let me see.', false, { since: Date.parse('2026-10-25T10:00:00Z') });
    vi.setSystemTime(new Date('2026-10-25T10:00:04Z'));
    t.text('Done.');
    expect(qs(t.entry, '.op-thinking .op-row-label').textContent).toBe('Thought for 4s');
  });

  test.each([
    [2600, 'Thought for 3s'],
    [400, 'Thought for <1s'],
    [null, 'Thought'],
  ])('replayed with a recorded %s ms, it reads "%s"', (durationMs, label) => {
    const t = turn();
    t.think('', false, { durationMs });
    t.text('Done.');
    expect(qs(t.entry, '.op-thinking .op-row-label').textContent).toBe(label);
  });

  test('a tool call that follows takes the pulse from it', () => {
    const t = turn();
    t.think('Hmm.');
    const thought = qs(t.entry, '.op-thinking');
    const chip = t.tool(call('t1', 'Read'));

    expect(thought.dataset.state).toBe('done');
    expect(thought.classList.contains('pulse')).toBe(false);
    expect(chip.el.classList.contains('pulse')).toBe(true);
    expect(t.entry.querySelectorAll('.pulse')).toHaveLength(1);
  });

  test('redacted reasoning settles to a row that does not open', () => {
    const t = turn();
    t.think('');
    t.closeThinking();
    const row = qs(t.entry, '.op-thinking');
    expect(row.classList.contains('empty')).toBe(true);
    expect(qs(row, '.op-row-head', HTMLButtonElement).disabled).toBe(true);
  });
});

describe('placing', () => {
  test('text between two calls of one kind keeps them apart', () => {
    const t = turn();
    t.tool(call('t1', 'Read'));
    t.text('One down.');
    t.tool(call('t2', 'Read'));

    expect(blocksOf(t.entry)).toEqual(['tool', 'text', 'tool']);
  });
});
