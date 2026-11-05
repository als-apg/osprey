// @ts-check
/**
 * The ARIEL "Ask agent" reader action:
 *   npx vitest run tests/interfaces/ariel/ask-agent.test.mjs
 *
 * Embedded only, it registers a reader-seam action on the `a` key that posts
 * a citation block (entry id, author, date, then the reader's selected text,
 * else the subject) to the host as `osprey-paste-to-terminal`, addressed to
 * this page's own origin. It never sends and raises no ARIEL toast; the host
 * owns the feedback. Standalone, nothing is registered.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    entriesApi: { ...actual.entriesApi, get: vi.fn(), thread: vi.fn() },
  };
});

import { entriesApi } from '../../../src/osprey/interfaces/ariel/static/js/api.js';
import { seam } from '../../../src/osprey/interfaces/ariel/static/js/reader-seam.js';
import {
  showInReader,
  runReaderKey,
} from '../../../src/osprey/interfaces/ariel/static/js/reader.js';
import {
  askAgentAction,
  citation,
  registerAskAgent,
} from '../../../src/osprey/interfaces/ariel/static/js/ask-agent.js';
import { formatTimestamp } from '../../../src/osprey/interfaces/ariel/static/js/components.js';

const ENTRY = {
  entry_id: 'e-42',
  timestamp: '2026-01-02T03:04:00Z',
  author: 'Jane Operator',
  source_system: 'sys',
  subject: 'Beam dump at 03:04',
  raw_text: 'Beam dump at 03:04\nRF trip in sector 3, recovered after reset.',
  keywords: [],
  attachments: [],
};

/** @param {string} search */
function setSearch(search) {
  window.history.replaceState(null, '', `/${search}`);
}

function resetSeam() {
  seam.actions.splice(0);
  seam.keys.clear();
}

/** @returns {Promise<HTMLElement>} a reader showing ENTRY */
async function shownReader() {
  const container = document.createElement('div');
  container.id = 'ariel-reader-body';
  document.body.appendChild(container);
  /** @type {any} */ (entriesApi.get).mockResolvedValue(ENTRY);
  /** @type {any} */ (entriesApi.thread).mockResolvedValue({ entries: [] });
  await showInReader(container, ENTRY.entry_id, {});
  return container;
}

/** @param {Node} node */
function selectContents(node) {
  const range = document.createRange();
  range.selectNodeContents(node);
  const sel = window.getSelection();
  sel?.removeAllRanges();
  sel?.addRange(range);
}

let post = vi.fn();

beforeEach(() => {
  resetSeam();
  post = vi.fn();
  vi.spyOn(window.parent, 'postMessage').mockImplementation(post);
});

afterEach(() => {
  vi.restoreAllMocks();
  window.getSelection()?.removeAllRanges();
  document.body.innerHTML = '';
  setSearch('');
  resetSeam();
});

describe('citation', () => {
  test('carries id, author and date, then the subject without a selection', () => {
    const text = citation(ENTRY, '');
    const lines = text.split('\n');
    expect(lines[0]).toContain('e-42');
    expect(lines[0]).toContain('Jane Operator');
    expect(lines[0]).toContain(formatTimestamp(ENTRY.timestamp));
    expect(text).toContain('Beam dump at 03:04');
    expect(text).not.toContain('RF trip');
    expect(text.indexOf('e-42')).toBeLessThan(text.indexOf('Jane Operator'));
    expect(text.indexOf('Jane Operator')).toBeLessThan(text.lastIndexOf('Beam dump at 03:04'));
  });

  test('uses the selection instead of the subject when there is one', () => {
    const text = citation(ENTRY, 'RF trip in sector 3');
    expect(text).toContain('RF trip in sector 3');
    expect(text.split('\n').slice(1).join('\n')).not.toContain('Beam dump');
  });

  test('derives the subject from raw_text and names an unknown author', () => {
    const text = citation({ ...ENTRY, subject: undefined, author: '' }, '');
    expect(text).toContain('Beam dump at 03:04');
    expect(text).toContain('Unknown');
  });

  test('quotes every line of a multi-line selection', () => {
    const text = citation(ENTRY, 'one\ntwo');
    expect(text).toContain('> one\n> two');
  });
});

describe('registerAskAgent', () => {
  test('standalone ARIEL registers nothing', () => {
    setSearch('');
    registerAskAgent();
    expect(seam.actions).toHaveLength(0);
  });

  test('embedded, it registers the action on the a key', () => {
    setSearch('?embedded=true');
    registerAskAgent();
    expect(seam.actions).toEqual([askAgentAction]);
    expect(askAgentAction.key).toBe('a');
    expect(askAgentAction.id).toBe('ask-agent');
  });
});

describe('Ask agent action', () => {
  beforeEach(() => setSearch('?embedded=true'));

  test('the header shows its button when embedded', async () => {
    registerAskAgent();
    const reader = await shownReader();
    expect(reader.querySelector('[data-reader-action="ask-agent"]')).not.toBeNull();
  });

  test('the a key posts the subject citation to the same-origin parent', async () => {
    registerAskAgent();
    const reader = await shownReader();
    expect(runReaderKey(reader, 'a')).toBe(true);
    expect(post).toHaveBeenCalledTimes(1);
    const [message, origin] = post.mock.calls[0];
    expect(origin).toBe(window.location.origin);
    expect(message).toEqual({ type: 'osprey-paste-to-terminal', text: citation(ENTRY, '') });
  });

  test('a selection inside the reader is cited instead of the subject', async () => {
    registerAskAgent();
    const reader = await shownReader();
    const body = /** @type {HTMLElement} */ (reader.querySelector('.entry-detail-text'));
    selectContents(body);
    runReaderKey(reader, 'a');
    const { text } = post.mock.calls[0][0];
    expect(text).toContain('RF trip in sector 3');
    expect(text).toBe(citation(ENTRY, body.textContent?.trim() ?? ''));
  });

  test('a selection outside the reader is ignored', async () => {
    registerAskAgent();
    const reader = await shownReader();
    const outside = document.createElement('p');
    outside.textContent = 'unrelated page text';
    document.body.appendChild(outside);
    selectContents(outside);
    runReaderKey(reader, 'a');
    const { text } = post.mock.calls[0][0];
    expect(text).not.toContain('unrelated');
    expect(text).toBe(citation(ENTRY, ''));
  });

  test('a button click posts the same citation', async () => {
    registerAskAgent();
    const reader = await shownReader();
    const button = /** @type {HTMLElement} */ (
      reader.querySelector('[data-reader-action="ask-agent"]')
    );
    button.click();
    expect(post).toHaveBeenCalledTimes(1);
    expect(post.mock.calls[0][0].type).toBe('osprey-paste-to-terminal');
  });

  test('it raises no ARIEL toast', async () => {
    registerAskAgent();
    const reader = await shownReader();
    runReaderKey(reader, 'a');
    expect(document.querySelector('.toast, .osprey-toast, [role="status"]')).toBeNull();
  });

  test('run never posts when not embedded', () => {
    setSearch('');
    const container = document.createElement('div');
    askAgentAction.run({
      container,
      entry: ENTRY,
      entryId: ENTRY.entry_id,
      callbacks: {},
      isCurrent: () => true,
    });
    expect(post).not.toHaveBeenCalled();
  });
});
