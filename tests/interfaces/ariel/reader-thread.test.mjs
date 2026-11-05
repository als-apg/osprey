// @ts-check
/**
 * The ARIEL reader's thread section (C-THREAD):
 *   npx vitest run tests/interfaces/ariel/reader-thread.test.mjs
 *
 * "Follows up" shows the parent as a `data-thread-entry` link, or its bare id
 * when the parent is not held; "Follow-ups (n)" lists the children and notes
 * a truncated list. No parent and no children renders nothing, and a failed
 * or non-object thread fetch renders nothing.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return { ...actual, entriesApi: { ...actual.entriesApi, thread: vi.fn() } };
});

import { entriesApi } from '../../../src/osprey/interfaces/ariel/static/js/api.js';
import {
  renderThreadLink,
  renderThreadSection,
  threadSection,
} from '../../../src/osprey/interfaces/ariel/static/js/reader-thread.js';

const thread = /** @type {import('vitest').Mock} */ (/** @type {any} */ (entriesApi).thread);

/**
 * @param {string} id
 * @param {object} [extra]
 * @returns {any}
 */
function row(id, extra = {}) {
  return {
    entry_id: id,
    subject: `subject of ${id}`,
    timestamp: '2026-01-01T00:00:00Z',
    author: 'alice',
    ...extra,
  };
}

/** @param {string} html */
function dom(html) {
  const div = document.createElement('div');
  div.innerHTML = html;
  return div;
}

/**
 * The section's render, with only the context field it reads.
 * @param {string} entryId
 */
function renderFor(entryId) {
  return threadSection.render(/** @type {any} */ ({ entryId }));
}

describe('renderThreadLink', () => {
  test('is a button that selects the entry, with subject, time and author', () => {
    const li = dom(renderThreadLink(row('e1'))).querySelector('li');
    const button = li?.querySelector('button.entry-thread-link');
    expect(button?.getAttribute('data-thread-entry')).toBe('e1');
    expect(button?.querySelector('.entry-thread-subject')?.textContent).toBe('subject of e1');
    const meta = button?.querySelectorAll('.entry-thread-meta') ?? [];
    expect(meta).toHaveLength(2);
    expect(meta[0].textContent).not.toBe('');
    expect(meta[1].textContent).toBe('alice');
  });

  test('falls back to the id for a missing subject and to nothing for a missing author', () => {
    const div = dom(renderThreadLink(row('e2', { subject: '', author: undefined })));
    expect(div.querySelector('.entry-thread-subject')?.textContent).toBe('e2');
    expect(div.querySelectorAll('.entry-thread-meta')[1].textContent).toBe('');
  });

  test('stringifies a numeric id and treats a missing id as empty', () => {
    expect(
      dom(renderThreadLink(row('x', { entry_id: 42 }))).querySelector('button')?.dataset.threadEntry
    ).toBe('42');
    expect(
      dom(renderThreadLink(row('x', { entry_id: undefined }))).querySelector('button')?.dataset
        .threadEntry
    ).toBe('');
  });

  test('escapes the id, subject and author', () => {
    const html = renderThreadLink(
      row('"><img src=x onerror=alert(1)>', {
        subject: '<b>bold</b>',
        author: '<script>x</script>',
      })
    );
    const div = dom(html);
    expect(div.querySelector('img')).toBeNull();
    expect(div.querySelector('b')).toBeNull();
    expect(div.querySelector('script')).toBeNull();
    expect(div.querySelector('button')?.dataset.threadEntry).toBe('"><img src=x onerror=alert(1)>');
    expect(div.querySelector('.entry-thread-subject')?.textContent).toBe('<b>bold</b>');
  });
});

describe('renderThreadSection', () => {
  test('renders nothing without a parent or follow-ups', () => {
    expect(renderThreadSection({ parent_id: null, parent: null, children: [] })).toBe('');
    expect(renderThreadSection({})).toBe('');
    expect(renderThreadSection(null)).toBe('');
    expect(renderThreadSection({ parent_id: null, children: 'not-a-list' })).toBe('');
  });

  test('shows a held parent as a link under "Follows up"', () => {
    const div = dom(renderThreadSection({ parent_id: 'p1', parent: row('p1'), children: [] }));
    const section = div.querySelector('[data-entry-thread]');
    expect(section?.classList.contains('entry-thread')).toBe(true);
    const headings = [...div.querySelectorAll('h4')].map((h) => h.textContent);
    expect(headings).toEqual(['Follows up']);
    expect(
      div.querySelector('.entry-thread-list [data-thread-entry]')?.getAttribute('data-thread-entry')
    ).toBe('p1');
    expect(div.querySelector('.entry-thread-missing')).toBeNull();
  });

  test('shows the bare, escaped id of a parent that is not held', () => {
    const div = dom(renderThreadSection({ parent_id: '<i>p9</i>', parent: null, children: [] }));
    const missing = div.querySelector('.entry-thread-missing');
    expect(missing?.textContent).toBe('<i>p9</i>');
    expect(div.querySelector('i')).toBeNull();
    expect(div.querySelector('[data-thread-entry]')).toBeNull();
  });

  test('lists follow-ups with their count, in order', () => {
    const div = dom(
      renderThreadSection({ parent_id: null, children: [row('c1'), row('c2'), row('c3')] })
    );
    expect([...div.querySelectorAll('h4')].map((h) => h.textContent)).toEqual(['Follow-ups (3)']);
    const ids = [...div.querySelectorAll('[data-thread-entry]')].map((b) =>
      b.getAttribute('data-thread-entry')
    );
    expect(ids).toEqual(['c1', 'c2', 'c3']);
    expect(div.querySelector('[data-thread-truncated]')).toBeNull();
  });

  test('notes older follow-ups when the server truncated the list', () => {
    const div = dom(
      renderThreadSection({ parent_id: null, children: [row('c1')], truncated: true })
    );
    expect(div.querySelector('[data-thread-truncated]')?.textContent).toBe(
      'Older follow-ups not shown'
    );
  });

  test('does not note truncation without follow-ups', () => {
    expect(renderThreadSection({ parent_id: null, children: [], truncated: true })).toBe('');
  });

  test('puts "Follows up" before "Follow-ups" when both are present', () => {
    const div = dom(
      renderThreadSection({ parent_id: 'p1', parent: row('p1'), children: [row('c1')] })
    );
    expect([...div.querySelectorAll('h4')].map((h) => h.textContent)).toEqual([
      'Follows up',
      'Follow-ups (1)',
    ]);
    const ids = [...div.querySelectorAll('[data-thread-entry]')].map((b) =>
      b.getAttribute('data-thread-entry')
    );
    expect(ids).toEqual(['p1', 'c1']);
  });
});

describe('threadSection', () => {
  beforeEach(() => {
    thread.mockReset();
  });

  test('is the "thread" section', () => {
    expect(threadSection.id).toBe('thread');
  });

  test('fetches the entry thread and renders it', async () => {
    const data = { parent_id: null, parent: null, children: [row('c1')] };
    thread.mockResolvedValue(data);
    const html = await renderFor('e1');
    expect(thread).toHaveBeenCalledWith('e1');
    expect(html).toBe(renderThreadSection(data));
    expect(
      dom(/** @type {string} */ (html)).querySelector('[data-thread-entry="c1"]')
    ).not.toBeNull();
  });

  test('renders nothing for an empty thread', async () => {
    thread.mockResolvedValue({ parent_id: null, parent: null, children: [] });
    expect(await renderFor('e1')).toBe('');
  });

  test('renders nothing when the fetch fails', async () => {
    thread.mockRejectedValue(new Error('boom'));
    expect(await renderFor('e1')).toBe('');
  });

  test('renders nothing for a non-object response', async () => {
    for (const value of [null, undefined, 'text', 7]) {
      thread.mockResolvedValue(value);
      expect(await renderFor('e1')).toBe('');
    }
  });
});
