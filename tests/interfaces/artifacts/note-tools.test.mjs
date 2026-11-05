// @ts-check
/**
 * Unit tests for a markdown note's strip tools (note-tools.js): Contents,
 * Find in note and Copy as Markdown:
 *   npx vitest run tests/interfaces/artifacts/note-tools.test.mjs
 *
 * The note renders in the gallery's own document, so the tools act on real
 * DOM here. happy-dom has no CSS custom highlights, so a small registry and
 * `Highlight` stand in for them where the marks are checked; `fetch` and the
 * clipboard are stubbed.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  copyMarkdown,
  findRanges,
  noteHeadings,
  openContents,
  openFind,
} from '../../../src/osprey/interfaces/artifacts/static/js/note-tools.js';
import { closeAllPopovers } from '/design-system/js/popover.js';

const NOTE = `
  <div class="osprey-md-rendered">
    <h1>Dump and refill</h1>
    <p>The stored current fell to 0 at 14:31 and stayed at 0 until 14:53.</p>
    <h2>What the data shows</h2>
    <p>A local bump of 0.146 mm. Outside the bump no BPM reads more.</p>
    <h3>Next</h3>
  </div>`;

/** @type {HTMLElement} */
let note;
/** @type {HTMLButtonElement} */
let button;

beforeEach(() => {
  document.body.innerHTML = `<button id="tool" type="button">tool</button>
    <div class="md-preview-container">${NOTE}</div>`;
  note = /** @type {HTMLElement} */ (document.querySelector('.md-preview-container'));
  button = /** @type {HTMLButtonElement} */ (document.getElementById('tool'));
  Element.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  closeAllPopovers();
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  document.body.replaceChildren();
});

/** @param {string} value */
function type(value) {
  const input = /** @type {HTMLInputElement} */ (document.querySelector('#note-find input'));
  input.value = value;
  input.dispatchEvent(new Event('input'));
  return input;
}

/** @param {boolean} [shiftKey] */
function enter(shiftKey = false) {
  const input = /** @type {HTMLInputElement} */ (document.querySelector('#note-find input'));
  input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', shiftKey }));
}

const count = () => document.querySelector('#note-find .note-find-count')?.textContent;

describe('Contents', () => {
  test("lists the note's headings in order", () => {
    expect(noteHeadings(note).map((h) => h.textContent)).toEqual([
      'Dump and refill',
      'What the data shows',
      'Next',
    ]);
  });

  test('opens a menu of the headings under its button; a pick scrolls to the heading', () => {
    openContents(button, note);

    const rows = Array.from(document.querySelectorAll('[role="menu"] [role="menuitem"]'));
    expect(rows.map((row) => row.textContent)).toEqual([
      'Dump and refill',
      'What the data shows',
      'Next',
    ]);
    expect(button.getAttribute('aria-expanded')).toBe('true');
    /** @type {HTMLElement} */ (rows[1]).click();
    expect(vi.mocked(Element.prototype.scrollIntoView).mock.contexts[0]).toBe(
      note.querySelector('h2')
    );
  });

  test('a note without headings says so, on a row that does nothing', () => {
    note.innerHTML = '<p>Just text.</p>';
    openContents(button, note);

    const rows = Array.from(document.querySelectorAll('[role="menu"] [role="menuitem"]'));
    expect(rows.map((row) => row.textContent)).toEqual(['No headings']);
    expect(rows[0].getAttribute('aria-disabled')).toBe('true');
  });
});

describe('Find in note', () => {
  test('finds every match in the text, whatever its case', () => {
    expect(findRanges(note, 'BUMP').map((range) => range.toString())).toEqual(['bump', 'bump']);
    expect(findRanges(note, '0').length).toBe(3);
    expect(findRanges(note, '   ')).toEqual([]);
  });

  test('opens a search field under its button; typing counts the matches', () => {
    openFind(button, note);
    const pop = /** @type {HTMLElement} */ (document.getElementById('note-find'));
    expect(pop.hidden).toBe(false);
    expect(document.activeElement).toBe(pop.querySelector('input'));

    type('bump');
    expect(count()).toBe('1 of 2');
    type('quench');
    expect(count()).toBe('No match');
    type('');
    expect(count()).toBe('');
  });

  test('Enter goes to the next match, Shift+Enter to the one before, and both wrap', () => {
    openFind(button, note);
    type('bump');

    enter();
    expect(count()).toBe('2 of 2');
    enter();
    expect(count()).toBe('1 of 2');
    enter(true);
    expect(count()).toBe('2 of 2');
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
  });

  test('marks the matches and the current one, and clears the marks when it closes', async () => {
    /** @type {Map<string, any>} */
    const highlights = new Map();
    vi.stubGlobal('CSS', { highlights });
    vi.stubGlobal(
      'Highlight',
      class {
        /** @param {Range[]} ranges */
        constructor(...ranges) {
          this.ranges = ranges;
        }
      }
    );
    openFind(button, note);
    type('bump');

    expect(highlights.get('osprey-note-find').ranges).toHaveLength(2);
    expect(highlights.get('osprey-note-find-current').ranges[0].toString()).toBe('bump');

    openFind(button, note);
    expect(/** @type {HTMLElement} */ (document.getElementById('note-find')).hidden).toBe(true);
    await vi.waitFor(() => expect(highlights.size).toBe(0));
  });

  test('opened again with text in it, it marks that text at once', () => {
    openFind(button, note);
    type('bump');
    openFind(button, note);
    openFind(button, note);
    expect(count()).toBe('1 of 2');
  });
});

describe('Copy as Markdown', () => {
  test('copies the source the note renders from', async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, text: async () => '# Dump and refill\n' }));
    vi.stubGlobal('fetch', fetchMock);
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });

    expect(await copyMarkdown({ id: 'n 1', filename: 'note.md' })).toBe(true);

    expect(fetchMock).toHaveBeenCalledWith('/files/n%201/note.md');
    expect(writeText).toHaveBeenCalledWith('# Dump and refill\n');
  });

  test('copies nothing when the source cannot be read', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: false, text: async () => '' }))
    );
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });

    expect(await copyMarkdown({ id: 'n1', filename: 'note.md' })).toBe(false);
    expect(writeText).not.toHaveBeenCalled();
  });
});
