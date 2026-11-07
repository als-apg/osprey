// @ts-check
/**
 * OSPREY Artifact Gallery — the tools of a markdown note's content strip:
 * Contents, Find in note and Copy as Markdown.
 *
 * The note renders in the gallery's own document (preview-content.js), so the
 * tools act on it directly:
 *   - Contents lists the note's headings in a menu under its button
 *     (context-menu.js); a pick scrolls to the heading.
 *   - Find in note opens a search field under its button (popover.js):
 *     typing marks every match in the note and scrolls to the first; Enter
 *     goes to the next, Shift+Enter to the one before, and the field counts
 *     them. The marks are CSS custom highlights (`::highlight(osprey-note-find)`
 *     in gallery.css), so the note's own markup never changes; a browser
 *     without them still scrolls to each match. A match inside one run of
 *     text is found; one split by formatting (`foo **bar**`) is not.
 *   - Copy as Markdown puts the note's source, the file it renders from, on
 *     the clipboard.
 *
 * @module note-tools
 */

import { copyText } from '/design-system/js/clipboard.js';
import { openContextMenu } from '/design-system/js/context-menu.js';
import { icon } from '/design-system/js/icons.js';
import { togglePopover } from '/design-system/js/popover.js';
import { fileUrl } from './state.js';

/** The highlight every match wears, and the one the current match adds. */
const ALL_MATCHES = 'osprey-note-find';
const CURRENT_MATCH = 'osprey-note-find-current';

/**
 * @typedef {object} Finder
 * @property {HTMLElement} pop
 * @property {HTMLInputElement} input
 * @property {HTMLElement} count
 */

/** @type {Finder | null} */
let finder = null;

/** The note Find searches, its matches and the current one. */
let search = {
  /** @type {HTMLElement | null} */
  note: null,
  /** @type {Range[]} */
  ranges: [],
  current: -1,
};

/**
 * The note's headings, in order.
 * @param {HTMLElement} note
 * @returns {HTMLElement[]}
 */
export function noteHeadings(note) {
  return Array.from(note.querySelectorAll('h1, h2, h3, h4'));
}

/**
 * Contents: the note's headings in a menu under `button`; a pick scrolls to
 * the heading. A second press closes the menu.
 * @param {HTMLElement} button
 * @param {HTMLElement} note
 */
export function openContents(button, note) {
  const headings = noteHeadings(note);
  const rect = button.getBoundingClientRect();
  openContextMenu({
    ariaLabel: 'Contents',
    x: rect.left,
    y: rect.bottom,
    anchorEl: button,
    triggerEl: button,
    items: headings.length
      ? headings.map((heading) => ({
          label: (heading.textContent ?? '').trim(),
          run: () => heading.scrollIntoView({ block: 'start' }),
        }))
      : [{ label: 'No headings', disabled: true }],
  });
}

/**
 * Every place `query` occurs in the note's text, case-insensitively.
 * @param {HTMLElement} note
 * @param {string} query
 * @returns {Range[]}
 */
export function findRanges(note, query) {
  const needle = query.trim().toLowerCase();
  if (!needle) return [];
  /** @type {Range[]} */
  const ranges = [];
  const walker = document.createTreeWalker(note, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = (node.textContent ?? '').toLowerCase();
    for (let at = text.indexOf(needle); at >= 0; at = text.indexOf(needle, at + needle.length)) {
      const range = document.createRange();
      range.setStart(node, at);
      range.setEnd(node, at + needle.length);
      ranges.push(range);
    }
  }
  return ranges;
}

/**
 * Mark the matches, the current one apart; no ranges clears the marks. A
 * browser without CSS custom highlights shows none.
 * @param {Range[]} ranges
 * @param {number} current
 */
function mark(ranges, current) {
  const registry = /** @type {any} */ (globalThis).CSS?.highlights;
  const Highlight = /** @type {any} */ (globalThis).Highlight;
  if (!registry || typeof Highlight !== 'function') return;
  if (ranges.length) registry.set(ALL_MATCHES, new Highlight(...ranges));
  else registry.delete(ALL_MATCHES);
  if (ranges[current]) registry.set(CURRENT_MATCH, new Highlight(ranges[current]));
  else registry.delete(CURRENT_MATCH);
}

/** Show the current match: mark it, count it and scroll to it. */
function showMatch() {
  if (!finder) return;
  const { ranges, current } = search;
  mark(ranges, current);
  finder.count.textContent = ranges.length
    ? `${current + 1} of ${ranges.length}`
    : finder.input.value.trim()
      ? 'No match'
      : '';
  ranges[current]?.startContainer.parentElement?.scrollIntoView({ block: 'center' });
}

/** @param {string} query */
function runFind(query) {
  search.ranges = search.note ? findRanges(search.note, query) : [];
  search.current = search.ranges.length ? 0 : -1;
  showMatch();
}

/** @param {1 | -1} delta */
function stepMatch(delta) {
  const count = search.ranges.length;
  if (!count) return;
  search.current = (search.current + delta + count) % count;
  showMatch();
}

/** @returns {Finder} the page's Find popover, made on first use */
function findPopover() {
  if (finder?.pop.isConnected) return finder;
  const pop = document.createElement('div');
  pop.id = 'note-find';
  pop.className = 'osprey-popover note-find';
  pop.setAttribute('role', 'dialog');
  pop.setAttribute('aria-label', 'Find in note');
  pop.hidden = true;
  const field = document.createElement('label');
  field.className = 'osprey-search-field';
  const input = document.createElement('input');
  input.type = 'search';
  input.placeholder = 'Find in note';
  input.setAttribute('aria-label', 'Find in note');
  field.append(icon('search'), input);
  const count = document.createElement('span');
  count.className = 'note-find-count';
  count.setAttribute('aria-live', 'polite');
  pop.append(field, count);
  document.body.append(pop);
  input.addEventListener('input', () => runFind(input.value));
  input.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    stepMatch(event.shiftKey ? -1 : 1);
  });
  // popover.js closes it (Escape, an outside press); the marks go with it.
  new MutationObserver(() => {
    if (pop.hidden) mark([], -1);
  }).observe(pop, { attributes: true, attributeFilter: ['hidden'] });
  finder = { pop, input, count };
  return finder;
}

/**
 * Find in note: open the search field under `button`, or close it. Opened
 * again with text in it, it marks that text in `note` at once.
 * @param {HTMLElement} button
 * @param {HTMLElement} note
 */
export function openFind(button, note) {
  const { pop, input } = findPopover();
  search = { note, ranges: [], current: -1 };
  togglePopover(button, pop, { place: 'below-end' });
  if (!pop.hidden && input.value) runFind(input.value);
}

/**
 * Copy as Markdown: the note's source on the clipboard.
 * @param {any} a the markdown artifact
 * @returns {Promise<boolean>} whether it was copied
 */
export async function copyMarkdown(a) {
  try {
    const resp = await fetch(fileUrl(a));
    return resp.ok && copyText(await resp.text());
  } catch {
    return false;
  }
}
