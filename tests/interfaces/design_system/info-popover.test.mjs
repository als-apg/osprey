// @ts-check
/**
 * Unit tests for the info popover (design_system/static/js/info-popover.js):
 *   npx vitest run tests/interfaces/design_system/info-popover.test.mjs
 *
 * happy-dom does no layout; placement is popover.js's own concern and is
 * tested there. These tests cover what this module adds: the content, the
 * Copy buttons and the toggle on the anchor.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { openInfoPopover } from '/design-system/js/info-popover.js';
import { closeAllPopovers } from '/design-system/js/popover.js';

/** @param {string} id @returns {HTMLElement} */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

/** @returns {HTMLElement} the page's info popover */
const popover = () => /** @type {HTMLElement} */ (document.querySelector('.osprey-info-popover'));

const ROWS = [
  { label: 'Created', value: '2026-11-04 02:48' },
  { label: 'Size', value: '49.2 KB' },
  { label: 'Path', value: '/data/artifacts/tunes.html', copy: true },
];

/** @type {PropertyDescriptor | undefined} */
let clipboardDescriptor;

beforeEach(() => {
  closeAllPopovers();
  document.body.innerHTML = `
    <button id="info" type="button" aria-label="Artifact details">i</button>
    <button id="chip" type="button" class="osprey-chip">2 issues</button>
    <main id="outside">content</main>
  `;
  clipboardDescriptor = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
});

afterEach(() => {
  if (clipboardDescriptor) Object.defineProperty(navigator, 'clipboard', clipboardDescriptor);
  // @ts-expect-error - removing a test-only stub
  else delete navigator.clipboard;
  vi.useRealTimers();
});

describe('openInfoPopover', () => {
  test('shows the title, the text and the rows as a definition list', () => {
    openInfoPopover($('info'), {
      title: 'Betatron tunes',
      text: 'Tunes from the SR lattice.',
      rows: ROWS,
    });
    const pop = popover();
    expect(pop.hidden).toBe(false);
    expect(pop.getAttribute('role')).toBe('dialog');
    expect(pop.textContent).toContain('Betatron tunes');
    expect(pop.textContent).toContain('Tunes from the SR lattice.');
    const terms = Array.from(pop.querySelectorAll('dt'), (dt) => dt.textContent);
    const values = Array.from(pop.querySelectorAll('dd'), (dd) => dd.textContent);
    expect(terms).toEqual(['Created', 'Size', 'Path']);
    expect(values).toEqual(['2026-11-04 02:48', '49.2 KB', '/data/artifacts/tunes.html']);
    expect($('info').getAttribute('aria-expanded')).toBe('true');
  });

  test('only a row with copy gets a Copy button, and it copies the value', async () => {
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    openInfoPopover($('info'), { rows: ROWS });

    const buttons = popover().querySelectorAll('dd button');
    expect(buttons).toHaveLength(1);
    /** @type {HTMLButtonElement} */ (buttons[0]).click();
    await vi.waitFor(() => expect(writeText).toHaveBeenCalledWith('/data/artifacts/tunes.html'));
  });

  test('a second open on the same anchor closes it', () => {
    openInfoPopover($('info'), { title: 'Details' });
    openInfoPopover($('info'), { title: 'Details' });
    expect(popover().hidden).toBe(true);
    expect($('info').getAttribute('aria-expanded')).toBe('false');
  });

  test('an open on another anchor replaces the content', () => {
    openInfoPopover($('info'), { title: 'Artifact' });
    openInfoPopover($('chip'), {
      title: 'Bundle issues',
      rows: [{ label: 'kind', value: 'link' }],
    });
    expect(document.querySelectorAll('.osprey-info-popover')).toHaveLength(1);
    expect(popover().hidden).toBe(false);
    expect(popover().textContent).toContain('Bundle issues');
    expect(popover().textContent).not.toContain('Artifact');
    expect($('info').getAttribute('aria-expanded')).toBe('false');
    expect($('chip').getAttribute('aria-expanded')).toBe('true');
  });

  test('the returned close() closes it, and Escape does too', () => {
    const { close } = openInfoPopover($('info'), { text: 'One paragraph.' });
    close();
    expect(popover().hidden).toBe(true);

    openInfoPopover($('info'), { text: 'One paragraph.' });
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(popover().hidden).toBe(true);
  });
});

describe('a notice and links', () => {
  test('a row with onClick shows its value as a link that closes the popover, then runs', () => {
    const onClick = vi.fn(() => expect(popover().hidden).toBe(true));
    openInfoPopover($('info'), {
      rows: [
        { label: 'Session', value: 'Tune scan', onClick },
        { label: 'Size', value: '49.2 KB' },
      ],
    });
    const links = popover().querySelectorAll('.osprey-info-popover-link');
    expect(Array.from(links, (link) => link.textContent)).toEqual(['Tune scan']);
    /** @type {HTMLButtonElement} */ (links[0]).click();
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  test('the notice leads the card: its text, its links and its action', () => {
    const onClick = vi.fn();
    const onAction = vi.fn();
    openInfoPopover($('info'), {
      notice: {
        parts: ['From another session: "', { text: 'Tune scan', onClick }, '"'],
        actionLabel: 'Got it',
        onAction,
      },
      title: 'Betatron tunes',
    });
    const notice = /** @type {HTMLElement} */ (popover().firstElementChild);
    expect(notice.className).toBe('osprey-info-popover-notice');
    expect(notice.textContent).toBe('From another session: "Tune scan"Got it');
    const action = /** @type {HTMLButtonElement} */ (
      Array.from(notice.querySelectorAll('button')).find((b) => b.textContent === 'Got it')
    );
    action.click();
    expect(onAction).toHaveBeenCalledTimes(1);
    // The action leaves the card open: the caller sends what it shows next.
    expect(popover().hidden).toBe(false);
    /** @type {HTMLButtonElement} */ (notice.querySelector('.osprey-info-popover-link')).click();
    expect(onClick).toHaveBeenCalledTimes(1);
    expect(popover().hidden).toBe(true);
  });

  test('update shows new content in place while open, and nothing once closed', () => {
    const handle = openInfoPopover($('info'), {
      notice: { parts: ['Look'], actionLabel: 'Got it', onAction: () => {} },
      rows: [{ label: 'Session', value: 'Tune scan' }],
    });
    const action = /** @type {HTMLButtonElement} */ (popover().querySelector('.osprey-btn'));
    action.focus();
    expect(handle.update({ rows: [{ label: 'Session', value: 'Tune scan' }] })).toBe(true);
    expect(popover().querySelector('.osprey-info-popover-notice')).toBeNull();
    expect(popover().querySelector('dd')?.textContent).toBe('Tune scan');
    // Focus that was on the dropped button stays in the card.
    expect(popover().contains(document.activeElement)).toBe(true);

    handle.close();
    expect(handle.update({ title: 'Other' })).toBe(false);
    expect(popover().textContent).not.toContain('Other');
  });
});
