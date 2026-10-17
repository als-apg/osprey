// @ts-check
/**
 * Unit tests for the rail's ⋯ More menu and the Docs/Feedback hide gesture
 * (rail-more-menu.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/rail-more-menu.test.mjs
 *
 * What is pinned:
 *   - the rows ⋯ More offers follow the page: "Show Docs" / "Show Feedback"
 *     only for a control that is hidden AND exists here, System only where the
 *     deployment serves it, its pin only while it is off the rail;
 *   - each row runs the verb its owner exposes (the tour, the activity log,
 *     openPanelOffRail, setPanelVisibility, setRailUtilityVisible) and closes first;
 *   - the popover grammar of the "+" menu: aria-expanded, outside click,
 *     Escape returning focus, arrow keys;
 *   - right-click on Docs or Feedback offers one "Hide" row, and a hide that
 *     landed is offered back through the Undo notice while a refused one is not.
 *
 * The verbs' owners and the deployment context are mocked at their module
 * seams; the context-menu presenter and the Undo notice are real, so the DOM
 * this suite reads is the one an operator would see.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const seams = vi.hoisted(() => ({
  openPanelOffRail: vi.fn(),
  setPanelVisibility: vi.fn(),
  setRailUtilityVisible: vi.fn(async () => true),
  startTour: vi.fn(),
  systemHealthAvailable: true,
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js', () => ({
  openPanelOffRail: seams.openPanelOffRail,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-commands.js', () => ({
  setPanelVisibility: seams.setPanelVisibility,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/bar-customize.js', () => ({
  setRailUtilityVisible: seams.setRailUtilityVisible,
}));
// bar-sync.js adopts the layout document at import time, which needs a server;
// the one fact read from it here is the deployment context's SYSTEM flag.
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/bar-sync.js', () => ({
  deploymentContext: () => ({ systemHealthAvailable: seams.systemHealthAvailable }),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/tour.js', () => ({
  startTour: seams.startTour,
}));

import {
  initRailMoreMenu,
  moreMenuRows,
  SYSTEM_PANEL_ID,
} from '../../../src/osprey/interfaces/web_terminal/static/js/rail-more-menu.js';
import { activityLogUrl } from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-link.js';
import { closeUndoNotice } from '../../../src/osprey/interfaces/web_terminal/static/js/undo-notice.js';

const STATIC_DIR = join(import.meta.dirname, '../../../src/osprey/interfaces/web_terminal/static');
const indexHtml = readFileSync(join(STATIC_DIR, 'index.html'), 'utf8');

const KEY = '11111111-2222-3333-4444-555555555555';

/** The shipped rail region, verbatim, so the suite drives the real markup. */
function railRegionMarkup() {
  const open = '<div class="panel-rail-region">';
  const start = indexHtml.indexOf(open);
  const tags = /<div\b[^>]*>|<\/div\s*>/g;
  tags.lastIndex = start;
  let depth = 0;
  /** @type {RegExpExecArray|null} */
  let match = null;
  while ((match = tags.exec(indexHtml)) !== null) {
    depth += match[0].startsWith('</') ? -1 : 1;
    if (depth === 0) return indexHtml.slice(start, tags.lastIndex);
  }
  throw new Error('unbalanced <div> nesting inside .panel-rail-region');
}

/**
 * Mount the rail, stamp the flags the server would on <html>, and set what the
 * deployment serves.
 * @param {{system?: boolean, docsUrl?: boolean, hidden?: ('docs'|'feedback')[],
 *   systemOnRail?: boolean}} [state]
 */
function mount({ system = true, docsUrl = true, hidden = [], systemOnRail = false } = {}) {
  const html = document.documentElement;
  seams.systemHealthAvailable = system;
  delete html.dataset.railDocs;
  delete html.dataset.railFeedback;
  if (hidden.includes('docs')) html.dataset.railDocs = 'hidden';
  if (hidden.includes('feedback')) html.dataset.railFeedback = 'hidden';
  document.body.innerHTML = railRegionMarkup();
  const docs = /** @type {HTMLAnchorElement} */ (document.getElementById('panel-docs-link'));
  if (docsUrl) {
    docs.setAttribute('href', 'https://docs.example.org/');
    docs.hidden = false;
  }
  if (systemOnRail) {
    const entry = document.createElement('button');
    entry.className = 'panel-rail-button';
    entry.dataset.panelId = SYSTEM_PANEL_ID;
    document.getElementById('panel-rail')?.appendChild(entry);
  }
}

/** @param {string} id @returns {HTMLElement} */
function byId(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`expected #${id}`);
  return el;
}

const button = () => byId('rail-more-btn');
const menu = () => byId('rail-more-menu');
const rowKeys = () =>
  Array.from(menu().querySelectorAll('.rail-more-row'), (r) => /** @type {HTMLElement} */ (r).dataset.row);

/** @param {string} key @param {string} [part] */
function row(key, part = '.rail-more-item') {
  const el = menu().querySelector(`.rail-more-row[data-row="${key}"] ${part}`);
  if (!(el instanceof HTMLElement)) throw new Error(`no ${part} on row ${key}`);
  return el;
}

/** @param {string} key */
function press(key, init = {}) {
  document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, ...init }));
}

/** @param {HTMLElement} node */
function rightClick(node) {
  const e = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 10, clientY: 10 });
  node.dispatchEvent(e);
  return e;
}

const contextLabels = () =>
  Array.from(document.querySelectorAll('.rail-context-menu .rail-context-label'), (el) => el.textContent);

beforeEach(() => {
  vi.clearAllMocks();
  seams.setRailUtilityVisible.mockImplementation(async () => true);
});

afterEach(() => {
  press('Escape');
  closeUndoNotice();
  document.querySelector('.rail-context-menu')?.remove();
  document.body.innerHTML = '';
});

describe('moreMenuRows', () => {
  test('offers System, the tour and the activity log by default', () => {
    mount();
    expect(moreMenuRows(() => KEY).map((r) => r.key)).toEqual(['system', 'tour', 'activity']);
  });

  test('offers a hidden control back, Docs before Feedback', () => {
    mount({ hidden: ['feedback', 'docs'] });
    expect(moreMenuRows(() => KEY).map((r) => r.label)).toEqual([
      'Show Docs',
      'Show Feedback',
      'System',
      'Take tour',
      'Activity log',
    ]);
  });

  test('does not offer Docs back where the deployment has no docs link', () => {
    mount({ docsUrl: false, hidden: ['docs'] });
    expect(moreMenuRows(() => KEY).map((r) => r.key)).not.toContain('show-docs');
  });

  test('leaves System out where the deployment does not serve it', () => {
    mount({ system: false });
    expect(moreMenuRows(() => KEY).map((r) => r.key)).toEqual(['tour', 'activity']);
  });

  test('carries the pin only while System is off the rail', () => {
    mount();
    expect(moreMenuRows(() => KEY).find((r) => r.key === 'system')?.pin).toBeTypeOf('function');
    mount({ systemOnRail: true });
    expect(moreMenuRows(() => KEY).find((r) => r.key === 'system')?.pin).toBeUndefined();
  });
});

describe('the ⋯ More popover', () => {
  test('opens on a click, says so, and focuses the first row', () => {
    mount();
    initRailMoreMenu({ getSessionId: () => KEY });

    button().click();

    expect(menu().classList.contains('open')).toBe(true);
    expect(button().getAttribute('aria-expanded')).toBe('true');
    expect(rowKeys()).toEqual(['system', 'tour', 'activity']);
    expect(document.activeElement).toBe(row('system'));
    for (const item of menu().querySelectorAll('button')) {
      expect(item.getAttribute('role')).toBe('menuitem');
    }
  });

  test('is rebuilt on every open, so it follows the flags', () => {
    mount();
    initRailMoreMenu({ getSessionId: () => KEY });
    button().click();
    button().click();
    document.documentElement.dataset.railFeedback = 'hidden';

    button().click();

    expect(rowKeys()[0]).toBe('show-feedback');
  });

  test('closes on an outside click', () => {
    mount();
    initRailMoreMenu({ getSessionId: () => KEY });
    button().click();

    document.body.click();

    expect(menu().classList.contains('open')).toBe(false);
    expect(button().getAttribute('aria-expanded')).toBe('false');
  });

  test('closes on Escape and hands focus back to the button', () => {
    mount();
    initRailMoreMenu({ getSessionId: () => KEY });
    button().click();

    press('Escape');

    expect(menu().classList.contains('open')).toBe(false);
    expect(document.activeElement).toBe(button());
  });

  test('moves between rows with the arrow keys, wrapping at both ends', () => {
    mount();
    initRailMoreMenu({ getSessionId: () => KEY });
    button().click();

    press('ArrowDown');
    expect(document.activeElement).toBe(row('system', '.rail-more-pin'));
    press('ArrowUp');
    press('ArrowUp');
    expect(document.activeElement).toBe(row('activity'));
  });

  test('each row closes the menu, then runs its verb', () => {
    mount({ hidden: ['docs'] });
    const open = vi.fn();
    vi.stubGlobal('open', open);
    initRailMoreMenu({ getSessionId: () => KEY });

    button().click();
    row('show-docs').click();
    expect(menu().classList.contains('open')).toBe(false);
    expect(seams.setRailUtilityVisible).toHaveBeenCalledWith('docs', true);

    button().click();
    row('system').click();
    expect(seams.openPanelOffRail).toHaveBeenCalledWith(SYSTEM_PANEL_ID);

    button().click();
    row('system', '.rail-more-pin').click();
    expect(seams.setPanelVisibility).toHaveBeenCalledWith(SYSTEM_PANEL_ID, true);

    button().click();
    row('tour').click();
    expect(seams.startTour).toHaveBeenCalledTimes(1);

    button().click();
    row('activity').click();
    expect(open).toHaveBeenCalledWith(activityLogUrl(KEY), '_blank', 'noopener');
    vi.unstubAllGlobals();
  });

  test('right-click on System offers "Pin to rail" while it is off the rail', () => {
    mount();
    initRailMoreMenu({ getSessionId: () => KEY });
    button().click();

    const e = rightClick(row('system'));

    expect(e.defaultPrevented).toBe(true);
    expect(contextLabels()).toEqual(['Pin to rail']);
    /** @type {HTMLElement} */ (document.querySelector('.rail-context-item')).click();
    expect(seams.setPanelVisibility).toHaveBeenCalledWith(SYSTEM_PANEL_ID, true);
  });

  test('a System row already on the rail has neither pin nor pin menu', () => {
    mount({ systemOnRail: true });
    initRailMoreMenu({ getSessionId: () => KEY });
    button().click();

    expect(menu().querySelector('.rail-more-pin')).toBeNull();
    expect(rightClick(row('system')).defaultPrevented).toBe(false);
    expect(document.querySelector('.rail-context-menu')).toBeNull();
  });
});

describe('hiding Docs and Feedback', () => {
  test.each([
    ['panel-docs-link', 'docs', 'Docs'],
    ['panel-feedback-btn', 'feedback', 'Feedback'],
  ])('right-click on #%s offers one Hide row', async (id, name, label) => {
    mount();
    initRailMoreMenu({ getSessionId: () => KEY });

    const e = rightClick(byId(id));
    expect(e.defaultPrevented).toBe(true);
    expect(contextLabels()).toEqual([`Hide ${label}`]);

    /** @type {HTMLElement} */ (document.querySelector('.rail-context-item')).click();
    await vi.waitFor(() => expect(document.getElementById('undo-notice')).not.toBeNull());

    expect(seams.setRailUtilityVisible).toHaveBeenCalledWith(name, false);
    expect(document.querySelector('.undo-notice-text')?.textContent).toBe(`${label} hidden`);
    /** @type {HTMLElement} */ (document.querySelector('.undo-notice-btn')).click();
    expect(seams.setRailUtilityVisible).toHaveBeenLastCalledWith(name, true);
  });

  test('the keyboard opens the same menu (menu key, Shift+F10)', () => {
    mount();
    initRailMoreMenu({ getSessionId: () => KEY });
    const feedback = byId('panel-feedback-btn');

    const key = new KeyboardEvent('keydown', { key: 'F10', shiftKey: true, bubbles: true, cancelable: true });
    feedback.dispatchEvent(key);

    expect(key.defaultPrevented).toBe(true);
    expect(contextLabels()).toEqual(['Hide Feedback']);
  });

  test('a refused hide shows no Undo notice of its own', async () => {
    seams.setRailUtilityVisible.mockImplementation(async () => false);
    mount();
    initRailMoreMenu({ getSessionId: () => KEY });

    rightClick(byId('panel-docs-link'));
    /** @type {HTMLElement} */ (document.querySelector('.rail-context-item')).click();
    await vi.waitFor(() => expect(seams.setRailUtilityVisible).toHaveBeenCalled());
    await Promise.resolve();

    expect(document.getElementById('undo-notice')).toBeNull();
  });

  test('⋯ More has no hide gesture: it is the way back', () => {
    mount();
    initRailMoreMenu({ getSessionId: () => KEY });

    expect(rightClick(button()).defaultPrevented).toBe(false);
    expect(document.querySelector('.rail-context-menu')).toBeNull();
  });
});
