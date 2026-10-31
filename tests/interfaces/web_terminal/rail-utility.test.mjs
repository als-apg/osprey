/**
 * OSPREY Web Terminal — the sidebar footer (Docs, Feedback, ☰ More).
 *
 * Four things are pinned here, because each of them is a defect the panel
 * column has already shipped once before:
 *   1. The footer is NOT part of `#panel-rail`. panel-rail.js's createRail()
 *      re-places that list's rows on every render, so a control parked inside
 *      it would be moved or dropped the first time a panel renders.
 *   2. The Docs anchor carries rel="noopener" and gets its href from the
 *      deployment config (initRailUtility), never from hardcoded markup.
 *   3. A Docs anchor initRailUtility hid stays hidden: the row's display rule
 *      must not beat the `hidden` attribute.
 *   4. Each control is NAMED in the row, and the visible name is inside the
 *      accessible one. A mark on its own made the operator hover to learn
 *      what it opens, and an aria-label that drops the visible word takes the
 *      control away from speech input (WCAG 2.5.3).
 *
 * Geometry is not assertable here — happy-dom has no layout engine. What this
 * file guards is the markup, and the stylesheet rule no layout measurement
 * can see.
 *
 *   npx vitest run tests/interfaces/web_terminal/rail-utility.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  initRailUtility,
  onFeedbackClick,
} from '../../../src/osprey/interfaces/web_terminal/static/js/rail-utility.js';
import { qs } from '../_support/dom.mjs';

// `import.meta.dirname` is a plain string, so it sidesteps happy-dom's
// override of the global URL (which breaks fileURLToPath(new URL(...)) here).
const STATIC_DIR = join(import.meta.dirname, '../../../src/osprey/interfaces/web_terminal/static');
const indexHtml = readFileSync(join(STATIC_DIR, 'index.html'), 'utf8');
const primitivesCss = readFileSync(
  join(STATIC_DIR, '../../design_system/static/css/primitives.css'),
  'utf8'
);

/** The three footer controls, top to bottom, with their visible names. */
const FOOTER_CONTROLS = [
  ['#panel-docs-link', 'Docs'],
  ['#panel-feedback-btn', 'Feedback'],
  ['#more-btn', 'More'],
];

/**
 * The `<nav … id="sidebar">…</nav>` subtree, verbatim from the shipped
 * template.
 *
 * Taking the fixture from the real file rather than retyping it is the point:
 * a test that hand-rolls its own markup passes happily while the template it
 * claims to describe drifts away underneath it.
 *
 * @returns {string}
 */
function sidebarMarkup() {
  const start = indexHtml.indexOf('<nav class="sidebar" id="sidebar"');
  expect(start, 'index.html has no nav#sidebar').toBeGreaterThan(-1);
  const end = indexHtml.indexOf('</nav>', start);
  expect(end, 'nav#sidebar is never closed').toBeGreaterThan(start);
  return indexHtml.slice(start, end + '</nav>'.length);
}

/**
 * Parse the sidebar into a detached element (no <body> mutation).
 * @returns {HTMLElement}
 */
function parseSidebar() {
  const host = document.createElement('div');
  host.innerHTML = sidebarMarkup();
  return qs(host, '#sidebar');
}

/** @returns {HTMLElement} the parsed `.sidebar-foot` */
function parseFooter() {
  return qs(parseSidebar(), '.sidebar-foot');
}

/**
 * The declaration block of the first primitives.css rule whose selector list
 * names `selector`.
 * @param {string} selector
 * @returns {string}
 */
function ruleBody(selector) {
  const start = primitivesCss.indexOf(selector);
  expect(start, `primitives.css has no rule for \`${selector}\``).toBeGreaterThan(-1);
  const open = primitivesCss.indexOf('{', start);
  expect(primitivesCss.slice(start, open)).not.toMatch(/[;}]/);
  const close = primitivesCss.indexOf('}', open);
  return primitivesCss.slice(open + 1, close);
}

/** @returns {HTMLAnchorElement} */
function docsLink() {
  return qs(document, '#panel-docs-link', HTMLAnchorElement);
}

/** @returns {HTMLButtonElement} */
function feedbackButton() {
  return qs(document, '#panel-feedback-btn', HTMLButtonElement);
}

describe('sidebar footer markup', () => {
  test('is a child of the sidebar after the panel rows, not a panel row', () => {
    const sidebar = parseSidebar();
    const footer = qs(sidebar, '.sidebar-foot');

    expect(footer.parentElement).toBe(sidebar);
    expect(footer.previousElementSibling?.id).toBe('panel-rail');
    expect(sidebar.lastElementChild).toBe(footer);
    // The two ways it could wrongly become a panel row: nested in the list
    // that createRail() re-places, or wearing the row class the renderer owns.
    expect(sidebar.querySelector('#panel-rail .sidebar-foot')).toBeNull();
    expect(footer.querySelector('.panel-rail-button')).toBeNull();
  });

  test('holds Docs, Feedback, Sign out and More, in that order', () => {
    const footer = parseFooter();

    // The raw template still carries the Jinja-gated Sign out row; a render
    // without a signed-in user drops it and the other three keep their order.
    expect(Array.from(footer.children, (child) => child.id)).toEqual([
      'panel-docs-link',
      'panel-feedback-btn',
      'logout-btn',
      'more-btn',
    ]);
  });

  test('exposes a Docs anchor that is safe to open in a new tab', () => {
    const link = qs(parseFooter(), '#panel-docs-link', HTMLAnchorElement);

    expect(link.tagName).toBe('A');
    expect(link.getAttribute('target')).toBe('_blank');
    expect(link.getAttribute('rel')).toContain('noopener');
    expect(link.getAttribute('aria-label')).toBe('Docs');
    // The longer phrase the label abbreviates stays reachable as the tooltip.
    expect(link.getAttribute('title')).toBe('Documentation');
    // The mark is decorative; the accessible name is the aria-label above.
    expect(qs(link, '.osprey-nav-icon').getAttribute('aria-hidden')).toBe('true');
  });

  test('ships the Docs anchor href-less and hidden', () => {
    const link = qs(parseFooter(), '#panel-docs-link', HTMLAnchorElement);

    // The target is per-deployment (web.docs_url). Hardcoding one here would
    // send an air-gapped control room to an unreachable public site.
    expect(link.hasAttribute('href')).toBe(false);
    expect(link.hasAttribute('hidden')).toBe(true);
  });

  test('exposes a Feedback button with an accessible name', () => {
    const button = qs(parseFooter(), '#panel-feedback-btn', HTMLButtonElement);

    expect(button.getAttribute('type')).toBe('button');
    expect(button.getAttribute('aria-label')).toBe('Feedback');
    expect(button.getAttribute('title')).toBe('Send feedback');
    expect(qs(button, '.osprey-nav-icon').getAttribute('aria-hidden')).toBe('true');
  });

  test('exposes a More button that announces the menu it opens', () => {
    const button = qs(parseFooter(), '#more-btn', HTMLButtonElement);

    expect(button.getAttribute('type')).toBe('button');
    expect(button.getAttribute('aria-label')).toBe('More');
    expect(button.getAttribute('aria-haspopup')).toBe('menu');
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(button.getAttribute('aria-controls')).toBe('more-menu');
    expect(qs(button, '.osprey-nav-icon').getAttribute('aria-hidden')).toBe('true');
  });

  test('the More menu is a body-level menu popover, outside the sidebar', () => {
    // popover.js places body-level popovers; one inside the sidebar would be
    // clipped by the sidebar's own scroll box.
    const open = /<div\b[^>]*\bid="more-menu"[^>]*>/.exec(indexHtml);
    expect(open, 'index.html has no #more-menu').not.toBeNull();
    const tag = /** @type {RegExpExecArray} */ (open)[0];
    expect(tag).toMatch(/\brole="menu"/);
    expect(tag).toMatch(/\bclass="[^"]*\bosprey-popover\b/);
    expect(/** @type {RegExpExecArray} */ (open).index).toBeGreaterThan(
      indexHtml.indexOf('</nav>', indexHtml.indexOf('id="sidebar"'))
    );
  });

  test('names every control in the row, not just in the tooltip', () => {
    // The marks alone made the operator hover to find out what they open —
    // every other row in the sidebar carries its name.
    const footer = parseFooter();

    for (const [id, label] of FOOTER_CONTROLS) {
      expect(qs(footer, `${id} .osprey-nav-label`).textContent).toBe(label);
    }
  });

  test('keeps the visible label inside the accessible name', () => {
    // WCAG 2.5.3 (Label in Name): a speech-input user says what they see, so
    // an aria-label that drops the visible word makes the control
    // unaddressable by voice.
    const footer = parseFooter();

    for (const [id] of FOOTER_CONTROLS) {
      const control = qs(footer, id);
      const visible = qs(control, '.osprey-nav-label').textContent ?? '';
      const name = control.getAttribute('aria-label') ?? '';
      expect(name.toLowerCase()).toContain(visible.toLowerCase());
    }
  });
});

describe('initRailUtility', () => {
  beforeEach(() => {
    document.body.innerHTML = sidebarMarkup();
  });

  test('points the Docs anchor at the configured docs site and shows it', () => {
    initRailUtility({ docs_url: 'https://docs.example.org/osprey' });

    expect(docsLink().getAttribute('href')).toBe('https://docs.example.org/osprey');
    expect(docsLink().hidden).toBe(false);
  });

  test('keeps a relative docs path exactly as configured', () => {
    initRailUtility({ docs_url: '/docs/osprey/' });

    expect(docsLink().getAttribute('href')).toBe('/docs/osprey/');
  });

  test('trims a padded config value', () => {
    initRailUtility({ docs_url: '  https://docs.example.org/osprey  ' });

    expect(docsLink().getAttribute('href')).toBe('https://docs.example.org/osprey');
  });

  test.each([
    ['an empty string', { docs_url: '' }],
    ['whitespace only', { docs_url: '   ' }],
    ['a missing key', {}],
    ['a non-string value', { docs_url: 42 }],
    ['a null payload', null],
    ['an absent payload', undefined],
  ])('hides the anchor when the payload carries %s', (_label, payload) => {
    initRailUtility(payload);

    expect(docsLink().hidden).toBe(true);
    expect(docsLink().hasAttribute('href')).toBe(false);
  });

  test('re-hides an anchor that a later payload blanks', () => {
    initRailUtility({ docs_url: 'https://docs.example.org/osprey' });
    initRailUtility({ docs_url: '' });

    expect(docsLink().hidden).toBe(true);
    expect(docsLink().hasAttribute('href')).toBe(false);
  });

  test('a hidden anchor does not paint: the row display rule yields to [hidden]', () => {
    // `[hidden] { display: none }` is a UA rule and loses to any class
    // selector, so a Docs anchor hidden by initRailUtility would still paint.
    expect(ruleBody('.osprey-nav-row[hidden]')).toMatch(/display:\s*none;/);
  });

  test('no-ops on a page without the sidebar', () => {
    document.body.innerHTML = '';

    expect(() => initRailUtility({ docs_url: 'https://docs.example.org/' })).not.toThrow();
  });
});

describe('onFeedbackClick', () => {
  beforeEach(() => {
    document.body.innerHTML = sidebarMarkup();
  });

  test('dispatches the callback on a click', () => {
    let calls = 0;
    onFeedbackClick(() => {
      calls += 1;
    });

    feedbackButton().click();

    expect(calls).toBe(1);
  });

  test('dispatches to every registered callback', () => {
    /** @type {string[]} */
    const seen = [];
    onFeedbackClick(() => seen.push('first'));
    onFeedbackClick(() => seen.push('second'));

    feedbackButton().click();

    expect(seen).toEqual(['first', 'second']);
  });

  test('suppresses the click default', () => {
    onFeedbackClick(() => {});
    const click = new MouseEvent('click', { bubbles: true, cancelable: true });

    feedbackButton().dispatchEvent(click);

    expect(click.defaultPrevented).toBe(true);
  });

  test('returns an unsubscribe that stops further dispatches', () => {
    let calls = 0;
    const off = onFeedbackClick(() => {
      calls += 1;
    });

    off();
    feedbackButton().click();

    expect(calls).toBe(0);
  });

  test('no-ops on a page without the sidebar', () => {
    document.body.innerHTML = '';

    expect(() => onFeedbackClick(() => {})()).not.toThrow();
  });
});
