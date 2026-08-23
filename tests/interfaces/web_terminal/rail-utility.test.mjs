/**
 * OSPREY Web Terminal — rail utility cluster (Documentation + Feedback).
 *
 * Four things are pinned here, because each of them is a defect the rail has
 * already shipped once before:
 *   1. The cluster is a SIBLING of the entry <nav>. panel-rail.js's
 *      createRail() calls railEl.replaceChildren(), so a control parked inside
 *      the nav silently disappears the first time a panel renders.
 *   2. The Documentation anchor carries rel="noopener" and gets its href from
 *      the deployment config, never from hardcoded markup.
 *   3. terminal.css ships BOTH orientations. The top rail is a 40px strip; a
 *      62x52 button with no override spills out of it over the workspace.
 *   4. Each control is NAMED in the cell, and the visible name is inside the
 *      accessible one. A mark on its own made the operator hover to learn
 *      what it opens, and an aria-label that drops the visible word takes the
 *      control away from speech input (WCAG 2.5.3).
 *
 * Geometry itself is not assertable here — happy-dom has no layout engine, so
 * the real boxes are measured in the Playwright suite. What this file guards
 * is that the rules and the markup exist and say the right thing.
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
const STATIC_DIR = join(
  import.meta.dirname,
  '../../../src/osprey/interfaces/web_terminal/static'
);
const indexHtml = readFileSync(join(STATIC_DIR, 'index.html'), 'utf8');
const terminalCss = readFileSync(join(STATIC_DIR, 'css/terminal.css'), 'utf8');

/**
 * The `<div class="panel-rail-region">…</div>` subtree, verbatim from the
 * shipped template, by counting <div> depth from its opening tag.
 *
 * Taking the fixture from the real file rather than retyping it is the point:
 * a test that hand-rolls its own markup passes happily while the template it
 * claims to describe drifts away underneath it.
 *
 * @returns {string}
 */
function railRegionMarkup() {
  const open = '<div class="panel-rail-region">';
  const start = indexHtml.indexOf(open);
  expect(start, 'index.html has no .panel-rail-region').toBeGreaterThan(-1);
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
 * Parse the rail region into a detached element (no <body> mutation).
 * @returns {HTMLElement}
 */
function parseRailRegion() {
  const host = document.createElement('div');
  host.innerHTML = railRegionMarkup();
  return qs(host, '.panel-rail-region');
}

/**
 * The declaration block of the first CSS rule whose selector list ends with
 * `selector`.
 * @param {string} selector
 * @returns {string}
 */
function ruleBody(selector) {
  const start = terminalCss.indexOf(`${selector} {`);
  expect(start, `terminal.css has no rule for \`${selector}\``).toBeGreaterThan(-1);
  const open = terminalCss.indexOf('{', start);
  const close = terminalCss.indexOf('}', open);
  return terminalCss.slice(open + 1, close);
}

/** @returns {HTMLAnchorElement} */
function docsLink() {
  return qs(document, '#panel-docs-link', HTMLAnchorElement);
}

/** @returns {HTMLButtonElement} */
function feedbackButton() {
  return qs(document, '#panel-feedback-btn', HTMLButtonElement);
}

describe('utility cluster markup', () => {
  test('is a sibling of the rail <nav>, not a rail entry', () => {
    const region = parseRailRegion();
    const cluster = qs(region, '#panel-utility');

    expect(cluster.parentElement).toBe(region);
    expect(cluster.classList.contains('panel-utility')).toBe(true);
    // The two ways it could wrongly become an entry: nested in the nav that
    // createRail() wipes, or wearing the entry class the rail renderer owns.
    expect(region.querySelector('#panel-rail #panel-utility')).toBeNull();
    expect(cluster.querySelector('.panel-rail-button')).toBeNull();
  });

  test('sits after the entry list and the "+" cell', () => {
    const region = parseRailRegion();
    const children = Array.from(region.children);
    const ids = children.map((child) => child.id);

    expect(ids).toEqual(['panel-rail', 'panel-add', 'panel-utility']);
    expect(children[children.length - 1].id).toBe('panel-utility');
  });

  test('exposes a Documentation anchor that is safe to open in a new tab', () => {
    const cluster = qs(parseRailRegion(), '#panel-utility');
    const link = qs(cluster, '#panel-docs-link', HTMLAnchorElement);

    expect(link.tagName).toBe('A');
    expect(link.getAttribute('target')).toBe('_blank');
    expect(link.getAttribute('rel')).toContain('noopener');
    expect(link.getAttribute('aria-label')).toBe('Docs');
    // The longer phrase the label abbreviates stays reachable as the tooltip.
    expect(link.getAttribute('title')).toBe('Documentation');
    // The mark is decorative; the accessible name is the aria-label above.
    expect(qs(link, '.panel-utility-icon').getAttribute('aria-hidden')).toBe('true');
  });

  test('ships the Documentation anchor href-less and hidden', () => {
    const link = qs(parseRailRegion(), '#panel-docs-link', HTMLAnchorElement);

    // The target is per-deployment (web.docs_url). Hardcoding one here would
    // send an air-gapped control room to an unreachable public site.
    expect(link.hasAttribute('href')).toBe(false);
    expect(link.hasAttribute('hidden')).toBe(true);
  });

  test('exposes a Feedback button with an accessible name', () => {
    const cluster = qs(parseRailRegion(), '#panel-utility');
    const button = qs(cluster, '#panel-feedback-btn', HTMLButtonElement);

    expect(button.getAttribute('type')).toBe('button');
    expect(button.getAttribute('aria-label')).toBe('Feedback');
    expect(button.getAttribute('title')).toBe('Send feedback');
    expect(qs(button, '.panel-utility-icon').getAttribute('aria-hidden')).toBe('true');
  });

  test('names both controls in the cell, not just in the tooltip', () => {
    // The marks alone made the operator hover to find out what they open —
    // every other cell in the column carries its name.
    const cluster = qs(parseRailRegion(), '#panel-utility');

    for (const [id, label] of [
      ['#panel-docs-link', 'Docs'],
      ['#panel-feedback-btn', 'Feedback'],
    ]) {
      expect(qs(cluster, `${id} .panel-utility-label`).textContent).toBe(label);
    }
  });

  test('keeps the visible label inside the accessible name', () => {
    // WCAG 2.5.3 (Label in Name): a speech-input user says what they see, so
    // an aria-label that drops the visible word ("Documentation" over a cell
    // reading DOCS) makes the control unaddressable by voice.
    const cluster = qs(parseRailRegion(), '#panel-utility');

    for (const id of ['#panel-docs-link', '#panel-feedback-btn']) {
      const control = qs(cluster, id);
      const visible = qs(control, '.panel-utility-label').textContent ?? '';
      const name = control.getAttribute('aria-label') ?? '';
      expect(name.toLowerCase()).toContain(visible.toLowerCase());
    }
  });
});

describe('utility cluster styling', () => {
  test('pins the cluster to the far end of the left column', () => {
    const body = ruleBody('.panel-utility');

    // .panel-rail has no flex:1, so the free space has to be absorbed here.
    expect(body).toMatch(/margin-top:\s*auto;/);
  });

  test('detaches the cluster from the panel-tab group', () => {
    const body = ruleBody('.panel-utility');

    expect(body).toMatch(/border-top:\s*1px solid var\(--border-default\)/);
  });

  test('mirrors the .panel-add-btn cell geometry', () => {
    const body = ruleBody('.panel-utility-btn');

    expect(body).toMatch(/width:\s*62px;/);
    expect(body).toMatch(/height:\s*52px;/);
    expect(body).toMatch(/border-radius:\s*var\(--radius-xs\);/);
  });

  test('stacks the mark over the label like a panel entry', () => {
    // Without the column direction the label lands beside an 18px mark in a
    // 62px cell and ellipsises away to nothing.
    const body = ruleBody('.panel-utility-btn');

    expect(body).toMatch(/flex-direction:\s*column;/);
  });

  test('keeps the label on one line inside the 62px cell', () => {
    const body = ruleBody('.panel-utility-label');

    expect(body).toMatch(/white-space:\s*nowrap;/);
    expect(body).toMatch(/text-overflow:\s*ellipsis;/);
    // BUILTIN_PANEL_LABELS ships its labels pre-upper-cased; these are
    // authored as words so the accessible name reads as words, which leaves
    // the casing to the stylesheet.
    expect(body).toMatch(/text-transform:\s*uppercase;/);
  });

  test('lets the hidden attribute win over the button display rule', () => {
    // `[hidden] { display: none }` is a UA rule and loses to any class
    // selector, so a docs anchor hidden by initRailUtility would still paint.
    expect(ruleBody('.panel-utility-btn[hidden]')).toMatch(/display:\s*none;/);
  });

  test('re-pins the cluster to the right end under a top rail', () => {
    const body = ruleBody('html[data-rail-position="top"] .panel-utility');

    expect(body).toMatch(/margin-left:\s*auto;/);
    expect(body).toMatch(/margin-top:\s*0;/);
    expect(body).toMatch(/height:\s*100%;/);
  });

  test('reshapes the buttons to fit the 40px top strip', () => {
    const body = ruleBody('html[data-rail-position="top"] .panel-utility-btn');

    // The strip is 40px tall and does not clip: a 52px button spills over
    // the dockview area below it.
    expect(body).toMatch(/height:\s*100%;/);
    expect(body).toMatch(/border-radius:\s*0;/);
    // ...and the stack turns the same corner the panel pills do, so the label
    // sits beside the mark in a cell wide enough to hold it.
    expect(body).toMatch(/flex-direction:\s*row;/);
    expect(body).toMatch(/width:\s*auto;/);
  });

  test('gives both controls a drawn SVG mark, not a font glyph', () => {
    const cluster = qs(parseRailRegion(), '#panel-utility');

    for (const id of ['#panel-docs-link', '#panel-feedback-btn']) {
      const svg = qs(cluster, `${id} .panel-utility-icon svg[viewBox="0 0 16 16"]`);
      expect(svg.getAttribute('stroke')).toBe('currentColor');
    }
  });
});

describe('initRailUtility', () => {
  beforeEach(() => {
    document.body.innerHTML = railRegionMarkup();
  });

  test('points the Documentation anchor at the configured docs site', () => {
    initRailUtility({ docs_url: 'https://docs.example.org/osprey' });

    expect(docsLink().getAttribute('href')).toBe('https://docs.example.org/osprey');
    expect(docsLink().hidden).toBe(false);
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

  test('no-ops on a page without the rail', () => {
    document.body.innerHTML = '';

    expect(() => initRailUtility({ docs_url: 'https://docs.example.org/' })).not.toThrow();
  });
});

describe('onFeedbackClick', () => {
  beforeEach(() => {
    document.body.innerHTML = railRegionMarkup();
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

  test('returns an unsubscribe that stops further dispatches', () => {
    let calls = 0;
    const off = onFeedbackClick(() => {
      calls += 1;
    });

    off();
    feedbackButton().click();

    expect(calls).toBe(0);
  });

  test('no-ops on a page without the rail', () => {
    document.body.innerHTML = '';

    expect(() => onFeedbackClick(() => {})()).not.toThrow();
  });
});
