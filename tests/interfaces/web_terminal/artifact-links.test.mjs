/**
 * Unit tests for artifact-links.js: which artifact ids in an agent answer
 * become workspace controls, and what a click on one does.
 *
 *   npx vitest run tests/interfaces/web_terminal/artifact-links.test.mjs
 *
 * app-artifact-links.test.mjs covers app.js's wiring of these to the gallery.
 */

import { describe, expect, test, vi } from 'vitest';
import {
  linkArtifactIds,
  wireArtifactLinks,
} from '../../../src/osprey/interfaces/web_terminal/static/js/artifact-links.js';

/**
 * @param {string} html
 * @returns {HTMLElement}
 */
function body(html) {
  const el = document.createElement('div');
  el.innerHTML = html;
  return el;
}

/** @param {HTMLElement} el */
function ids(el) {
  return Array.from(el.querySelectorAll('.op-artifact-link')).map(
    (b) => /** @type {HTMLElement} */ (b).dataset.artifactId
  );
}

describe('linkArtifactIds', () => {
  test('turns each phrase into a control that reads as written', () => {
    const el = body('<p>Saved artifact_id: 28BFAA39E90E and artifact_id=d5c208e480af.</p>');
    linkArtifactIds(el);

    expect(ids(el)).toEqual(['28bfaa39e90e', 'd5c208e480af']);
    expect(el.textContent).toBe('Saved artifact_id: 28BFAA39E90E and artifact_id=d5c208e480af.');
  });

  test('links an id in an inline code span right after the lead-in, keeping the code', () => {
    const el = body('<p>Notebook artifact_id: <code>341ccd5efdcc</code> is ready</p>');
    linkArtifactIds(el);

    expect(ids(el)).toEqual(['341ccd5efdcc']);
    expect(el.querySelector('.op-artifact-link code')?.textContent).toBe('341ccd5efdcc');
    expect(el.textContent).toBe('Notebook artifact_id: 341ccd5efdcc is ready');
  });

  test.each([
    ['a code block', '<pre><code>artifact_id: 28bfaa39e90e</code></pre>'],
    ['a link', '<p><a href="#">artifact_id: 28bfaa39e90e</a></p>'],
    ['a button', '<p><button>artifact_id: 28bfaa39e90e</button></p>'],
    ['math', '<p><span class="katex">artifact_id: 28bfaa39e90e</span></p>'],
    ['a drawing', '<svg viewBox="0 0 9 9"><text>artifact_id: 28bfaa39e90e</text></svg>'],
    ['a code span holding more than an id', '<p>artifact_id: <code>28bfaa39e90e x</code></p>'],
    ['an id that is not hex', '<p>artifact_id: plot-of-orbit</p>'],
    ['an id shorter than the store mints', '<p>artifact_id: abc12</p>'],
    ['the bare word without an id', '<p>every artifact_id in the gallery</p>'],
  ])('leaves %s alone', (_what, html) => {
    const el = body(html);
    linkArtifactIds(el);
    expect(ids(el)).toEqual([]);
  });

  test('one pass links a plain phrase and a code-span id in the same text', () => {
    const el = body('<p>artifact_id: 28bfaa39e90e and artifact_id: <code>d5c208e480af</code></p>');
    linkArtifactIds(el);

    expect(ids(el)).toEqual(['28bfaa39e90e', 'd5c208e480af']);
  });

  test('linking a body again adds nothing', () => {
    const el = body('<p>artifact_id: 28bfaa39e90e and artifact_id: <code>d5c208e480af</code></p>');
    linkArtifactIds(el);
    const once = el.innerHTML;
    linkArtifactIds(el);

    expect(el.innerHTML).toBe(once);
  });
});

describe('wireArtifactLinks', () => {
  /**
   * @param {import('../../../src/osprey/interfaces/web_terminal/static/js/artifact-links.js').FocusOutcome} outcome - what the gallery answers
   */
  async function clickWith(outcome) {
    const el = body('<p>artifact_id: 28bfaa39e90e</p>');
    linkArtifactIds(el);
    const focusArtifact = vi.fn(async () => outcome);
    wireArtifactLinks(el, { focusArtifact });
    const b = /** @type {HTMLButtonElement} */ (el.querySelector('.op-artifact-link'));
    b.click();
    await Promise.resolve();
    await Promise.resolve();
    return { b, focusArtifact };
  }

  test('a click asks the gallery for that artifact', async () => {
    const { focusArtifact } = await clickWith('shown');
    expect(focusArtifact).toHaveBeenCalledWith('28bfaa39e90e');
  });

  test('an id the gallery does not hold marks the control, and a later success clears it', async () => {
    const { b } = await clickWith('missing');
    expect(b.dataset.missing).toBe('true');

    const shownAgain = await clickWith('shown');
    expect(shownAgain.b.dataset.missing).toBeUndefined();
  });

  test.each([
    ['missing', 'This artifact is not in the gallery'],
    ['expired', 'Your session has expired. Sign in again.'],
    ['refused', 'The server refused this request'],
    ['unreachable', 'The gallery could not be reached'],
  ])("a %s focus says so in the control's title", async (outcome, title) => {
    const { b } = await clickWith(/** @type {any} */ (outcome));
    expect(b.title).toBe(title);
  });

  test('only a missing artifact is struck through; a failure to ask is not', async () => {
    const { b } = await clickWith('expired');
    expect(b.dataset.missing).toBeUndefined();
    expect(b.dataset.unavailable).toBe('expired');
  });

  test("a success after a failure restores the control's title and state", async () => {
    const el = body('<p>artifact_id: 28bfaa39e90e</p>');
    linkArtifactIds(el);
    const outcomes = ['unreachable', 'shown'];
    wireArtifactLinks(el, { focusArtifact: async () => /** @type {any} */ (outcomes.shift()) });
    const b = /** @type {HTMLButtonElement} */ (el.querySelector('.op-artifact-link'));
    b.click();
    await vi.waitFor(() => expect(b.dataset.unavailable).toBe('unreachable'));
    b.click();
    await vi.waitFor(() => expect(b.dataset.unavailable).toBeUndefined());
    expect(b.title).toBe('Show this artifact in the workspace');
  });

  test('a click outside any control asks for nothing', () => {
    const el = body('<p>artifact_id: 28bfaa39e90e and more text</p>');
    linkArtifactIds(el);
    const focusArtifact = vi.fn(async () => /** @type {const} */ ('shown'));
    wireArtifactLinks(el, { focusArtifact });
    /** @type {HTMLElement} */ (el.querySelector('p')).click();

    expect(focusArtifact).not.toHaveBeenCalled();
  });
});
