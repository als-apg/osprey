// @ts-check
/**
 * Front-end proofs for the ARIEL advanced-options panel's touched-parameter
 * contract (Task 4.1):
 *
 *   - `getAdvancedParams()` emits ONLY knobs the operator actually moved, so a
 *     search never overrides the deployment's own configuration for a knob
 *     nobody touched — the panel pre-seeds every non-null descriptor default
 *     into its value map, and those seeds must stay invisible on the wire;
 *   - clearing a text/select/date control writes a literal null, which stays
 *     filtered out even though the name is now touched;
 *   - Reset returns the panel to the untouched state;
 *   - `getEffectiveParam(name)` reports what a search would really run with —
 *     the operator's value if touched, else the mode descriptor's default —
 *     and `undefined` for a name the current mode does not declare.
 *
 *   npx vitest run tests/interfaces/ariel/advanced-options.test.mjs
 *
 * Runs under happy-dom (vitest.config.js). The module keeps its state at module
 * scope, so each test re-imports it fresh via vi.resetModules(). Nothing is
 * mocked: the fixture declares no dynamic_select, so no fetch fires.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

const MODULE_PATH = '../../../src/osprey/interfaces/ariel/static/js/advanced-options.js';

/** Capabilities payload shaped like the backend's /api/capabilities response. */
const CAPABILITIES = /** @type {any} */ ({
  categories: {
    direct: {
      label: 'Direct',
      modes: [
        {
          name: 'hybrid',
          label: 'Hybrid',
          description: 'Keyword and semantic combined',
          parameters: [
            {
              name: 'rerank',
              label: 'Rerank Results',
              description: 'Re-score candidates with the cross-encoder',
              type: 'bool',
              default: true,
              section: 'Retrieval',
            },
            {
              name: 'candidate_limit',
              label: 'Candidate Limit',
              description: 'How many candidates to retrieve before reranking',
              type: 'int',
              default: 40,
              min: 1,
              max: 200,
              step: 1,
              section: 'Retrieval',
            },
            {
              name: 'sort',
              label: 'Sort Order',
              description: 'Result ordering',
              type: 'select',
              default: null,
              options: [
                { value: '', label: 'Relevance' },
                { value: 'newest', label: 'Newest first' },
              ],
              section: 'Retrieval',
            },
          ],
        },
      ],
    },
  },
  default_mode: 'hybrid',
  shared_parameters: [
    {
      name: 'author',
      label: 'Author',
      description: 'Restrict to entries by this author',
      type: 'text',
      default: null,
      placeholder: 'e.g. thellert',
      section: 'Filters',
    },
  ],
  vocabulary: { enabled: false, concepts: 0, expand_by_default: false },
});

/** The advanced-options surface as static/index.html lays it out. */
function mountFixture() {
  document.body.innerHTML = `
    <div id="search-mode-tabs" class="search-mode-tabs"></div>
    <div class="search-options-bar">
      <button type="button" id="advanced-toggle-btn">Filters &amp; Options</button>
    </div>
    <div id="advanced-panel" class="advanced-panel hidden">
      <div class="advanced-header">
        <button type="button" id="advanced-reset-btn">Reset</button>
        <button type="button" id="advanced-close-btn">Close</button>
      </div>
      <div class="advanced-sections" id="advanced-sections"></div>
    </div>
  `;
}

/**
 * @param {string} id - Element id
 * @returns {HTMLElement} The element (the fixture guarantees it exists)
 */
function el(id) {
  return /** @type {HTMLElement} */ (document.getElementById(id));
}

/**
 * @param {string} name - Parameter name
 * @returns {HTMLInputElement} The rendered control for that parameter
 */
function control(name) {
  return /** @type {HTMLInputElement} */ (el(`param-${name}`));
}

describe('advanced options: touched parameters and effective values', () => {
  /** @type {typeof import('../../../src/osprey/interfaces/ariel/static/js/advanced-options.js')} */
  let mod;

  beforeEach(async () => {
    vi.resetModules();
    mountFixture();
    mod = await import(MODULE_PATH);
    mod.initAdvancedOptions(CAPABILITIES);
    // attachParamListeners only runs from renderAdvancedPanel, which only runs
    // when the panel is opened.
    el('advanced-toggle-btn').click();
  });

  test('the panel starts on the deployment mode with its controls rendered', () => {
    expect(mod.getCurrentMode()).toBe('hybrid');
    expect(control('rerank').checked).toBe(true);
    expect(control('candidate_limit').value).toBe('40');
  });

  test('an untouched panel sends nothing, despite pre-seeded defaults', () => {
    expect(mod.getAdvancedParams()).toEqual({});
  });

  test('flipping one toggle sends that knob and only that knob', () => {
    const toggle = control('rerank');
    toggle.checked = false;
    toggle.dispatchEvent(new Event('change'));

    expect(mod.getAdvancedParams()).toEqual({ rerank: false });
  });

  test('moving a slider sends only the slider, not its untouched neighbours', () => {
    const slider = control('candidate_limit');
    slider.value = '80';
    slider.dispatchEvent(new Event('input'));

    expect(mod.getAdvancedParams()).toEqual({ candidate_limit: 80 });
  });

  test('Reset returns the panel to the untouched state', () => {
    const toggle = control('rerank');
    toggle.checked = false;
    toggle.dispatchEvent(new Event('change'));
    expect(mod.getAdvancedParams()).toEqual({ rerank: false });

    el('advanced-reset-btn').click();

    expect(mod.getAdvancedParams()).toEqual({});
    expect(mod.getEffectiveParam('rerank')).toBe(true);
  });

  test('a text filter typed and then cleared sends nothing', () => {
    const input = control('author');
    input.value = 'thellert';
    input.dispatchEvent(new Event('input'));
    expect(mod.getAdvancedParams()).toEqual({ author: 'thellert' });

    input.value = '';
    input.dispatchEvent(new Event('input'));

    expect(mod.getAdvancedParams()).toEqual({});
  });

  test('a select set and then cleared sends nothing', () => {
    const select = /** @type {HTMLSelectElement} */ (
      /** @type {unknown} */ (control('sort'))
    );
    select.value = 'newest';
    select.dispatchEvent(new Event('change'));
    expect(mod.getAdvancedParams()).toEqual({ sort: 'newest' });

    select.value = '';
    select.dispatchEvent(new Event('change'));

    expect(mod.getAdvancedParams()).toEqual({});
  });

  test('getEffectiveParam reports the configured default until the knob moves', () => {
    expect(mod.getEffectiveParam('rerank')).toBe(true);
    expect(mod.getEffectiveParam('candidate_limit')).toBe(40);

    const toggle = control('rerank');
    toggle.checked = false;
    toggle.dispatchEvent(new Event('change'));

    expect(mod.getEffectiveParam('rerank')).toBe(false);
    // ...without leaking into the wire payload of its neighbours.
    expect(mod.getEffectiveParam('candidate_limit')).toBe(40);
  });

  test('getEffectiveParam is undefined for names the current mode does not declare', () => {
    expect(mod.getEffectiveParam('not_a_parameter')).toBeUndefined();
    // `author` is a shared filter, not a mode descriptor: it has no mode-level
    // default to report.
    expect(mod.getEffectiveParam('author')).toBeUndefined();
  });
});

/**
 * Task 4.2: the descriptor's description is documentation, and documentation
 * that only appears on hover is documentation nobody reads. Every control
 * renders its description inline, and no renderer keeps the old `title=`.
 */
const HINT_CAPABILITIES = /** @type {any} */ ({
  categories: {
    direct: {
      label: 'Direct',
      modes: [
        {
          name: 'hybrid',
          label: 'Hybrid',
          parameters: [
            {
              name: 'rerank',
              label: 'Rerank Results',
              description: 'Re-score candidates with the cross-encoder',
              type: 'bool',
              default: true,
              section: 'Retrieval',
            },
            {
              name: 'min_score',
              label: 'Minimum Score',
              description: 'Drop results scoring below this threshold',
              type: 'float',
              default: 0.25,
              min: 0,
              max: 1,
              step: 0.05,
              section: 'Retrieval',
            },
            {
              name: 'undocumented',
              label: 'Undocumented Knob',
              type: 'int',
              default: 3,
              min: 1,
              max: 9,
              section: 'Retrieval',
            },
          ],
        },
      ],
    },
  },
  default_mode: 'hybrid',
  shared_parameters: [],
  vocabulary: { enabled: false, concepts: 0, expand_by_default: false },
});

describe('advanced options: visible parameter hints', () => {
  beforeEach(async () => {
    vi.resetModules();
    mountFixture();
    const mod = await import(MODULE_PATH);
    mod.initAdvancedOptions(HINT_CAPABILITIES);
    el('advanced-toggle-btn').click();
  });

  /** @returns {string[]} Text of every rendered hint, in document order. */
  function hintTexts() {
    return Array.from(
      el('advanced-sections').querySelectorAll('.param-hint'),
      (node) => (node.textContent || '').trim()
    );
  }

  test('a bool descriptor renders its description under the toggle', () => {
    expect(hintTexts()).toContain('Re-score candidates with the cross-encoder');
  });

  test('a float descriptor renders its description under the slider', () => {
    expect(hintTexts()).toContain('Drop results scoring below this threshold');
  });

  test('a descriptor without a description renders no empty hint', () => {
    // escapeHtml() stringifies undefined, so the absent description would
    // otherwise render as a blank <p> taking up a row.
    expect(hintTexts()).toEqual([
      'Re-score candidates with the cross-encoder',
      'Drop results scoring below this threshold',
    ]);
  });

  test('no rendered control keeps the old hover-only title attribute', () => {
    expect(el('advanced-sections').querySelectorAll('[title]')).toHaveLength(0);
  });
});
