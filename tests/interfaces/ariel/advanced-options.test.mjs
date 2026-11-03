// @ts-check
/**
 * Front-end proofs for the Filters view (advanced-options.js).
 *
 * The view holds no authoritative state: it renders from the capabilities,
 * mode and filters passed in and reports changes. What these tests pin down:
 *
 *   - `renderFilters` reports ONLY knobs the operator actually moved, so a
 *     search never overrides the deployment's own configuration for a knob
 *     nobody touched — the rendered controls show descriptor defaults, and
 *     those must never appear in a report;
 *   - clearing a text/select/date control removes the name from the report;
 *   - Reset reports the untouched state (`{}`);
 *   - a `locked` parameter renders disabled with its locked value and is never
 *     reported;
 *   - the view never mutates the filters it was given;
 *   - `effectiveParam(capabilities, mode, filters, name)` reports what a search
 *     would really run with — the touched value if any, else the mode
 *     descriptor's default — and `undefined` for a name the mode does not
 *     declare;
 *   - `renderModeChoices` marks the passed mode and reports a pick;
 *   - descriptions render as visible hints, never as hover-only titles;
 *   - reports are quiet: one per slider drag (on release), a text field after
 *     300 ms of no typing or on Enter, never the same value twice;
 *   - the author field is a pick-list fed by /api/filter-options/authors that
 *     typing narrows, and a pick yields the same filter as `toggleValue`;
 *   - `max_results` is not shown.
 *
 *   npx vitest run tests/interfaces/ariel/advanced-options.test.mjs
 *
 * Runs under happy-dom (vitest.config.js). The fixtures declare no
 * dynamic_select; the author suggestions' fetch is stubbed where it fires.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import {
  splitFilters,
  toggleValue,
} from '../../../src/osprey/interfaces/ariel/static/js/filter-state.js';
import {
  AUTHORS_ENDPOINT,
  matchSuggestions,
  effectiveParam,
  declaredParamNames,
  defaultMode,
  renderFilters,
  renderModeChoices,
} from '../../../src/osprey/interfaces/ariel/static/js/advanced-options.js';

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
      placeholder: 'e.g. alice',
      section: 'Filters',
    },
  ],
  vocabulary: { enabled: false, concepts: 0, expand_by_default: false },
});

/** A plain shared text parameter (not the author pick-list). */
const TAG = {
  name: 'tag',
  label: 'Tag',
  type: 'text',
  default: null,
  section: 'Filters',
};

/**
 * @param {string} name - Parameter name
 * @returns {HTMLInputElement} The rendered control for that parameter
 */
function control(name) {
  return /** @type {HTMLInputElement} */ (document.getElementById(`param-${name}`));
}

/**
 * Render the Filters view into a fresh root and record its reports.
 * @param {{capabilities?: any, mode?: string, filters?: any, locked?: any}} [opts]
 */
function mount({ capabilities = CAPABILITIES, mode = 'hybrid', filters = {}, locked } = {}) {
  document.body.innerHTML = '<div id="filters-root"></div>';
  const root = /** @type {HTMLElement} */ (document.getElementById('filters-root'));
  const onChange = vi.fn();
  renderFilters(root, { capabilities, mode, filters, locked, onChange });
  /** @returns {any} The most recent report */
  const last = () => onChange.mock.calls.at(-1)?.[0];
  return { root, onChange, last };
}

describe('filters view: touched parameters', () => {
  test('the controls show the passed filters over the descriptor defaults', () => {
    mount({ filters: { candidate_limit: 120 } });
    expect(control('rerank').checked).toBe(true);
    expect(control('candidate_limit').value).toBe('120');
  });

  test('rendering alone reports nothing', () => {
    const { onChange } = mount();
    expect(onChange).not.toHaveBeenCalled();
  });

  test('flipping one toggle reports that knob and only that knob', () => {
    const { last } = mount();
    const toggle = control('rerank');
    toggle.checked = false;
    toggle.dispatchEvent(new Event('change'));

    expect(last()).toEqual({ rerank: false });
  });

  test('moving a slider reports only the slider, not its untouched neighbours', () => {
    const { last } = mount();
    const slider = control('candidate_limit');
    slider.value = '80';
    slider.dispatchEvent(new Event('input'));
    slider.dispatchEvent(new Event('change'));

    expect(last()).toEqual({ candidate_limit: 80 });
    expect(document.getElementById('param-candidate_limit-value')?.textContent).toBe('80');
  });

  test("a change keeps every other mode's touched values it was given", () => {
    const { last } = mount({ filters: { similarity: 0.7 } });
    const toggle = control('rerank');
    toggle.checked = false;
    toggle.dispatchEvent(new Event('change'));

    expect(last()).toEqual({ similarity: 0.7, rerank: false });
  });

  test('successive changes before a re-render accumulate', () => {
    const { last } = mount();
    const toggle = control('rerank');
    toggle.checked = false;
    toggle.dispatchEvent(new Event('change'));
    const slider = control('candidate_limit');
    slider.value = '80';
    slider.dispatchEvent(new Event('change'));

    expect(last()).toEqual({ rerank: false, candidate_limit: 80 });
  });

  test('the filters passed in are never mutated', () => {
    const filters = Object.freeze({ candidate_limit: 50 });
    mount({ filters });
    const toggle = control('rerank');
    toggle.checked = false;
    toggle.dispatchEvent(new Event('change'));

    expect(filters).toEqual({ candidate_limit: 50 });
  });

  test('Reset reports the untouched state', () => {
    const { root, last } = mount({ filters: { rerank: false, similarity: 0.7 } });
    /** @type {HTMLElement} */ (root.querySelector('[data-action="reset-filters"]')).click();

    expect(last()).toEqual({});
  });

  test('a text filter typed and then cleared reports no value', () => {
    vi.useFakeTimers();
    try {
      const { last } = mount();
      const input = control('author');
      input.value = 'alice';
      input.dispatchEvent(new Event('input'));
      vi.advanceTimersByTime(300);
      expect(last()).toEqual({ author: 'alice' });

      input.value = '';
      input.dispatchEvent(new Event('input'));
      vi.advanceTimersByTime(300);

      expect(last()).toEqual({});
    } finally {
      vi.useRealTimers();
    }
  });

  test('a select set and then cleared reports no value', () => {
    const { last } = mount();
    const select = /** @type {HTMLSelectElement} */ (/** @type {unknown} */ (control('sort')));
    select.value = 'newest';
    select.dispatchEvent(new Event('change'));
    expect(last()).toEqual({ sort: 'newest' });

    select.value = '';
    select.dispatchEvent(new Event('change'));

    expect(last()).toEqual({});
  });

  test('a mode without parameters or shared ones renders the empty state', () => {
    const { root } = mount({
      capabilities: { ...CAPABILITIES, shared_parameters: [] },
      mode: 'not_a_mode',
    });
    expect(root.textContent).toContain('No advanced options for this mode.');
  });
});

describe('filters view: quiet reports', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  test('a slider drag reports once, on release, while its readout follows the drag', () => {
    const { onChange, last } = mount();
    const slider = control('candidate_limit');
    for (const v of ['50', '60', '70', '80']) {
      slider.value = v;
      slider.dispatchEvent(new Event('input'));
    }
    expect(onChange).not.toHaveBeenCalled();
    expect(document.getElementById('param-candidate_limit-value')?.textContent).toBe('80');

    slider.dispatchEvent(new Event('change'));
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(last()).toEqual({ candidate_limit: 80 });
  });

  test('a text field reports once, 300 ms after the last keystroke', () => {
    const { onChange, last } = mount({
      capabilities: { ...CAPABILITIES, shared_parameters: [TAG] },
    });
    const input = control('tag');
    for (const v of ['q', 'qu', 'que', 'quench']) {
      input.value = v;
      input.dispatchEvent(new Event('input'));
      vi.advanceTimersByTime(200);
    }
    expect(onChange).not.toHaveBeenCalled();
    vi.advanceTimersByTime(100);
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(last()).toEqual({ tag: 'quench' });
  });

  test('Enter reports at once and the pending timer adds no second report', () => {
    const { onChange, last } = mount();
    const input = control('author');
    input.value = 'alice';
    input.dispatchEvent(new Event('input'));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(last()).toEqual({ author: 'alice' });

    vi.advanceTimersByTime(1000);
    input.dispatchEvent(new Event('change'));
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  test('a pending report of a control rendered away is dropped', () => {
    const { root, onChange } = mount();
    const input = control('author');
    input.value = 'alice';
    input.dispatchEvent(new Event('input'));
    root.innerHTML = '';
    vi.advanceTimersByTime(300);
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe('filters view: author pick-list', () => {
  const AUTHORS = ['Alice Smith', 'alicia', 'Bob', 'Carol'];

  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          field: 'authors',
          options: AUTHORS.map((a) => ({ value: a, label: a })),
        }),
      }))
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  /** @returns {string[]} The suggestions the author field currently offers */
  const suggestions = () => {
    const input = control('author');
    const list = document.getElementById(/** @type {string} */ (input.getAttribute('list')));
    return Array.from(list?.querySelectorAll('option') ?? []).map((o) => o.value);
  };

  test('the author field offers every author from the authors endpoint', async () => {
    mount();
    control('author').dispatchEvent(new Event('pointerdown'));
    await vi.waitFor(() => expect(suggestions()).toEqual(AUTHORS));
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe(AUTHORS_ENDPOINT);
  });

  test('typing narrows the suggestions, ignoring case', async () => {
    mount();
    const input = control('author');
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'a' }));
    await vi.waitFor(() => expect(suggestions()).toEqual(AUTHORS));
    input.value = 'ALI';
    input.dispatchEvent(new Event('input'));
    expect(suggestions()).toEqual(['Alice Smith', 'alicia']);
  });

  test('picking an author reports at once and yields the same filter as toggleValue', () => {
    const { onChange, last } = mount();
    const input = control('author');
    input.value = 'Bob';
    input.dispatchEvent(new InputEvent('input', { inputType: 'insertReplacementText' }));
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(splitFilters(last()).filter).toEqual(toggleValue({}, 'author', 'Bob'));
  });

  test('a locked author field fetches no suggestions', () => {
    mount({ locked: { author: 'bob' } });
    control('author').dispatchEvent(new Event('pointerdown'));
    expect(fetch).not.toHaveBeenCalled();
  });

  test('matchSuggestions caps what it returns and blank matches all', () => {
    expect(matchSuggestions(AUTHORS, '')).toEqual(AUTHORS);
    expect(matchSuggestions(AUTHORS, 'o', 1)).toEqual(['Bob']);
  });
});

describe('filters view: hidden knobs', () => {
  test('max_results is not shown in the panel', () => {
    const maxResults = {
      name: 'max_results',
      label: 'Max Results',
      type: 'int',
      default: 10,
      min: 1,
      max: 100,
      section: 'General',
    };
    mount({ capabilities: { ...CAPABILITIES, shared_parameters: [maxResults, TAG] } });
    expect(control('max_results')).toBeNull();
    expect(control('tag')).not.toBeNull();
  });
});

describe('filters view: locked parameters', () => {
  test('a locked parameter renders disabled with its locked value', () => {
    mount({ filters: { author: 'alice' }, locked: { author: 'bob' } });
    expect(control('author').disabled).toBe(true);
    expect(control('author').value).toBe('bob');
    expect(control('rerank').disabled).toBe(false);
  });

  test('a locked parameter is never reported, even if its control fires', () => {
    const { onChange, last } = mount({ locked: { author: 'bob' } });
    const input = control('author');
    input.value = 'mallory';
    input.dispatchEvent(new Event('input'));
    expect(onChange).not.toHaveBeenCalled();

    const toggle = control('rerank');
    toggle.checked = false;
    toggle.dispatchEvent(new Event('change'));
    expect(last()).toEqual({ rerank: false });
  });
});

describe('effectiveParam', () => {
  test('reports the configured default until the knob is touched', () => {
    expect(effectiveParam(CAPABILITIES, 'hybrid', {}, 'rerank')).toBe(true);
    expect(effectiveParam(CAPABILITIES, 'hybrid', {}, 'candidate_limit')).toBe(40);
    expect(effectiveParam(CAPABILITIES, 'hybrid', { rerank: false }, 'rerank')).toBe(false);
    // ...without leaking into its neighbours.
    expect(effectiveParam(CAPABILITIES, 'hybrid', { rerank: false }, 'candidate_limit')).toBe(40);
  });

  test('a null touched value falls back to the default', () => {
    expect(effectiveParam(CAPABILITIES, 'hybrid', { rerank: null }, 'rerank')).toBe(true);
  });

  test('is undefined for names the mode does not declare', () => {
    expect(effectiveParam(CAPABILITIES, 'hybrid', {}, 'not_a_parameter')).toBeUndefined();
    // `author` is a shared filter, not a mode descriptor: it has no mode-level
    // default to report.
    expect(effectiveParam(CAPABILITIES, 'hybrid', { author: 'x' }, 'author')).toBeUndefined();
    expect(effectiveParam(CAPABILITIES, 'not_a_mode', { rerank: true }, 'rerank')).toBeUndefined();
  });

  test('is pure: the same inputs give the same answer whatever was rendered', () => {
    mount({ filters: { rerank: false } });
    expect(effectiveParam(CAPABILITIES, 'hybrid', {}, 'rerank')).toBe(true);
  });
});

describe('declaredParamNames and defaultMode', () => {
  test("a mode's names plus the shared ones", () => {
    expect([...declaredParamNames(CAPABILITIES, 'hybrid')].sort()).toEqual([
      'author',
      'candidate_limit',
      'rerank',
      'sort',
    ]);
    expect([...declaredParamNames(CAPABILITIES, 'not_a_mode')]).toEqual(['author']);
  });

  test("the deployment's default mode when advertised, else the first mode", () => {
    expect(defaultMode(CAPABILITIES)).toBe('hybrid');
    expect(defaultMode({ ...CAPABILITIES, default_mode: 'gone' })).toBe('hybrid');
    expect(defaultMode(null)).toBe('keyword');
  });
});

describe('mode choices', () => {
  const TWO_MODES = /** @type {any} */ ({
    categories: {
      direct: {
        label: 'Direct',
        modes: [
          { name: 'keyword', label: 'Keyword', description: 'Text search' },
          { name: 'hybrid', label: 'Hybrid' },
        ],
      },
    },
  });

  /** @param {string} mode */
  function mountChoices(mode) {
    document.body.innerHTML = '<div id="modes-root"></div>';
    const root = /** @type {HTMLElement} */ (document.getElementById('modes-root'));
    const onSelect = vi.fn();
    renderModeChoices(root, { capabilities: TWO_MODES, mode, onSelect });
    /** @param {string} name */
    const tab = (name) => /** @type {HTMLElement} */ (root.querySelector(`[data-mode="${name}"]`));
    return { root, onSelect, tab };
  }

  test('marks the passed mode as the chosen one', () => {
    const { tab } = mountChoices('hybrid');
    expect(tab('hybrid').classList.contains('active')).toBe(true);
    expect(tab('hybrid').getAttribute('aria-pressed')).toBe('true');
    expect(tab('keyword').classList.contains('active')).toBe(false);
  });

  test('a pick is reported, and the view does not move the mark itself', () => {
    const { tab, onSelect } = mountChoices('hybrid');
    tab('keyword').click();
    expect(onSelect).toHaveBeenCalledWith('keyword');
    expect(tab('hybrid').classList.contains('active')).toBe(true);
  });

  test('picking the current mode reports nothing', () => {
    const { tab, onSelect } = mountChoices('hybrid');
    tab('hybrid').click();
    expect(onSelect).not.toHaveBeenCalled();
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

describe('filters view: visible parameter hints', () => {
  /** @type {HTMLElement} */
  let root;

  beforeEach(() => {
    root = mount({ capabilities: HINT_CAPABILITIES }).root;
  });

  /** @returns {string[]} Text of every rendered hint, in document order. */
  function hintTexts() {
    return Array.from(root.querySelectorAll('.param-hint'), (node) =>
      (node.textContent || '').trim()
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
    expect(root.querySelectorAll('[title]')).toHaveLength(0);
  });
});
