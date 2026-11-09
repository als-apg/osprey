import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  bindControls,
  chooseModel,
  choosePlane,
  clearBump,
  matchBpms,
  panelOf,
  refresh,
  setSelection,
  shapeWeights,
  showState,
} from '../panel/panel.js';

// The bump table and the controls around it: rows keyed by BPM element, the
// corrector rule, and the plane/model/Clear controls that discard a plan first.

/**
 * @typedef {import('../panel/panel.js').Member} Member
 * @typedef {import('../panel/panel.js').Model} Model
 * @typedef {import('../panel/panel.js').Plan} Plan
 * @typedef {import('../panel/panel.js').BumpState} BumpState
 */

const MARKUP = `
  <span data-field="plane"><button data-plane="x"></button><button data-plane="y"></button></span>
  <select data-field="model" hidden></select>
  <span data-field="orm"><i></i><span data-field="orm-text"></span><button data-field="recompute"></button></span>
  <span data-field="no-model" hidden>no closed-orbit model</span>
  <p data-field="posture" hidden></p>
  <p data-field="notice" hidden></p>
  <p data-field="error" hidden></p>
  <div data-field="body"></div>
  <div data-field="confirm" hidden></div>`;

/**
 * @param {string} element
 * @param {'x' | 'y'} plane
 * @param {number} s
 * @returns {Member}
 */
function bpm(element, plane, s) {
  return {
    element,
    plane,
    index: s,
    s_m: s,
    unit: 'mm',
    position_addr: `${element}:POS:${plane}`,
    golden_addr: `${element}:GOLD:${plane}`,
  };
}

/**
 * @param {string} element
 * @param {'x' | 'y'} plane
 * @param {number} s
 * @returns {Member}
 */
function corrector(element, plane, s) {
  return {
    element,
    plane,
    index: s,
    s_m: s,
    unit: 'A',
    setpoint_addr: `${element}:SP`,
    readback_addr: `${element}:RB`,
  };
}

const BPMS = [
  bpm('B1', 'x', 10),
  bpm('B1', 'y', 10),
  bpm('B2', 'x', 30),
  bpm('B2', 'y', 30),
  bpm('B3', 'x', 50),
  bpm('B3', 'y', 50),
];

const CORRECTORS = [
  corrector('H1', 'x', 20),
  corrector('H2', 'x', 40),
  corrector('H3', 'x', 60),
  corrector('V1', 'y', 25),
];

/**
 * @param {string} name
 * @param {string[]} [unsolved]
 * @returns {Model}
 */
function model(name, unsolved = []) {
  return {
    name,
    length_m: 100,
    orm: { status: 'ready', age_s: 3, version: 1, error: null, unsolved },
    members: { bpms: BPMS, correctors: CORRECTORS },
  };
}

/** @returns {{position: Record<string, number | null>, golden: Record<string, number | null>}} */
function orbit() {
  /** @type {Record<string, number | null>} */
  const position = {};
  /** @type {Record<string, number | null>} */
  const golden = {};
  for (const member of BPMS) {
    if (!member.position_addr || !member.golden_addr) continue;
    position[member.position_addr] = 1.25;
    golden[member.golden_addr] = 1;
  }
  return { position, golden };
}

/**
 * @param {Partial<Plan>} [over]
 * @returns {Plan}
 */
function plan(over = {}) {
  return {
    id: 'p1',
    model: 'RING',
    plane: 'x',
    status: 'ready',
    targets: { B2: 0.5 },
    correctors: [{ element: 'H1' }, { element: 'H2' }],
    predicted: { B1: 1, B2: 1.5, B3: 1 },
    ...over,
  };
}

/**
 * @param {Partial<BumpState>} [over]
 * @returns {BumpState}
 */
function state(over = {}) {
  return {
    models: [model('RING')],
    posture: { allowed: true, reason: null, phrase: '' },
    orbit: orbit(),
    plan: null,
    ...over,
  };
}

/**
 * A fetcher recording every call; the state route answers `current()`, the
 * discard route answers `discardStatus` (with `discardText` as its body).
 * @param {() => BumpState} current
 * @param {{discardStatus?: number, discardText?: string}} [options]
 */
function backend(current, options = {}) {
  const { discardStatus = 204, discardText = '' } = options;
  /** @type {string[]} */
  const log = [];
  const fetcher = vi.fn(async (/** @type {string} */ url, /** @type {RequestInit} */ init) => {
    log.push(`${init?.method ?? 'GET'} ${url}`);
    if (url === 'api/discard') {
      return new Response(discardStatus === 204 ? null : discardText, { status: discardStatus });
    }
    return new Response(JSON.stringify(current()), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  });
  return { fetcher: /** @type {typeof fetch} */ (/** @type {unknown} */ (fetcher)), log };
}

/**
 * @param {Element} root
 * @param {string} name
 * @returns {HTMLElement}
 */
function field(root, name) {
  const found = root.querySelector(`[data-field="${name}"]`);
  if (!(found instanceof HTMLElement)) throw new Error(`no ${name}`);
  return found;
}

/**
 * @param {Element} root
 * @returns {string[]}
 */
function rowElements(root) {
  return [...field(root, 'bump').querySelectorAll('tbody tr')].map(
    (row) => row.getAttribute('data-element') ?? ''
  );
}

/**
 * @param {Element} root
 * @param {string} element
 * @returns {HTMLTableRowElement}
 */
function row(root, element) {
  const found = field(root, 'bump').querySelector(`tbody tr[data-element="${element}"]`);
  if (!(found instanceof HTMLTableRowElement)) throw new Error(`no row for ${element}`);
  return found;
}

/**
 * @param {Element} root
 * @param {string} element
 * @returns {HTMLInputElement}
 */
function input(root, element) {
  const found = row(root, element).querySelector('input');
  if (!(found instanceof HTMLInputElement)) throw new Error(`no input for ${element}`);
  return found;
}

/**
 * @param {Element} root
 * @param {string} element
 * @param {string} cell
 */
function cell(root, element, cell) {
  return row(root, element).querySelector(`[data-cell="${cell}"]`)?.textContent ?? '';
}

/**
 * @param {HTMLSelectElement} select
 * @param {string} value
 */
function choose(select, value) {
  select.value = value;
  select.dispatchEvent(new Event('change', { bubbles: true }));
}

/**
 * Press the button of a segmented control.
 * @param {Element} root
 * @param {string} name the control's data-field
 * @param {string} attr the attribute its buttons carry their value in
 * @param {string} value
 */
function pressSeg(root, name, attr, value) {
  const button = field(root, name).querySelector(`[${attr}="${value}"]`);
  if (!button) throw new Error(`no ${value} in ${name}`);
  click(button);
}

/**
 * The value of the pressed button of a segmented control.
 * @param {Element} root
 * @param {string} name
 * @param {string} attr
 */
function pressed(root, name, attr) {
  return field(root, name).querySelector('[aria-pressed="true"]')?.getAttribute(attr) ?? null;
}

/**
 * @param {Element} target
 * @param {string} key
 * @param {boolean} [shiftKey]
 */
function press(target, key, shiftKey = false) {
  target.dispatchEvent(new KeyboardEvent('keydown', { key, shiftKey, bubbles: true }));
}

/**
 * Type into the picker, as an input event.
 * @param {Element} root
 * @param {string} text
 */
function typePick(root, text) {
  const pick = /** @type {HTMLInputElement} */ (field(root, 'pick'));
  pick.focus();
  pick.value = text;
  pick.dispatchEvent(new Event('input', { bubbles: true }));
  return pick;
}

/**
 * The picker's option rows: name and whether it shows as in the bump.
 * @param {Element} root
 */
function options(root) {
  return [...field(root, 'pick-list').querySelectorAll('[role="option"][data-name]')].map(
    (row) =>
      `${row.getAttribute('data-name')}${row.getAttribute('aria-selected') === 'true' ? '✓' : ''}`
  );
}

/** @param {Element} target */
function click(target) {
  target.dispatchEvent(new MouseEvent('click', { bubbles: true }));
}

/**
 * @param {Element} root
 * @param {string} element
 */
function stripTick(root, element) {
  const found = root.querySelector(`svg .ob-cor[data-element="${element}"]`);
  if (!found) throw new Error(`no tick for ${element}`);
  return found;
}

describe('the orbit-bump table and controls', () => {
  /** @type {HTMLElement} */
  let root;

  beforeEach(() => {
    root = document.createElement('main');
    root.innerHTML = MARKUP;
    document.body.append(root);
  });

  afterEach(() => {
    root.remove();
  });

  describe('the bump table', () => {
    it('starts empty with the facility unit in its header', () => {
      showState(root, state());
      expect(rowElements(root)).toEqual([]);
      expect(field(root, 'bump-empty').hidden).toBe(false);
      expect(field(root, 'bump').querySelector('thead')?.textContent).toContain('now mm');
      expect(field(root, 'bump').querySelector('thead')?.textContent).toContain('target mm');
    });

    it('adds a BPM from a strip click with a zero offset and its reading now', () => {
      showState(root, state());
      const dot = root.querySelector('svg .ob-bpm[data-element="B2"]');
      if (!dot) throw new Error('no dot');
      click(dot);
      expect(rowElements(root)).toEqual(['B2']);
      expect(input(root, 'B2').value).toBe('0');
      // Now is position − golden in the facility's unit: 1.25 − 1.
      expect(cell(root, 'B2', 'now')).toContain('0.250');
      expect(field(root, 'bump-count').textContent).toBe('1');
      expect(field(root, 'bump-empty').hidden).toBe(true);
    });

    it('does not offer a BPM without a reading in the picker', () => {
      const readings = orbit();
      readings.position['B1:POS:x'] = null;
      showState(root, state({ orbit: readings }));
      typePick(root, '');
      expect(options(root)).toEqual(['B2', 'B3']);
    });

    it('nudges a row by the step with its buttons and the arrow keys', () => {
      showState(root, state());
      setSelection(root, { targets: { B2: 0 } });
      const more = row(root, 'B2').querySelector('[data-step-dir="1"]');
      if (!more) throw new Error('no + button');
      click(more);
      expect(panelOf(root).selection.targets.B2).toBeCloseTo(0.01, 9);
      press(input(root, 'B2'), 'ArrowDown', true);
      expect(panelOf(root).selection.targets.B2).toBeCloseTo(-0.09, 9);
      expect(input(root, 'B2').value).toBe('-0.09');
      expect(input(root, 'B2').step).toBe('0.01');
      expect(cell(root, 'B2', 's')).toBe('30.0 m');
    });

    it('highlights the row whose input has focus', () => {
      showState(root, state());
      setSelection(root, { targets: { B1: 0, B2: 0 } });
      input(root, 'B2').focus();
      expect(row(root, 'B2').classList.contains('ob-focus')).toBe(true);
      expect(row(root, 'B1').classList.contains('ob-focus')).toBe(false);
    });

    it('lists rows in ring order and removes one with its remove button', () => {
      showState(root, state());
      setSelection(root, { targets: { B3: 0.2, B1: -0.1 } });
      expect(rowElements(root)).toEqual(['B1', 'B3']);
      const remove = row(root, 'B1').querySelector('button[data-remove]');
      if (!remove) throw new Error('no remove button');
      click(remove);
      expect(panelOf(root).selection.targets).toEqual({ B3: 0.2 });
      expect(rowElements(root)).toEqual(['B3']);
    });

    it('takes a typed offset into the selection', () => {
      showState(root, state());
      setSelection(root, { targets: { B2: 0 } });
      const box = input(root, 'B2');
      box.value = '0.4';
      box.dispatchEvent(new Event('input', { bubbles: true }));
      expect(panelOf(root).selection.targets).toEqual({ B2: 0.4 });
      // A half-typed value leaves the last good offset in place.
      box.value = '';
      box.dispatchEvent(new Event('input', { bubbles: true }));
      expect(panelOf(root).selection.targets).toEqual({ B2: 0.4 });
    });

    it('shows the predicted position of the current plan relative to golden', () => {
      showState(root, state({ plan: plan() }));
      expect(rowElements(root)).toEqual(['B2']);
      expect(cell(root, 'B2', 'predicted')).toContain('0.500');
    });

    it("keeps an input's value and focus across refresh(), updating the other cells", async () => {
      let readings = orbit();
      const { fetcher } = backend(() => state({ orbit: readings }));
      await refresh(root, fetcher);
      setSelection(root, { targets: { B2: 0 } });
      const box = input(root, 'B2');
      box.focus();
      box.value = '0.3';
      box.dispatchEvent(new Event('input', { bubbles: true }));
      box.value = '0.3';
      readings = orbit();
      readings.position['B2:POS:x'] = 1.75;
      await refresh(root, fetcher);
      await refresh(root, fetcher);
      expect(input(root, 'B2')).toBe(box);
      expect(box.value).toBe('0.3');
      expect(document.activeElement).toBe(box);
      expect(cell(root, 'B2', 'now')).toContain('0.750');
    });

    it('keeps the row element itself across polls', async () => {
      const { fetcher } = backend(() => state());
      await refresh(root, fetcher);
      setSelection(root, { targets: { B1: 0.1, B2: 0.2 } });
      const before = [row(root, 'B1'), row(root, 'B2')];
      await refresh(root, fetcher);
      expect([row(root, 'B1'), row(root, 'B2')]).toEqual(before);
      expect(row(root, 'B1')).toBe(before[0]);
    });
  });

  describe('the corrector rule', () => {
    it('defaults to auto and offers auto, tight and manual as a segmented control', () => {
      showState(root, state());
      expect(pressed(root, 'rule', 'data-rule')).toBe('auto');
      const offered = [...field(root, 'rule').querySelectorAll('[data-rule]')].map((button) => [
        button.getAttribute('data-rule'),
        button.textContent,
      ]);
      expect(offered).toEqual([
        ['auto', 'auto'],
        ['mml:[-1 1]', 'tight'],
        ['manual', 'manual'],
      ]);
      expect(root.querySelector('select:not([data-field="model"])')).toBeNull();
    });

    it('changes the selection rule, without discarding a plan', () => {
      const { fetcher, log } = backend(() => state());
      bindControls(root, fetcher);
      showState(root, state({ plan: plan() }));
      pressSeg(root, 'rule', 'data-rule', 'mml:[-1 1]');
      expect(panelOf(root).selection.rule).toBe('mml:[-1 1]');
      expect(pressed(root, 'rule', 'data-rule')).toBe('mml:[-1 1]');
      expect(log).toEqual([]);
    });

    describe('the overflow menu', () => {
      const items = () => [...document.querySelectorAll('[data-field="rule-menu"] [data-rule]')];
      const more = () => field(root, 'rule-more');
      const item = (/** @type {string} */ value) =>
        /** @type {HTMLElement} */ (items().find((i) => i.getAttribute('data-rule') === value));

      afterEach(() => document.body.replaceChildren());

      beforeEach(() => {
        document.body.append(root);
      });

      it('offers the two wider lists behind a ⋯ button, labelled', () => {
        showState(root, state());
        expect(more().title).toBe('more corrector rules');
        expect(items().map((i) => [i.getAttribute('data-rule'), i.textContent])).toEqual([
          ['mml:[-2 -1 1 2]', '±2 (−2 −1 1 2)'],
          ['mml:[-3 -1 1 3]', '±3 (−3 −1 1 3)'],
        ]);
        expect(items().every((i) => i.getAttribute('role') === 'menuitemradio')).toBe(true);
        expect(more().getAttribute('aria-pressed')).toBe('false');
      });

      it.each(['mml:[-2 -1 1 2]', 'mml:[-3 -1 1 3]'])('chooses %s', (value) => {
        showState(root, state());
        setSelection(root, { rule: 'manual', correctors: ['H2'] });
        click(more());
        click(item(value));
        expect(panelOf(root).selection.rule).toBe(value);
        expect(panelOf(root).selection.correctors).toEqual([]);
        expect(pressed(root, 'rule', 'data-rule')).toBeNull();
        expect(more().getAttribute('aria-pressed')).toBe('true');
        expect(items().map((i) => i.getAttribute('aria-checked'))).toEqual(
          value === 'mml:[-2 -1 1 2]' ? ['true', 'false'] : ['false', 'true']
        );
      });

      it('goes back to a toggle button and clears the mark', () => {
        showState(root, state());
        click(item('mml:[-3 -1 1 3]'));
        pressSeg(root, 'rule', 'data-rule', 'auto');
        expect(pressed(root, 'rule', 'data-rule')).toBe('auto');
        expect(more().getAttribute('aria-pressed')).toBe('false');
        expect(items().map((i) => i.getAttribute('aria-checked'))).toEqual(['false', 'false']);
      });

      it('shows a hydrated wider list as such, not as manual', () => {
        showState(root, state({ plan: plan({ rule: 'mml:[-2 -1 1 2]' }) }));
        expect(panelOf(root).selection.rule).toBe('mml:[-2 -1 1 2]');
        expect(panelOf(root).selection.correctors).toEqual([]);
        expect(pressed(root, 'rule', 'data-rule')).toBeNull();
        expect(more().getAttribute('aria-pressed')).toBe('true');
        expect(item('mml:[-2 -1 1 2]').getAttribute('aria-checked')).toBe('true');
      });

      it('still hydrates a plan without a wider rule as manual', () => {
        showState(root, state({ plan: plan() }));
        expect(panelOf(root).selection.rule).toBe('manual');
        expect(pressed(root, 'rule', 'data-rule')).toBe('manual');
        expect(more().getAttribute('aria-pressed')).toBe('false');
      });
    });

    it('drops manual correctors when the rule leaves manual', () => {
      showState(root, state());
      setSelection(root, { rule: 'manual', correctors: ['H2'] });
      pressSeg(root, 'rule', 'data-rule', 'auto');
      expect(panelOf(root).selection.correctors).toEqual([]);
    });

    it('makes an unsolved corrector unselectable and names it in the title tooltip', () => {
      showState(root, state({ models: [model('RING', ['H2', 'V1'])] }));
      setSelection(root, { rule: 'manual' });
      click(stripTick(root, 'H2'));
      expect(panelOf(root).selection.correctors).toEqual([]);
      click(stripTick(root, 'H1'));
      expect(panelOf(root).selection.correctors).toEqual(['H1']);
      const title = field(root, 'cor-title').title;
      // Only the selected plane's unsolved correctors are named.
      expect(title).toContain('H2');
      expect(title).not.toContain('V1');
    });

    it("names the plan's skipped unsolved correctors in the title tooltip", () => {
      showState(
        root,
        state({ models: [model('RING', ['H3'])], plan: plan({ skipped_unsolved: ['H3'] }) })
      );
      const title = field(root, 'cor-title').title;
      expect(title).toContain('plan #p1');
      expect(title).toContain('skipped');
      expect(title).toContain('H3');
    });

    it('has no title tooltip when every corrector solved and there is no plan', () => {
      showState(root, state());
      expect(field(root, 'cor-title').hasAttribute('title')).toBe(false);
    });
  });

  describe('the BPM picker', () => {
    it('reads a number, a range, a prefix and a name', () => {
      const names = ['BPM1', 'BPM2', 'BPM3', 'BPM12', 'BPM21', 'BPM22', 'BPM23'];
      expect(matchBpms('22', names)).toEqual(['BPM22']);
      expect(matchBpms('BPM22', names)).toEqual(['BPM22']);
      expect(matchBpms('bpm 3', names)).toEqual(['BPM3']);
      expect(matchBpms('5-6', names)).toEqual(['BPM21', 'BPM22']);
      expect(matchBpms('6–5', names)).toEqual(['BPM21', 'BPM22']);
      expect(matchBpms('2*', names)).toEqual(['BPM2', 'BPM21', 'BPM22', 'BPM23']);
      expect(matchBpms('', names)).toEqual(names);
      expect(matchBpms('99', names)).toEqual([]);
      // A number no name carries is the n-th BPM in ring order.
      expect(matchBpms('2', ['SR01:BPM_A', 'SR01:BPM_B'])).toEqual(['SR01:BPM_B']);
      expect(matchBpms('bpm_b', ['SR01:BPM_A', 'SR01:BPM_B'])).toEqual(['SR01:BPM_B']);
    });

    it('opens a listbox on focus that marks the BPMs already in the bump', () => {
      showState(root, state());
      setSelection(root, { targets: { B2: 0.2 } });
      const pick = typePick(root, '');
      expect(pick.getAttribute('role')).toBe('combobox');
      expect(pick.getAttribute('aria-expanded')).toBe('true');
      expect(field(root, 'pick-list').hidden).toBe(false);
      expect(field(root, 'pick-list').getAttribute('role')).toBe('listbox');
      expect(options(root)).toEqual(['B1', 'B2✓', 'B3']);
      // Each row carries s and the offset (the target for a bump BPM).
      const b2 = field(root, 'pick-list').querySelector('[data-name="B2"]');
      expect(b2?.textContent).toContain('30.0 m');
      expect(b2?.textContent).toContain('+0.200');
    });

    it('toggles a clicked row without closing the list', () => {
      showState(root, state());
      typePick(root, '');
      const rowOf = (/** @type {string} */ name) => {
        const found = field(root, 'pick-list').querySelector(`[data-name="${name}"]`);
        if (!found) throw new Error(`no option ${name}`);
        return found;
      };
      click(rowOf('B3'));
      expect(panelOf(root).selection.targets).toEqual({ B3: 0 });
      expect(field(root, 'pick-list').hidden).toBe(false);
      expect(options(root)).toEqual(['B1', 'B2', 'B3✓']);
      click(rowOf('B3'));
      expect(panelOf(root).selection.targets).toEqual({});
    });

    it('arms rows with the arrows and picks the armed row on Enter', () => {
      showState(root, state());
      const pick = typePick(root, '');
      press(pick, 'ArrowDown');
      press(pick, 'ArrowDown');
      const armed = field(root, 'pick-list').querySelector('.armed');
      expect(armed?.getAttribute('data-name')).toBe('B2');
      expect(pick.getAttribute('aria-activedescendant')).toBe(armed?.id);
      press(pick, 'Enter');
      expect(panelOf(root).selection.targets).toEqual({ B2: 0 });
    });

    it('toggles a single match, or adds every match, on an unarmed Enter', () => {
      showState(root, state());
      const pick = typePick(root, '3');
      press(pick, 'Enter');
      expect(panelOf(root).selection.targets).toEqual({ B3: 0 });
      expect(pick.value).toBe('');
      typePick(root, '1-2');
      // Several matches: the first row offers to add them all.
      const all = field(root, 'pick-list').querySelector('[data-all]');
      expect(all?.textContent).toContain('add all 2 shown');
      press(pick, 'Enter');
      expect(Object.keys(panelOf(root).selection.targets).sort()).toEqual(['B1', 'B2', 'B3']);
    });

    it('closes on Escape and on a press outside', () => {
      showState(root, state());
      const pick = typePick(root, '');
      press(pick, 'Escape');
      expect(field(root, 'pick-list').hidden).toBe(true);
      expect(pick.getAttribute('aria-expanded')).toBe('false');
      typePick(root, '');
      expect(field(root, 'pick-list').hidden).toBe(false);
      document.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
      expect(field(root, 'pick-list').hidden).toBe(true);
    });
  });

  describe('the shape row and the nudge step', () => {
    it('spreads the amplitude flat, as a peak or as a cosine', () => {
      expect(shapeWeights(3, 'flat')).toEqual([1, 1, 1]);
      expect(shapeWeights(5, 'peak')).toEqual([0, 0.5, 1, 0.5, 0]);
      const cos = shapeWeights(5, 'cos');
      expect(cos[0]).toBeCloseTo(0, 9);
      expect(cos[1]).toBeCloseTo(0.5, 9);
      expect(cos[2]).toBeCloseTo(1, 9);
      // Under three BPMs every shape is flat.
      expect(shapeWeights(2, 'peak')).toEqual([1, 1]);
    });

    it('redistributes the targets when the amplitude or the shape changes', () => {
      showState(root, state());
      setSelection(root, { targets: { B1: 0, B2: 0, B3: 0 } });
      const amp = /** @type {HTMLInputElement} */ (field(root, 'amp'));
      amp.value = '0.4';
      amp.dispatchEvent(new Event('change', { bubbles: true }));
      expect(panelOf(root).selection.targets).toEqual({ B1: 0.4, B2: 0.4, B3: 0.4 });
      pressSeg(root, 'shape', 'data-shape', 'peak');
      expect(panelOf(root).selection.targets).toEqual({ B1: 0, B2: 0.4, B3: 0 });
      expect(pressed(root, 'shape', 'data-shape')).toBe('peak');
      // The amplitude steps by the nudge step.
      const more = field(root, 'shape-row').querySelector('[data-step-dir="1"]');
      if (!more) throw new Error('no + button');
      click(more);
      expect(panelOf(root).selection.targets.B2).toBeCloseTo(0.41, 9);
    });

    it('follows a typed row in the amplitude readout', () => {
      showState(root, state());
      setSelection(root, { targets: { B1: 0, B2: 0 } });
      const box = input(root, 'B2');
      box.value = '-0.3';
      box.dispatchEvent(new Event('change', { bubbles: true }));
      expect(panelOf(root).amp).toBeCloseTo(-0.3, 9);
      expect(/** @type {HTMLInputElement} */ (field(root, 'amp')).value).toBe('-0.300');
    });

    it('widens the bump by one BPM on each side', () => {
      showState(root, state());
      setSelection(root, { targets: { B2: 0 } });
      const amp = /** @type {HTMLInputElement} */ (field(root, 'amp'));
      amp.value = '0.2';
      amp.dispatchEvent(new Event('change', { bubbles: true }));
      click(field(root, 'widen'));
      expect(panelOf(root).selection.targets).toEqual({ B1: 0.2, B2: 0.2, B3: 0.2 });
    });

    it('sets the nudge step from the options menu', () => {
      bindControls(root, backend(() => state()).fetcher);
      showState(root, state());
      const step = field(root, 'step');
      expect(pressed(root, 'step', 'data-step')).toBe('0.01');
      expect([...step.querySelectorAll('button')].map((b) => b.textContent)).toEqual([
        '1 µm',
        '10 µm',
        '100 µm',
      ]);
      pressSeg(root, 'step', 'data-step', '0.001');
      expect(panelOf(root).step).toBe(0.001);
      setSelection(root, { targets: { B2: 0 } });
      expect(input(root, 'B2').step).toBe('0.001');
      // The menu opens from its hamburger and holds the legend.
      click(field(root, 'menu-button'));
      expect(field(root, 'menu').hidden).toBe(false);
      expect(field(root, 'menu').textContent).toContain('predicted');
    });
  });

  describe('discard before clearing', () => {
    it('Clear discards the plan first, then empties the selection', async () => {
      const { fetcher, log } = backend(() => state());
      showState(root, state({ plan: plan() }));
      expect(panelOf(root).selection.targets).toEqual({ B2: 0.5 });
      const cleared = await clearBump(root, fetcher);
      expect(cleared).toBe(true);
      expect(log).toEqual(['POST api/discard']);
      expect(panelOf(root).selection.targets).toEqual({});
      expect(panelOf(root).selection.correctors).toEqual([]);
      expect(panelOf(root).selection.rule).toBe('auto');
      expect(panelOf(root).state?.plan).toBeNull();
      expect(rowElements(root)).toEqual([]);
    });

    it('the Clear button discards through the bound fetcher', async () => {
      const { fetcher, log } = backend(() => state());
      bindControls(root, fetcher);
      showState(root, state({ plan: plan() }));
      click(field(root, 'clear'));
      await vi.waitFor(() => expect(panelOf(root).selection.targets).toEqual({}));
      expect(log).toEqual(['POST api/discard']);
    });

    it('a plane switch discards the plan first, then clears', async () => {
      const { fetcher, log } = backend(() => state());
      bindControls(root, fetcher);
      showState(root, state({ plan: plan() }));
      const vertical = root.querySelector('[data-plane="y"]');
      if (!vertical) throw new Error('no plane button');
      click(vertical);
      await vi.waitFor(() => expect(panelOf(root).selection.plane).toBe('y'));
      expect(log).toEqual(['POST api/discard']);
      expect(panelOf(root).selection.targets).toEqual({});
      expect(panelOf(root).state?.plan).toBeNull();
    });

    it('a model switch discards the plan first, then clears', async () => {
      const { fetcher, log } = backend(() => state());
      bindControls(root, fetcher);
      const two = { models: [model('RING'), model('LATTICE2')] };
      showState(root, state({ ...two, plan: plan() }));
      choose(/** @type {HTMLSelectElement} */ (field(root, 'model')), 'LATTICE2');
      await vi.waitFor(() => expect(panelOf(root).selection.model).toBe('LATTICE2'));
      expect(log).toEqual(['POST api/discard']);
      expect(panelOf(root).selection.targets).toEqual({});
    });

    it('the selection is cleared only after the discard answers', async () => {
      /** @type {(value: Response) => void} */
      let answer = () => {};
      const fetcher = /** @type {typeof fetch} */ (
        /** @type {unknown} */ (
          vi.fn(
            () =>
              new Promise((resolve) => {
                answer = resolve;
              })
          )
        )
      );
      showState(root, state({ plan: plan() }));
      const pending = choosePlane(root, 'y', fetcher);
      expect(panelOf(root).selection.plane).toBe('x');
      expect(panelOf(root).selection.targets).toEqual({ B2: 0.5 });
      answer(new Response(null, { status: 204 }));
      expect(await pending).toBe(true);
      expect(panelOf(root).selection.plane).toBe('y');
    });

    it('posts no discard without a plan', async () => {
      const { fetcher, log } = backend(() => state());
      showState(root, state());
      setSelection(root, { targets: { B1: 0.1 } });
      expect(await choosePlane(root, 'y', fetcher)).toBe(true);
      expect(panelOf(root).selection.targets).toEqual({});
      showState(root, state({ models: [model('RING'), model('LATTICE2')] }));
      expect(await chooseModel(root, 'LATTICE2', fetcher)).toBe(true);
      setSelection(root, { targets: { B1: 0.1 } });
      expect(await clearBump(root, fetcher)).toBe(true);
      expect(panelOf(root).selection.targets).toEqual({});
      expect(log).toEqual([]);
    });

    it('keeps the plan and the selection and shows the error when the discard fails', async () => {
      const { fetcher, log } = backend(() => state(), {
        discardStatus: 409,
        discardText: 'the plan is being applied',
      });
      showState(root, state({ plan: plan() }));
      expect(await choosePlane(root, 'y', fetcher)).toBe(false);
      expect(log).toEqual(['POST api/discard']);
      expect(panelOf(root).selection.plane).toBe('x');
      expect(panelOf(root).selection.targets).toEqual({ B2: 0.5 });
      expect(panelOf(root).state?.plan?.id).toBe('p1');
      expect(field(root, 'error').hidden).toBe(false);
      expect(field(root, 'error').textContent).toBe('the plan is being applied');
      expect(await clearBump(root, fetcher)).toBe(false);
      expect(panelOf(root).selection.targets).toEqual({ B2: 0.5 });
    });

    it('a failed model switch puts the dropdown back on the kept model', async () => {
      const { fetcher } = backend(() => state(), { discardStatus: 409, discardText: 'busy' });
      bindControls(root, fetcher);
      showState(root, state({ models: [model('RING'), model('LATTICE2')], plan: plan() }));
      const select = /** @type {HTMLSelectElement} */ (field(root, 'model'));
      choose(select, 'LATTICE2');
      await vi.waitFor(() => expect(field(root, 'error').hidden).toBe(false));
      expect(panelOf(root).selection.model).toBe('RING');
      expect(select.value).toBe('RING');
    });

    it('a poll sent before the discard does not bring the plan back', async () => {
      /** @type {((value: Response) => void)[]} */
      const held = [];
      const fetcher = /** @type {typeof fetch} */ (
        /** @type {unknown} */ (
          vi.fn((/** @type {string} */ url) => {
            if (url === 'api/discard') return Promise.resolve(new Response(null, { status: 204 }));
            return new Promise((resolve) => held.push(resolve));
          })
        )
      );
      showState(root, state({ plan: plan() }));
      const poll = refresh(root, fetcher);
      expect(await clearBump(root, fetcher)).toBe(true);
      held[0](
        new Response(JSON.stringify(state({ plan: plan() })), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      );
      await poll;
      expect(panelOf(root).state?.plan).toBeNull();
    });
  });
});
