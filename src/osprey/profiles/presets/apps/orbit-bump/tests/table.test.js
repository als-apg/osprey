import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  bindControls,
  chooseModel,
  choosePlane,
  clearBump,
  panelOf,
  refresh,
  setSelection,
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

    it('adds a BPM from the dropdown, which then no longer offers it', () => {
      showState(root, state());
      const add = /** @type {HTMLSelectElement} */ (field(root, 'add-bpm'));
      const offered = () => [...add.options].map((option) => option.value).filter(Boolean);
      expect(offered()).toEqual(['B1', 'B2', 'B3']);
      choose(add, 'B3');
      expect(panelOf(root).selection.targets).toEqual({ B3: 0 });
      expect(rowElements(root)).toEqual(['B3']);
      expect(add.value).toBe('');
      expect(offered()).toEqual(['B1', 'B2']);
    });

    it('does not offer a BPM without a reading', () => {
      const readings = orbit();
      readings.position['B1:POS:x'] = null;
      showState(root, state({ orbit: readings }));
      const add = /** @type {HTMLSelectElement} */ (field(root, 'add-bpm'));
      expect([...add.options].map((option) => option.value).filter(Boolean)).toEqual(['B2', 'B3']);
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
    it('defaults to auto and offers the MML lists and manual', () => {
      showState(root, state());
      const rule = /** @type {HTMLSelectElement} */ (field(root, 'rule'));
      expect(rule.value).toBe('auto');
      expect([...rule.options].map((option) => option.value)).toEqual([
        'auto',
        'mml:[-1 1]',
        'mml:[-2 -1 1 2]',
        'mml:[-3 -1 1 3]',
        'manual',
      ]);
    });

    it('changes the selection rule, without discarding a plan', () => {
      const { fetcher, log } = backend(() => state());
      bindControls(root, fetcher);
      showState(root, state({ plan: plan() }));
      const rule = /** @type {HTMLSelectElement} */ (field(root, 'rule'));
      choose(rule, 'mml:[-2 -1 1 2]');
      expect(panelOf(root).selection.rule).toBe('mml:[-2 -1 1 2]');
      expect(log).toEqual([]);
    });

    it('drops manual correctors when the rule leaves manual', () => {
      showState(root, state());
      setSelection(root, { rule: 'manual', correctors: ['H2'] });
      choose(/** @type {HTMLSelectElement} */ (field(root, 'rule')), 'auto');
      expect(panelOf(root).selection.correctors).toEqual([]);
    });

    it('makes an unsolved corrector unselectable and lists it', () => {
      showState(root, state({ models: [model('RING', ['H2', 'V1'])] }));
      setSelection(root, { rule: 'manual' });
      click(stripTick(root, 'H2'));
      expect(panelOf(root).selection.correctors).toEqual([]);
      click(stripTick(root, 'H1'));
      expect(panelOf(root).selection.correctors).toEqual(['H1']);
      const note = field(root, 'unsolved');
      expect(note.hidden).toBe(false);
      // Only the selected plane's unsolved correctors are listed.
      expect(note.textContent).toContain('H2');
      expect(note.textContent).not.toContain('V1');
    });

    it("lists the plan's skipped unsolved correctors", () => {
      showState(
        root,
        state({ models: [model('RING', ['H3'])], plan: plan({ skipped_unsolved: ['H3'] }) })
      );
      const note = field(root, 'unsolved');
      expect(note.hidden).toBe(false);
      expect(note.textContent).toContain('skipped');
      expect(note.textContent).toContain('H3');
    });

    it('hides the unsolved note when every corrector solved', () => {
      showState(root, state());
      expect(field(root, 'unsolved').hidden).toBe(true);
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
