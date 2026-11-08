import { beforeEach, describe, expect, it } from 'vitest';
import { panelOf, setSelection, showState } from '../panel/panel.js';

// The strip: drawn from the selection and the state only, into the body field.

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

const LENGTH = 100;

/**
 * @param {string} element
 * @param {'x' | 'y'} plane
 * @param {number} index
 * @param {number} s
 * @param {string | null} [unit]
 */
function bpm(element, plane, index, s, unit = 'mm') {
  return {
    element,
    plane,
    index,
    s_m: s,
    unit,
    position_addr: `${element}:POS:${plane}`,
    golden_addr: `${element}:GOLD:${plane}`,
  };
}

/**
 * @param {string} element
 * @param {'x' | 'y'} plane
 * @param {number} index
 * @param {number} s
 */
function corrector(element, plane, index, s) {
  return {
    element,
    plane,
    index,
    s_m: s,
    unit: 'A',
    setpoint_addr: `${element}:SP`,
    readback_addr: `${element}:RB`,
  };
}

const BPMS = [
  bpm('B1', 'x', 1, 10),
  bpm('B1', 'y', 1, 10),
  bpm('B2', 'x', 4, 30),
  bpm('B2', 'y', 4, 30),
  bpm('B3', 'x', 7, 50),
  bpm('B3', 'y', 7, 50),
  bpm('B4', 'x', 9, 80),
  bpm('B4', 'y', 9, 80),
];

const CORRECTORS = [
  corrector('H1', 'x', 2, 20),
  corrector('H2', 'x', 5, 40),
  corrector('H3', 'x', 8, 70),
  corrector('V1', 'y', 3, 25),
  corrector('V2', 'y', 6, 45),
];

/**
 * @param {{bpms?: Member[], correctors?: Member[]}} [members]
 * @returns {Model}
 */
function model(members = {}) {
  return {
    name: 'RING',
    length_m: LENGTH,
    orm: { status: 'ready', age_s: 3, version: 1, error: null, unsolved: [] },
    members: { bpms: members.bpms ?? BPMS, correctors: members.correctors ?? CORRECTORS },
  };
}

/** @returns {{position: Record<string, number | null>, golden: Record<string, number | null>}} */
function orbit() {
  /** @type {Record<string, number | null>} */
  const position = {};
  /** @type {Record<string, number | null>} */
  const golden = {};
  for (const member of BPMS) {
    position[member.position_addr] = 0.2;
    golden[member.golden_addr] = 0.1;
  }
  return { position, golden };
}

/**
 * @param {Partial<BumpState>} [over]
 * @returns {BumpState}
 */
function state(over = {}) {
  return {
    models: [model()],
    posture: { allowed: true, reason: null, phrase: '' },
    orbit: orbit(),
    plan: null,
    ...over,
  };
}

/** @param {Element} root */
function svg(root) {
  const strip = root.querySelector('[data-field="body"] svg[data-field="strip"]');
  if (!strip) throw new Error('no strip');
  return strip;
}

/**
 * @param {Element} root
 * @param {string} element
 */
function dot(root, element) {
  const found = svg(root).querySelector(`.ob-bpm[data-element="${element}"]`);
  if (!found) throw new Error(`no dot for ${element}`);
  return found;
}

/**
 * @param {Element} root
 * @param {string} element
 */
function tick(root, element) {
  const found = svg(root).querySelector(`.ob-cor[data-element="${element}"]`);
  if (!found) throw new Error(`no tick for ${element}`);
  return found;
}

/** @param {Element} target */
function click(target) {
  target.dispatchEvent(new MouseEvent('click', { bubbles: true }));
}

/**
 * The x pixel of an element's centre on the strip.
 * @param {Element} element
 */
function xOf(element) {
  if (element.tagName.toLowerCase() === 'circle') return Number(element.getAttribute('cx'));
  return Number(element.getAttribute('x')) + Number(element.getAttribute('width')) / 2;
}

describe('the orbit-bump strip', () => {
  /** @type {HTMLElement} */
  let root;

  beforeEach(() => {
    root = document.createElement('main');
    root.innerHTML = MARKUP;
  });

  it('renders every BPM and corrector of the selected plane at its s-position', () => {
    showState(root, state());
    const dots = [...svg(root).querySelectorAll('.ob-bpm')];
    const ticks = [...svg(root).querySelectorAll('.ob-cor')];
    expect(dots.map((d) => d.getAttribute('data-element'))).toEqual(['B1', 'B2', 'B3', 'B4']);
    expect(ticks.map((t) => t.getAttribute('data-element'))).toEqual(['H1', 'H2', 'H3']);
    // Ordered along s, and proportional to s (geometry in metres, not member count).
    const xs = ['B1', 'B2', 'B3', 'B4'].map((name) => xOf(dot(root, name)));
    const [x10, x30, x50, x80] = xs;
    expect(x10).toBeLessThan(x30);
    expect((x30 - x10) / (x50 - x30)).toBeCloseTo(1, 5);
    expect((x80 - x50) / (x50 - x30)).toBeCloseTo(1.5, 5);
    // Correctors share the same s-axis: H1 at 20 m sits halfway between B1 and B2.
    expect(xOf(tick(root, 'H1'))).toBeCloseTo((x10 + x30) / 2, 5);
  });

  it('switches to the other plane when the selection changes', () => {
    showState(root, state());
    setSelection(root, { plane: 'y' });
    const ticks = [...svg(root).querySelectorAll('.ob-cor')];
    expect(ticks.map((t) => t.getAttribute('data-element'))).toEqual(['V1', 'V2']);
    expect(svg(root).querySelectorAll('.ob-bpm')).toHaveLength(4);
  });

  it('opens on the whole ring', () => {
    showState(root, state());
    expect(panelOf(root).view).toBeNull();
    expect(svg(root).getAttribute('data-s0')).toBe('0');
    expect(svg(root).getAttribute('data-s1')).toBe(String(LENGTH));
  });

  it("labels the axis with the BPM members' unit, unscaled", () => {
    showState(root, state());
    expect(svg(root).querySelector('[data-field="unit"]')?.textContent).toBe('mm');
    const um = model({ bpms: BPMS.map((member) => ({ ...member, unit: 'um' })) });
    showState(root, state({ models: [um] }));
    expect(svg(root).querySelector('[data-field="unit"]')?.textContent).toBe('um');
  });

  it('draws the golden orbit at the zero line and the orbit relative to it', () => {
    showState(root, state());
    const zero = svg(root).querySelector('.ob-golden');
    expect(zero).not.toBeNull();
    const y0 = Number(zero?.getAttribute('y1'));
    // Every BPM reads golden + 0.1: all dots sit above the zero line, at the same height.
    const ys = ['B1', 'B2', 'B3', 'B4'].map((name) => Number(dot(root, name).getAttribute('cy')));
    for (const y of ys) {
      expect(y).toBeLessThan(y0);
      expect(y).toBeCloseTo(ys[0], 5);
    }
    expect(svg(root).querySelector('.ob-orbit')).not.toBeNull();
  });

  it('adds a clicked BPM to the bump and marks its target', () => {
    showState(root, state());
    expect(svg(root).querySelectorAll('.ob-target')).toHaveLength(0);
    click(dot(root, 'B2'));
    expect(panelOf(root).selection.targets).toEqual({ B2: 0 });
    const markers = [...svg(root).querySelectorAll('.ob-target')];
    expect(markers.map((m) => m.getAttribute('data-element'))).toEqual(['B2']);
    expect(dot(root, 'B2').classList.contains('on')).toBe(true);
  });

  it('keeps an existing offset when a BPM already in the bump is clicked', () => {
    showState(root, state());
    setSelection(root, { targets: { B2: 0.3 } });
    click(dot(root, 'B2'));
    expect(panelOf(root).selection.targets).toEqual({ B2: 0.3 });
  });

  it("marks the targets and predicted orbit of a hydrated agent's plan", () => {
    /** @type {Plan} */
    const plan = {
      id: 'agent-1',
      model: 'RING',
      plane: 'x',
      status: 'ready',
      targets: { B2: 0.5, B3: -0.25 },
      correctors: [{ element: 'H1' }, { element: 'H3' }],
      predicted: { B1: 0.1, B2: 0.6, B3: -0.15, B4: 0.1 },
    };
    showState(root, state({ plan }));
    const markers = [...svg(root).querySelectorAll('.ob-target')];
    expect(markers.map((m) => m.getAttribute('data-element'))).toEqual(['B2', 'B3']);
    // The B2 target (+0.5) sits above the zero line, the B3 one (−0.25) below it.
    const y0 = Number(svg(root).querySelector('.ob-golden')?.getAttribute('y1'));
    const yOf = (/** @type {Element} */ m) => Number(m.getAttribute('data-y'));
    expect(yOf(markers[0])).toBeLessThan(y0);
    expect(yOf(markers[1])).toBeGreaterThan(y0);
    expect(svg(root).querySelector('.ob-predicted')).not.toBeNull();
    expect(tick(root, 'H1').classList.contains('on')).toBe(true);
    expect(tick(root, 'H2').classList.contains('on')).toBe(false);
  });

  it("draws each plan corrector's step as a bar on the corrector row, signed", () => {
    /** @type {Plan} */
    const plan = {
      id: 'p2',
      model: 'RING',
      plane: 'x',
      status: 'ready',
      targets: { B2: 0.5 },
      correctors: [
        { element: 'H1', delta: 1 },
        { element: 'H2', delta: -0.5 },
      ],
    };
    showState(root, state({ plan }));
    const bar = (/** @type {string} */ element) => {
      const found = svg(root).querySelector(`.ob-delta[data-element="${element}"]`);
      if (!found) throw new Error(`no bar for ${element}`);
      return found;
    };
    const row = Number(tick(root, 'H1').getAttribute('y'));
    const up = bar('H1');
    const down = bar('H2');
    expect(Number(up.getAttribute('y')) + Number(up.getAttribute('height'))).toBeCloseTo(row);
    expect(Number(down.getAttribute('y'))).toBeCloseTo(row);
    expect(Number(down.getAttribute('height'))).toBeCloseTo(Number(up.getAttribute('height')) / 2);
  });

  it('draws no predicted orbit without a plan', () => {
    showState(root, state());
    expect(svg(root).querySelector('.ob-predicted')).toBeNull();
  });

  it('shows a BPM without golden as a hollow dot that cannot be picked', () => {
    const readings = orbit();
    readings.golden['B3:GOLD:x'] = null;
    showState(root, state({ orbit: readings }));
    const hollow = dot(root, 'B3');
    expect(hollow.classList.contains('hollow')).toBe(true);
    expect(hollow.getAttribute('fill')).toBe('none');
    click(hollow);
    expect(panelOf(root).selection.targets).toEqual({});
    // The orbit line breaks at it: two segments, one either side.
    const line = svg(root).querySelector('.ob-orbit')?.getAttribute('d') ?? '';
    expect(line.match(/M/g)).toHaveLength(2);
  });

  it('shows a BPM without a position reading as a hollow dot that cannot be picked', () => {
    const readings = orbit();
    readings.position['B1:POS:x'] = null;
    showState(root, state({ orbit: readings }));
    expect(dot(root, 'B1').classList.contains('hollow')).toBe(true);
    click(dot(root, 'B1'));
    expect(panelOf(root).selection.targets).toEqual({});
  });

  it('picks a corrector only under the manual rule', () => {
    showState(root, state());
    click(tick(root, 'H2'));
    expect(panelOf(root).selection.correctors).toEqual([]);
    setSelection(root, { rule: 'manual' });
    click(tick(root, 'H2'));
    expect(panelOf(root).selection.correctors).toEqual(['H2']);
    expect(tick(root, 'H2').classList.contains('on')).toBe(true);
    click(tick(root, 'H2'));
    expect(panelOf(root).selection.correctors).toEqual([]);
  });

  it('autoscales the vertical range to the largest value on view', () => {
    showState(root, state());
    const small = Number(dot(root, 'B1').getAttribute('cy'));
    const readings = orbit();
    readings.position['B4:POS:x'] = 10.1;
    showState(root, state({ orbit: readings }));
    const y0 = Number(svg(root).querySelector('.ob-golden')?.getAttribute('y1'));
    // B1's 0.1 offset shrinks towards the zero line once a 10.0 offset is on view.
    expect(Math.abs(Number(dot(root, 'B1').getAttribute('cy')) - y0)).toBeLessThan(
      Math.abs(small - y0)
    );
    expect(Number(dot(root, 'B4').getAttribute('cy'))).toBeGreaterThanOrEqual(0);
  });

  it('draws nothing when no model is offered', () => {
    showState(root, state({ models: [] }));
    expect(root.querySelector('svg[data-field="strip"]')?.childElementCount ?? 0).toBe(0);
  });
});
