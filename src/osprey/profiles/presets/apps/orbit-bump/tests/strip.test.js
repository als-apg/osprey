import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { closeContextMenu } from '/design-system/js/context-menu.js';
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

  describe('the aims of the other BPMs', () => {
    /** @param {Element} root */
    const aims = (root) =>
      [...svg(root).querySelectorAll('.ob-aim')].map((ring) => [
        ring.getAttribute('data-element'),
        Number(ring.getAttribute('cy')),
      ]);

    it('rings every other BPM at its reading under hold, the default', () => {
      showState(root, state());
      expect(aims(root)).toEqual([]);
      setSelection(root, { targets: { B2: 0.3 } });
      const found = aims(root);
      expect(found.map(([element]) => element)).toEqual(['B1', 'B3', 'B4']);
      for (const [element, cy] of found) {
        expect(cy).toBeCloseTo(Number(dot(root, String(element)).getAttribute('cy')), 9);
      }
      const ring = svg(root).querySelector('.ob-aim');
      expect(ring?.getAttribute('fill')).toBe('none');
      // The bump target keeps its filled dot.
      expect(dot(root, 'B2').getAttribute('fill')).not.toBe('none');
    });

    it('rings every other BPM on the golden line under zero', () => {
      showState(root, state());
      setSelection(root, { targets: { B2: 0.3 }, others: 'zero' });
      const y0 = Number(svg(root).querySelector('.ob-golden')?.getAttribute('y1'));
      const found = aims(root);
      expect(found.map(([element]) => element)).toEqual(['B1', 'B3', 'B4']);
      for (const [, cy] of found) expect(cy).toBeCloseTo(y0, 9);
    });

    it('draws no aim under free, nor for a BPM without a reading', () => {
      showState(root, state());
      setSelection(root, { targets: { B2: 0.3 }, others: 'free' });
      expect(aims(root)).toEqual([]);
      const readings = orbit();
      readings.position['B3:POS:x'] = null;
      showState(root, state({ orbit: readings }));
      setSelection(root, { others: 'hold' });
      expect(aims(root).map(([element]) => element)).toEqual(['B1', 'B4']);
    });
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

  it('takes a BPM already in the bump out of it on a click', () => {
    showState(root, state());
    setSelection(root, { targets: { B2: 0.3, B3: 0.1 } });
    click(dot(root, 'B2'));
    expect(panelOf(root).selection.targets).toEqual({ B3: 0.1 });
    expect(dot(root, 'B2').classList.contains('on')).toBe(false);
  });

  it('extends the bump from the last picked BPM on a shift-click', () => {
    showState(root, state());
    click(dot(root, 'B1'));
    dot(root, 'B3').dispatchEvent(new MouseEvent('click', { bubbles: true, shiftKey: true }));
    expect(Object.keys(panelOf(root).selection.targets).sort()).toEqual(['B1', 'B2', 'B3']);
    // Extending again from B3 leaves B1–B3 as they are.
    dot(root, 'B4').dispatchEvent(new MouseEvent('click', { bubbles: true, shiftKey: true }));
    expect(Object.keys(panelOf(root).selection.targets).sort()).toEqual(['B1', 'B2', 'B3', 'B4']);
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

  it('rings the handle of a bump BPM the plan misses', () => {
    /** @type {Plan} */
    const plan = {
      id: 'agent-2',
      model: 'RING',
      plane: 'x',
      status: 'ready',
      targets: { B2: 0.5, B3: 0.5 },
      correctors: [{ element: 'H1' }, { element: 'H3' }],
      predicted: { B1: 0, B2: 0.5, B3: 0.2, B4: 0 },
      missed: ['B3'],
    };
    showState(root, state({ plan }));
    expect(handle(root, 'B3').classList.contains('ob-missed')).toBe(true);
    expect(handle(root, 'B3').querySelector('title')?.textContent).toBe('B3: target missed');
    expect(handle(root, 'B3').querySelector('.ob-miss-ring')).not.toBeNull();
    expect(handle(root, 'B2').classList.contains('ob-missed')).toBe(false);
  });

  it("draws each plan corrector's step as a signed bar about a zero axis above the ticks", () => {
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
    const axis = svg(root).querySelector('[data-field="delta-axis"]');
    expect(axis).not.toBeNull();
    const zero = Number(axis?.getAttribute('y1'));
    expect(zero).toBeLessThan(Number(tick(root, 'H1').getAttribute('y')));
    expect(svg(root).textContent).toContain('ΔA');
    const up = bar('H1');
    const down = bar('H2');
    expect(Number(up.getAttribute('y')) + Number(up.getAttribute('height'))).toBeCloseTo(zero);
    expect(Number(down.getAttribute('y'))).toBeCloseTo(zero);
    expect(Number(down.getAttribute('height'))).toBeCloseTo(Number(up.getAttribute('height')) / 2);
    // The plan's correctors are marked picked, and each tick names its step.
    expect(tick(root, 'H1').classList.contains('picked')).toBe(true);
    expect(tick(root, 'H3').classList.contains('picked')).toBe(false);
    expect(tick(root, 'H2').getAttribute('data-tip')).toBe('H2 · Δ −0.500 A');
  });

  it('draws no Δ axis without a plan', () => {
    showState(root, state());
    expect(svg(root).querySelector('[data-field="delta-axis"]')).toBeNull();
    expect(svg(root).querySelector('.ob-delta')).toBeNull();
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

/**
 * Give the strip a layout box (happy-dom does none) where client pixels are
 * the strip's user units.
 * @param {Element} root
 */
function layout(root) {
  const strip = svg(root);
  const box = { left: 0, top: 0, width: 1000, height: 242, right: 1000, bottom: 242, x: 0, y: 0 };
  strip.getBoundingClientRect = () => /** @type {DOMRect} */ ({ ...box, toJSON: () => ({}) });
}

/**
 * @param {Element} target
 * @param {string} type
 * @param {number} clientX
 * @param {number} clientY
 */
function pointer(target, type, clientX, clientY) {
  target.dispatchEvent(new PointerEvent(type, { clientX, clientY, pointerId: 1, bubbles: true }));
}

/**
 * @param {Element} root
 * @param {string} element
 */
function handle(root, element) {
  const found = svg(root).querySelector(`.ob-target[data-element="${element}"]`);
  if (!(found instanceof SVGElement)) throw new Error(`no handle for ${element}`);
  return found;
}

/**
 * @param {Element} target
 * @param {string} key
 * @param {boolean} [shiftKey]
 */
function press(target, key, shiftKey = false) {
  target.dispatchEvent(new KeyboardEvent('keydown', { key, shiftKey, bubbles: true }));
}

/** @param {Element} target */
function rightClick(target) {
  target.dispatchEvent(
    new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 300, clientY: 60 })
  );
}

/** The rows of the open context menu. */
function menuRows() {
  return /** @type {HTMLElement[]} */ ([
    ...document.querySelectorAll('.osprey-context-menu [role^="menuitem"]'),
  ]);
}

/** @param {string} label */
function menuRow(label) {
  const found = menuRows().find((row) => row.textContent?.startsWith(label));
  if (!found) throw new Error(`no menu row ${label}`);
  return found;
}

describe('picking and shaping on the strip', () => {
  /** @type {HTMLElement} */
  let root;

  beforeEach(() => {
    root = document.createElement('main');
    root.innerHTML = MARKUP;
    document.body.append(root);
    showState(root, state());
    layout(root);
  });

  afterEach(() => {
    closeContextMenu();
    root.remove();
  });

  it('adds every BPM a drag across the empty plot spans, at a zero offset', () => {
    const bg = svg(root).querySelector('.ob-bg');
    if (!bg) throw new Error('no background');
    // From H1 (20 m) to H3 (70 m): B2 (30 m) and B3 (50 m).
    pointer(bg, 'pointerdown', xOf(tick(root, 'H1')), 60);
    pointer(svg(root), 'pointermove', 500, 60);
    expect(svg(root).querySelector('.ob-marquee')).not.toBeNull();
    pointer(svg(root), 'pointerup', xOf(tick(root, 'H3')), 60);
    expect(panelOf(root).selection.targets).toEqual({ B2: 0, B3: 0 });
    expect(svg(root).querySelector('.ob-marquee')).toBeNull();
    expect(panelOf(root).view).toBeNull();
    // The bump's band and its grip, named by its ends.
    expect(svg(root).querySelector('.ob-band')).not.toBeNull();
    expect(svg(root).querySelector('.ob-grip-text')?.textContent).toBe('B2–B3 · 2 BPM  ⇕');
  });

  it('leaves out a BPM without a reading from a drag', () => {
    const readings = orbit();
    readings.position['B2:POS:x'] = null;
    showState(root, state({ orbit: readings }));
    layout(root);
    const bg = svg(root).querySelector('.ob-bg');
    if (!bg) throw new Error('no background');
    pointer(bg, 'pointerdown', 40, 60);
    pointer(svg(root), 'pointerup', 990, 60);
    expect(Object.keys(panelOf(root).selection.targets).sort()).toEqual(['B1', 'B3', 'B4']);
  });

  it('drags a target handle vertically in nudge steps, on the scale it started on', () => {
    setSelection(root, { targets: { B2: 0 } });
    const y0 = Number(handle(root, 'B2').getAttribute('data-y'));
    // The plot's half-height (69 units) is the scale: 0.1 × 1.15 = 0.115 mm.
    pointer(handle(root, 'B2'), 'pointerdown', 300, y0);
    pointer(svg(root), 'pointermove', 300, y0 - 34.5);
    // 0.0575 mm, snapped to the 0.01 mm step.
    expect(panelOf(root).selection.targets.B2).toBeCloseTo(0.06, 9);
    expect(svg(root).classList.contains('dragging')).toBe(true);
    pointer(svg(root), 'pointermove', 300, y0 + 72);
    expect(panelOf(root).selection.targets.B2).toBeCloseTo(-0.12, 9);
    pointer(svg(root), 'pointerup', 300, y0 + 72);
    expect(svg(root).classList.contains('dragging')).toBe(false);
    expect(panelOf(root).amp).toBeCloseTo(-0.12, 9);
    expect(handle(root, 'B2').querySelector('.ob-handle-label')?.textContent).toBe('−0.120');
  });

  it('moves every target by the same amount when the grip is dragged', () => {
    setSelection(root, { targets: { B2: 0.1, B3: -0.05 } });
    const grip = svg(root).querySelector('[data-grip]');
    if (!grip) throw new Error('no grip');
    pointer(grip, 'pointerdown', 400, 170);
    pointer(svg(root), 'pointermove', 400, 140);
    pointer(svg(root), 'pointerup', 400, 140);
    const { B2, B3 } = panelOf(root).selection.targets;
    expect(B2 - 0.1).toBeGreaterThan(0);
    expect(B2 - 0.1).toBeCloseTo(B3 + 0.05, 9);
  });

  it('nudges a focused handle with the arrow keys, ×10 with shift, and removes it', () => {
    setSelection(root, { targets: { B2: 0, B3: 0 } });
    handle(root, 'B2').focus();
    press(handle(root, 'B2'), 'ArrowUp');
    expect(panelOf(root).selection.targets.B2).toBeCloseTo(0.01, 9);
    press(handle(root, 'B2'), 'ArrowDown', true);
    expect(panelOf(root).selection.targets.B2).toBeCloseTo(-0.09, 9);
    // The redrawn handle keeps the focus.
    expect(document.activeElement).toBe(handle(root, 'B2'));
    press(handle(root, 'B2'), 'ArrowRight');
    expect(document.activeElement).toBe(handle(root, 'B3'));
    press(handle(root, 'B3'), 'Delete');
    expect(panelOf(root).selection.targets).toEqual({ B2: -0.09 });
    expect(handle(root, 'B2').getAttribute('role')).toBe('slider');
    expect(handle(root, 'B2').getAttribute('tabindex')).toBe('0');
  });

  it('opens a context menu on a dot that adds it to the bump', () => {
    rightClick(dot(root, 'B2'));
    expect(document.querySelector('.osprey-context-menu')).not.toBeNull();
    expect(menuRows().some((row) => row.textContent?.startsWith('Whole ring'))).toBe(true);
    expect(menuRow('Whole ring').getAttribute('aria-disabled')).toBe('true');
    menuRow('Add to bump').click();
    expect(panelOf(root).selection.targets).toEqual({ B2: 0 });
  });

  it('inverts the bump from the grip menu, and sets the nudge step from any menu', () => {
    setSelection(root, { targets: { B2: 0.1, B3: -0.05 } });
    const grip = svg(root).querySelector('[data-grip]');
    if (!grip) throw new Error('no grip');
    rightClick(grip);
    menuRow('Invert the bump').click();
    expect(panelOf(root).selection.targets).toEqual({ B2: -0.1, B3: 0.05 });
    rightClick(svg(root).querySelector('[data-grip]') ?? grip);
    menuRow('100 µm').click();
    expect(panelOf(root).step).toBe(0.1);
  });
});
