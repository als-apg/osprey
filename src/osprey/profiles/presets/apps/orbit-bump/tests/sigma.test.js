import { beforeEach, describe, expect, it, vi } from 'vitest';
import { bindControls, panelOf, setSelection, showState } from '../panel/panel.js';

// The singular-value spectrum in the Correctors head: bars, the cutoff, and the
// count it sets.

/**
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
 * @param {number} s
 */
function bpm(element, s) {
  return {
    element,
    plane: /** @type {'x'} */ ('x'),
    index: s,
    s_m: s,
    unit: 'mm',
    position_addr: `${element}:POS`,
    golden_addr: `${element}:GOLD`,
  };
}

/**
 * @param {string} element
 * @param {number} s
 */
function corrector(element, s) {
  return {
    element,
    plane: /** @type {'x'} */ ('x'),
    index: s,
    s_m: s,
    unit: 'A',
    setpoint_addr: `${element}:SP`,
    readback_addr: `${element}:RB`,
  };
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
    rule: 'auto',
    targets: { B2: 0.5 },
    correctors: [{ element: 'H1' }, { element: 'H2' }],
    sigma_values: [10, 1, 0.01, 1e-6],
    sigma_kept: 3,
    sigma_total: 4,
    sigma_targets: 0,
    sigma_keep: null,
    ...over,
  };
}

/**
 * @param {Partial<BumpState>} [over]
 * @returns {BumpState}
 */
function state(over = {}) {
  return {
    models: [
      {
        name: 'RING',
        length_m: 100,
        orm: { status: 'ready', age_s: 3, version: 1, error: null, unsolved: [] },
        members: {
          bpms: [bpm('B1', 10), bpm('B2', 30), bpm('B3', 50)],
          correctors: [corrector('H1', 20), corrector('H2', 40)],
        },
      },
    ],
    posture: { allowed: true, reason: null, phrase: '' },
    orbit: {
      position: { 'B1:POS': 1, 'B2:POS': 1, 'B3:POS': 1 },
      golden: { 'B1:GOLD': 1, 'B2:GOLD': 1, 'B3:GOLD': 1 },
    },
    plan: null,
    ...over,
  };
}

/**
 * A fetcher that answers Compute with `answer(body)` and anything else with
 * an empty state; records every Compute body.
 * @param {(body: any) => Plan} answer
 */
function backend(answer) {
  /** @type {any[]} */
  const computes = [];
  const fetcher = vi.fn(async (/** @type {string} */ url, /** @type {RequestInit} */ init) => {
    const body = typeof init?.body === 'string' ? JSON.parse(init.body) : null;
    if (url === 'api/compute') {
      computes.push(body);
      return new Response(JSON.stringify(answer(body)), {
        status: 201,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    return new Response(null, { status: 204 });
  });
  return { fetcher: /** @type {typeof fetch} */ (/** @type {unknown} */ (fetcher)), computes };
}

/**
 * @param {Element} root
 * @returns {HTMLElement}
 */
function control(root) {
  const found = root.querySelector('[data-field="sigma"]');
  if (!(found instanceof HTMLElement)) throw new Error('no spectrum');
  return found;
}

/** @param {Element} root */
function plot(root) {
  const found = control(root).querySelector('svg');
  if (!found) throw new Error('no spectrum plot');
  return found;
}

/** @param {Element} root */
function bars(root) {
  return [...plot(root).querySelectorAll('.ob-sigma-bar')];
}

/** @param {Element} root */
function keptCount(root) {
  return bars(root).filter((bar) => bar.classList.contains('kept')).length;
}

/**
 * Give the plot a layout box (happy-dom does none) 96 px wide at the left edge.
 * @param {Element} root
 */
function layout(root) {
  const box = { left: 0, top: 0, width: 96, height: 18, right: 96, bottom: 18, x: 0, y: 0 };
  plot(root).getBoundingClientRect = () => /** @type {DOMRect} */ ({ ...box, toJSON: () => ({}) });
}

/**
 * @param {Element} target
 * @param {string} type
 * @param {number} clientX
 */
function pointer(target, type, clientX) {
  target.dispatchEvent(
    new PointerEvent(type, { clientX, clientY: 9, button: 0, pointerId: 1, bubbles: true })
  );
}

/**
 * @param {Element} target
 * @param {string} key
 */
function key(target, key) {
  target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
  target.dispatchEvent(new KeyboardEvent('keyup', { key, bubbles: true }));
}

describe('the singular-value spectrum', () => {
  /** @type {HTMLElement} */
  let root;

  beforeEach(() => {
    root = document.createElement('main');
    root.innerHTML = MARKUP;
  });

  /**
   * @param {typeof fetch} fetcher
   * @param {BumpState} shown
   */
  function mount(fetcher, shown) {
    bindControls(root, fetcher);
    showState(root, shown);
    layout(root);
  }

  it('is empty and disabled before the first Compute', () => {
    mount(backend(() => plan()).fetcher, state());
    const slider = control(root);
    expect(slider.getAttribute('role')).toBe('slider');
    expect(slider.getAttribute('aria-disabled')).toBe('true');
    expect(slider.hasAttribute('tabindex')).toBe(false);
    expect(slider.hasAttribute('aria-valuenow')).toBe(false);
    expect(bars(root)).toHaveLength(0);
    expect(root.querySelector('[data-field="sigma-k"]')?.textContent).toBe('');
  });

  it('sits in its own row under the Correctors head, not in the head', () => {
    mount(backend(() => plan()).fetcher, state());
    const row = root.querySelector('[data-field="spectrum-row"]');
    expect(row?.contains(control(root))).toBe(true);
    expect(row?.previousElementSibling?.tagName).toBe('H3');
    expect(root.querySelector('[data-field="cor-card"] > h3')?.contains(control(root))).toBe(false);
  });

  it('draws one bar per singular value of the plan, the kept ones filled, with the count', () => {
    mount(backend(() => plan()).fetcher, state({ plan: plan() }));
    const slider = control(root);
    expect(bars(root)).toHaveLength(4);
    expect(keptCount(root)).toBe(3);
    expect(bars(root).map((bar) => bar.classList.contains('kept'))).toEqual([
      true,
      true,
      true,
      false,
    ]);
    expect(root.querySelector('[data-field="sigma-k"]')?.textContent).toBe('σ 3/4');
    expect(slider.getAttribute('aria-disabled')).toBe('false');
    expect(slider.tabIndex).toBe(0);
    expect(slider.getAttribute('aria-valuemin')).toBe('1');
    expect(slider.getAttribute('aria-valuemax')).toBe('4');
    expect(slider.getAttribute('aria-valuenow')).toBe('3');
    // A log scale: each bar is shorter than the one before it.
    const heights = bars(root).map((bar) => Number(bar.getAttribute('height')));
    expect(heights[0]).toBeGreaterThan(heights[1]);
    expect(heights[1]).toBeGreaterThan(heights[2]);
    expect(heights[2]).toBeGreaterThan(heights[3]);
    expect(plot(root).querySelector('.ob-sigma-cut')).not.toBeNull();
  });

  it('gives each bar its value and its ratio to the largest', () => {
    mount(backend(() => plan()).fetcher, state({ plan: plan() }));
    expect(bars(root)[1].getAttribute('data-tip')).toBe('σ2 1 · σ/σmax 0.1');
    expect(bars(root)[3].getAttribute('data-tip')).toBe('σ4 1.00e-6 · σ/σmax 1.00e-7');
  });

  it('marks the target step and the closure step under hit targets', () => {
    const shown = plan({ priority: 'targets', sigma_values: [4, 2, 3, 1], sigma_targets: 2 });
    mount(backend(() => shown).fetcher, state({ plan: shown }));
    expect(plot(root).querySelector('.ob-sigma-split')).not.toBeNull();
    expect(bars(root)[1].getAttribute('data-tip')).toBe('σ2 2 · σ/σmax 0.5 · target step');
    // A closure value is read against the largest closure value.
    expect(bars(root)[3].getAttribute('data-tip')).toBe('σ4 1 · σ/σmax 0.333 · closure step');
  });

  it('leaves the singular values out of the Correctors title tooltip', () => {
    mount(backend(() => plan()).fetcher, state({ plan: plan() }));
    const title = root.querySelector('[data-field="cor-title"]')?.getAttribute('title') ?? '';
    expect(title).toContain('plan #p1');
    expect(title).not.toContain('σ');
  });

  it('previews a dragged cutoff, then sets the count and computes again on release', async () => {
    const { fetcher, computes } = backend((body) =>
      plan({ id: 'p2', sigma_keep: body.sigma_keep, sigma_kept: body.sigma_keep })
    );
    mount(fetcher, state({ plan: plan() }));
    pointer(plot(root), 'pointerdown', 24);
    expect(keptCount(root)).toBe(1);
    expect(computes).toHaveLength(0);
    pointer(plot(root), 'pointermove', 50);
    expect(keptCount(root)).toBe(2);
    pointer(plot(root), 'pointerup', 50);
    expect(panelOf(root).selection.sigma_keep).toBe(2);
    await vi.waitFor(() => expect(panelOf(root).lastSeenPlanId).toBe('p2'));
    expect(computes).toHaveLength(1);
    expect(computes[0].sigma_keep).toBe(2);
    expect(keptCount(root)).toBe(2);
    expect(control(root).getAttribute('aria-valuenow')).toBe('2');
  });

  it('clamps a drag to 1 and to the spectrum', () => {
    mount(backend(() => plan()).fetcher, state({ plan: plan() }));
    pointer(plot(root), 'pointerdown', -40);
    expect(keptCount(root)).toBe(1);
    pointer(plot(root), 'pointermove', 400);
    expect(keptCount(root)).toBe(4);
    pointer(plot(root), 'pointercancel', 400);
    expect(keptCount(root)).toBe(3);
    expect(panelOf(root).selection.sigma_keep).toBeNull();
  });

  it('steps the count with the arrow keys, Home and End, computing on each release', async () => {
    const { fetcher, computes } = backend((body) =>
      plan({ id: `k${body.sigma_keep}`, sigma_keep: body.sigma_keep, sigma_kept: body.sigma_keep })
    );
    mount(fetcher, state({ plan: plan() }));
    const slider = control(root);
    key(slider, 'ArrowLeft');
    expect(panelOf(root).selection.sigma_keep).toBe(2);
    await vi.waitFor(() => expect(panelOf(root).lastSeenPlanId).toBe('k2'));
    key(slider, 'End');
    await vi.waitFor(() => expect(panelOf(root).lastSeenPlanId).toBe('k4'));
    // At the end already: the count does not move and nothing is computed.
    key(slider, 'ArrowRight');
    key(slider, 'Home');
    await vi.waitFor(() => expect(panelOf(root).lastSeenPlanId).toBe('k1'));
    expect(computes.map((body) => body.sigma_keep)).toEqual([2, 4, 1]);
  });

  it('takes no keys and no drag without a spectrum', () => {
    const { fetcher, computes } = backend(() => plan());
    mount(fetcher, state());
    key(control(root), 'ArrowLeft');
    pointer(plot(root), 'pointerdown', 24);
    pointer(plot(root), 'pointerup', 24);
    expect(panelOf(root).selection.sigma_keep).toBeNull();
    expect(computes).toHaveLength(0);
  });

  it('carries the count of a hydrated plan into the selection, and leaves it out by default', async () => {
    const { fetcher, computes } = backend(() => plan({ id: 'p3' }));
    mount(fetcher, state({ plan: plan({ id: 'agent', sigma_keep: 2, sigma_kept: 2 }) }));
    expect(panelOf(root).selection.sigma_keep).toBe(2);
    expect(keptCount(root)).toBe(2);
    setSelection(root, { sigma_keep: null });
    const button = root.querySelector('[data-field="compute"]');
    if (!(button instanceof HTMLButtonElement)) throw new Error('no Compute');
    button.click();
    await vi.waitFor(() => expect(computes).toHaveLength(1));
    expect('sigma_keep' in computes[0]).toBe(false);
  });
});
