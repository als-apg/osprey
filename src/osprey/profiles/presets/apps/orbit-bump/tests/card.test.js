import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  applyPlan,
  bindControls,
  compute,
  openConfirm,
  panelOf,
  refresh,
  setSelection,
  showState,
} from '../panel/panel.js';

// Compute → plan card → Apply… confirm card → apply outcomes.

/**
 * @typedef {import('../panel/panel.js').Member} Member
 * @typedef {import('../panel/panel.js').Model} Model
 * @typedef {import('../panel/panel.js').Plan} Plan
 * @typedef {import('../panel/panel.js').PlanCorrector} PlanCorrector
 * @typedef {import('../panel/panel.js').BumpState} BumpState
 * @typedef {import('../panel/panel.js').OrmStatus} OrmStatus
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
 * @returns {Member}
 */
function bpm(element, s) {
  return {
    element,
    plane: 'x',
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
 * @returns {Member}
 */
function corrector(element, s) {
  return {
    element,
    plane: 'x',
    index: s,
    s_m: s,
    unit: 'A',
    setpoint_addr: `${element}:SP`,
    readback_addr: `${element}:RB`,
  };
}

const BPMS = [bpm('B1', 10), bpm('B2', 30), bpm('B3', 50)];
const CORRECTORS = [corrector('H1', 20), corrector('H2', 40)];

/**
 * @param {string} name
 * @param {OrmStatus} [status]
 * @returns {Model}
 */
function model(name, status = 'ready') {
  return {
    name,
    length_m: 100,
    orm: { status, age_s: 3, version: 1, error: null, unsolved: [] },
    members: { bpms: BPMS, correctors: CORRECTORS },
  };
}

/**
 * @param {Partial<PlanCorrector>} [over]
 * @returns {PlanCorrector}
 */
function row(over = {}) {
  return {
    element: 'H1',
    setpoint_addr: 'H1:SP',
    readback_addr: 'H1:RB',
    now: 1,
    readback_at_compute: 1,
    new: 1.5,
    delta: 0.5,
    unit: 'A',
    verdict: {
      allowed: true,
      reason: null,
      declared: true,
      enforced: true,
      mode: 'exclusive',
      limits: { min: -10, max: 10, max_step: null },
    },
    tolerance: 0.01,
    ...over,
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
    targets: { B2: 0.5 },
    setpoint_digest: 'abc123',
    correctors: [
      row(),
      row({ element: 'H2', setpoint_addr: 'H2:SP', now: -2, new: -2.25, delta: -0.25 }),
    ],
    predicted: { B1: 1, B2: 1.5, B3: 1 },
    leakage: [
      { element: 'B1', predicted_residual: 0.002 },
      { element: 'B3', predicted_residual: -0.004 },
    ],
    predicted_closure_max: 0.004,
    sigma_kept: 2,
    sigma_total: 2,
    skipped_unsolved: [],
    dropped_bpms: [],
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
    orbit: {
      position: { 'B1:POS': 1, 'B2:POS': 1, 'B3:POS': 1 },
      golden: { 'B1:GOLD': 1, 'B2:GOLD': 1, 'B3:GOLD': 1 },
    },
    plan: null,
    ...over,
  };
}

/**
 * A fetcher routing by url; records every call with its parsed body.
 * @param {{
 *   current?: () => BumpState,
 *   compute?: () => Response,
 *   apply?: () => Response,
 * }} routes
 */
function backend(routes) {
  /** @type {{url: string, method: string, body: any}[]} */
  const calls = [];
  const fetcher = vi.fn(async (/** @type {string} */ url, /** @type {RequestInit} */ init) => {
    const body = typeof init?.body === 'string' ? JSON.parse(init.body) : null;
    calls.push({ url, method: init?.method ?? 'GET', body });
    if (url === 'api/compute' && routes.compute) return routes.compute();
    if (url === 'api/apply' && routes.apply) return routes.apply();
    return json(routes.current ? routes.current() : state());
  });
  return { fetcher: /** @type {typeof fetch} */ (/** @type {unknown} */ (fetcher)), calls };
}

/**
 * @param {unknown} body
 * @param {number} [status]
 */
function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/**
 * @param {string} text
 * @param {number} status
 */
function text(text, status) {
  return new Response(text, { status });
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
 * @param {string} name
 * @returns {HTMLButtonElement}
 */
function button(root, name) {
  const found = field(root, name);
  if (!(found instanceof HTMLButtonElement)) throw new Error(`${name} is no button`);
  return found;
}

/**
 * @param {Element} root
 * @param {string} element
 * @returns {HTMLElement}
 */
function planRow(root, element) {
  const found = field(root, 'card').querySelector(`tr[data-element="${element}"]`);
  if (!(found instanceof HTMLElement)) throw new Error(`no plan row for ${element}`);
  return found;
}

/**
 * @param {Element} root
 * @param {string} element
 * @param {string} cell
 */
function cell(root, element, cell) {
  return planRow(root, element).querySelector(`[data-cell="${cell}"]`)?.textContent ?? '';
}

describe('the compute and apply card', () => {
  /** @type {HTMLElement} */
  let root;

  beforeEach(() => {
    root = document.createElement('main');
    root.innerHTML = MARKUP;
  });

  /**
   * @param {typeof fetch} fetcher
   * @param {BumpState} [shown]
   */
  function mount(fetcher, shown = state()) {
    bindControls(root, fetcher);
    showState(root, shown);
  }

  // -- Compute button -------------------------------------------------------

  it('disables Compute until the ORM is ready', () => {
    const { fetcher } = backend({});
    mount(fetcher, state({ models: [model('RING', 'building')] }));
    setSelection(root, { targets: { B2: 0.5 } });
    expect(button(root, 'compute').disabled).toBe(true);
    showState(root, state());
    expect(button(root, 'compute').disabled).toBe(false);
  });

  it('disables Compute without a target', () => {
    const { fetcher } = backend({});
    mount(fetcher);
    expect(button(root, 'compute').disabled).toBe(true);
    setSelection(root, { targets: { B2: 0.5 } });
    expect(button(root, 'compute').disabled).toBe(false);
  });

  it('sends no Compute without a model', async () => {
    const { fetcher, calls } = backend({});
    mount(fetcher, state({ models: [] }));
    setSelection(root, { targets: { B2: 0.5 } });
    expect(button(root, 'compute').hidden).toBe(true);
    await compute(root);
    expect(calls.filter((call) => call.url === 'api/compute')).toHaveLength(0);
  });

  it('posts the selected model, plane, targets and rule, and takes the plan id', async () => {
    const { fetcher, calls } = backend({ compute: () => json(plan({ id: 'p7' }), 201) });
    mount(fetcher);
    setSelection(root, { targets: { B2: 0.5 } });
    button(root, 'compute').click();
    await vi.waitFor(() => expect(panelOf(root).lastSeenPlanId).toBe('p7'));
    const posted = calls.find((call) => call.url === 'api/compute');
    expect(posted?.method).toBe('POST');
    expect(posted?.body).toEqual({
      model: 'RING',
      plane: 'x',
      targets: { B2: 0.5 },
      rule: 'auto',
      priority: 'closure',
      others: 'hold',
      all_correctors: false,
    });
    expect(field(root, 'card').hidden).toBe(false);
  });

  it('sends the manual correctors only under the manual rule', async () => {
    const { fetcher, calls } = backend({ compute: () => json(plan(), 201) });
    mount(fetcher);
    setSelection(root, { targets: { B2: 0.5 }, rule: 'manual', correctors: ['H1', 'H2'] });
    await compute(root);
    expect(calls.find((call) => call.url === 'api/compute')?.body).toEqual({
      model: 'RING',
      plane: 'x',
      targets: { B2: 0.5 },
      rule: 'manual',
      priority: 'closure',
      others: 'hold',
      all_correctors: false,
      correctors: ['H1', 'H2'],
    });
  });

  it('carries the chosen model in a two-model Compute body', async () => {
    const { fetcher, calls } = backend({
      compute: () => json(plan({ model: 'OTHER' }), 201),
    });
    mount(fetcher, state({ models: [model('RING'), model('OTHER')] }));
    setSelection(root, { model: 'OTHER', targets: { B2: 0.5 } });
    await compute(root);
    expect(calls.find((call) => call.url === 'api/compute')?.body.model).toBe('OTHER');
  });

  it('shows a read-only notice and hides Compute and Apply on a 403', async () => {
    const { fetcher } = backend({ compute: () => text('writes are off for this target', 403) });
    mount(fetcher, state({ plan: plan() }));
    setSelection(root, { targets: { B2: 0.5 } });
    await compute(root);
    expect(field(root, 'notice').hidden).toBe(false);
    expect(field(root, 'notice').textContent).toContain('read-only');
    expect(field(root, 'notice').textContent).toContain('writes are off for this target');
    expect(button(root, 'compute').hidden).toBe(true);
    expect(button(root, 'apply').hidden).toBe(true);
  });

  it("shows a JSON refusal's detail, not the raw body, in the read-only notice", async () => {
    const { fetcher } = backend({
      compute: () => json({ detail: 'Writes are off; turn it on from the chip.' }, 403),
    });
    mount(fetcher, state({ plan: plan() }));
    setSelection(root, { targets: { B2: 0.5 } });
    await compute(root);
    expect(field(root, 'notice').textContent).toBe(
      'read-only: Writes are off; turn it on from the chip.'
    );
  });

  it("shows a JSON 409's detail from apply", async () => {
    const { fetcher } = backend({
      apply: () => json({ detail: 'plan p1 is not the current plan' }, 409),
    });
    mount(fetcher, state({ plan: plan() }));
    openConfirm(root);
    await applyPlan(root);
    expect(field(root, 'error').textContent).toBe('plan p1 is not the current plan');
  });

  it('shows a 409 from Compute verbatim', async () => {
    const { fetcher } = backend({
      compute: () => text('the ORM of RING is building; Compute when it is ready', 409),
    });
    mount(fetcher);
    setSelection(root, { targets: { B2: 0.5 } });
    await compute(root);
    expect(field(root, 'error').textContent).toBe(
      'the ORM of RING is building; Compute when it is ready'
    );
    expect(button(root, 'compute').hidden).toBe(false);
  });

  it('leaves the rule and the card unchanged when the poll after Compute carries the plan', async () => {
    const computed = plan({ id: 'p9' });
    const { fetcher } = backend({
      compute: () => json(computed, 201),
      current: () => state({ plan: computed }),
    });
    mount(fetcher);
    setSelection(root, { targets: { B2: 0.5 }, rule: 'mml:[-1 1]' });
    await compute(root);
    const before = field(root, 'card').innerHTML;
    await refresh(root, fetcher);
    expect(panelOf(root).selection.rule).toBe('mml:[-1 1]');
    expect(panelOf(root).selection.correctors).toEqual([]);
    expect(field(root, 'card').innerHTML).toBe(before);
  });

  // -- plan card ------------------------------------------------------------

  it('lists now, delta, new and the declared limit per row in the Correctors box', () => {
    const { fetcher } = backend({});
    mount(fetcher, state({ plan: plan() }));
    expect(field(root, 'card').closest('.osprey-card')?.textContent).toContain('Correctors');
    expect([...planRow(root, 'H1').children].map((td) => td.getAttribute('data-cell'))).toEqual([
      'corrector',
      'now',
      'delta',
      'new',
      'verdict',
      'outcome',
    ]);
    expect(cell(root, 'H1', 'now')).toBe('+1.000');
    expect(cell(root, 'H1', 'new')).toBe('+1.500');
    expect(cell(root, 'H1', 'delta')).toBe('+0.500');
    expect(cell(root, 'H1', 'verdict')).toBe('limit -10..10');
    expect(cell(root, 'H2', 'new')).toBe('−2.250');
  });

  it('labels a row without a declared limit with the limits mode', () => {
    const { fetcher } = backend({});
    const free = row({
      verdict: {
        allowed: true,
        reason: null,
        declared: false,
        enforced: true,
        mode: 'optional',
        limits: null,
      },
    });
    mount(fetcher, state({ plan: plan({ correctors: [free] }) }));
    expect(cell(root, 'H1', 'verdict')).toBe('no declared limit (mode: optional)');
  });

  it('labels a row when limits checking is off', () => {
    const { fetcher } = backend({});
    const off = row({
      verdict: {
        allowed: true,
        reason: null,
        declared: false,
        enforced: false,
        mode: null,
        limits: null,
      },
    });
    mount(fetcher, state({ plan: plan({ correctors: [off] }) }));
    expect(cell(root, 'H1', 'verdict')).toBe('limits checking off');
  });

  it('highlights a refusal with its reason and the overshoot', () => {
    const { fetcher } = backend({});
    const refused = row({
      new: 12.5,
      delta: 11.5,
      verdict: {
        allowed: false,
        reason: 'MAX_EXCEEDED: value above the maximum',
        declared: true,
        enforced: true,
        mode: 'exclusive',
        limits: { min: -10, max: 10, max_step: null },
      },
    });
    mount(fetcher, state({ plan: plan({ status: 'not_applicable', correctors: [refused] }) }));
    const verdict = planRow(root, 'H1').querySelector('[data-cell="verdict"]');
    expect(verdict?.classList.contains('ob-refused')).toBe(true);
    expect(verdict?.textContent).toContain('MAX_EXCEEDED: value above the maximum');
    expect(verdict?.textContent).toContain('over by 2.500');
    expect(button(root, 'apply').disabled).toBe(true);
  });

  it('computes the overshoot of a step cap from the delta', () => {
    const { fetcher } = backend({});
    const refused = row({
      new: 4,
      delta: 3,
      verdict: {
        allowed: false,
        reason: 'STEP_EXCEEDED: step too large',
        declared: true,
        enforced: true,
        mode: 'exclusive',
        limits: { min: -10, max: 10, max_step: 1 },
      },
    });
    mount(fetcher, state({ plan: plan({ status: 'not_applicable', correctors: [refused] }) }));
    expect(cell(root, 'H1', 'verdict')).toContain('over by 2.000');
  });

  it('notes a row without a tolerance in its tooltip and on the confirm card', () => {
    const { fetcher } = backend({});
    mount(fetcher, state({ plan: plan({ correctors: [row({ tolerance: null })] }) }));
    expect(planRow(root, 'H1').title).toBe('no tolerance: not staleness-checked');
    openConfirm(root);
    expect(field(root, 'confirm').textContent).toContain('no tolerance: not staleness-checked');
  });

  it('puts the closure line with sigma, leakage, dropped BPMs and skipped correctors in the title tooltip', () => {
    const { fetcher } = backend({});
    mount(
      fetcher,
      state({
        plan: plan({
          sigma_kept: 1,
          sigma_total: 2,
          dropped_bpms: ['B3'],
          skipped_unsolved: ['H9'],
        }),
      })
    );
    const closure = field(root, 'cor-title').title;
    expect(closure).toContain('plan #p1');
    expect(closure).toContain('σ kept 1/2');
    expect(closure).toContain('max predicted leakage +0.004 mm');
    expect(closure).toContain('dropped BPMs: B3');
    expect(closure).toContain('skipped (ORM unsolved): H9');
  });

  it('disables Apply on a not applicable plan and says why', () => {
    const { fetcher } = backend({});
    mount(
      fetcher,
      state({
        plan: plan({ status: 'not_applicable', reason: 'C3 refused: MAX_EXCEEDED' }),
      })
    );
    expect(button(root, 'apply').disabled).toBe(true);
    expect(field(root, 'card').textContent).toContain('C3 refused: MAX_EXCEEDED');
  });

  it('names the setpoint count on Apply and the plan at the end of the actions row', () => {
    const { fetcher } = backend({});
    mount(fetcher, state({ plan: plan() }));
    expect(button(root, 'apply').textContent).toBe('Apply 2 setpoints…');
    expect(field(root, 'plan-status').textContent).toBe('plan #p1');
    expect(field(root, 'plan-status').hidden).toBe(false);
  });

  it('says nothing in the status chip without targets or a plan', () => {
    const { fetcher } = backend({});
    mount(fetcher);
    expect(field(root, 'plan-status').hidden).toBe(true);
    setSelection(root, { targets: { B2: 0.5 } });
    expect(field(root, 'plan-status').textContent).toBe('edited · not computed');
    expect(field(root, 'cor-empty').hidden).toBe(false);
  });

  it('gates Apply once the targets are edited after Compute, until Compute runs again', async () => {
    const { fetcher } = backend({ compute: () => json(plan({ id: 'p2', targets: { B2: 0.6 } })) });
    mount(fetcher, state({ plan: plan() }));
    expect(button(root, 'apply').disabled).toBe(false);
    setSelection(root, { targets: { B2: 0.6 } });
    expect(field(root, 'plan-status').textContent).toBe('edited · not computed');
    expect(button(root, 'apply').disabled).toBe(true);
    expect(button(root, 'apply').title).toBe('edited since plan #p1: Compute again');
    openConfirm(root);
    expect(field(root, 'confirm').hidden).toBe(true);
    // A rule change is an edit too.
    setSelection(root, { targets: { B2: 0.5 }, rule: 'auto', correctors: [] });
    expect(field(root, 'plan-status').textContent).toBe('edited · not computed');
    setSelection(root, { targets: { B2: 0.6 } });
    await compute(root);
    expect(field(root, 'plan-status').textContent).toBe('plan #p2');
    expect(button(root, 'apply').disabled).toBe(false);
    expect(button(root, 'apply').hasAttribute('title')).toBe(false);
  });

  it('keeps Compute enabled with writes off and gates only Apply', () => {
    const { fetcher } = backend({});
    mount(
      fetcher,
      state({
        plan: plan(),
        posture: { allowed: false, reason: 'off', phrase: 'Writes are off' },
      })
    );
    setSelection(root, { targets: { B2: 0.5 } });
    expect(button(root, 'compute').hidden).toBe(false);
    expect(button(root, 'compute').disabled).toBe(false);
    expect(button(root, 'apply').disabled).toBe(true);
    expect(button(root, 'apply').title).toBe('Writes are off');
  });

  it('shows the posture phrase on a disabled Apply', () => {
    const { fetcher } = backend({});
    mount(
      fetcher,
      state({
        plan: plan(),
        posture: { allowed: false, reason: 'blocked', phrase: 'Writes are off: read-only session' },
      })
    );
    expect(button(root, 'apply').disabled).toBe(true);
    expect(button(root, 'apply').title).toBe('Writes are off: read-only session');
  });

  it('shows no card and no Apply without a plan', () => {
    const { fetcher } = backend({});
    mount(fetcher);
    expect(field(root, 'card').hidden).toBe(true);
    expect(button(root, 'apply').hidden).toBe(true);
  });

  // -- confirm and apply ----------------------------------------------------

  it('opens a confirm card listing every setpoint and the closure', () => {
    const { fetcher } = backend({});
    mount(fetcher, state({ plan: plan() }));
    button(root, 'apply').click();
    const confirm = field(root, 'confirm');
    expect(confirm.hidden).toBe(false);
    expect(confirm.querySelector('[data-cell="channel"]')?.textContent).toBe('H1:SP');
    expect(confirm.textContent).toContain('H2:SP');
    expect(confirm.textContent).toContain('+1.500');
    expect(confirm.textContent).toContain('−2.250');
    expect(confirm.textContent).toContain('σ kept 2/2');
  });

  it('posts the plan id and setpoint digest on confirm and shows the outcomes', async () => {
    const applied = plan({
      status: 'partially_applied',
      correctors: [
        row({ outcome: 'confirmed', readback: 1.5, settled: true }),
        row({ element: 'H2', setpoint_addr: 'H2:SP', outcome: 'not_attempted' }),
      ],
      residual_after: { B2: 0.01, B1: -0.002 },
    });
    const { fetcher, calls } = backend({ apply: () => json(applied) });
    mount(fetcher, state({ plan: plan() }));
    openConfirm(root);
    button(root, 'confirm-apply').click();
    await vi.waitFor(() =>
      expect(field(root, 'plan-status').textContent).toBe('partially_applied')
    );
    expect(calls.find((call) => call.url === 'api/apply')?.body).toEqual({
      plan_id: 'p1',
      setpoint_digest: 'abc123',
    });
    expect(field(root, 'confirm').hidden).toBe(true);
    expect(cell(root, 'H1', 'outcome')).toBe('confirmed');
    expect(cell(root, 'H2', 'outcome')).toBe('not attempted');
    const after = field(root, 'cor-title').title;
    expect(after).toContain('residual after (mm)');
    expect(after).toContain('B2 +0.010');
    expect(after).toContain('B1 −0.002');
  });

  it.each(['mismatch', 'refused', 'failed', 'unconfirmed'])('shows the %s outcome', (outcome) => {
    const { fetcher } = backend({});
    const done = plan({
      status: 'partially_applied',
      correctors: [row({ outcome: /** @type {any} */ (outcome) })],
    });
    mount(fetcher, state({ plan: done }));
    expect(cell(root, 'H1', 'outcome')).toBe(outcome);
  });

  it.each([
    ['writing', 'applying · writing'],
    ['settling', 'applying · settling'],
  ])('shows an applying plan in its %s phase', (phase, label) => {
    const { fetcher } = backend({});
    mount(
      fetcher,
      state({
        plan: plan({ status: 'applying', phase: /** @type {'writing' | 'settling'} */ (phase) }),
      })
    );
    expect(field(root, 'plan-status').textContent).toBe(label);
    expect(button(root, 'apply').disabled).toBe(true);
  });

  it.each([
    [409, 'plan p1 is not the current plan; the current plan is p2'],
    [403, 'Writes are off: read-only session'],
  ])('shows a %s from apply verbatim', async (status, message) => {
    const { fetcher } = backend({ apply: () => text(message, status) });
    mount(fetcher, state({ plan: plan() }));
    openConfirm(root);
    await applyPlan(root);
    expect(field(root, 'error').hidden).toBe(false);
    expect(field(root, 'error').textContent).toBe(message);
  });

  it('closes the confirm card on cancel without posting', () => {
    const { fetcher, calls } = backend({});
    mount(fetcher, state({ plan: plan() }));
    openConfirm(root);
    button(root, 'confirm-cancel').click();
    expect(field(root, 'confirm').hidden).toBe(true);
    expect(calls.filter((call) => call.url === 'api/apply')).toHaveLength(0);
  });

  it('opens no confirm card for a plan that is not ready', () => {
    const { fetcher } = backend({});
    mount(fetcher, state({ plan: plan({ status: 'not_applicable' }) }));
    openConfirm(root);
    expect(field(root, 'confirm').hidden).toBe(true);
  });

  // -- targets the plan misses ----------------------------------------------

  describe('targets the plan misses', () => {
    const missing = () =>
      plan({ missed: ['B2', 'B3'], miss: { B2: -0.1104, B3: 0.02 }, target_miss_max: 0.1104 });

    it('warns in the actions row with the largest miss, naming the BPMs in its tooltip', () => {
      const { fetcher } = backend({});
      mount(fetcher, state({ plan: missing() }));
      const chip = field(root, 'miss');
      expect(chip.hidden).toBe(false);
      expect(chip.classList.contains('osprey-chip--warn')).toBe(true);
      expect(field(root, 'miss-text').textContent).toBe('targets missed · up to 0.110 mm');
      expect(chip.title).toBe('missed: B2, B3');
    });

    it('shows no warning for a plan that meets its targets', () => {
      const { fetcher } = backend({});
      mount(fetcher, state({ plan: plan({ missed: [], target_miss_max: 0.0004 }) }));
      expect(field(root, 'miss').hidden).toBe(true);
    });

    it('keeps Apply possible and names the miss on the confirm card', () => {
      const { fetcher } = backend({});
      mount(fetcher, state({ plan: missing() }));
      expect(button(root, 'apply').disabled).toBe(false);
      openConfirm(root);
      expect(field(root, 'confirm-miss').textContent).toBe(
        'targets missed · up to 0.110 mm at B2, B3'
      );
    });

    it('names the correctors left at a limit in the closure line', () => {
      const { fetcher } = backend({});
      mount(fetcher, state({ plan: plan({ bounded: ['H1'] }) }));
      openConfirm(root);
      expect(field(root, 'confirm').textContent).toContain('at limit: H1');
    });
  });
});
