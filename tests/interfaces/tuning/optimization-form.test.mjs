// @ts-check
/**
 * Render + wiring tests for the OSPREY Tuning optimization form
 * (optimization-form.js).
 *
 * Covers the three public/behavioral surfaces of the module:
 *
 *   1. `renderTable` — row structure (count, selected class, checkbox state,
 *      LHS vs BO column set, empty-state colspan) and hostile-`pv_name`
 *      escaping (the rendered cell must be structurally inert).
 *   2. `validateForm` — reached through the wired Start button: valid inputs
 *      start the run; each invalid case blocks it and surfaces the exact
 *      validation message.
 *   3. Control wiring — Start/Pause/Resume/Cancel buttons invoke the right API
 *      calls and drive the expected state transitions and disabled/visibility
 *      toggles.
 *
 * The shared `state.js` store is a module-level singleton whose constructor
 * reads `sessionStorage`, and `initOptimizationForm()` registers persistent
 * listeners. For per-test isolation each test clears `sessionStorage`, rebuilds
 * the DOM fixture, re-imports the module graph under `vi.resetModules()`
 * (so the store, api client, and form share one fresh registry), then wires a
 * fresh form.
 *
 * Pure DOM guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/tuning/optimization-form.test.mjs
 *
 * @module tests/interfaces/tuning/optimization-form
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

const STATE_PATH = '../../../src/osprey/interfaces/tuning/static/js/state.js';
const API_PATH = '../../../src/osprey/interfaces/tuning/static/js/api.js';
const FORM_PATH = '../../../src/osprey/interfaces/tuning/static/js/optimization-form.js';

/** @type {typeof import('../../../src/osprey/interfaces/tuning/static/js/state.js').state} */
let store;
/** @type {typeof import('../../../src/osprey/interfaces/tuning/static/js/api.js').api} */
let apiClient;
/** @type {typeof import('../../../src/osprey/interfaces/tuning/static/js/optimization-form.js').renderTable} */
let renderTable;
/** @type {typeof import('../../../src/osprey/interfaces/tuning/static/js/optimization-form.js').initOptimizationForm} */
let initOptimizationForm;

// ---- Typed DOM accessors (getElementById returns HTMLElement | null) ----

/** @param {string} id  @returns {HTMLElement} */
const byId = (id) => /** @type {HTMLElement} */ (document.getElementById(id));
/** @param {string} id  @returns {HTMLButtonElement} */
const btn = (id) => /** @type {HTMLButtonElement} */ (document.getElementById(id));
/** @param {string} id  @returns {HTMLSelectElement} */
const select = (id) => /** @type {HTMLSelectElement} */ (document.getElementById(id));

/** Flush the microtask + timer queue so awaited async handlers settle.
 *  @returns {Promise<void>} */
function flush() {
  return new Promise((resolve) => {
    setTimeout(() => resolve(), 0);
  });
}

/**
 * Assert a rendered container is structurally inert — no injected element and
 * no inline `on*` handler attribute. Mirrors the tuning security suite:
 * inspect the parsed DOM, not serialized `innerHTML`.
 * @param {HTMLElement} container  @returns {void}
 */
function assertInert(container) {
  expect(container.querySelector('img, svg, script')).toBeNull();
  for (const node of container.querySelectorAll('*')) {
    for (const attr of node.attributes) {
      expect(attr.name.toLowerCase().startsWith('on')).toBe(false);
    }
  }
}

/**
 * Full DOM fixture: every element id `initOptimizationForm()` looks up (~30),
 * wrapped in `#optimization-form` (queried by `updateControlButtons`), with the
 * two `<tbody>` sinks wrapped in `<table>` and the validation banner. Objective
 * `<select>`s carry an `obj-x` option so `.value` can be set in tests.
 * @returns {string}
 */
function fixtureHtml() {
  const objectiveOptions = '<option value="">Select...</option><option value="obj-x">obj-x</option>';
  return `
    <div id="optimization-form">
      <select id="lhs-objective">${objectiveOptions}</select>
      <input id="lhs-samples" value="20">
      <table><tbody id="lhs-table-body"></tbody></table>
      <input id="lhs-add-pv">
      <button id="lhs-check-value"></button>
      <button id="lhs-add-btn"></button>
      <div id="lhs-add-result"></div>
      <button id="lhs-select-all"></button>
      <button id="lhs-deselect-all"></button>
      <input type="checkbox" id="lhs-check-all">
      <button id="lhs-reset-btn"></button>
      <button id="lhs-update-btn"></button>

      <select id="bo-objective">${objectiveOptions}</select>
      <select id="bo-algorithm"><option value="expected_improvement">EI</option></select>
      <input id="bo-top-points" value="5">
      <input id="bo-iterations" value="30">
      <table><tbody id="bo-table-body"></tbody></table>
      <input id="bo-add-pv">
      <button id="bo-check-value"></button>
      <button id="bo-add-btn"></button>
      <div id="bo-add-result"></div>
      <button id="bo-select-all"></button>
      <button id="bo-deselect-all"></button>
      <input type="checkbox" id="bo-check-all">
      <button id="bo-reset-btn"></button>
      <button id="bo-update-btn"></button>

      <button id="start-btn"></button>
      <button id="pause-btn"></button>
      <button id="resume-btn"></button>
      <button id="cancel-btn"></button>
    </div>
    <div id="validation-alert" style="display:none"><span id="validation-text"></span></div>
  `;
}

/** Build one variable-table row.
 *  @param {Partial<Record<string, unknown>> & { pv_name: string }} overrides */
function makeVar(overrides) {
  return {
    current_value: 1,
    min: 0,
    max: 2,
    step_size: null,
    bo_range_factor: 1,
    selected: false,
    ...overrides,
  };
}

beforeEach(async () => {
  sessionStorage.clear();
  document.body.innerHTML = '';
  vi.resetModules();
  // optimization-form does not itself plot, but stub the vendored global so any
  // transitively-touched plot path stays inert under Vitest.
  /** @type {any} */ (globalThis).Plotly = { react() {}, newPlot() {}, purge() {}, relayout() {} };

  // Import state + api before the form so the whole graph shares one fresh
  // registry: the form's internal `import { state/api }` resolve to these.
  const stateMod = await import(STATE_PATH);
  const apiMod = await import(API_PATH);
  const formMod = await import(FORM_PATH);
  store = stateMod.state;
  apiClient = apiMod.api;
  renderTable = formMod.renderTable;
  initOptimizationForm = formMod.initOptimizationForm;
});

describe('renderTable — row structure', () => {
  /** @returns {HTMLTableSectionElement} */
  function freshTbody() {
    const table = document.createElement('table');
    const tbody = document.createElement('tbody');
    table.appendChild(tbody);
    return tbody;
  }

  it('renders one row per variable with the selected class and checkbox state', () => {
    store.setVariableTableData([
      makeVar({ pv_name: 'PV:A', current_value: 1.5, min: 0, max: 10, step_size: 0.5, selected: true }),
      makeVar({ pv_name: 'PV:B', current_value: null, min: null, max: null, selected: false }),
    ]);
    const tbody = freshTbody();

    renderTable(tbody, false);

    const rows = tbody.querySelectorAll('tr');
    expect(rows.length).toBe(2);
    expect(rows[0].classList.contains('selected')).toBe(true);
    expect(rows[1].classList.contains('selected')).toBe(false);

    const names = [...tbody.querySelectorAll('.pv-name')].map((c) => c.textContent);
    expect(names).toEqual(['PV:A', 'PV:B']);

    const firstCheck = /** @type {HTMLInputElement} */ (rows[0].querySelector('input[type="checkbox"]'));
    const secondCheck = /** @type {HTMLInputElement} */ (rows[1].querySelector('input[type="checkbox"]'));
    expect(firstCheck.checked).toBe(true);
    expect(secondCheck.checked).toBe(false);
  });

  it('omits the BO range-factor column for the LHS table and includes it for BO', () => {
    store.setVariableTableData([makeVar({ pv_name: 'PV:A' })]);
    const lhsBody = freshTbody();
    const boBody = freshTbody();

    renderTable(lhsBody, false);
    renderTable(boBody, true);

    expect(lhsBody.querySelector('input[data-field="bo_range_factor"]')).toBeNull();
    expect(boBody.querySelector('input[data-field="bo_range_factor"]')).not.toBeNull();
  });

  it('renders an empty-state row with the phase-appropriate colspan', () => {
    store.setVariableTableData([]);
    const tbody = freshTbody();

    renderTable(tbody, false);
    expect(tbody.querySelector('tr.empty-row td')?.getAttribute('colspan')).toBe('6');

    renderTable(tbody, true);
    expect(tbody.querySelector('tr.empty-row td')?.getAttribute('colspan')).toBe('7');
  });

  it('escapes a hostile pv_name so no live element is injected', () => {
    const payload = '"><img src=x onerror=alert(1)>';
    store.setVariableTableData([makeVar({ pv_name: payload, selected: false })]);
    const tbody = freshTbody();

    renderTable(tbody, true); // showBoRange exercises every data-pv attribute copy

    assertInert(tbody);
    // The payload survives only as inert text: textContent decodes the escaped
    // entities back to the literal string.
    expect(tbody.textContent).toContain(payload);
  });
});

describe('validateForm — reached through the wired Start button', () => {
  beforeEach(() => {
    document.body.innerHTML = fixtureHtml();
    initOptimizationForm();
  });

  it('starts the run for a valid form and applies the returned job/state', async () => {
    const startSpy = vi.spyOn(apiClient, 'startOptimization').mockResolvedValue({ job_id: 'new-job' });
    store.setEnvironment('sim'); // enables Start (idle + env)
    select('lhs-objective').value = 'obj-x';
    store.setVariableTableData([
      makeVar({ pv_name: 'PV:A', min: 0, max: 10, step_size: 0.5, bo_range_factor: 1, selected: true }),
    ]);

    btn('start-btn').click();
    await flush();

    expect(startSpy).toHaveBeenCalledTimes(1);
    expect(startSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        environment: 'sim',
        objective: 'obj-x',
        lhs_samples: 20,
        bo_algorithm: 'expected_improvement',
        bo_top_points: 5,
        bo_iterations: 30,
        variables: [{ name: 'PV:A', min: 0, max: 10, step_size: 0.5, bo_range_factor: 1 }],
      }),
    );
    expect(store.jobId).toBe('new-job');
    expect(store.optimizationState.status).toBe('RUNNING');
  });

  /** @type {{ name: string, setup: () => void, message: string }[]} */
  const invalidCases = [
    {
      name: 'no objective is selected',
      setup: () => {},
      message: 'Please select an objective variable.',
    },
    {
      name: 'no variable is selected',
      setup: () => {
        select('lhs-objective').value = 'obj-x';
      },
      message: 'Please add and select at least one variable.',
    },
    {
      name: 'a selected variable is missing bounds',
      setup: () => {
        select('lhs-objective').value = 'obj-x';
        store.setVariableTableData([makeVar({ pv_name: 'PV:A', min: null, max: 5, selected: true })]);
      },
      message: 'Variable PV:A is missing min/max bounds.',
    },
    {
      name: 'a selected variable has min >= max',
      setup: () => {
        select('lhs-objective').value = 'obj-x';
        store.setVariableTableData([makeVar({ pv_name: 'PV:A', min: 5, max: 5, selected: true })]);
      },
      message: 'Variable PV:A: min must be less than max.',
    },
  ];

  for (const testCase of invalidCases) {
    it(`blocks start and shows the message when ${testCase.name}`, async () => {
      const startSpy = vi.spyOn(apiClient, 'startOptimization').mockResolvedValue({ job_id: 'x' });
      store.setEnvironment('sim'); // enable the button so the click reaches validateForm
      testCase.setup();

      btn('start-btn').click();
      await flush();

      expect(startSpy).not.toHaveBeenCalled();
      expect(byId('validation-text').textContent).toBe(testCase.message);
      expect(byId('validation-alert').style.display).toBe('');
    });
  }
});

describe('control wiring — Start / Pause / Resume / Cancel', () => {
  beforeEach(() => {
    document.body.innerHTML = fixtureHtml();
    initOptimizationForm();
  });

  it('Pause calls the API with the active job and transitions to PAUSED', async () => {
    const pauseSpy = vi.spyOn(apiClient, 'pause').mockResolvedValue({});
    store.setEnvironment('sim');
    store.setJobId('job-9');
    store.setOptimizationState({ status: 'RUNNING' });

    expect(btn('pause-btn').disabled).toBe(false);
    btn('pause-btn').click();
    await flush();

    expect(pauseSpy).toHaveBeenCalledWith('job-9');
    expect(store.optimizationState.status).toBe('PAUSED');
  });

  it('Resume calls the API and transitions back to RUNNING', async () => {
    const resumeSpy = vi.spyOn(apiClient, 'resume').mockResolvedValue({});
    store.setEnvironment('sim');
    store.setJobId('job-9');
    store.setOptimizationState({ status: 'PAUSED' });

    expect(btn('resume-btn').disabled).toBe(false);
    btn('resume-btn').click();
    await flush();

    expect(resumeSpy).toHaveBeenCalledWith('job-9');
    expect(store.optimizationState.status).toBe('RUNNING');
  });

  it('Cancel calls the API and transitions to CANCELLED', async () => {
    const cancelSpy = vi.spyOn(apiClient, 'cancel').mockResolvedValue({});
    store.setEnvironment('sim');
    store.setJobId('job-9');
    store.setOptimizationState({ status: 'RUNNING' });

    expect(btn('cancel-btn').disabled).toBe(false);
    btn('cancel-btn').click();
    await flush();

    expect(cancelSpy).toHaveBeenCalledWith('job-9');
    expect(store.optimizationState.status).toBe('CANCELLED');
  });

  it('toggles button disabled/visibility as the run state changes', () => {
    store.setEnvironment('sim');
    expect(btn('start-btn').disabled).toBe(false);
    // No job yet: pause/resume are unavailable.
    expect(btn('pause-btn').disabled).toBe(true);

    store.setJobId('job-1');
    store.setOptimizationState({ status: 'RUNNING' });

    expect(btn('start-btn').disabled).toBe(true);
    expect(btn('pause-btn').disabled).toBe(false);
    expect(btn('resume-btn').disabled).toBe(true);
    expect(btn('pause-btn').style.display).toBe('');
    expect(btn('resume-btn').style.display).toBe('none');
  });
});
