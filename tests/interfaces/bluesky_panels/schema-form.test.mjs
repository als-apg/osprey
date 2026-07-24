/**
 * Unit tests for the plan panel's JSON-Schema → 2-D form renderer
 * (schema-form.js).
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/bluesky_panels/schema-form.test.mjs
 *
 * The two fixtures below are the verbatim ``model_json_schema()`` output of the
 * shipped ``orm`` and ``grid_scan`` plans' Pydantic parameter models — the real
 * shapes the panel must render. They exercise the typed editors end to end:
 * chip editors for device lists, the editable table for ``axes`` (an array of
 * ``$ref`` objects), bounded number inputs, the boolean toggle, the layout
 * option, and the ``form-change`` structural-edit event — driving each control
 * like an operator would and reading the nested ``plan_args`` back out.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

import {
  renderSchemaForm,
  resolveNode,
  OMIT,
} from '../../../src/osprey/interfaces/bluesky_panels/panels/plan/schema-form.js';

const ORM_SCHEMA = {
  properties: {
    correctors: {
      description: 'Corrector device names to sweep, one at a time.',
      items: { type: 'string' },
      minItems: 1,
      title: 'Correctors',
      type: 'array',
      'x-widget': 'channel-list',
    },
    detectors: {
      description: 'BPM detector device names to read at every point.',
      items: { type: 'string' },
      minItems: 1,
      title: 'BPMs',
      type: 'array',
      'x-widget': 'channel-list',
    },
    span_a: {
      description: 'Maximum corrector kick, in amps, at the far end of the sweep.',
      exclusiveMinimum: 0,
      maximum: 10.0,
      title: 'Max kick (A)',
      type: 'number',
    },
    num: {
      description: 'Number of evenly-spaced current points per corrector.',
      minimum: 3,
      title: 'Number of steps',
      type: 'integer',
    },
    sweep: {
      default: 'bidirectional',
      description: 'bidirectional sweeps [-span_a, +span_a]; monodirectional sweeps [0, +span_a].',
      enum: ['bidirectional', 'monodirectional'],
      title: 'Sweep direction',
      type: 'string',
      'x-widget': 'segmented',
    },
  },
  required: ['correctors', 'detectors', 'span_a', 'num'],
  title: 'PARAMS',
  type: 'object',
};

const GRID_SCAN_SCHEMA = {
  $defs: {
    GridAxis: {
      description: "One setpoint device's sweep range and point count, for one grid dimension.",
      properties: {
        setpoint: { title: 'Setpoint', type: 'string' },
        start: { title: 'Start', type: 'number' },
        stop: { title: 'Stop', type: 'number' },
        num_points: {
          description: 'Points along this axis.',
          minimum: 2,
          title: 'Num Points',
          type: 'integer',
        },
      },
      required: ['setpoint', 'start', 'stop', 'num_points'],
      title: 'GridAxis',
      type: 'object',
    },
  },
  properties: {
    detectors: {
      description: 'Device names to read at each grid point.',
      items: { type: 'string' },
      minItems: 1,
      title: 'Detectors',
      type: 'array',
    },
    axes: {
      description: 'One entry per grid dimension.',
      items: { $ref: '#/$defs/GridAxis' },
      minItems: 1,
      title: 'Axes',
      type: 'array',
    },
    snake_axes: {
      default: false,
      description: 'Snake back-and-forth across successive axes.',
      title: 'Snake Axes',
      type: 'boolean',
    },
  },
  required: ['detectors', 'axes'],
  title: 'PARAMS',
  type: 'object',
};

/** @type {HTMLElement} */
let form;

beforeEach(() => {
  document.body.innerHTML = '<form id="f"></form>';
  form = /** @type {any} */ (document.getElementById('f'));
});

/**
 * Loose-typed query helpers so interactions read cleanly under strict checkJs.
 * @param {any} root
 * @param {string} sel
 * @returns {any}
 */
function $(root, sel) {
  return root.querySelector(sel);
}
/**
 * @param {any} root
 * @param {string} sel
 * @returns {any[]}
 */
function $$(root, sel) {
  return [...root.querySelectorAll(sel)];
}

/**
 * Type text into a chips input and press Enter (how an operator adds a chip).
 * @param {any} chipsEl
 * @param {string} text
 */
function addChip(chipsEl, text) {
  const input = $(chipsEl, '.chips-input');
  input.value = text;
  input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
}

/**
 * Type text into a channel-list input and press Enter (how an operator adds a
 * channel to a channel-list widget).
 * @param {any} listEl
 * @param {string} text
 */
function addChannel(listEl, text) {
  const input = $(listEl, '.channel-add');
  input.value = text;
  input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
}

describe('resolveNode', () => {
  test('dereferences a $ref into $defs and keeps sibling keys', () => {
    const node = resolveNode(GRID_SCAN_SCHEMA, {
      $ref: '#/$defs/GridAxis',
      title: 'Axis 1',
    });
    expect(node.type).toBe('object');
    expect(node.title).toBe('Axis 1'); // sibling wins over the target's title
    expect(Object.keys(node.properties || {})).toEqual(['setpoint', 'start', 'stop', 'num_points']);
  });

  test('unwraps the non-null branch of an Optional anyOf', () => {
    const node = resolveNode({}, { anyOf: [{ type: 'string' }, { type: 'null' }], default: null });
    expect(node.type).toBe('string');
  });
});

describe('orm schema (channel lists + segmented sweep + bounded scalars + layout)', () => {
  test('renders channel-list editors for device lists, bounded scalars, and a segmented sweep', () => {
    renderSchemaForm(form, ORM_SCHEMA);

    // Five labeled rows, in schema order.
    const names = $$(form, '.field-name').map((n) => n.textContent);
    expect(names).toEqual(['Correctors', 'BPMs', 'Max kick (A)', 'Number of steps', 'Sweep direction']);

    // Both device lists render as the vertical channel-list widget — not the
    // wrapping chip well, and not a lone text box.
    expect($$(form, '.channel-list').length).toBe(2);
    expect($$(form, '.chips').length).toBe(0);

    // span_a: number input carrying the schema bounds.
    const numberInputs = $$(form, 'input[type="number"]');
    const spanA = numberInputs.find((i) => i.step === 'any');
    expect(spanA.min).toBe('0');
    expect(spanA.max).toBe('10');

    // num: integer input (step 1) with min 3.
    const numInput = numberInputs.find((i) => i.step === '1');
    expect(numInput.min).toBe('3');

    // sweep: a two-option segmented control, seeded active on its default.
    const segmented = $(form, '.segmented');
    expect(segmented).toBeTruthy();
    const options = $$(segmented, '.segmented-option');
    expect(options.map((o) => o.textContent)).toEqual(['bidirectional', 'monodirectional']);
    expect(options[0].getAttribute('aria-checked')).toBe('true');
    expect(options[1].getAttribute('aria-checked')).toBe('false');

    // Required markers on the four required fields (sweep is optional).
    expect($$(form, '.field-required').length).toBe(4);
  });

  test('a layout option places fields into side-by-side rows', () => {
    renderSchemaForm(form, ORM_SCHEMA, {
      layout: [['correctors', 'detectors'], ['span_a', 'num'], ['sweep']],
    });
    const rows = $$(form, '.form-row');
    expect(rows.length).toBe(3);
    expect(rows[0].style.getPropertyValue('--cols')).toBe('2');
    expect($$(rows[0], '.field-name').map((n) => n.textContent)).toEqual(['Correctors', 'BPMs']);
    expect($$(rows[1], '.field-name').map((n) => n.textContent)).toEqual([
      'Max kick (A)',
      'Number of steps',
    ]);
    expect($$(rows[2], '.field-name').map((n) => n.textContent)).toEqual(['Sweep direction']);
  });

  test('channel entries commit on Enter, split pasted lists, and collect as arrays', () => {
    const collect = renderSchemaForm(form, ORM_SCHEMA);
    const [correctors, detectors] = $$(form, '.channel-list');

    addChannel(correctors, 'HCM1');
    addChannel(correctors, 'HCM2, HCM3'); // comma-separated paste → two entries
    addChannel(detectors, 'BPM1');

    expect($$(correctors, '.channel-item').length).toBe(3);

    const spanA = $$(form, 'input[type="number"]').find((i) => i.step === 'any');
    const num = $$(form, 'input[type="number"]').find((i) => i.step === '1');
    spanA.value = '2.5';
    num.value = '7';

    expect(collect()).toEqual({
      correctors: ['HCM1', 'HCM2', 'HCM3'],
      detectors: ['BPM1'],
      span_a: 2.5,
      num: 7,
      // The segmented control always contributes; untouched, it is its default.
      sweep: 'bidirectional',
    });
  });

  test('the channel-list header reflects the current count', () => {
    renderSchemaForm(form, ORM_SCHEMA);
    const [correctors] = $$(form, '.channel-list');
    expect($(correctors, '.channel-count').textContent).toBe('0 channels');
    addChannel(correctors, 'HCM1');
    expect($(correctors, '.channel-count').textContent).toBe('1 channel');
    addChannel(correctors, 'HCM2 HCM3'); // whitespace-separated paste → two more
    expect($(correctors, '.channel-count').textContent).toBe('3 channels');
  });

  test('channel × removes an entry; Backspace on empty input removes the last', () => {
    const collect = renderSchemaForm(form, ORM_SCHEMA);
    const [correctors] = $$(form, '.channel-list');
    addChannel(correctors, 'HCM1');
    addChannel(correctors, 'HCM2');
    addChannel(correctors, 'HCM3');

    // × on the first entry.
    $(correctors, '.chan-x').click();
    // Backspace with an empty input pops the last entry.
    $(correctors, '.channel-add').dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Backspace', bubbles: true })
    );

    expect(collect().correctors).toEqual(['HCM2']);
  });

  test('structural channel edits dispatch a bubbling form-change event', () => {
    renderSchemaForm(form, ORM_SCHEMA);
    const listener = vi.fn();
    form.addEventListener('form-change', listener);
    addChannel($$(form, '.channel-list')[0], 'HCM1');
    expect(listener).toHaveBeenCalled();
  });

  test('the segmented control switches the active option and collects it', () => {
    const collect = renderSchemaForm(form, ORM_SCHEMA);
    const listener = vi.fn();
    form.addEventListener('form-change', listener);

    const options = $$(form, '.segmented-option');
    options[1].click(); // monodirectional

    expect(options[0].getAttribute('aria-checked')).toBe('false');
    expect(options[1].getAttribute('aria-checked')).toBe('true');
    expect(listener).toHaveBeenCalled();
    expect(collect().sweep).toBe('monodirectional');
  });

  test('omits blank optional fields, but a segmented control always contributes', () => {
    const collect = renderSchemaForm(form, ORM_SCHEMA);
    // Nothing typed: the required device/scalar fields are omitted so plan-side
    // defaults apply, but the segmented sweep always carries its active value.
    expect(collect()).toEqual({ sweep: 'bidirectional' });
  });

  test('rejects a fractional value on an integer field', () => {
    const collect = renderSchemaForm(form, ORM_SCHEMA);
    const num = $$(form, 'input[type="number"]').find((i) => i.step === '1');
    num.value = '7.4';
    expect(collect().num).toBeUndefined();
  });
});

describe('grid_scan schema (axes table + toggle)', () => {
  test('renders axes as an editable table seeded with one blank row', () => {
    renderSchemaForm(form, GRID_SCAN_SCHEMA);

    const table = $(form, '.obj-table');
    expect(table).toBeTruthy();

    // Header: the four GridAxis columns (plus the remove column).
    const headers = $$(table, 'thead th span:first-child').map((n) => n.textContent);
    expect(headers).toEqual(['Setpoint', 'Start', 'Stop', 'Num Points']);

    // minItems: 1 → one starter row so the columns are visible immediately…
    expect($$(table, 'tbody tr').length).toBe(1);

    // snake_axes renders as a toggle, seeded from its default (false).
    const check = $(form, '.switch-input');
    expect(check.checked).toBe(false);
  });

  test('collects detectors + nested axes rows + toggled snake_axes', () => {
    const collect = renderSchemaForm(form, GRID_SCAN_SCHEMA);

    addChip($(form, '.chips'), 'BPM1');

    // Fill the starter row, then add and fill a second axis.
    $(form, '.table-add').click();
    const rows = $$(form, '.obj-table tbody tr');
    expect(rows.length).toBe(2);
    /**
     * @param {any} row
     * @param {string[]} values
     */
    const fill = (row, values) => {
      const inputs = $$(row, 'input');
      values.forEach((v, i) => {
        inputs[i].value = v;
      });
    };
    fill(rows[0], ['QF1', '-1.5', '1.5', '11']);
    fill(rows[1], ['QD2', '0', '2', '5']);

    $(form, '.switch-input').checked = true;

    expect(collect()).toEqual({
      detectors: ['BPM1'],
      axes: [
        { setpoint: 'QF1', start: -1.5, stop: 1.5, num_points: 11 },
        { setpoint: 'QD2', start: 0, stop: 2, num_points: 5 },
      ],
      snake_axes: true,
    });
  });

  test('a fully-blank table row is scaffolding, not data', () => {
    const collect = renderSchemaForm(form, GRID_SCAN_SCHEMA);
    // The seeded blank row must not emit an empty axes object.
    expect(collect().axes).toBeUndefined();
  });

  test('removing a table row drops it from the collected list', () => {
    const collect = renderSchemaForm(form, GRID_SCAN_SCHEMA);
    $(form, '.table-add').click();
    const rows = $$(form, '.obj-table tbody tr');
    /**
     * @param {any} row
     * @param {string[]} values
     */
    const fill = (row, values) => {
      const inputs = $$(row, 'input');
      values.forEach((v, i) => {
        inputs[i].value = v;
      });
    };
    fill(rows[0], ['QF1', '-1', '1', '3']);
    fill(rows[1], ['QD2', '0', '2', '5']);

    $(rows[0], '.row-x').click();

    expect(collect().axes).toEqual([{ setpoint: 'QD2', start: 0, stop: 2, num_points: 5 }]);
  });
});

describe('empty schema', () => {
  test('renders a "no parameters" note and collects an empty object', () => {
    const collect = renderSchemaForm(form, { type: 'object', properties: {} });
    expect($(form, '.param-empty')).toBeTruthy();
    expect(collect()).toEqual({});
  });

  test('OMIT is exported for parent collectors', () => {
    expect(typeof OMIT).toBe('symbol');
  });
});

describe('applyValues (field registry / programmatic draft application)', () => {
  const SCALAR_SCHEMA = {
    properties: {
      label: { title: 'Label', type: 'string' },
    },
    required: ['label'],
    title: 'PARAMS',
    type: 'object',
  };

  const ENUM_SCHEMA = {
    properties: {
      mode: {
        default: 'a',
        enum: ['a', 'b', 'c'],
        title: 'Mode',
        type: 'string',
      },
    },
    title: 'PARAMS',
    type: 'object',
  };

  // An integer-membered enum (Pydantic emits these for `IntEnum` params) —
  // exercises buildEnum's `String(opt) === String(value)` matching against
  // non-string members.
  const INT_ENUM_SCHEMA = {
    properties: {
      level: {
        enum: [1, 2, 3],
        title: 'Level',
        type: 'integer',
      },
    },
    title: 'PARAMS',
    type: 'object',
  };

  // A nested-object field (`options`) and an array of non-flat objects
  // (`items`, whose item schema mixes a scalar with a nested sub-object, so
  // it fails `isFlatObject` and falls to the generic buildArrayRows/
  // buildObject stack rather than the axes table) — exercises two levels of
  // recursive setValue composition through buildArrayRows → buildObject →
  // buildObject.
  const NESTED_SCHEMA = {
    $defs: {
      Sub: {
        properties: {
          rate: { title: 'Rate', type: 'number' },
          active: { title: 'Active', type: 'boolean' },
        },
        required: ['rate'],
        title: 'Sub',
        type: 'object',
      },
      Item: {
        properties: {
          name: { title: 'Name', type: 'string' },
          sub: { $ref: '#/$defs/Sub', title: 'Sub' },
        },
        required: ['name'],
        title: 'Item',
        type: 'object',
      },
    },
    properties: {
      options: { $ref: '#/$defs/Sub', title: 'Options' },
      items: {
        items: { $ref: '#/$defs/Item' },
        title: 'Items',
        type: 'array',
      },
    },
    required: ['items'],
    title: 'PARAMS',
    type: 'object',
  };

  /**
   * Attach `input`/`change` counters to `root` and return a getter for the
   * combined count — `applyValues` must never trip either, since it is a
   * programmatic apply, not a user edit.
   * @param {any} root
   */
  function countNativeEvents(root) {
    let count = 0;
    root.addEventListener('input', () => {
      count += 1;
    });
    root.addEventListener('change', () => {
      count += 1;
    });
    return () => count;
  }

  test('applies a plain string scalar and round-trips through collect()', () => {
    const collect = renderSchemaForm(form, SCALAR_SCHEMA);
    const nativeCount = countNativeEvents(form);
    collect.applyValues({ label: 'HCM1' });
    expect(nativeCount()).toBe(0);
    expect(collect()).toEqual({ label: 'HCM1' });
  });

  test('applies a number field and an integer field, coercing display text', () => {
    const collect = renderSchemaForm(form, ORM_SCHEMA);
    const nativeCount = countNativeEvents(form);
    collect.applyValues({ span_a: 4, num: 6 });
    expect(nativeCount()).toBe(0);
    const spanA = $$(form, 'input[type="number"]').find((i) => i.step === 'any');
    const num = $$(form, 'input[type="number"]').find((i) => i.step === '1');
    expect(spanA.value).toBe('4');
    expect(num.value).toBe('6');
    expect(collect()).toEqual({ span_a: 4, num: 6, sweep: 'bidirectional' });
  });

  test('applies a boolean toggle without a click/change event', () => {
    const collect = renderSchemaForm(form, GRID_SCAN_SCHEMA);
    const nativeCount = countNativeEvents(form);
    collect.applyValues({ snake_axes: true });
    expect(nativeCount()).toBe(0);
    expect($(form, '.switch-input').checked).toBe(true);
    expect(collect().snake_axes).toBe(true);
  });

  test('applies an enum select', () => {
    const collect = renderSchemaForm(form, ENUM_SCHEMA);
    const nativeCount = countNativeEvents(form);
    collect.applyValues({ mode: 'c' });
    expect(nativeCount()).toBe(0);
    expect(collect()).toEqual({ mode: 'c' });
  });

  test('applies chip lists as a whole-value replacement', () => {
    const collect = renderSchemaForm(form, GRID_SCAN_SCHEMA);
    const nativeCount = countNativeEvents(form);
    collect.applyValues({ detectors: ['BPM1', 'BPM2'] });
    expect(nativeCount()).toBe(0);
    expect($$(form, '.chip').length).toBe(2);
    expect(collect().detectors).toEqual(['BPM1', 'BPM2']);
  });

  test('applies channel-list and segmented fields via the orm fixture', () => {
    const collect = renderSchemaForm(form, ORM_SCHEMA);
    const nativeCount = countNativeEvents(form);
    collect.applyValues({
      correctors: ['HCM1', 'HCM2'],
      detectors: ['BPM1'],
      sweep: 'monodirectional',
    });
    expect(nativeCount()).toBe(0);

    const [correctors, detectors] = $$(form, '.channel-list');
    expect($$(correctors, '.channel-item').length).toBe(2);
    expect($$(detectors, '.channel-item').length).toBe(1);

    const options = $$(form, '.segmented-option');
    expect(options[0].getAttribute('aria-checked')).toBe('false');
    expect(options[1].getAttribute('aria-checked')).toBe('true');

    expect(collect()).toEqual({
      correctors: ['HCM1', 'HCM2'],
      detectors: ['BPM1'],
      sweep: 'monodirectional',
    });
  });

  test('applies the axes table as a whole-value replacement (row count follows the incoming array)', () => {
    const collect = renderSchemaForm(form, GRID_SCAN_SCHEMA);
    // Seeded with one blank starter row (minItems: 1) before any apply.
    expect($$(form, '.obj-table tbody tr').length).toBe(1);

    const nativeCount = countNativeEvents(form);
    collect.applyValues({
      axes: [
        { setpoint: 'QF1', start: -1, stop: 1, num_points: 5 },
        { setpoint: 'QD2', start: 0, stop: 2, num_points: 7 },
      ],
    });
    expect(nativeCount()).toBe(0);
    expect($$(form, '.obj-table tbody tr').length).toBe(2);
    expect(collect().axes).toEqual([
      { setpoint: 'QF1', start: -1, stop: 1, num_points: 5 },
      { setpoint: 'QD2', start: 0, stop: 2, num_points: 7 },
    ]);
  });

  test('applies a nested object (buildObject) by delegating recursively to its children', () => {
    const collect = renderSchemaForm(form, NESTED_SCHEMA);
    const nativeCount = countNativeEvents(form);
    collect.applyValues({ options: { rate: 2.5, active: true } });
    expect(nativeCount()).toBe(0);
    expect(collect().options).toEqual({ rate: 2.5, active: true });
  });

  test('applies an array of non-flat objects (buildArrayRows), rebuilding rows with their nested sub-objects', () => {
    const collect = renderSchemaForm(form, NESTED_SCHEMA);
    const nativeCount = countNativeEvents(form);
    collect.applyValues({
      items: [
        { name: 'first', sub: { rate: 1, active: false } },
        { name: 'second', sub: { rate: 2, active: true } },
      ],
    });
    expect(nativeCount()).toBe(0);
    expect($$(form, '.array-row').length).toBe(2);
    expect(collect().items).toEqual([
      { name: 'first', sub: { rate: 1, active: false } },
      { name: 'second', sub: { rate: 2, active: true } },
    ]);
  });

  test('dispatches exactly one form-change event per applyValues pass', () => {
    const collect = renderSchemaForm(form, ORM_SCHEMA);
    const listener = vi.fn();
    form.addEventListener('form-change', listener);
    collect.applyValues({ span_a: 1, num: 3, sweep: 'monodirectional' });
    expect(listener).toHaveBeenCalledTimes(1);
  });

  test('keys absent from values are left untouched across separate applyValues calls', () => {
    const collect = renderSchemaForm(form, ORM_SCHEMA);
    collect.applyValues({ span_a: 5 });
    expect(collect()).toEqual({ span_a: 5, sweep: 'bidirectional' });

    // A disjoint second call must not clobber the field set by the first.
    collect.applyValues({ num: 4 });
    expect(collect()).toEqual({ span_a: 5, num: 4, sweep: 'bidirectional' });
  });

  test('exposes a top-level field registry keyed by schema property name', () => {
    const collect = renderSchemaForm(form, ORM_SCHEMA);
    expect(Object.keys(collect.fields).sort()).toEqual(
      ['correctors', 'detectors', 'num', 'span_a', 'sweep'].sort()
    );
    expect(typeof collect.fields.span_a.setValue).toBe('function');
    expect(collect.fields.span_a.el).toBeTruthy();
  });

  test('the returned collector is still callable directly — the collectPlanArgs() contract is unchanged', () => {
    const collect = renderSchemaForm(form, SCALAR_SCHEMA);
    expect(typeof collect).toBe('function');
    const label = $(form, 'input[type="text"]');
    label.value = 'direct-call';
    expect(collect()).toEqual({ label: 'direct-call' });
  });

  test('applying the axes table follows the incoming row count down to zero, leaving the blank starter row', () => {
    const collect = renderSchemaForm(form, GRID_SCAN_SCHEMA);

    collect.applyValues({
      axes: [
        { setpoint: 'QF1', start: -1, stop: 1, num_points: 5 },
        { setpoint: 'QD2', start: 0, stop: 2, num_points: 7 },
      ],
    });
    expect($$(form, '.obj-table tbody tr').length).toBe(2);
    expect(collect().axes).toEqual([
      { setpoint: 'QF1', start: -1, stop: 1, num_points: 5 },
      { setpoint: 'QD2', start: 0, stop: 2, num_points: 7 },
    ]);

    collect.applyValues({ axes: [{ setpoint: 'QF1', start: -1, stop: 1, num_points: 5 }] });
    expect($$(form, '.obj-table tbody tr').length).toBe(1);
    expect(collect().axes).toEqual([{ setpoint: 'QF1', start: -1, stop: 1, num_points: 5 }]);

    // An empty array is a whole-value replacement too: minItems: 1 means the
    // table falls back to its one blank starter row, which collects as
    // scaffolding, not data — matching a fresh render of an empty/required
    // table, not a zero-row table.
    collect.applyValues({ axes: [] });
    expect($$(form, '.obj-table tbody tr').length).toBe(1);
    expect(collect().axes).toBeUndefined();
  });

  test('applying a channel-list discards pending uncommitted input text and supports clearing back to empty', () => {
    const collect = renderSchemaForm(form, ORM_SCHEMA);
    const [correctors] = $$(form, '.channel-list');
    const addInput = $(correctors, '.channel-add');

    // Typed but not yet committed (no Enter/blur) when the apply lands.
    addInput.value = 'PENDING';
    collect.applyValues({ correctors: ['HCM9'] });

    expect(addInput.value).toBe('');
    expect($$(correctors, '.channel-item').map((li) => $(li, '.channel-name').textContent)).toEqual([
      'HCM9',
    ]);
    expect(collect().correctors).toEqual(['HCM9']);

    // A subsequent apply-to-empty clears the list entirely.
    collect.applyValues({ correctors: [] });
    expect($$(correctors, '.channel-item').length).toBe(0);
    expect($(correctors, '.channel-count').textContent).toBe('0 channels');
    expect(collect().correctors).toBeUndefined();
  });

  test('a partial nested-object apply resets its omitted always-contributing child to its schema default', () => {
    const collect = renderSchemaForm(form, NESTED_SCHEMA);

    // Set both children first, `active` explicitly true.
    collect.applyValues({ options: { rate: 1, active: true } });
    expect(collect().options).toEqual({ rate: 1, active: true });

    // A second apply omitting `active` must reset it to its schema default
    // (no `default` on the boolean field, so the constructor-seed state is
    // `false`) rather than leaving the previous `true` in place.
    collect.applyValues({ options: { rate: 2 } });
    expect(collect().options).toEqual({ rate: 2, active: false });
  });

  test('applies an integer-membered enum and round-trips the numeric member', () => {
    const collect = renderSchemaForm(form, INT_ENUM_SCHEMA);
    const nativeCount = countNativeEvents(form);
    collect.applyValues({ level: 2 });
    expect(nativeCount()).toBe(0);
    expect(collect()).toEqual({ level: 2 });
    expect(typeof collect().level).toBe('number');
  });

  test('applyValues(null/undefined/non-object) is a safe no-op that fires no form-change', () => {
    const collect = renderSchemaForm(form, SCALAR_SCHEMA);
    const listener = vi.fn();
    form.addEventListener('form-change', listener);

    expect(() => collect.applyValues(null)).not.toThrow();
    expect(() => collect.applyValues(undefined)).not.toThrow();
    expect(() => collect.applyValues('not-an-object')).not.toThrow();

    expect(listener).not.toHaveBeenCalled();
    expect(collect()).toEqual({}); // untouched — label is still blank
  });
});
