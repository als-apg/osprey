/**
 * Unit tests for the plan panel's JSON-Schema → 2-D form renderer
 * (schema-form.js).
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/services/bluesky_panels/schema-form.test.mjs
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
} from '../../../src/osprey/services/bluesky_panels/panels/plan/schema-form.js';

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
