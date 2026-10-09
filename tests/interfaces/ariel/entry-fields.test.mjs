// @ts-check
/**
 * Front-end proofs for the ARIEL create form's facility-declared entry fields
 * (`static/js/entry-fields.js`):
 *
 *   - declared fields render into one section per descriptor `section`, in
 *     declaration order, inside the mount placed after "Metadata";
 *   - each type gets its plain input (number, checkbox, date, select, text,
 *     dynamic select), `default` pre-fills, required fields are marked;
 *   - a descriptor named `logbook`/`shift` replaces that built-in input in
 *     place, keeping its form name so the built-in path submits the same value;
 *   - a dynamic select fetches its `options_endpoint` with its `depends_on`
 *     values and refetches when one of them changes;
 *   - `collectEntryFieldValues()` returns the declared values, native-typed,
 *     leaving blanks out; `markEntryFieldError()` marks the named input;
 *   - the rendered-promise settles once the fields are rendered, or once the
 *     form is known to declare none.
 *
 *   npx vitest run tests/interfaces/ariel/entry-fields.test.mjs
 *
 * The module keeps its state at module scope, so each test re-imports it fresh
 * via vi.resetModules(). `fetch` is stubbed per test.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const MODULE_PATH = '../../../src/osprey/interfaces/ariel/static/js/entry-fields.js';

/** The create form's Metadata section, as index.html ships it, plus the mount. */
const FORM_HTML = `
  <form id="create-entry-form">
    <div class="form-section">
      <div class="form-section-title">Metadata</div>
      <div class="form-row">
        <div class="input-group">
          <label class="input-label" for="entry-author">Author</label>
          <input type="text" id="entry-author" name="author" class="input">
        </div>
        <div class="input-group">
          <label class="input-label" for="entry-logbook">Logbook</label>
          <input type="text" id="entry-logbook" name="logbook" class="input">
        </div>
        <div class="input-group">
          <label class="input-label" for="entry-shift">Shift</label>
          <input type="text" id="entry-shift" name="shift" class="input">
        </div>
      </div>
    </div>
    <div id="entry-fields"></div>
    <div class="form-section"><div class="form-section-title">Tags</div></div>
  </form>
`;

/** Declarations shaped like `publish-info.entry_fields` for the example adapter. */
const FIELDS = /** @type {any[]} */ ([
  {
    name: 'book',
    label: 'Book',
    description: 'Which book',
    type: 'select',
    default: 'ops',
    section: 'Facility',
    options: [
      { value: 'ops', label: 'Operations' },
      { value: 'physics', label: 'Physics' },
    ],
    required: true,
  },
  {
    name: 'day',
    label: 'Day',
    description: 'Shift day',
    type: 'date',
    default: '2026-10-01',
    section: 'Facility',
  },
  {
    name: 'scan',
    label: 'Scan',
    description: 'Scan of that day',
    type: 'dynamic_select',
    default: null,
    section: 'Facility',
    options_endpoint: '/entry-fields/scan/options',
    depends_on: ['day'],
  },
  {
    name: 'beam_current',
    label: 'Beam current',
    description: 'mA',
    type: 'float',
    default: 500.5,
    section: 'Machine',
    min: 0,
    max: 600,
    step: 0.1,
  },
  {
    name: 'turns',
    label: 'Turns',
    description: '',
    type: 'int',
    default: null,
    section: 'Machine',
  },
  {
    name: 'top_off',
    label: 'Top-off',
    description: '',
    type: 'bool',
    default: true,
    section: 'Machine',
  },
  {
    name: 'note',
    label: 'Note',
    description: '',
    type: 'text',
    default: null,
    section: 'Machine',
    placeholder: 'Anything else',
  },
]);

/**
 * Stub `fetch` to answer every options request with `options`, recording URLs.
 * @param {(url: URL) => Array<{value: string, label: string}>} optionsFor
 * @returns {string[]} The requested URLs, in order.
 */
function stubOptionsFetch(optionsFor) {
  /** @type {string[]} */
  const calls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ input) => {
      calls.push(String(input));
      const url = new URL(String(input), window.location.origin);
      const field = url.pathname.split('/').at(-2);
      return new Response(JSON.stringify({ field, options: optionsFor(url) }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    })
  );
  return calls;
}

/** @param {string} name */
function control(name) {
  return /** @type {HTMLInputElement & HTMLSelectElement} */ (
    document.querySelector(`[data-entry-field="${name}"]`)
  );
}

/** @returns {Promise<any>} */
async function load() {
  return import(MODULE_PATH);
}

beforeEach(() => {
  vi.resetModules();
  document.body.innerHTML = FORM_HTML;
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('rendering', () => {
  test('one section per descriptor section, after Metadata, in declaration order', async () => {
    stubOptionsFetch(() => []);
    const mod = await load();
    await mod.renderEntryFields(FIELDS);

    const mount = /** @type {HTMLElement} */ (document.getElementById('entry-fields'));
    const sections = [...mount.querySelectorAll('[data-entry-field-section]')];
    expect(sections.map((s) => s.querySelector('.form-section-title')?.textContent)).toEqual([
      'Facility',
      'Machine',
    ]);
    // The mount sits after the Metadata section and before Tags.
    const titles = [...document.querySelectorAll('.form-section-title')].map((t) => t.textContent);
    expect(titles).toEqual(['Metadata', 'Facility', 'Machine', 'Tags']);
    const facility = sections[0];
    expect(
      [...facility.querySelectorAll('[data-entry-field]')].map(
        (el) => /** @type {HTMLElement} */ (el).dataset.entryField
      )
    ).toEqual(['book', 'day', 'scan']);
  });

  test('plain input per type, with default pre-filled', async () => {
    stubOptionsFetch(() => []);
    const mod = await load();
    await mod.renderEntryFields(FIELDS);

    expect(control('book').tagName).toBe('SELECT');
    expect(control('book').value).toBe('ops');
    expect([...control('book').options].map((o) => o.value)).toContain('physics');

    expect(control('day').type).toBe('date');
    expect(control('day').value).toBe('2026-10-01');

    expect(control('scan').tagName).toBe('SELECT');

    expect(control('beam_current').type).toBe('number');
    expect(control('beam_current').value).toBe('500.5');
    expect(control('beam_current').min).toBe('0');
    expect(control('beam_current').max).toBe('600');
    expect(control('beam_current').step).toBe('0.1');

    expect(control('turns').type).toBe('number');
    expect(control('turns').step).toBe('1');
    expect(control('turns').value).toBe('');

    expect(control('top_off').type).toBe('checkbox');
    expect(control('top_off').checked).toBe(true);

    expect(control('note').type).toBe('text');
    expect(control('note').placeholder).toBe('Anything else');
  });

  test('required fields are marked; optional ones are not', async () => {
    stubOptionsFetch(() => []);
    const mod = await load();
    await mod.renderEntryFields(FIELDS);

    expect(control('book').required).toBe(true);
    const bookLabel = document.querySelector(`label[for="${control('book').id}"]`);
    expect(bookLabel?.textContent).toBe('Book *');
    expect(control('day').required).toBe(false);
    expect(document.querySelector(`label[for="${control('day').id}"]`)?.textContent).toBe('Day');
  });

  test('labels and option text are rendered as text, never as markup', async () => {
    stubOptionsFetch(() => []);
    const mod = await load();
    await mod.renderEntryFields([
      {
        name: 'x',
        label: '<img src=x onerror=alert(1)>',
        description: '',
        type: 'select',
        default: null,
        section: '<b>S</b>',
        options: [{ value: 'a', label: '<i>A</i>' }],
      },
    ]);
    expect(document.querySelector('img')).toBeNull();
    expect(document.querySelector('b')).toBeNull();
    expect(document.querySelector('i')).toBeNull();
  });

  test('no declarations render nothing and change nothing', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const before = document.body.innerHTML;
    const mod = await load();
    await mod.renderEntryFields([]);

    expect(document.body.innerHTML).toBe(before);
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(mod.collectEntryFieldValues()).toEqual({});
  });

  test('rendering again replaces the previous fields', async () => {
    stubOptionsFetch(() => []);
    const mod = await load();
    await mod.renderEntryFields(FIELDS);
    await mod.renderEntryFields([FIELDS[0]]);

    expect(document.querySelectorAll('[data-entry-field]').length).toBe(1);
    expect(document.querySelectorAll('[data-entry-field-section]').length).toBe(1);
  });
});

describe('logbook / shift override', () => {
  const LOGBOOK = {
    name: 'logbook',
    label: 'Logbook',
    description: '',
    type: 'select',
    default: 'ops',
    section: 'Facility',
    options: [
      { value: 'ops', label: 'Operations' },
      { value: 'physics', label: 'Physics' },
    ],
  };

  test('a declared logbook replaces the built-in input in place', async () => {
    stubOptionsFetch(() => []);
    const mod = await load();
    await mod.renderEntryFields([LOGBOOK]);

    const builtIns = document.querySelectorAll('#create-entry-form [name="logbook"]');
    expect(builtIns.length).toBe(1);
    const sel = /** @type {HTMLSelectElement} */ (builtIns[0]);
    expect(sel.tagName).toBe('SELECT');
    expect(sel.dataset.entryField).toBe('logbook');
    expect(sel.id).toBe('entry-logbook');
    // In place: still between Author and Shift in the Metadata row.
    const row = /** @type {HTMLElement} */ (document.querySelector('.form-row'));
    const names = [...row.querySelectorAll('input, select')].map(
      (el) => /** @type {HTMLInputElement} */ (el).name
    );
    expect(names).toEqual(['author', 'logbook', 'shift']);
    // A section holding only the override is not rendered.
    expect(document.querySelectorAll('[data-entry-field-section]').length).toBe(0);
    // The built-in path's FormData reads the declared value.
    const fd = new FormData(/** @type {HTMLFormElement} */ (document.getElementById('create-entry-form')));
    expect(fd.get('logbook')).toBe('ops');
    expect(mod.collectEntryFieldValues()).toEqual({ logbook: 'ops' });
  });

  test('a declared shift replaces the built-in shift input', async () => {
    stubOptionsFetch(() => []);
    const mod = await load();
    await mod.renderEntryFields([
      { name: 'shift', label: 'Shift', description: '', type: 'text', default: 'owl', section: 'X' },
    ]);
    const shift = /** @type {HTMLInputElement} */ (document.getElementById('entry-shift'));
    expect(shift.dataset.entryField).toBe('shift');
    expect(shift.value).toBe('owl');
    expect(shift.name).toBe('shift');
  });

  test('re-rendering without the override restores the built-in input', async () => {
    stubOptionsFetch(() => []);
    const mod = await load();
    await mod.renderEntryFields([LOGBOOK]);
    await mod.renderEntryFields([]);
    const logbook = /** @type {HTMLInputElement} */ (document.getElementById('entry-logbook'));
    expect(logbook.tagName).toBe('INPUT');
    expect(logbook.dataset.entryField).toBeUndefined();
  });
});

describe('dynamic select', () => {
  test('fetches its options endpoint with its depends_on values', async () => {
    const calls = stubOptionsFetch((url) => [
      { value: `scan-${url.searchParams.get('day')}`, label: 'S' },
    ]);
    const mod = await load();
    await mod.renderEntryFields(FIELDS);

    expect(calls.length).toBe(1);
    const url = new URL(calls[0], window.location.origin);
    expect(url.pathname).toBe('/api/entry-fields/scan/options');
    expect(Object.fromEntries(url.searchParams)).toEqual({ day: '2026-10-01' });
    expect([...control('scan').options].map((o) => o.value)).toEqual(['', 'scan-2026-10-01']);
  });

  test('refetches with the new value when a depends_on field changes', async () => {
    const calls = stubOptionsFetch((url) => [
      { value: `scan-${url.searchParams.get('day')}`, label: 'S' },
    ]);
    const mod = await load();
    await mod.renderEntryFields(FIELDS);

    const day = control('day');
    day.value = '2026-10-03';
    day.dispatchEvent(new Event('change', { bubbles: true }));

    await vi.waitFor(() => expect(calls.length).toBe(2));
    const url = new URL(calls[1], window.location.origin);
    expect(url.searchParams.get('day')).toBe('2026-10-03');
    await vi.waitFor(() =>
      expect([...control('scan').options].map((o) => o.value)).toContain('scan-2026-10-03')
    );
  });

  test('a change to a field it does not depend on fetches nothing', async () => {
    const calls = stubOptionsFetch(() => []);
    const mod = await load();
    await mod.renderEntryFields(FIELDS);
    control('note').value = 'x';
    control('note').dispatchEvent(new Event('change', { bubbles: true }));
    control('book').value = 'physics';
    control('book').dispatchEvent(new Event('change', { bubbles: true }));
    await new Promise((r) => setTimeout(r, 0));
    expect(calls.length).toBe(1);
  });

  test('a blank parent is left out of the query', async () => {
    const calls = stubOptionsFetch(() => []);
    const mod = await load();
    await mod.renderEntryFields(FIELDS.map((f) => (f.name === 'day' ? { ...f, default: null } : f)));
    const url = new URL(calls[0], window.location.origin);
    expect([...url.searchParams.keys()]).toEqual([]);
  });

  test('keeps the selected value when the refetched options still hold it', async () => {
    stubOptionsFetch(() => [
      { value: 'a', label: 'A' },
      { value: 'b', label: 'B' },
    ]);
    const mod = await load();
    await mod.renderEntryFields(FIELDS);
    control('scan').value = 'b';
    control('day').value = '2026-10-05';
    control('day').dispatchEvent(new Event('change', { bubbles: true }));
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));
    expect(control('scan').value).toBe('b');
  });

  test('a stale response never overwrites a newer one', async () => {
    /** @type {Array<(r: Response) => void>} */
    const pending = [];
    /** @type {string[]} */
    const calls = [];
    vi.stubGlobal(
      'fetch',
      vi.fn((/** @type {string} */ input) => {
        calls.push(String(input));
        return new Promise((resolve) => pending.push(resolve));
      })
    );
    const mod = await load();
    const rendered = mod.renderEntryFields(FIELDS);
    await vi.waitFor(() => expect(pending.length).toBe(1));
    /** @param {string} v */
    const answer = (v) =>
      new Response(JSON.stringify({ field: 'scan', options: [{ value: v, label: v }] }), {
        status: 200,
      });
    pending[0](answer('first'));
    await rendered;

    control('day').value = '2026-10-02';
    control('day').dispatchEvent(new Event('change', { bubbles: true }));
    control('day').value = '2026-10-03';
    control('day').dispatchEvent(new Event('change', { bubbles: true }));
    await vi.waitFor(() => expect(pending.length).toBe(3));
    pending[2](answer('newest'));
    await vi.waitFor(() =>
      expect([...control('scan').options].map((o) => o.value)).toContain('newest')
    );
    pending[1](answer('stale'));
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));
    expect([...control('scan').options].map((o) => o.value)).not.toContain('stale');
  });

  test('an options failure leaves the form rendered and the promise settled', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ detail: 'down' }), { status: 502 }))
    );
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const mod = await load();
    await mod.renderEntryFields(FIELDS);
    await expect(mod.entryFieldsRendered()).resolves.toBeUndefined();
    expect(control('scan')).not.toBeNull();
    expect(control('scan').disabled).toBe(false);
  });
});

describe('collection', () => {
  test('returns declared values native-typed, blanks left out', async () => {
    stubOptionsFetch(() => [{ value: 'scan-7', label: 'Scan 7' }]);
    const mod = await load();
    await mod.renderEntryFields(FIELDS);

    expect(mod.collectEntryFieldValues()).toEqual({
      book: 'ops',
      day: '2026-10-01',
      beam_current: 500.5,
      top_off: true,
    });

    control('scan').value = 'scan-7';
    control('turns').value = '12';
    control('top_off').checked = false;
    control('note').value = 'hello';
    expect(mod.collectEntryFieldValues()).toEqual({
      book: 'ops',
      day: '2026-10-01',
      scan: 'scan-7',
      beam_current: 500.5,
      turns: 12,
      top_off: false,
      note: 'hello',
    });
  });

  test('a declared logbook override is collected; the built-ins are not', async () => {
    stubOptionsFetch(() => []);
    const mod = await load();
    await mod.renderEntryFields([FIELDS[0]]);
    /** @type {HTMLInputElement} */ (document.getElementById('entry-logbook')).value = 'L';
    expect(mod.collectEntryFieldValues()).toEqual({ book: 'ops' });
  });
});

describe('markEntryFieldError', () => {
  test('marks the named input with the message and clears on edit', async () => {
    stubOptionsFetch(() => []);
    const mod = await load();
    await mod.renderEntryFields(FIELDS);

    expect(mod.markEntryFieldError('book', 'Pick a book')).toBe(true);
    const book = control('book');
    expect(book.getAttribute('aria-invalid')).toBe('true');
    const msg = book.parentElement?.querySelector('[data-entry-field-error]');
    expect(msg?.textContent).toBe('Pick a book');
    expect(document.activeElement).toBe(book);

    // Marking again replaces, never stacks, the message.
    mod.markEntryFieldError('book', 'Still wrong');
    expect(book.parentElement?.querySelectorAll('[data-entry-field-error]').length).toBe(1);

    book.value = 'physics';
    book.dispatchEvent(new Event('change', { bubbles: true }));
    expect(book.hasAttribute('aria-invalid')).toBe(false);
    expect(book.parentElement?.querySelector('[data-entry-field-error]')).toBeNull();
  });

  test('returns false for a name that is not rendered', async () => {
    stubOptionsFetch(() => []);
    const mod = await load();
    await mod.renderEntryFields(FIELDS);
    expect(mod.markEntryFieldError('nope', 'x')).toBe(false);
  });
});

describe('rendered promise', () => {
  test('is pending before render and settles after the fields exist', async () => {
    stubOptionsFetch(() => [{ value: 'z', label: 'Z' }]);
    const mod = await load();
    let settled = false;
    const ready = mod.entryFieldsRendered().then(() => {
      settled = true;
    });
    await new Promise((r) => setTimeout(r, 0));
    expect(settled).toBe(false);

    mod.renderEntryFields(FIELDS);
    await ready;
    // Fields and the first dynamic options are in place once it settles.
    expect(control('book')).not.toBeNull();
    expect([...control('scan').options].map((o) => o.value)).toContain('z');
  });

  test('settles when the form is known to declare none', async () => {
    const mod = await load();
    const ready = mod.entryFieldsRendered();
    mod.renderEntryFields([]);
    await expect(ready).resolves.toBeUndefined();
  });
});
