// @ts-check
/**
 * Front-end proofs that the ARIEL create view carries the facility-declared
 * entry fields from publish-info through to the create request:
 *
 *   - index.html ships one empty `#entry-fields` mount right after "Metadata";
 *   - opening the create view renders `publish-info.entry_fields` into it, and
 *     an empty list (or an unreadable publish-info) leaves the form unchanged;
 *   - both create paths (JSON and multipart upload) submit the declared values
 *     inside `metadata`, on top of any draft metadata; with nothing declared the
 *     payload is exactly the built-in one;
 *   - a loaded draft pre-fills the declared inputs from its `fields`, also when
 *     it loads before publish-info answers, and the operator's edits win;
 *   - an `invalid_entry_field` / `entry_field_options_unavailable` error marks
 *     the named input and keeps the form populated; `ApiError` carries `field`.
 *
 *   npx vitest run tests/interfaces/ariel/entry-form-submit.test.mjs
 *
 * entry-fields.js and entries-form.js keep state at module scope, so each test
 * re-imports the modules fresh via vi.resetModules(). `fetch` is stubbed per test.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const STATIC = '../../../src/osprey/interfaces/ariel/static';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/entries-detail.js', () => ({
  showEntry: vi.fn(),
  openEntry: vi.fn(),
  closeEntryModal: vi.fn(),
  showImageLightbox: vi.fn(),
  getCurrentEntry: vi.fn(),
  initEntryDetail: vi.fn(),
}));

/** The create form as index.html ships it, trimmed to what submission reads. */
const FORM_HTML = `
  <form id="create-entry-form">
    <input type="text" id="entry-subject" name="subject" value="Beam dump">
    <textarea id="entry-details" name="details">Lost beam at 10:02</textarea>
    <div class="form-section">
      <div class="form-section-title">Metadata</div>
      <div class="form-row">
        <div class="input-group">
          <label class="input-label" for="entry-author">Author</label>
          <input type="text" id="entry-author" name="author" class="input" value="ops">
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
    <div class="form-section">
      <div class="form-section-title">Tags</div>
      <div id="entry-tags"></div>
    </div>
    <div id="publish-helper"></div>
    <div id="publish-credentials">
      <input type="text" id="entry-auth-user" name="auth_user">
      <input type="password" id="entry-auth-password" name="auth_password">
    </div>
    <input type="file" id="entry-files" multiple>
    <div id="file-preview"></div>
    <button type="submit">Save</button>
  </form>
`;

/** Declarations shaped like `publish-info.entry_fields`. */
const FIELDS = /** @type {any[]} */ ([
  {
    name: 'book',
    label: 'Book',
    type: 'select',
    default: 'ops',
    section: 'Facility',
    options: [
      { value: 'ops', label: 'Operations' },
      { value: 'physics', label: 'Physics' },
    ],
    required: true,
  },
  { name: 'beam_current', label: 'Beam current', type: 'float', section: 'Facility' },
  { name: 'beam_on', label: 'Beam on', type: 'bool', default: true, section: 'Facility' },
]);

/**
 * @typedef {{url: string, init: any}} Call
 */

/**
 * Stub `fetch`, routing by URL. `create` answers both create paths.
 * @param {{
 *   publishInfo?: any,
 *   publishInfoStatus?: number,
 *   create?: {status: number, body: any},
 *   draft?: any,
 *   publishInfoGate?: Promise<void>,
 * }} routes
 * @returns {Call[]} Every request made, in order
 */
function stubFetch(routes) {
  /** @type {Call[]} */
  const calls = [];
  const json = (/** @type {any} */ body, status = 200) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json' },
    });
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {any} */ input, /** @type {any} */ init) => {
      const url = String(input);
      calls.push({ url, init });
      if (url.includes('/api/publish-info')) {
        if (routes.publishInfoGate) await routes.publishInfoGate;
        return json(
          routes.publishInfo ?? {
            supports_write: true,
            requires_auth: false,
            source_system: 'TestLog',
            entry_fields: [],
          },
          routes.publishInfoStatus ?? 200,
        );
      }
      if (url.includes('/api/drafts/')) return json(routes.draft ?? {});
      if (url.endsWith('/api/entries') || url.endsWith('/api/entries/upload')) {
        const create = routes.create ?? { status: 200, body: { entry_id: 'e-1' } };
        return json(create.body, create.status);
      }
      if (url.startsWith('/att/')) {
        return new Response(new Blob(['png-bytes'], { type: 'image/png' }));
      }
      return json({ detail: `unrouted ${url}` }, 404);
    }),
  );
  return calls;
}

/** Fresh copies of the modules the create view uses, sharing one registry. */
async function loadModules() {
  vi.resetModules();
  const entries = await import(`${STATIC}/js/entries.js`);
  const form = await import(`${STATIC}/js/entries-form.js`);
  const fields = await import(`${STATIC}/js/entry-fields.js`);
  const api = await import(`${STATIC}/js/api.js`);
  return { entries, form, fields, api };
}

/** @returns {HTMLFormElement} */
function theForm() {
  return /** @type {HTMLFormElement} */ (document.getElementById('create-entry-form'));
}

/**
 * The create form's markup minus the publishing section, which publish-info
 * adapts independently of the declared fields.
 * @returns {string}
 */
function formWithoutPublishing() {
  const clone = /** @type {HTMLFormElement} */ (theForm().cloneNode(true));
  clone.querySelector('#publish-helper')?.remove();
  clone.querySelector('#publish-credentials')?.remove();
  return clone.innerHTML;
}

/**
 * Submit the create form through its handler.
 * @param {any} formModule
 */
async function submit(formModule) {
  const form = theForm();
  await formModule.handleCreateEntry(
    /** @type {any} */ ({ preventDefault() {}, target: form }),
  );
}

/**
 * The create request among `calls`.
 * @param {Call[]} calls
 * @returns {Call}
 */
function createCall(calls) {
  const call = calls.find(
    (c) => c.url.endsWith('/api/entries') || c.url.endsWith('/api/entries/upload'),
  );
  if (!call) throw new Error('no create request was made');
  return call;
}

/** Stage one draft attachment so submission takes the multipart path. */
function stageAttachment() {
  const preview = /** @type {HTMLElement} */ (document.getElementById('file-preview'));
  preview.dataset.draftAttachments = JSON.stringify([{ url: '/att/1', filename: 'plot.png' }]);
}

/**
 * @param {string} name
 * @returns {HTMLInputElement|HTMLSelectElement}
 */
function control(name) {
  const el = document.querySelector(`#entry-fields [data-entry-field="${name}"]`);
  if (!el) throw new Error(`no rendered input named ${name}`);
  return /** @type {HTMLInputElement|HTMLSelectElement} */ (el);
}

beforeEach(() => {
  document.body.innerHTML = FORM_HTML;
  vi.stubGlobal('alert', vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe('index.html mount point', () => {
  test('one empty #entry-fields sits right after the Metadata section, before Tags', () => {
    const html = readFileSync(join(import.meta.dirname, STATIC, 'index.html'), 'utf8');
    // A template's content is inert, so parsing the page loads none of its resources.
    const template = document.createElement('template');
    template.innerHTML = html;
    const mounts = template.content.querySelectorAll('#entry-fields');
    expect(mounts.length).toBe(1);
    const mount = /** @type {Element} */ (mounts[0]);
    expect(mount.children.length).toBe(0);
    expect(mount.textContent?.trim()).toBe('');
    const before = mount.previousElementSibling;
    expect(before?.classList.contains('form-section')).toBe(true);
    expect(before?.querySelector('.form-section-title')?.textContent?.trim()).toBe('Metadata');
    const after = mount.nextElementSibling;
    expect(after?.querySelector('.form-section-title')?.textContent?.trim()).toBe('Tags');
    expect(mount.closest('#create-entry-form')).not.toBeNull();
  });
});

describe('create view renders the declared fields', () => {
  test('publish-info entry_fields render into the mount', async () => {
    stubFetch({
      publishInfo: {
        supports_write: true,
        requires_auth: false,
        source_system: 'TestLog',
        entry_fields: FIELDS,
      },
    });
    const { entries, fields } = await loadModules();
    await entries.adaptCreateForm();
    await fields.entryFieldsRendered();

    expect(control('book').value).toBe('ops');
    expect(control('beam_current')).toBeTruthy();
    expect(/** @type {HTMLInputElement} */ (control('beam_on')).checked).toBe(true);
    expect(document.getElementById('publish-helper')?.textContent).toContain('TestLog');
  });

  test('an empty entry_fields list renders nothing and changes nothing', async () => {
    stubFetch({});
    const before = formWithoutPublishing();
    const { entries, fields } = await loadModules();
    await entries.adaptCreateForm();
    await fields.entryFieldsRendered();

    // Only the publishing section adapts to publish-info; the rest is untouched.
    expect(formWithoutPublishing()).toBe(before);
    expect(document.getElementById('entry-fields')?.children.length).toBe(0);
  });

  test('an unreadable publish-info leaves the form unchanged and settles the render', async () => {
    stubFetch({ publishInfo: { detail: 'down' }, publishInfoStatus: 503 });
    const before = theForm().innerHTML;
    const { entries, fields } = await loadModules();
    await entries.adaptCreateForm();
    await fields.entryFieldsRendered();
    expect(theForm().innerHTML).toBe(before);
  });
});

describe('submission merges the declared values into metadata', () => {
  /** @param {any} entries @param {any} fields */
  async function renderDeclared(entries, fields) {
    await entries.adaptCreateForm();
    await fields.entryFieldsRendered();
    control('book').value = 'physics';
    control('beam_current').value = '401.5';
  }

  const DECLARED = { book: 'physics', beam_current: 401.5, beam_on: true };

  test('the JSON create path sends the declared values in metadata', async () => {
    const calls = stubFetch({
      publishInfo: { supports_write: true, requires_auth: false, entry_fields: FIELDS },
    });
    const { entries, form, fields } = await loadModules();
    await renderDeclared(entries, fields);
    await submit(form);

    const call = createCall(calls);
    expect(call.url.endsWith('/api/entries')).toBe(true);
    const body = JSON.parse(call.init.body);
    expect(body.metadata).toEqual(DECLARED);
    expect(body.subject).toBe('Beam dump');
  });

  test('the multipart upload path sends the declared values in metadata', async () => {
    const calls = stubFetch({
      publishInfo: { supports_write: true, requires_auth: false, entry_fields: FIELDS },
    });
    const { entries, form, fields } = await loadModules();
    await renderDeclared(entries, fields);
    stageAttachment();
    await submit(form);

    const call = createCall(calls);
    expect(call.url.endsWith('/api/entries/upload')).toBe(true);
    const sent = /** @type {FormData} */ (call.init.body);
    expect(JSON.parse(String(sent.get('metadata')))).toEqual(DECLARED);
  });

  test('declared values win over draft metadata keys, which are kept', async () => {
    const calls = stubFetch({
      publishInfo: { supports_write: true, requires_auth: false, entry_fields: FIELDS },
      draft: {
        subject: 'Beam dump',
        details: 'Lost beam at 10:02',
        metadata: { session_metadata: { id: 's-1' }, book: 'ops' },
      },
    });
    const { entries, form, fields } = await loadModules();
    await renderDeclared(entries, fields);
    await form.loadDraft('d-1');
    control('book').value = 'physics';
    await submit(form);

    const body = JSON.parse(createCall(calls).init.body);
    expect(body.metadata).toEqual({ session_metadata: { id: 's-1' }, ...DECLARED });
  });

  test('with nothing declared, the JSON path sends metadata null as before', async () => {
    const calls = stubFetch({});
    const { entries, form, fields } = await loadModules();
    await entries.adaptCreateForm();
    await fields.entryFieldsRendered();
    await submit(form);

    const body = JSON.parse(createCall(calls).init.body);
    expect(body.metadata).toBeNull();
  });

  test('with nothing declared, the upload path sends no metadata part', async () => {
    const calls = stubFetch({});
    const { entries, form, fields } = await loadModules();
    await entries.adaptCreateForm();
    await fields.entryFieldsRendered();
    stageAttachment();
    await submit(form);

    const sent = /** @type {FormData} */ (createCall(calls).init.body);
    expect(sent.has('metadata')).toBe(false);
  });
});

describe('a draft pre-fills the declared inputs', () => {
  const PUBLISH_INFO = { supports_write: true, requires_auth: false, entry_fields: FIELDS };

  test('the draft value fills the input and the operator edit is what is submitted', async () => {
    const calls = stubFetch({
      publishInfo: PUBLISH_INFO,
      draft: {
        subject: 'Beam dump',
        details: 'Lost beam at 10:02',
        fields: { book: 'physics', beam_current: 401.5, beam_on: false },
        metadata: { session_metadata: { id: 's-1' }, book: 'physics' },
      },
    });
    const { entries, form, fields } = await loadModules();
    await entries.adaptCreateForm();
    await fields.entryFieldsRendered();
    await form.loadDraft('d-1');

    expect(control('book').value).toBe('physics');
    expect(control('beam_current').value).toBe('401.5');
    expect(/** @type {HTMLInputElement} */ (control('beam_on')).checked).toBe(false);

    control('book').value = 'ops';
    await submit(form);

    const body = JSON.parse(createCall(calls).init.body);
    expect(body.metadata).toEqual({
      session_metadata: { id: 's-1' },
      book: 'ops',
      beam_current: 401.5,
      beam_on: false,
    });
  });

  test('a draft loaded before publish-info answers still pre-fills', async () => {
    /** @type {() => void} */
    let release = () => {};
    const publishInfoGate = new Promise((/** @type {(v?: void) => void} */ resolve) => {
      release = () => resolve();
    });
    const calls = stubFetch({
      publishInfo: PUBLISH_INFO,
      publishInfoGate,
      draft: { subject: 'Beam dump', details: 'Lost beam at 10:02', fields: { book: 'physics' } },
    });
    const { entries, form } = await loadModules();
    const adapted = entries.adaptCreateForm();
    const loaded = form.loadDraft('d-1');
    // Let the draft request answer while publish-info is still held.
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(document.querySelector('#entry-fields [data-entry-field="book"]')).toBeNull();

    release();
    await adapted;
    await loaded;

    expect(control('book').value).toBe('physics');
    await submit(form);
    const body = JSON.parse(createCall(calls).init.body);
    expect(body.metadata.book).toBe('physics');
  });

  test('a declared name absent from the draft fields keeps its default, not stale metadata', async () => {
    const calls = stubFetch({
      publishInfo: PUBLISH_INFO,
      draft: {
        subject: 'Beam dump',
        details: 'Lost beam at 10:02',
        metadata: { session_metadata: { id: 's-1' }, beam_current: 12 },
      },
    });
    const { entries, form, fields } = await loadModules();
    await entries.adaptCreateForm();
    await fields.entryFieldsRendered();
    await form.loadDraft('d-1');
    await submit(form);

    const body = JSON.parse(createCall(calls).init.body);
    expect(body.metadata).toEqual({ session_metadata: { id: 's-1' }, book: 'ops', beam_on: true });
  });
});

describe('an entry-field error marks the named input', () => {
  /**
   * @param {{status: number, body: any}} create
   * @param {boolean} [upload]
   */
  async function submitRejected(create, upload = false) {
    stubFetch({
      publishInfo: { supports_write: true, requires_auth: false, entry_fields: FIELDS },
      create,
    });
    const { entries, form, fields } = await loadModules();
    await entries.adaptCreateForm();
    await fields.entryFieldsRendered();
    control('beam_current').value = '9999';
    if (upload) stageAttachment();
    await submit(form);
  }

  test('invalid_entry_field marks the input and keeps the form populated', async () => {
    await submitRejected({
      status: 422,
      body: { detail: 'Beam current must be at most 500', code: 'invalid_entry_field', field: 'beam_current' },
    });

    const input = control('beam_current');
    expect(input.getAttribute('aria-invalid')).toBe('true');
    const note = document.querySelector('[data-entry-field-error="beam_current"]');
    expect(note?.textContent).toBe('Beam current must be at most 500');
    expect(document.activeElement).toBe(input);
    expect(input.value).toBe('9999');
    const subject = /** @type {HTMLInputElement} */ (document.getElementById('entry-subject'));
    expect(subject.value).toBe('Beam dump');
    expect(window.alert).not.toHaveBeenCalled();
    const button = /** @type {HTMLButtonElement} */ (theForm().querySelector('button[type="submit"]'));
    expect(button.disabled).toBe(false);
  });

  test('entry_field_options_unavailable on the upload path marks the input', async () => {
    await submitRejected(
      {
        status: 502,
        body: { detail: 'Book choices unavailable', code: 'entry_field_options_unavailable', field: 'book' },
      },
      true,
    );

    expect(control('book').getAttribute('aria-invalid')).toBe('true');
    expect(document.querySelector('[data-entry-field-error="book"]')?.textContent).toBe(
      'Book choices unavailable',
    );
    expect(window.alert).not.toHaveBeenCalled();
  });

  test('an error naming no rendered input falls back to the generic alert', async () => {
    await submitRejected({
      status: 422,
      body: { detail: 'Unknown field', code: 'invalid_entry_field', field: 'nope' },
    });

    expect(window.alert).toHaveBeenCalledWith('Failed to create entry: Unknown field');
    expect(document.querySelector('[aria-invalid="true"]')).toBeNull();
  });

  test('ApiError carries the field named by the response', async () => {
    stubFetch({
      create: {
        status: 422,
        body: { detail: 'bad', code: 'invalid_entry_field', field: 'book' },
      },
    });
    const { api } = await loadModules();
    const error = await api.entriesApi
      .create({ subject: 's', details: 'd' })
      .catch((/** @type {any} */ e) => e);
    expect(error).toBeInstanceOf(api.ApiError);
    expect(error.code).toBe('invalid_entry_field');
    expect(error.field).toBe('book');
    expect(error.status).toBe(422);
    expect(new api.ApiError(400, 'x', 'c').field).toBeUndefined();
  });
});
