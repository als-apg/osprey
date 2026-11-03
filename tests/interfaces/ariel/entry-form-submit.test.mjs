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
 *     the named input and keeps the form populated; `ApiError` carries `field`;
 *   - entries.js hosts the form in `#ariel-compose` off the store's `compose`
 *     key: opening shows it (empty, or with `compose.draft` loaded) and sets
 *     `narrowView:'list'`; Cancel writes `compose:null`; a successful create
 *     closes compose, selects the new entry for the reader and remembers the
 *     author, with no `alert()`.
 *
 *   npx vitest run tests/interfaces/ariel/entry-form-submit.test.mjs
 *
 * entry-fields.js, entries-form.js and store.js keep state at module scope, so
 * each test re-imports the modules fresh via vi.resetModules(). `fetch` is
 * stubbed per test.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const STATIC = '../../../src/osprey/interfaces/ariel/static';

/** The compose host and create form as index.html ships them, trimmed to what submission reads. */
const FORM_HTML = `
  <section id="ariel-compose" hidden>
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
    <button type="button" id="create-cancel-btn">Cancel</button>
    <button type="submit">Save</button>
  </form>
  </section>
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
          routes.publishInfoStatus ?? 200
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
    })
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
  const store = await import(`${STATIC}/js/store.js`);
  return { entries, form, fields, api, store };
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
  await formModule.handleCreateEntry(/** @type {any} */ ({ preventDefault() {}, target: form }));
}

/**
 * The create request among `calls`.
 * @param {Call[]} calls
 * @returns {Call}
 */
function createCall(calls) {
  const call = calls.find(
    (c) => c.url.endsWith('/api/entries') || c.url.endsWith('/api/entries/upload')
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
  localStorage.clear();
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

  test('the multipart upload carries the header a plain form cannot send', async () => {
    const calls = stubFetch({
      publishInfo: { supports_write: true, requires_auth: false, entry_fields: FIELDS },
    });
    const { entries, form, fields } = await loadModules();
    await renderDeclared(entries, fields);
    stageAttachment();
    await submit(form);

    const call = createCall(calls);
    expect(call.url.endsWith('/api/entries/upload')).toBe(true);
    const headers = new Headers(call.init.headers);
    expect(headers.get('X-Osprey-Upload')).toBe('1');
    // The browser must write multipart/form-data with its own boundary.
    expect(headers.has('Content-Type')).toBe(false);
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
      body: {
        detail: 'Beam current must be at most 500',
        code: 'invalid_entry_field',
        field: 'beam_current',
      },
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
    const button = /** @type {HTMLButtonElement} */ (
      theForm().querySelector('button[type="submit"]')
    );
    expect(button.disabled).toBe(false);
  });

  test('entry_field_options_unavailable on the upload path marks the input', async () => {
    await submitRejected(
      {
        status: 502,
        body: {
          detail: 'Book choices unavailable',
          code: 'entry_field_options_unavailable',
          field: 'book',
        },
      },
      true
    );

    expect(control('book').getAttribute('aria-invalid')).toBe('true');
    expect(document.querySelector('[data-entry-field-error="book"]')?.textContent).toBe(
      'Book choices unavailable'
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

describe('the compose host', () => {
  /** @type {{destroy(): void}|null} */
  let host = null;

  afterEach(() => {
    host?.destroy();
    host = null;
  });

  /** Let every pending microtask (and the awaits chained behind it) run. */
  const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

  /** @returns {HTMLElement} */
  function composeEl() {
    return /** @type {HTMLElement} */ (document.getElementById('ariel-compose'));
  }

  /** @param {string} name @returns {HTMLInputElement} */
  function field(name) {
    return /** @type {HTMLInputElement} */ (theForm().querySelector(`[name="${name}"]`));
  }

  /** @param {Parameters<typeof stubFetch>[0]} [routes] */
  async function boot(routes = {}) {
    const calls = stubFetch(routes);
    const mods = await loadModules();
    host = mods.entries.initEntries();
    await mods.fields.entryFieldsRendered();
    return { calls, ...mods };
  }

  test('stays hidden until compose is set', async () => {
    await boot();
    expect(composeEl().hidden).toBe(true);
  });

  test('opening empty shows the host, clears the form and sets narrowView list', async () => {
    const { store } = await boot();
    store.setState({ narrowView: 'reader' });
    field('subject').value = 'left over';

    store.setState({ compose: { draft: null } });

    expect(composeEl().hidden).toBe(false);
    expect(store.getState().narrowView).toBe('list');
    // Reset to the markup's own default: the operator's leftover text is gone.
    expect(field('subject').value).toBe(field('subject').defaultValue);
    expect(document.getElementById('draft-banner')).toBeNull();
  });

  test('opening with a draft loads that draft into the form', async () => {
    const { store, calls } = await boot({
      draft: { subject: 'From the agent', details: 'Drafted', tags: ['rf'] },
    });

    store.setState({ compose: { draft: 'd-7' } });
    await flush();
    await flush();

    expect(composeEl().hidden).toBe(false);
    expect(calls.some((c) => c.url.includes('/api/drafts/d-7'))).toBe(true);
    expect(field('subject').value).toBe('From the agent');
    expect(document.querySelector('#entry-tags .tag')?.getAttribute('data-value')).toBe('rf');
    expect(document.getElementById('draft-banner')?.dataset.draftId).toBe('d-7');
  });

  test('a compose already set at init is applied', async () => {
    stubFetch({ draft: { subject: 'Early draft' } });
    const mods = await loadModules();
    mods.store.setState({ compose: { draft: 'd-0' }, narrowView: 'reader' });
    host = mods.entries.initEntries();
    await mods.fields.entryFieldsRendered();
    await flush();
    await flush();

    expect(composeEl().hidden).toBe(false);
    expect(mods.store.getState().narrowView).toBe('list');
    expect(field('subject').value).toBe('Early draft');
  });

  test('closing hides the host; Cancel writes compose null', async () => {
    const { store } = await boot();
    store.setState({ compose: { draft: null } });

    /** @type {HTMLElement} */ (document.getElementById('create-cancel-btn')).click();

    expect(store.getState().compose).toBeNull();
    expect(composeEl().hidden).toBe(true);
  });

  test('a successful create closes compose, selects the entry and remembers the author', async () => {
    const { store, form } = await boot({ create: { status: 200, body: { entry_id: 'e-42' } } });
    store.setState({ compose: { draft: null } });
    field('subject').value = 'Typed subject';
    field('author').value = 'Ada';

    await submit(form);

    const state = store.getState();
    expect(state.compose).toBeNull();
    expect(state.sel).toBe('e-42');
    expect(state.narrowView).toBe('reader');
    expect(state.mineAuthor).toBe('Ada');
    expect(composeEl().hidden).toBe(true);
    expect(window.alert).not.toHaveBeenCalled();
    expect(field('subject').value).toBe(field('subject').defaultValue);
  });

  test('a successful create from a draft forgets the draft attachments', async () => {
    const { store, form } = await boot();
    store.setState({ compose: { draft: null } });
    stageAttachment();

    await submit(form);

    const preview = /** @type {HTMLElement} */ (document.getElementById('file-preview'));
    expect(preview.dataset.draftAttachments).toBeUndefined();
    expect(store.getState().sel).toBe('e-1');
  });

  test('an Anonymous author is not remembered', async () => {
    const { store, form } = await boot();
    store.setState({ compose: { draft: null } });
    field('subject').value = 'Beam dump';
    field('author').value = 'Anonymous';

    await submit(form);

    expect(store.getState().mineAuthor).toBeNull();
    expect(store.getState().sel).toBe('e-1');
  });

  test('a credentials error keeps compose open and the form populated', async () => {
    const { store, form } = await boot({
      create: { status: 401, body: { detail: 'auth', code: 'auth_required' } },
    });
    store.setState({ compose: { draft: null } });
    field('subject').value = 'Beam dump';

    await submit(form);

    expect(window.alert).toHaveBeenCalledWith(
      'Logbook credentials required to publish. Please enter your username and password.'
    );
    expect(store.getState().compose).toEqual({ draft: null });
    expect(store.getState().sel).toBeNull();
    expect(field('subject').value).toBe('Beam dump');
    expect(document.activeElement?.id).toBe('entry-auth-user');
  });

  test('a generic failure alerts and keeps compose open', async () => {
    const { store, form } = await boot({ create: { status: 500, body: { detail: 'boom' } } });
    store.setState({ compose: { draft: null } });

    await submit(form);

    expect(window.alert).toHaveBeenCalledWith('Failed to create entry: boom');
    expect(store.getState().compose).toEqual({ draft: null });
  });

  test('destroy stops reacting to compose', async () => {
    const { store } = await boot();
    host?.destroy();
    host = null;

    store.setState({ compose: { draft: null } });

    expect(composeEl().hidden).toBe(true);
  });
});

describe('entries.js is the compose host only', () => {
  test('it no longer re-exports the entry modal surface', async () => {
    stubFetch({});
    const { entries } = await loadModules();
    for (const name of [
      'showEntry',
      'openEntry',
      'closeEntryModal',
      'showImageLightbox',
      'getCurrentEntry',
    ]) {
      expect(entries[name], name).toBeUndefined();
      expect(entries.default[name], `default.${name}`).toBeUndefined();
    }
  });

  test('neither module imports the retired detail modal', () => {
    for (const file of ['entries.js', 'entries-form.js']) {
      const source = readFileSync(join(import.meta.dirname, STATIC, 'js', file), 'utf8');
      expect(source, file).not.toContain('entries-detail');
    }
  });
});
