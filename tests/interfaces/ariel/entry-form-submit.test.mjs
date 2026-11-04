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
 *     key: opening shows it (empty, or with `compose.draft` loaded) and never
 *     writes `narrowView`; Discard and Close write `compose:null`; a successful
 *     create closes compose, selects the new entry for the reader and remembers
 *     the author, with no `alert()`;
 *   - submit and Discard (Decisions 9, 10): failures toast (C-TOAST) and keep
 *     the sheet and the draft; success finishes the sheet, warns when the
 *     attachments stayed local and deletes a draft whose load succeeded (a 404
 *     ignored, other failures warn); Discard finishes, closes and deletes a
 *     draft with a success toast; each settled DELETE bumps `draftsRev` once;
 *   - the sheet titles itself by kind, maximises and restores without touching
 *     the store (and reopens restored), and its footer names the destination;
 *   - index.html hosts the sheet outside `.ariel-list-col`, so it escapes that
 *     column's containing block and its stacked `display:none`.
 *   - a tag holding `"` or `\` adds once, de-duplicates and removes without
 *     throwing (addTag compares `dataset.value`, never a built selector).
 *   - a follow-up (`compose.followsUp`) prefills `Follow-up: <subject>` (never
 *     doubled), the parent's logbook and tags and a "Follows up" row, and both
 *     submit routes carry `metadata.follows_up`; an unloadable parent still
 *     opens with its id shown; nothing auto-submits; a parent answering after a
 *     reopen fills nothing, and a new entry after a follow-up carries no parent.
 *   - a draft (Decision 8) holds its staged attachments in memory from open,
 *     previews them with picked files and never resends them; a stale draft
 *     answer fills nothing; the provenance banner needs `session_metadata`; a
 *     failed load toasts, and a gone draft with no snapshot becomes a new entry.
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
  <div class="ariel-compose-head">
    <span id="ariel-compose-title">New entry</span>
    <button type="button" id="ariel-compose-max" aria-pressed="false">Expand</button>
    <button type="button" id="ariel-compose-close">Close</button>
  </div>
  <form id="create-entry-form">
    <div class="ariel-compose-banner" id="ariel-compose-banner" hidden></div>
    <div class="ariel-compose-parent" id="ariel-compose-parent" hidden>
      <span class="ariel-compose-parent-label">Follows up</span>
      <span class="ariel-compose-parent-ref" id="ariel-compose-parent-ref"></span>
    </div>
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
    <div class="ariel-compose-foot">
      <button type="submit">Submit</button>
      <button type="button" id="create-cancel-btn">Discard</button>
      <span id="ariel-compose-saved"></span>
      <span id="ariel-compose-dest"></span>
    </div>
  </form>
  </section>
`;

/**
 * The provenance banner while it shows, else null.
 * @returns {HTMLElement|null}
 */
function provenanceBanner() {
  const banner = document.getElementById('ariel-compose-banner');
  return banner && !banner.hidden ? banner : null;
}

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
 *   parent?: {status: number, body: any},
 *   parentGate?: Promise<void>,
 *   drafts?: Record<string, any>,
 *   draftStatus?: number,
 *   draftGates?: Record<string, Promise<void>>,
 *   missingAttachments?: string[],
 *   deleteStatus?: number,
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
      if (url.includes('/api/drafts/') && url.includes('/attachments/')) {
        const name = decodeURIComponent(url.split('/').pop() ?? '');
        if (routes.missingAttachments?.includes(name)) return json({ detail: 'gone' }, 404);
        return new Response(new Blob([`bytes of ${name}`], { type: 'image/png' }));
      }
      if (url.includes('/api/drafts/') && init?.method === 'DELETE') {
        const status = routes.deleteStatus ?? 204;
        return status === 204 ? new Response(null, { status }) : json({ detail: 'no' }, status);
      }
      if (url.includes('/api/drafts/')) {
        const id = url.split('/api/drafts/')[1].split(/[/?]/)[0];
        const gate = routes.draftGates?.[id];
        if (gate) await gate;
        const body = routes.drafts?.[id] ?? routes.draft;
        if (body === undefined) return json({ detail: 'Draft not found' }, 404);
        return json(body, routes.draftStatus ?? 200);
      }
      if (url.endsWith('/api/entries') || url.endsWith('/api/entries/upload')) {
        const create = routes.create ?? { status: 200, body: { entry_id: 'e-1' } };
        return json(create.body, create.status);
      }
      if (/\/api\/entries\/[^/]+$/.test(url) && !url.endsWith('/upload')) {
        if (routes.parentGate) await routes.parentGate;
        const parent = routes.parent ?? { status: 404, body: { detail: 'not found' } };
        return json(parent.body, parent.status);
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
  clone.querySelector('#ariel-compose-dest')?.remove();
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

/** Pick one file in the form's file input so submission takes the multipart path. */
function pickFile(name = 'picked.png') {
  const input = /** @type {any} */ (document.getElementById('entry-files'));
  input.files = [new File(['png-bytes'], name, { type: 'image/png' })];
}

/**
 * The names of the files a multipart create sent.
 * @param {Call} call
 * @returns {string[]}
 */
function sentFiles(call) {
  return /** @type {FormData} */ (call.init.body)
    .getAll('files')
    .map((f) => /** @type {File} */ (f).name);
}

/** @returns {string} The text of the error toast on screen, or '' */
function errorToast() {
  return (
    document.querySelector('.osprey-toast-host [role="alert"] .osprey-toast')?.textContent ?? ''
  );
}

/** @returns {{kind: string, text: string}|null} The status-region toast on screen */
function statusToast() {
  const el = /** @type {HTMLElement|null} */ (
    document.querySelector('.osprey-toast-host [role="status"] .osprey-toast')
  );
  return el ? { kind: el.dataset.kind ?? '', text: el.textContent ?? '' } : null;
}

/** @returns {number} The preview cards on screen */
function previewCards() {
  return /** @type {HTMLElement} */ (document.getElementById('file-preview')).children.length;
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

  test('the compose sheet lives outside the list column, with its controls and footer', () => {
    const html = readFileSync(join(import.meta.dirname, STATIC, 'index.html'), 'utf8');
    const template = document.createElement('template');
    template.innerHTML = html;
    const page = template.content;
    const sheet = /** @type {Element} */ (page.querySelector('#ariel-compose'));
    expect(sheet).not.toBeNull();
    // Decision 10: not inside .ariel-list-col (its containing block, and its
    // stacked display:none), but a direct child of the shell.
    expect(sheet.closest('.ariel-list-col')).toBeNull();
    expect(sheet.parentElement?.id).toBe('ariel-shell');
    expect(sheet.hasAttribute('hidden')).toBe(true);
    // One head with the title, maximise and close.
    expect(sheet.querySelectorAll('.osprey-pane-head').length).toBe(1);
    const head = /** @type {Element} */ (sheet.querySelector('.osprey-pane-head'));
    for (const id of ['ariel-compose-title', 'ariel-compose-max', 'ariel-compose-close']) {
      expect(head.querySelector(`#${id}`)).not.toBeNull();
    }
    // The footer sits inside the form, so Submit stays the form's submit button.
    const foot = /** @type {Element} */ (
      sheet.querySelector('#create-entry-form .ariel-compose-foot')
    );
    expect(foot).not.toBeNull();
    expect(foot.querySelector('button[type="submit"]')?.textContent?.trim()).toBe('Submit');
    expect(foot.querySelector('#create-cancel-btn')?.textContent?.trim()).toBe('Discard');
    expect(foot.querySelector('#ariel-compose-saved')).not.toBeNull();
    expect(foot.querySelector('#ariel-compose-dest')).not.toBeNull();
    expect(sheet.querySelectorAll('button[type="submit"]').length).toBe(1);
    // The slots 4.4 and 4.5 fill start hidden.
    expect(sheet.querySelector('#ariel-compose-banner')?.hasAttribute('hidden')).toBe(true);
    expect(sheet.querySelector('#ariel-compose-parent')?.hasAttribute('hidden')).toBe(true);
    expect(sheet.querySelector('#file-preview')?.getAttribute('style')).toBeNull();
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
    // The destination line is the one part that answers an unreadable
    // publish-info ("Destination unknown"); the sheet suite pins it.
    const withoutDestination = () => {
      const clone = /** @type {HTMLFormElement} */ (theForm().cloneNode(true));
      clone.querySelector('#ariel-compose-dest')?.remove();
      return clone.innerHTML;
    };
    const before = withoutDestination();
    const { entries, fields } = await loadModules();
    await entries.adaptCreateForm();
    await fields.entryFieldsRendered();
    expect(withoutDestination()).toBe(before);
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
    pickFile();
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
    pickFile();
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
    pickFile();
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
    if (upload) pickFile();
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

  test('an error naming no rendered input falls back to the generic error toast', async () => {
    await submitRejected({
      status: 422,
      body: { detail: 'Unknown field', code: 'invalid_entry_field', field: 'nope' },
    });

    expect(errorToast()).toBe('Failed to create entry: Unknown field');
    expect(window.alert).not.toHaveBeenCalled();
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

  test('opening empty shows the host, clears the form and never writes narrowView', async () => {
    const { store } = await boot();
    store.setState({ narrowView: 'reader' });
    field('subject').value = 'left over';

    store.setState({ compose: { draft: null } });

    expect(composeEl().hidden).toBe(false);
    // C-COMPOSE: the sheet overlays the tile, so the pane behind it stays put.
    expect(store.getState().narrowView).toBe('reader');
    // Reset to the markup's own default: the operator's leftover text is gone.
    expect(field('subject').value).toBe(field('subject').defaultValue);
    expect(provenanceBanner()).toBeNull();
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
    // No session_metadata recorded: no provenance banner.
    expect(provenanceBanner()).toBeNull();
    expect(document.getElementById('session-info-panel')).toBeNull();
  });

  test('a draft that records session_metadata shows the provenance banner', async () => {
    const { store } = await boot({
      draft: { subject: 'From the agent', metadata: { session_metadata: { operator: 'Ada' } } },
    });

    store.setState({ compose: { draft: 'd-7' } });
    await flush();
    await flush();

    const banner = provenanceBanner();
    expect(banner?.dataset.draftId).toBe('d-7');
    expect(banner?.classList.contains('ariel-compose-banner')).toBe(true);
    // The sheet's one banner slot is filled, never a second banner.
    expect(document.querySelectorAll('.ariel-compose-banner')).toHaveLength(1);
    expect(document.getElementById('session-info-panel')?.textContent).toContain('Ada');
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
    expect(mods.store.getState().narrowView).toBe('reader');
    expect(field('subject').value).toBe('Early draft');
  });

  test('closing hides the host; Cancel writes compose null', async () => {
    const { store } = await boot();
    store.setState({ compose: { draft: null } });
    // Discard is enabled only once the open has finished (C-SHEET-LIFECYCLE).
    await flush();

    /** @type {HTMLElement} */ (document.getElementById('create-cancel-btn')).click();

    expect(store.getState().compose).toBeNull();
    expect(composeEl().hidden).toBe(true);
  });

  test('the close control writes compose null and leaves narrowView alone', async () => {
    const { store } = await boot();
    store.setState({ narrowView: 'reader', compose: { draft: null } });

    /** @type {HTMLElement} */ (document.getElementById('ariel-compose-close')).click();

    expect(store.getState().compose).toBeNull();
    expect(store.getState().narrowView).toBe('reader');
    expect(composeEl().hidden).toBe(true);
  });

  test.each([
    [{ draft: null }, 'New entry'],
    [{ draft: null, followsUp: 'e-9' }, 'Follow up'],
    [{ draft: 'draft-0123456789ab' }, 'Draft'],
  ])('the sheet titles a compose of %o as %s', async (compose, title) => {
    const { store } = await boot();
    store.setState({ compose });

    expect(document.getElementById('ariel-compose-title')?.textContent).toBe(title);
    expect(composeEl().getAttribute('aria-label')).toBe(title);
  });

  test('maximise and restore toggle the sheet without touching the store', async () => {
    const { store } = await boot();
    store.setState({ sel: 'e-3', narrowView: 'reader', compose: { draft: null } });
    const before = store.getState();
    const max = /** @type {HTMLElement} */ (document.getElementById('ariel-compose-max'));

    max.click();
    expect(composeEl().classList.contains('max')).toBe(true);
    expect(max.getAttribute('aria-pressed')).toBe('true');
    expect(store.getState()).toEqual(before);

    max.click();
    expect(composeEl().classList.contains('max')).toBe(false);
    expect(max.getAttribute('aria-pressed')).toBe('false');
    expect(store.getState()).toEqual(before);
  });

  test('maximise is per open: a closed and reopened sheet starts restored', async () => {
    const { store } = await boot();
    store.setState({ compose: { draft: null } });
    /** @type {HTMLElement} */ (document.getElementById('ariel-compose-max')).click();

    store.setState({ compose: null });
    store.setState({ compose: { draft: null } });

    expect(composeEl().classList.contains('max')).toBe(false);
    expect(document.getElementById('ariel-compose-max')?.getAttribute('aria-pressed')).toBe(
      'false'
    );
  });

  test.each([
    [{ supports_write: true, requires_auth: false, source_system: 'TestLog' }, '→ TestLog logbook'],
    [
      { supports_write: true, requires_auth: true, source_system: 'TestLog' },
      '→ TestLog logbook (credentials required)',
    ],
    [{ supports_write: true, requires_auth: false, source_system: null }, '→ facility logbook'],
    [
      { supports_write: false, requires_auth: false, source_system: 'TestLog' },
      '→ This Logbook only (facility logbook read-only)',
    ],
  ])('the footer names the destination for %o', async (info, text) => {
    await boot({ publishInfo: { ...info, entry_fields: [] } });
    expect(document.getElementById('ariel-compose-dest')?.textContent).toBe(text);
  });

  test('an unreadable publish-info names the destination unknown', async () => {
    await boot({ publishInfo: { detail: 'down' }, publishInfoStatus: 503 });
    expect(document.getElementById('ariel-compose-dest')?.textContent).toBe('Destination unknown');
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
    const { store, form, calls } = await boot({
      draft: { subject: 'Plot', attachment_paths: ['/tmp/plot.png'] },
    });
    store.setState({ compose: { draft: 'd-1' } });
    await flush();
    await flush();
    expect(previewCards()).toBe(1);

    await submit(form);
    expect(sentFiles(createCall(calls))).toEqual(['plot.png']);
    expect(store.getState().sel).toBe('e-1');
    expect(previewCards()).toBe(0);
    expect(form.loadedDraftId()).toBeNull();

    calls.length = 0;
    store.setState({ compose: { draft: null } });
    await flush();
    await submit(form);
    expect(createCall(calls).url.endsWith('/api/entries')).toBe(true);
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

    expect(errorToast()).toBe(
      'Logbook credentials required to publish. Enter your username and password.'
    );
    expect(window.alert).not.toHaveBeenCalled();
    expect(store.getState().compose).toEqual({ draft: null });
    expect(store.getState().sel).toBeNull();
    expect(field('subject').value).toBe('Beam dump');
    expect(document.activeElement?.id).toBe('entry-auth-user');
  });

  test('a generic failure toasts and keeps compose open', async () => {
    const { store, form } = await boot({ create: { status: 500, body: { detail: 'boom' } } });
    store.setState({ compose: { draft: null } });

    await submit(form);

    expect(errorToast()).toBe('Failed to create entry: boom');
    expect(window.alert).not.toHaveBeenCalled();
    expect(store.getState().compose).toEqual({ draft: null });
  });

  /** @returns {HTMLButtonElement[]} Submit and Discard */
  function gatedButtons() {
    return [
      /** @type {HTMLButtonElement} */ (theForm().querySelector('button[type="submit"]')),
      /** @type {HTMLButtonElement} */ (document.getElementById('create-cancel-btn')),
    ];
  }

  test('Submit and Discard stay disabled until the open has finished', async () => {
    let release = () => {};
    const draftGate = new Promise((resolve) => (release = () => resolve(undefined)));
    stubFetch({ draft: { subject: 'Slow draft' } });
    const realFetch = globalThis.fetch;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {any} */ input, /** @type {any} */ init) => {
        if (String(input).includes('/api/drafts/')) await draftGate;
        return realFetch(input, init);
      })
    );
    const mods = await loadModules();
    host = mods.entries.initEntries();
    await mods.fields.entryFieldsRendered();

    mods.store.setState({ compose: { draft: 'd-slow' } });
    await flush();
    for (const btn of gatedButtons()) expect(btn.disabled).toBe(true);
    expect(theForm().hasAttribute('inert')).toBe(true);

    release();
    await flush();
    await flush();
    expect(field('subject').value).toBe('Slow draft');
    for (const btn of gatedButtons()) expect(btn.disabled).toBe(false);
    expect(theForm().hasAttribute('inert')).toBe(false);
  });

  describe('a draft (Decision 8)', () => {
    const PLOT = {
      subject: 'Plot',
      tags: ['rf'],
      metadata: { session_metadata: { operator: 'Ada' } },
      attachment_paths: ['/tmp/plot.png'],
    };

    /** @returns {Promise<void>} Two macrotask turns: an open runs to its end */
    const settle = async () => {
      await flush();
      await flush();
    };

    test('staged attachments are in memory and previewed at open, and submit with it', async () => {
      const { store, form, calls } = await boot({ draft: PLOT });
      store.setState({ compose: { draft: 'd-1' } });
      await settle();

      expect(calls.some((c) => c.url.includes('/api/drafts/d-1/attachments/plot.png'))).toBe(true);
      expect(previewCards()).toBe(1);
      expect(form.loadedDraftId()).toBe('d-1');

      calls.length = 0;
      await submit(form);
      // Submit reads memory, never the draft file again.
      expect(calls.some((c) => c.url.includes('/attachments/'))).toBe(false);
      expect(sentFiles(createCall(calls))).toEqual(['plot.png']);
    });

    test('picking files previews them with the draft files and sends both', async () => {
      const { store, form, calls } = await boot({ draft: PLOT });
      store.setState({ compose: { draft: 'd-1' } });
      await settle();

      pickFile('scope.png');
      document.getElementById('entry-files')?.dispatchEvent(new Event('change'));
      expect(previewCards()).toBe(2);

      await submit(form);
      expect(sentFiles(createCall(calls))).toEqual(['scope.png', 'plot.png']);
    });

    test('an attachment that cannot be fetched toasts its name and is left out', async () => {
      const { store, form, calls } = await boot({
        draft: { ...PLOT, attachment_paths: ['/tmp/plot.png', '/tmp/log.txt'] },
        missingAttachments: ['log.txt'],
      });
      const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
      store.setState({ compose: { draft: 'd-1' } });
      await settle();

      expect(errorToast()).toContain('log.txt');
      expect(previewCards()).toBe(1);
      await submit(form);
      expect(sentFiles(createCall(calls))).toEqual(['plot.png']);
      errors.mockRestore();
    });

    test('draft A answering after draft B opened fills nothing', async () => {
      let release = () => {};
      const gate = new Promise((resolve) => (release = () => resolve(undefined)));
      const { store, form, calls } = await boot({
        drafts: {
          'd-a': { ...PLOT, subject: 'Draft A', tags: ['a'], attachment_paths: ['/tmp/a.png'] },
          'd-b': { ...PLOT, subject: 'Draft B', tags: ['b'], attachment_paths: ['/tmp/b.png'] },
        },
        draftGates: { 'd-a': gate },
      });

      store.setState({ compose: { draft: 'd-a' } });
      await flush();
      store.setState({ compose: { draft: 'd-b' } });
      await settle();
      release();
      await settle();

      expect(field('subject').value).toBe('Draft B');
      expect(
        Array.from(document.querySelectorAll('#entry-tags .tag'), (t) =>
          t.getAttribute('data-value')
        )
      ).toEqual(['b']);
      expect(provenanceBanner()?.dataset.draftId).toBe('d-b');
      expect(previewCards()).toBe(1);
      expect(form.loadedDraftId()).toBe('d-b');
      expect(calls.some((c) => c.url.includes('/attachments/a.png'))).toBe(false);

      await submit(form);
      expect(sentFiles(createCall(calls))).toEqual(['b.png']);
    });

    test('a draft answering after the sheet closed fills nothing', async () => {
      let release = () => {};
      const gate = new Promise((resolve) => (release = () => resolve(undefined)));
      const { store, form } = await boot({ draft: PLOT, draftGates: { 'd-1': gate } });

      store.setState({ compose: { draft: 'd-1' } });
      await flush();
      store.setState({ compose: null });
      release();
      await settle();

      expect(field('subject').value).toBe(field('subject').defaultValue);
      expect(document.querySelectorAll('#entry-tags .tag')).toHaveLength(0);
      expect(provenanceBanner()).toBeNull();
      expect(previewCards()).toBe(0);
      expect(form.loadedDraftId()).toBeNull();
      expect(errorToast()).toBe('');
    });

    test('a gone draft with no snapshot toasts and becomes a new entry', async () => {
      const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
      const { store, form, calls } = await boot();
      store.setState({ compose: { draft: 'd-gone' } });
      await settle();

      expect(errorToast()).toContain('gone');
      expect(store.getState().compose).toEqual({ draft: null });
      expect(composeEl().hidden).toBe(false);
      expect(form.loadedDraftId()).toBeNull();
      for (const btn of gatedButtons()) expect(btn.disabled).toBe(false);

      await submit(form);
      expect(createCall(calls).url.endsWith('/api/entries')).toBe(true);
      expect(calls.some((c) => c.init?.method === 'DELETE')).toBe(false);
      errors.mockRestore();
    });

    test('a 422 draft with a snapshot toasts, stays in its context and restores it', async () => {
      const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
      const { store, form } = await boot({ draft: { detail: 'bad id' }, draftStatus: 422 });
      const autosave = await import(`${STATIC}/js/compose-autosave.js`);
      autosave.saveCompose('draft:d-9', {
        subject: 'Typed before the draft expired',
        details: '',
        author: '',
        fields: {},
        tags: [],
        logbook: '',
        shift: '',
        savedAt: Date.now(),
      });

      store.setState({ compose: { draft: 'd-9' } });
      await settle();

      expect(errorToast()).toContain('gone');
      expect(store.getState().compose).toEqual({ draft: 'd-9' });
      expect(field('subject').value).toBe('Typed before the draft expired');
      expect(form.loadedDraftId()).toBeNull();
      errors.mockRestore();
    });

    test('any other load failure toasts and keeps the draft context, not loaded', async () => {
      const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
      const { store, form } = await boot({ draft: { detail: 'boom' }, draftStatus: 500 });
      store.setState({ compose: { draft: 'd-9' } });
      await settle();

      expect(errorToast()).toContain('Could not load the draft');
      expect(store.getState().compose).toEqual({ draft: 'd-9' });
      expect(form.loadedDraftId()).toBeNull();
      expect(errors).toHaveBeenCalled();
      errors.mockRestore();
    });
  });

  describe('submit outcome and Discard (Decisions 9, 10)', () => {
    const DRAFT = { subject: 'From the agent', details: 'Drafted' };

    /** @returns {Promise<void>} Two macrotask turns: an open runs to its end */
    const settle = async () => {
      await flush();
      await flush();
    };

    /** @param {Call[]} calls @returns {Call[]} The DELETE requests made */
    const deletes = (calls) => calls.filter((c) => c.init?.method === 'DELETE');

    /** Store a snapshot under `key`, as a typed-and-saved sheet would. */
    async function snapshot(/** @type {string} */ key) {
      const autosave = await import(`${STATIC}/js/compose-autosave.js`);
      autosave.saveCompose(key, {
        subject: 'Typed',
        details: '',
        author: '',
        fields: {},
        tags: [],
        logbook: '',
        shift: '',
        savedAt: Date.now(),
      });
      return autosave;
    }

    test('success finishes the sheet: its snapshot is cleared, not kept', async () => {
      const { store, form } = await boot();
      store.setState({ compose: { draft: null } });
      await settle();
      const autosave = await snapshot('new');

      await submit(form);

      expect(store.getState().compose).toBeNull();
      expect(autosave.loadCompose('new')).toBeNull();
    });

    test('success with attachments_published false warns that they stayed local', async () => {
      const { store, form } = await boot({
        create: { status: 200, body: { entry_id: 'e-5', attachments_published: false } },
      });
      store.setState({ compose: { draft: null } });
      await settle();
      pickFile();

      await submit(form);

      expect(store.getState().sel).toBe('e-5');
      expect(store.getState().compose).toBeNull();
      expect(statusToast()).toEqual({
        kind: 'warning',
        text: 'Entry saved, but its attachments stayed local: they were not published.',
      });
    });

    test.each([true, null])('success with attachments_published %s shows no toast', async (v) => {
      const { store, form } = await boot({
        create: { status: 200, body: { entry_id: 'e-5', attachments_published: v } },
      });
      store.setState({ compose: { draft: null } });
      await settle();

      await submit(form);

      expect(statusToast()).toBeNull();
      expect(errorToast()).toBe('');
    });

    test('success on a loaded draft deletes it once and bumps draftsRev', async () => {
      const { store, form, calls } = await boot({ draft: DRAFT });
      store.setState({ compose: { draft: 'd-1' } });
      await settle();

      await submit(form);

      const sent = deletes(calls);
      expect(sent).toHaveLength(1);
      expect(sent[0].url).toContain('/api/drafts/d-1');
      expect(store.getState().draftsRev).toBe(1);
      expect(statusToast()).toBeNull();
      expect(store.getState().compose).toBeNull();
    });

    test('a 404 on the draft DELETE after submit is ignored, draftsRev still bumped', async () => {
      const { store, form, calls } = await boot({ draft: DRAFT, deleteStatus: 404 });
      store.setState({ compose: { draft: 'd-1' } });
      await settle();

      await submit(form);

      expect(deletes(calls)).toHaveLength(1);
      expect(statusToast()).toBeNull();
      expect(errorToast()).toBe('');
      expect(store.getState().draftsRev).toBe(1);
    });

    test('any other DELETE failure after submit warns the draft is still listed', async () => {
      const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
      const { store, form } = await boot({ draft: DRAFT, deleteStatus: 500 });
      store.setState({ compose: { draft: 'd-1' } });
      await settle();

      await submit(form);

      expect(statusToast()?.kind).toBe('warning');
      expect(statusToast()?.text).toContain('still listed');
      expect(store.getState().draftsRev).toBe(1);
      expect(store.getState().sel).toBe('e-1');
      errors.mockRestore();
    });

    test('a non-2xx submit keeps the sheet, its content and the draft', async () => {
      const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
      const { store, form, calls } = await boot({
        draft: DRAFT,
        create: { status: 401, body: { detail: 'auth', code: 'auth_required' } },
      });
      store.setState({ compose: { draft: 'd-1' } });
      await settle();
      const autosave = await snapshot('draft:d-1');

      await submit(form);

      expect(store.getState().compose).toEqual({ draft: 'd-1' });
      expect(field('subject').value).toBe('From the agent');
      expect(form.loadedDraftId()).toBe('d-1');
      expect(deletes(calls)).toHaveLength(0);
      expect(store.getState().draftsRev).toBe(0);
      expect(autosave.loadCompose('draft:d-1')).not.toBeNull();
      expect(errorToast()).toContain('credentials required');
      expect(document.activeElement?.id).toBe('entry-auth-user');
      errors.mockRestore();
    });

    test('Discard on a new entry finishes and closes, deleting nothing', async () => {
      const { store, calls } = await boot();
      store.setState({ compose: { draft: null } });
      await settle();
      const autosave = await snapshot('new');

      /** @type {HTMLElement} */ (document.getElementById('create-cancel-btn')).click();
      await flush();

      expect(store.getState().compose).toBeNull();
      expect(composeEl().hidden).toBe(true);
      expect(autosave.loadCompose('new')).toBeNull();
      expect(deletes(calls)).toHaveLength(0);
      expect(store.getState().draftsRev).toBe(0);
      expect(statusToast()).toBeNull();
    });

    test('Discard on a draft finishes, closes, deletes it and toasts success', async () => {
      const { store, calls } = await boot({ draft: DRAFT });
      store.setState({ compose: { draft: 'd-1' } });
      await settle();
      const autosave = await snapshot('draft:d-1');

      /** @type {HTMLElement} */ (document.getElementById('create-cancel-btn')).click();
      expect(store.getState().compose).toBeNull();
      expect(store.getState().draftsRev).toBe(0); // bumped only once the DELETE settles
      await flush();

      const sent = deletes(calls);
      expect(sent).toHaveLength(1);
      expect(sent[0].url).toContain('/api/drafts/d-1');
      expect(autosave.loadCompose('draft:d-1')).toBeNull();
      expect(statusToast()).toEqual({ kind: 'success', text: 'Draft discarded' });
      expect(store.getState().draftsRev).toBe(1);
    });

    test('Discard on a draft whose DELETE fails warns it is still listed', async () => {
      const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
      const { store } = await boot({ draft: DRAFT, deleteStatus: 503 });
      store.setState({ compose: { draft: 'd-1' } });
      await settle();

      /** @type {HTMLElement} */ (document.getElementById('create-cancel-btn')).click();
      await flush();

      expect(store.getState().compose).toBeNull();
      expect(statusToast()?.kind).toBe('warning');
      expect(statusToast()?.text).toContain('still listed');
      expect(store.getState().draftsRev).toBe(1);
      errors.mockRestore();
    });

    test('Close on a draft keeps its snapshot and the draft', async () => {
      const { store, calls } = await boot({ draft: DRAFT });
      store.setState({ compose: { draft: 'd-1' } });
      await settle();
      const autosave = await snapshot('draft:d-1');

      /** @type {HTMLElement} */ (document.getElementById('ariel-compose-close')).click();
      await flush();

      expect(store.getState().compose).toBeNull();
      expect(autosave.loadCompose('draft:d-1')).not.toBeNull();
      expect(deletes(calls)).toHaveLength(0);
      expect(store.getState().draftsRev).toBe(0);
    });
  });

  describe('a follow-up', () => {
    const PARENT = {
      status: 200,
      body: {
        entry_id: 'p-1',
        subject: 'RF trip',
        metadata: { logbook: 'Operations', tags: ['rf', 'trip'] },
      },
    };

    /** @returns {HTMLElement} */
    const parentRow = () =>
      /** @type {HTMLElement} */ (document.getElementById('ariel-compose-parent'));
    const parentRef = () => document.getElementById('ariel-compose-parent-ref')?.textContent;
    const tagValues = () =>
      Array.from(
        /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll('#entry-tags .tag'))
      ).map((t) => t.dataset.value);
    const isCreate = (/** @type {Call} */ c) =>
      c.url.endsWith('/api/entries') || c.url.endsWith('/api/entries/upload');

    test('prefills subject, logbook, tags and the parent row, and nothing auto-submits', async () => {
      const { store, calls } = await boot({ parent: PARENT });

      store.setState({ compose: { draft: null, followsUp: 'p-1' } });
      await flush();
      await flush();

      expect(calls.some((c) => /\/api\/entries\/p-1$/.test(c.url))).toBe(true);
      expect(field('subject').value).toBe('Follow-up: RF trip');
      expect(field('logbook').value).toBe('Operations');
      expect(tagValues()).toEqual(['rf', 'trip']);
      expect(parentRow().hidden).toBe(false);
      expect(parentRef()).toBe('RF trip');
      expect(calls.some(isCreate)).toBe(false);
      for (const btn of gatedButtons()) expect(btn.disabled).toBe(false);
    });

    test('does not double an existing Follow-up prefix', async () => {
      const { store } = await boot({
        parent: {
          status: 200,
          body: { entry_id: 'p-2', subject: 'Follow-up: RF trip', metadata: {} },
        },
      });

      store.setState({ compose: { draft: null, followsUp: 'p-2' } });
      await flush();
      await flush();

      expect(field('subject').value).toBe('Follow-up: RF trip');
      // No logbook or tags on the parent: the defaults stay.
      expect(field('logbook').value).toBe(field('logbook').defaultValue);
      expect(tagValues()).toEqual([]);
    });

    test('submit carries metadata.follows_up on the JSON route', async () => {
      const { store, form, calls } = await boot({ parent: PARENT });
      store.setState({ compose: { draft: null, followsUp: 'p-1' } });
      await flush();
      await flush();

      await submit(form);

      const call = createCall(calls);
      expect(call.url.endsWith('/api/entries')).toBe(true);
      const body = JSON.parse(call.init.body);
      expect(body.metadata).toEqual({ follows_up: 'p-1' });
      expect(body.subject).toBe('Follow-up: RF trip');
      expect(body.tags).toEqual(['rf', 'trip']);
    });

    test('submit carries metadata.follows_up on the upload route', async () => {
      const { store, form, calls } = await boot({ parent: PARENT });
      store.setState({ compose: { draft: null, followsUp: 'p-1' } });
      await flush();
      await flush();
      pickFile();

      await submit(form);

      const call = createCall(calls);
      expect(call.url.endsWith('/api/entries/upload')).toBe(true);
      const metadata = JSON.parse(String(call.init.body.get('metadata')));
      expect(metadata).toEqual({ follows_up: 'p-1' });
    });

    test('an unloadable parent still opens the sheet with the id shown', async () => {
      const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
      const { store, form, calls } = await boot({
        parent: { status: 500, body: { detail: 'boom' } },
      });

      store.setState({ compose: { draft: null, followsUp: 'p-gone' } });
      await flush();
      await flush();

      expect(composeEl().hidden).toBe(false);
      expect(parentRow().hidden).toBe(false);
      expect(parentRef()).toBe('p-gone');
      expect(field('subject').value).toBe(field('subject').defaultValue);
      for (const btn of gatedButtons()) expect(btn.disabled).toBe(false);
      expect(calls.some(isCreate)).toBe(false);

      await submit(form);
      expect(JSON.parse(createCall(calls).init.body).metadata).toEqual({ follows_up: 'p-gone' });
      errors.mockRestore();
    });

    test('a parent answering after a reopen as a new entry fills nothing', async () => {
      let release = () => {};
      const parentGate = new Promise((resolve) => (release = () => resolve(undefined)));
      const { store, form, calls } = await boot({ parent: PARENT, parentGate });

      store.setState({ compose: { draft: null, followsUp: 'p-1' } });
      await flush();
      store.setState({ compose: null });
      store.setState({ compose: { draft: null } });
      await flush();

      release();
      await flush();
      await flush();

      expect(field('subject').value).toBe(field('subject').defaultValue);
      expect(field('logbook').value).toBe(field('logbook').defaultValue);
      expect(tagValues()).toEqual([]);
      expect(parentRow().hidden).toBe(true);
      expect(parentRef()).toBe('');

      await submit(form);
      expect(JSON.parse(createCall(calls).init.body).metadata).toBeNull();
    });

    test('a new entry opened after a follow-up shows no parent row and submits no follows_up', async () => {
      const { store, form, calls } = await boot({ parent: PARENT });
      store.setState({ compose: { draft: null, followsUp: 'p-1' } });
      await flush();
      await flush();
      expect(parentRow().hidden).toBe(false);

      store.setState({ compose: null });
      store.setState({ compose: { draft: null } });
      await flush();

      expect(parentRow().hidden).toBe(true);
      expect(parentRef()).toBe('');
      expect(tagValues()).toEqual([]);

      await submit(form);
      const body = JSON.parse(createCall(calls).init.body);
      expect(body.metadata).toBeNull();
    });
  });

  test('typing autosaves the sheet, stamps the footer, and a reopen restores it', async () => {
    const { store } = await boot();
    store.setState({ compose: { draft: null } });
    await flush();
    vi.useFakeTimers();
    try {
      field('subject').value = 'Kept across a close';
      field('subject').dispatchEvent(new Event('input', { bubbles: true }));
      vi.advanceTimersByTime(5000);
    } finally {
      vi.useRealTimers();
    }
    expect(document.getElementById('ariel-compose-saved')?.textContent).toMatch(
      /^saved locally \d{2}:\d{2}/
    );

    store.setState({ compose: null });
    store.setState({ compose: { draft: null } });
    expect(field('subject').value).toBe(field('subject').defaultValue);
    expect(document.getElementById('ariel-compose-saved')?.textContent).toBe('');
    await flush();

    expect(field('subject').value).toBe('Kept across a close');
  });

  test('a user tag add and remove each save; credentials never do', async () => {
    const { store, form } = await boot();
    store.setState({ compose: { draft: null } });
    await flush();
    const read = () => JSON.parse(localStorage.getItem('ariel-compose-autosave') || '{}').new;

    const user = /** @type {HTMLInputElement} */ (document.getElementById('entry-auth-user'));
    user.value = 'kirk';
    user.dispatchEvent(new Event('input', { bubbles: true }));
    store.setState({ compose: null });
    expect(read()).toBeUndefined();

    store.setState({ compose: { draft: null } });
    await flush();
    const input = document.createElement('input');
    input.id = 'entry-tags-input';
    theForm().append(input);
    input.value = 'rf';
    form.handleTagInput(/** @type {any} */ ({ key: 'Enter', preventDefault() {}, target: input }));
    store.setState({ compose: null });
    expect(read().tags).toEqual(['rf']);
    expect(JSON.stringify(read())).not.toContain('kirk');

    store.setState({ compose: { draft: null } });
    await flush();
    expect(document.querySelectorAll('#entry-tags .tag')).toHaveLength(1);
    /** @type {HTMLElement} */ (document.querySelector('#entry-tags [data-tag-remove]')).click();
    store.setState({ compose: null });
    expect(read().tags).toEqual([]);
  });

  test.each([['a"b'], ['a\\b']])('a tag holding %s adds once and removes', async (value) => {
    const { store, form } = await boot();
    store.setState({ compose: { draft: null } });
    await flush();
    const input = document.createElement('input');
    input.id = 'entry-tags-input';
    theForm().append(input);
    const enter = () => {
      input.value = value;
      form.handleTagInput(
        /** @type {any} */ ({ key: 'Enter', preventDefault() {}, target: input })
      );
    };
    expect(enter).not.toThrow();
    expect(enter).not.toThrow();
    const chips = document.querySelectorAll('#entry-tags .tag');
    expect(chips).toHaveLength(1);
    expect(/** @type {HTMLElement} */ (chips[0]).dataset.value).toBe(value);
    /** @type {HTMLElement} */ (document.querySelector('#entry-tags [data-tag-remove]')).click();
    expect(document.querySelectorAll('#entry-tags .tag')).toHaveLength(0);
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

describe('leaving the create form leaves nothing behind', () => {
  const DRAFT = {
    subject: 'Beam dump',
    details: 'Lost beam at 10:02',
    tags: ['rf'],
    metadata: { session_metadata: { id: 's-1' } },
    attachment_paths: ['/tmp/plot.png'],
  };

  test("a submitted draft's attachments are not sent again with the next entry", async () => {
    const calls = stubFetch({ draft: DRAFT });
    const { entries, form } = await loadModules();
    await entries.adaptCreateForm();
    await form.loadDraft('d-1');
    await submit(form);
    expect(createCall(calls).url.endsWith('/api/entries/upload')).toBe(true);
    expect(sentFiles(createCall(calls))).toEqual(['plot.png']);

    calls.length = 0;
    await submit(form);

    expect(createCall(calls).url.endsWith('/api/entries')).toBe(true);
  });

  test('resetCreateForm drops a loaded draft: tags, metadata, attachments, banner', async () => {
    const calls = stubFetch({ draft: DRAFT });
    const { entries, form } = await loadModules();
    await entries.adaptCreateForm();
    await form.loadDraft('d-1');
    expect(provenanceBanner()).not.toBeNull();
    expect(document.querySelectorAll('#entry-tags .tag')).toHaveLength(1);
    expect(previewCards()).toBe(1);
    expect(form.loadedDraftId()).toBe('d-1');

    form.resetCreateForm();

    expect(provenanceBanner()).toBeNull();
    expect(document.querySelectorAll('#entry-tags .tag')).toHaveLength(0);
    expect(previewCards()).toBe(0);
    expect(form.loadedDraftId()).toBeNull();

    calls.length = 0;
    await submit(form);
    const call = createCall(calls);
    expect(call.url.endsWith('/api/entries')).toBe(true);
    const body = JSON.parse(call.init.body);
    expect(body.metadata).toBeNull();
    expect(body.tags).toEqual([]);
  });

  test('resetCreateForm clears an entry-field error mark', async () => {
    stubFetch({
      publishInfo: { supports_write: true, requires_auth: false, entry_fields: FIELDS },
    });
    const { entries, form, fields } = await loadModules();
    await entries.adaptCreateForm();
    await fields.entryFieldsRendered();
    expect(fields.markEntryFieldError('book', 'Unknown book')).toBe(true);
    expect(document.querySelector('[aria-invalid="true"]')).not.toBeNull();

    form.resetCreateForm();

    expect(document.querySelector('[aria-invalid="true"]')).toBeNull();
    expect(document.querySelector('[data-entry-field-error]')).toBeNull();
  });

  test('a created entry opens in the reader', async () => {
    stubFetch({});
    const { form, store } = await loadModules();
    await submit(form);

    expect(store.getState().sel).toBe('e-1');
    expect(store.getState().narrowView).toBe('reader');
  });
});
