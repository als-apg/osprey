/**
 * The session tree's context menus (sidebar-artifact-menus.js): each row
 * kind's exact items, New session through switchTo, rename in place, the
 * transcript export, a disabled approval preview, and the delete guard on an
 * artifact, on a run head and on a past session.
 *
 *   npx vitest run tests/interfaces/web_terminal/sidebar-artifact-menus.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const sessionModel = vi.hoisted(() => ({
  switchTo: vi.fn(async () => true),
  rows: vi.fn(() => [{ transcript_id: 't1', session_key: 'k1', title: 'Beam study' }]),
  refresh: vi.fn(),
  forgetSession: vi.fn(),
}));
const listing = vi.hoisted(() => ({
  ARTIFACTS_BASE: '/panel/artifacts',
  fetchArtifact: /** @type {import('vitest').Mock<(id: string) => Promise<any>>} */ (
    vi.fn(async () => null)
  ),
  readTreeView: vi.fn(() => 'kind'),
}));
const confirm = vi.hoisted(() => ({ showConfirm: vi.fn() }));
const controller = vi.hoisted(() => ({ setTreeView: vi.fn(), openInAgent: vi.fn(async () => {}) }));
const sidebar = vi.hoisted(() => ({ toggleEverySession: vi.fn() }));

vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/session-model.js',
  () => sessionModel
);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifact-listing.js',
  () => listing
);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js', () => confirm);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js',
  () => controller
);

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sidebar.js', () => sidebar);

const menus = await import(`${JS}/sidebar-artifact-menus.js`);
const pointer = await import(`${JS}/session-pointer.js`);

const RUN = {
  id: 'run1',
  title: 'orbit scan',
  run_parts: [
    { id: 'run1', role: 'head' },
    { id: 'out1', role: 'output', form: 'plots' },
    { id: 'code1', role: 'code' },
  ],
};
const PLOT = { id: 'a1', title: 'BPM plot', pinned: false };

/** Build a tree: a session (open), a group head, a run head with parts, an artifact. */
function buildTree() {
  document.body.innerHTML = `
    <div data-session-tree>
      <div class="osprey-tree-session osprey-tree-d0" data-session-id="t1" data-count="3" aria-expanded="false">
        <span class="osprey-tree-disc" data-tree-disc></span>
        <span class="osprey-tree-title">Beam study</span>
      </div>
      <div class="osprey-tree-head" data-group-head="t1:plots" aria-expanded="true"></div>
      <div class="box">
        <button class="osprey-list-row osprey-tree-run" data-id="run1" data-run="run1" aria-expanded="false">
          <span data-tree-disc></span>
        </button>
        <button class="osprey-list-row" data-id="out1"></button>
        <button class="osprey-list-row" data-id="code1"></button>
        <button class="osprey-list-row" data-id="a1"></button>
      </div>
    </div>`;
}

/** @param {string} sel */
const $ = (sel) => /** @type {HTMLElement} */ (document.querySelector(sel));

function menuLabels() {
  return [...document.querySelectorAll('.rail-context-item .rail-context-label')].map(
    (n) => n.textContent ?? ''
  );
}

/** @param {string} label */
function menuRow(label) {
  return /** @type {HTMLElement} */ (
    [...document.querySelectorAll('.rail-context-item')].find(
      (r) => r.querySelector('.rail-context-label')?.textContent === label
    )
  );
}

/** Right-click a row and wait for its menu. @param {Element} el */
async function rightClick(el) {
  const ev = new MouseEvent('contextmenu', {
    bubbles: true,
    cancelable: true,
    clientX: 5,
    clientY: 5,
  });
  el.dispatchEvent(ev);
  await vi.waitFor(() => expect(document.querySelector('.rail-context-menu')).not.toBeNull());
  return ev;
}

beforeEach(() => {
  vi.clearAllMocks();
  listing.fetchArtifact.mockImplementation(async (id) =>
    id === 'run1' ? RUN : id === 'a1' ? PLOT : { id, title: id }
  );
  buildTree();
  menus.initTreeMenus();
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, status: 200, json: async () => ({}) }))
  );
});

afterEach(() => {
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  menus.teardownTreeMenus();
  document.body.innerHTML = '';
});

describe('the Agent row', () => {
  test('New session, Expand all — and New session goes through switchTo({new: true})', async () => {
    const items = menus.buildAgentRowMenuItems();
    /** @param {{label?: string}} i */
    const label = (i) => i.label;
    expect(items.filter(label).map(label)).toEqual(['New session', 'Expand all']);
    items[0].run();
    await vi.waitFor(() => expect(sessionModel.switchTo).toHaveBeenCalledWith({ new: true }));
  });
});

describe('the Agent row: Expand all / Collapse all', () => {
  /** @param {string} name */
  const item = (name) =>
    menus.buildAgentRowMenuItems().find((/** @type {any} */ i) => i.label === name);

  test('Expand all is the ⌥-click gesture: it expands every session, never just the current one', async () => {
    item('Expand all').run();
    await vi.waitFor(() => expect(sidebar.toggleEverySession).toHaveBeenCalledWith(true));
  });

  test('the label reads Collapse all only when every session is open, and collapses them all', async () => {
    $('[data-session-id="t1"]').setAttribute('aria-expanded', 'true');
    const labels = menus.buildAgentRowMenuItems().map((/** @type {any} */ i) => i.label);
    expect(labels.filter(Boolean)).toEqual(['New session', 'Collapse all']);
    item('Collapse all').run();
    await vi.waitFor(() => expect(sidebar.toggleEverySession).toHaveBeenCalledWith(false));
  });

  test('one session open among several still offers Expand all', () => {
    $('[data-session-id="t1"]').setAttribute('aria-expanded', 'true');
    const other = document.createElement('div');
    other.className = 'osprey-tree-session osprey-tree-d0';
    other.dataset.sessionId = 't2';
    other.setAttribute('aria-expanded', 'false');
    $('[data-session-tree]').append(other);
    expect(item('Expand all')).toBeDefined();
  });
});

describe('a session row', () => {
  test('offers exactly PROPOSAL’s items', async () => {
    const ev = await rightClick($('.osprey-tree-title'));
    expect(ev.defaultPrevented).toBe(true);
    expect(menuLabels()).toEqual([
      'Group by Kind',
      'Group by Activity',
      'Rename…',
      'Export transcript',
      'Copy ID',
      'Delete…',
    ]);
    expect(menuRow('Group by Kind').getAttribute('aria-checked')).toBe('true');
    expect(menuRow('Open in Agent')).toBeUndefined();
  });

  test('Use automatic name appears only on a hand-named session', async () => {
    $('[data-session-id="t1"]').dataset.titleSource = 'auto';
    await rightClick($('.osprey-tree-title'));
    expect(menuRow('Use automatic name')).toBeUndefined();
    document.querySelector('.rail-context-menu')?.remove();
    $('[data-session-id="t1"]').dataset.titleSource = 'user';
    await rightClick($('.osprey-tree-title'));
    expect(menuLabels()).toContain('Use automatic name');
    menuRow('Use automatic name').click();
    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    const [url, init] = /** @type {any} */ (globalThis.fetch).mock.calls[0];
    expect(url).toBe('/api/sessions/t1/title');
    expect(JSON.parse(init.body)).toEqual({ title: '' });
  });

  test('Rename… edits in place; an empty name puts the automatic one back', async () => {
    await rightClick($('.osprey-tree-title'));
    menuRow('Rename…').click();
    const input = /** @type {HTMLInputElement} */ ($('input.osprey-tree-rename'));
    expect(input.value).toBe('Beam study');
    input.value = '  ';
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    const [url, init] = /** @type {any} */ (globalThis.fetch).mock.calls[0];
    expect(url).toBe('/api/sessions/t1/title');
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body)).toEqual({ title: '' });
    expect($('.osprey-tree-title')).not.toBeNull();
    await vi.waitFor(() => expect(sessionModel.refresh).toHaveBeenCalled());
  });

  test('Escape cancels a rename without saving', async () => {
    await rightClick($('.osprey-tree-title'));
    menuRow('Rename…').click();
    const input = /** @type {HTMLInputElement} */ ($('input.osprey-tree-rename'));
    input.value = 'Other';
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(globalThis.fetch).not.toHaveBeenCalled();
    expect($('.osprey-tree-title').textContent).toBe('Beam study');
  });

  test('Export transcript downloads the session’s transcript.md', async () => {
    /** @type {(string | null)[]} */
    const clicked = [];
    const spy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(
      /** @this {HTMLAnchorElement} */ function () {
        clicked.push(this.getAttribute('href'));
      }
    );
    await rightClick($('.osprey-tree-title'));
    menuRow('Export transcript').click();
    expect(clicked).toEqual(['/api/sessions/t1/transcript.md']);
    spy.mockRestore();
  });

  test('Group by Activity applies the mode through the controller', async () => {
    await rightClick($('.osprey-tree-title'));
    menuRow('Group by Activity').click();
    await vi.waitFor(() => expect(controller.setTreeView).toHaveBeenCalledWith('activity'));
  });

  test('the menu key opens the same menu', async () => {
    const ev = new KeyboardEvent('keydown', {
      key: 'ContextMenu',
      bubbles: true,
      cancelable: true,
    });
    $('.osprey-tree-title').dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
    await vi.waitFor(() => expect(menuLabels()[0]).toBe('Group by Kind'));
  });
});

describe('Delete… on a session row', () => {
  test('a past session: enabled, danger, and asks with the session and its count', async () => {
    await rightClick($('.osprey-tree-title'));
    const item = menuRow('Delete…');
    expect(item.getAttribute('aria-disabled')).toBeNull();
    expect(item.classList.contains('danger')).toBe(true);
    item.click();
    await vi.waitFor(() => expect(confirm.showConfirm).toHaveBeenCalledTimes(1));
    expect(globalThis.fetch).not.toHaveBeenCalled();
    const spec = confirm.showConfirm.mock.calls[0][0];
    expect(spec.title).toBe('Delete session?');
    expect(spec.body).toEqual([
      ['This deletes the session ', { em: 'Beam study' }, ' and its 3 artifacts.'],
    ]);
    expect(spec.confirmLabel).toBe('Delete');
    expect(spec.focus).toBe('cancel');
    const ui = { done: vi.fn(), error: document.createElement('div') };
    await spec.onConfirm(ui);
    expect(globalThis.fetch).toHaveBeenCalledWith('/api/sessions/t1', { method: 'DELETE' });
    expect(ui.done).toHaveBeenCalled();
    expect(sessionModel.forgetSession).toHaveBeenCalledWith('t1');
  });

  test('without a summary the dialog says "and its artifacts"', async () => {
    delete $('.osprey-tree-session').dataset.count;
    await rightClick($('.osprey-tree-title'));
    menuRow('Delete…').click();
    await vi.waitFor(() => expect(confirm.showConfirm).toHaveBeenCalledTimes(1));
    expect(JSON.stringify(confirm.showConfirm.mock.calls[0][0].body)).toContain(
      ' and its artifacts.'
    );
  });

  test('the session the Agent is on: disabled, with the tip', async () => {
    $('.osprey-tree-session').classList.add('active');
    await rightClick($('.osprey-tree-title'));
    const item = menuRow('Delete…');
    expect(item.getAttribute('aria-disabled')).toBe('true');
    expect(item.title).toBe('Switch to another session to delete this one');
    item.click();
    await Promise.resolve();
    expect(confirm.showConfirm).not.toHaveBeenCalled();
  });

  test('a refusal keeps the dialog open with the hub’s reason', async () => {
    await rightClick($('.osprey-tree-title'));
    menuRow('Delete…').click();
    await vi.waitFor(() => expect(confirm.showConfirm).toHaveBeenCalledTimes(1));
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 409,
        json: async () => ({ detail: 'A turn is running in this session.' }),
      }))
    );
    const ui = { done: vi.fn(), error: document.createElement('div') };
    ui.error.hidden = true;
    await confirm.showConfirm.mock.calls[0][0].onConfirm(ui);
    expect(ui.done).not.toHaveBeenCalled();
    expect(ui.error.hidden).toBe(false);
    expect(ui.error.textContent).toContain('A turn is running');
    expect(sessionModel.forgetSession).not.toHaveBeenCalled();
  });
});

describe('a group head', () => {
  test('offers Collapse when open', async () => {
    await rightClick($('[data-group-head]'));
    expect(menuLabels()).toEqual(['Collapse']);
  });
});

describe('a run head', () => {
  test('Open result, Open approval preview (disabled: never asked), Open code, Delete…', async () => {
    await rightClick($('[data-run]'));
    expect(menuLabels()).toEqual(['Open result', 'Open approval preview', 'Open code', 'Delete…']);
    expect(menuRow('Open approval preview').getAttribute('aria-disabled')).toBe('true');
    expect(menuRow('Open result').getAttribute('aria-disabled')).toBeNull();
  });

  test('Open code clicks the run’s code part', async () => {
    const picked = vi.fn();
    $('[data-id="code1"]').addEventListener('click', picked);
    await rightClick($('[data-run]'));
    menuRow('Open code').click();
    expect(picked).toHaveBeenCalled();
  });

  test('Delete… asks first and says the run and its parts go', async () => {
    await rightClick($('[data-run]'));
    menuRow('Delete…').click();
    expect(globalThis.fetch).not.toHaveBeenCalled();
    expect(confirm.showConfirm).toHaveBeenCalledTimes(1);
    const spec = confirm.showConfirm.mock.calls[0][0];
    expect(spec.title).toBe('Delete run?');
    expect(JSON.stringify(spec.body)).toContain('and its parts (2)');
    expect(spec.focus).toBe('cancel');
    const ui = { done: vi.fn(), error: document.createElement('div') };
    await spec.onConfirm(ui);
    expect(globalThis.fetch).toHaveBeenCalledWith('/panel/artifacts/api/artifacts/run1', {
      method: 'DELETE',
    });
    expect(ui.done).toHaveBeenCalled();
  });
});

describe('an artifact row', () => {
  test('Open, Pin, Delete…, Copy ID', async () => {
    await rightClick($('[data-id="a1"]'));
    expect(menuLabels()).toEqual(['Open', 'Pin', 'Delete…', 'Copy ID']);
  });

  test('Delete… confirms before anything is deleted', async () => {
    await rightClick($('[data-id="a1"]'));
    menuRow('Delete…').click();
    expect(globalThis.fetch).not.toHaveBeenCalled();
    const spec = confirm.showConfirm.mock.calls[0][0];
    expect(spec.title).toBe('Delete artifact?');
    expect(JSON.stringify(spec.body)).not.toContain('its parts');
    spec.onDismiss?.();
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  test('a failed delete keeps the dialog open with the error', async () => {
    await rightClick($('[data-id="a1"]'));
    menuRow('Delete…').click();
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: false, status: 404 }))
    );
    const ui = { done: vi.fn(), error: document.createElement('div') };
    ui.error.hidden = true;
    await confirm.showConfirm.mock.calls[0][0].onConfirm(ui);
    expect(ui.done).not.toHaveBeenCalled();
    expect(ui.error.hidden).toBe(false);
  });

  test('Pin posts the flipped state', async () => {
    pointer.clearPointer();
    await rightClick($('[data-id="a1"]'));
    menuRow('Pin').click();
    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    const [url, init] = /** @type {any} */ (globalThis.fetch).mock.calls[0];
    expect(url).toBe('/panel/artifacts/api/artifacts/a1/pin');
    expect(JSON.parse(init.body)).toEqual({ pinned: true });
  });

  test("Pin names the operator's current session, not the row's", async () => {
    // The row's session is k1; the operator is on another key.
    pointer.setPointer('k-current');
    try {
      await rightClick($('[data-id="a1"]'));
      menuRow('Pin').click();
      await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
      const [, init] = /** @type {any} */ (globalThis.fetch).mock.calls[0];
      expect(JSON.parse(init.body)).toEqual({ pinned: true, session_key: 'k-current' });
    } finally {
      pointer.clearPointer();
    }
  });

  test('a row outside the tree is left to the browser', () => {
    const outside = document.createElement('button');
    outside.className = 'osprey-list-row';
    outside.dataset.id = 'a1';
    document.body.append(outside);
    const ev = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    outside.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(false);
  });
});

describe('behind the hub prefix (/u/alice)', () => {
  beforeEach(() => {
    /** @type {any} */ (window).__OSPREY_PREFIX__ = '/u/alice';
  });
  afterEach(() => {
    delete (/** @type {any} */ (window).__OSPREY_PREFIX__);
  });

  test('rename PUTs the title under the prefix', async () => {
    await menus.putTitle('t1', 'New name');
    const [url, init] = /** @type {any} */ (globalThis.fetch).mock.calls[0];
    expect(url).toBe('/u/alice/api/sessions/t1/title');
    expect(init.method).toBe('PUT');
  });

  test('delete DELETEs the session under the prefix', async () => {
    await rightClick($('.osprey-tree-title'));
    menuRow('Delete…').click();
    await vi.waitFor(() => expect(confirm.showConfirm).toHaveBeenCalledTimes(1));
    const ui = { done: vi.fn(), error: document.createElement('div') };
    await confirm.showConfirm.mock.calls[0][0].onConfirm(ui);
    expect(globalThis.fetch).toHaveBeenCalledWith('/u/alice/api/sessions/t1', {
      method: 'DELETE',
    });
  });

  test('the transcript export link carries the prefix', () => {
    let href = '';
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {
      href = document.body.querySelector('a[download]')?.getAttribute('href') ?? '';
    });
    try {
      menus.exportTranscript('t1');
    } finally {
      click.mockRestore();
    }
    expect(href).toBe('/u/alice/api/sessions/t1/transcript.md');
  });

  test('Pin goes under the prefix', async () => {
    await rightClick($('[data-id="a1"]'));
    menuRow('Pin').click();
    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    expect(/** @type {any} */ (globalThis.fetch).mock.calls[0][0]).toBe(
      '/u/alice/panel/artifacts/api/artifacts/a1/pin'
    );
  });
});
