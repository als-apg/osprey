/**
 * The command palette's Settings files group: the registry rows built from the
 * Settings drawer's file list, and the palette's concurrent read of that list.
 *
 *   npx vitest run tests/interfaces/web_terminal/palette-settings-files.test.mjs
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

import { buildRegistry } from '../../../src/osprey/interfaces/web_terminal/static/js/palette-registry.js';
import { fuzzyMatch } from '../../../src/osprey/interfaces/design_system/static/js/fuzzy.js';

/** Entries as listGalleryEntries returns them for a typical deployment. */
const ENTRIES = [
  {
    name: 'claude-md',
    label: 'claude-md',
    category: 'project instructions',
    tabId: 'tab-behavior',
    tabLabel: 'Behavior',
    description: 'Project instructions',
  },
  {
    name: 'rules/safety',
    label: 'safety',
    category: 'instructions',
    tabId: 'tab-behavior',
    tabLabel: 'Behavior',
    description: 'Safety boundaries and error handling',
  },
  {
    name: 'hooks/approval',
    label: 'approval',
    category: 'hooks',
    tabId: 'tab-safety',
    tabLabel: 'Safety',
    description: 'Human approval before hardware writes',
  },
  {
    name: 'code-review/rules',
    label: 'rules',
    category: 'code-review',
    tabId: 'tab-safety',
    tabLabel: 'Safety',
    description: "The code reviewer's instructions",
  },
];

/** @param {ReturnType<typeof buildRegistry>} items */
const files = (items) => /** @type {any[]} */ (items.filter((i) => i.group === 'Settings files'));

describe('buildRegistry: Settings files group', () => {
  it('emits one row per entry, labelled as its card, detail naming tab and section', () => {
    const items = buildRegistry({
      artifacts: { state: 'ok', entries: ENTRIES },
      revealArtifact: () => {},
    });
    const rows = files(items);
    expect(rows.map((r) => r.label)).toEqual(['claude-md', 'safety', 'approval', 'rules']);
    expect(rows[3].detail).toBe('Safety · code-review');
    expect(rows[3].searchText).toContain('code-review/rules');
    expect(rows[3].searchText).toContain("The code reviewer's instructions");
  });

  it('comes before the Settings group', () => {
    const items = buildRegistry({
      artifacts: { state: 'ok', entries: ENTRIES },
      revealArtifact: () => {},
      config: { state: 'ok', sections: { code_review: { prompt: 'x' } } },
      revealSetting: () => {},
    });
    const groups = [...new Set(items.map((i) => i.group))];
    expect(groups.slice(0, 2)).toEqual(['Settings files', 'Settings']);
  });

  it('"code review" ranks the code reviewer\'s rules row top', () => {
    const rows = files(
      buildRegistry({ artifacts: { state: 'ok', entries: ENTRIES }, revealArtifact: () => {} })
    );
    const ranked = rows
      .map((r) => ({ r, m: fuzzyMatch('code review', r.searchText) }))
      .filter((x) => x.m)
      .sort((a, b) => /** @type {any} */ (b.m).score - /** @type {any} */ (a.m).score);
    expect(ranked[0].r.label).toBe('rules');
  });

  it('run reveals the entry by name on its tab', () => {
    const revealArtifact = vi.fn();
    const rows = files(
      buildRegistry({ artifacts: { state: 'ok', entries: ENTRIES }, revealArtifact })
    );
    rows[3].run();
    expect(revealArtifact).toHaveBeenCalledWith('code-review/rules', 'tab-safety');
  });

  it.each([
    ['loading', 'Loading settings files…'],
    ['error', 'Settings files unavailable'],
  ])('%s yields one non-navigable decoration', (state, label) => {
    const rows = files(
      buildRegistry({ artifacts: /** @type {any} */ ({ state }), revealArtifact: () => {} })
    );
    expect(rows).toEqual([{ group: 'Settings files', status: state, label }]);
  });

  it('is withheld entirely without revealArtifact', () => {
    expect(files(buildRegistry({ artifacts: { state: 'ok', entries: ENTRIES } }))).toEqual([]);
    expect(files(buildRegistry({ artifacts: { state: 'loading' } }))).toEqual([]);
  });
});

// ---- palette.js: the concurrent list read ---- //

vi.stubGlobal(
  'fetch',
  vi.fn(async (/** @type {string} */ url) => {
    throw new Error(`unstubbed fetch: ${url}`);
  })
);

const { openPalette, closePalette } =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/palette.js');

/** Visible row labels under the Settings files heading, plus its status rows. */
function settingsFilesRows() {
  const list = document.querySelector('.command-palette-list');
  if (!list) return [];
  /** @type {string[]} */
  const out = [];
  let inGroup = false;
  for (const node of Array.from(list.children)) {
    if (node.classList.contains('command-palette-group-heading')) {
      inGroup = node.textContent === 'Settings files';
      continue;
    }
    if (!inGroup) continue;
    const label = node.querySelector('.command-palette-item-label');
    out.push((label || node).textContent || '');
  }
  return out;
}

const flush = () => new Promise((r) => setTimeout(r, 0));

/** @returns {{ promise: Promise<any>, resolve: (v: any) => void, reject: (e: any) => void }} */
function deferred() {
  /** @type {any} */
  let resolve;
  /** @type {any} */
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe('palette: Settings files read', () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = () => {};
    document.body.innerHTML = '';
  });
  afterEach(() => {
    closePalette();
  });

  it('shows loading, then the rows once the lister settles', async () => {
    const d = deferred();
    const loadArtifacts = vi.fn(() => d.promise);
    openPalette({ loadArtifacts, revealArtifact: () => {} });
    expect(settingsFilesRows()).toEqual(['Loading settings files…']);
    await flush();
    expect(loadArtifacts).toHaveBeenCalledTimes(1);
    d.resolve(ENTRIES);
    await flush();
    expect(settingsFilesRows()).toEqual(['claude-md', 'safety', 'approval', 'rules']);
  });

  it('a rejecting lister yields the unavailable row', async () => {
    openPalette({ loadArtifacts: () => Promise.reject(new Error('x')), revealArtifact: () => {} });
    await flush();
    expect(settingsFilesRows()).toEqual(['Settings files unavailable']);
  });

  it('withheld deps: no read, no group', async () => {
    const loadArtifacts = vi.fn(() => Promise.resolve(ENTRIES));
    openPalette({ loadArtifacts });
    await flush();
    expect(loadArtifacts).not.toHaveBeenCalled();
    expect(settingsFilesRows()).toEqual([]);
  });

  it('a read that settles after close, or after a newer open, is dropped', async () => {
    const first = deferred();
    const second = deferred();
    const loads = [first, second];
    const loadArtifacts = vi.fn(() => /** @type {any} */ (loads.shift()).promise);
    openPalette({ loadArtifacts, revealArtifact: () => {} });
    closePalette();
    openPalette({ loadArtifacts, revealArtifact: () => {} });
    first.resolve(ENTRIES);
    await flush();
    expect(settingsFilesRows()).toEqual(['Loading settings files…']);
    second.resolve(ENTRIES.slice(3));
    await flush();
    expect(settingsFilesRows()).toEqual(['rules']);
  });

  it('typing "code review" and Enter reveals the reviewer\'s rules', async () => {
    const revealArtifact = vi.fn();
    openPalette({ loadArtifacts: () => Promise.resolve(ENTRIES), revealArtifact });
    await flush();
    const input = /** @type {HTMLInputElement} */ (
      document.querySelector('.command-palette-input')
    );
    input.value = 'code review';
    input.dispatchEvent(new Event('input'));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(revealArtifact).toHaveBeenCalledWith('code-review/rules', 'tab-safety');
  });
});
