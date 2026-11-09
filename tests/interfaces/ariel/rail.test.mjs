// @ts-check
/**
 * Rail contract: mailbox rows, the Mine name and saved searches.
 *
 * Run: npx vitest run tests/interfaces/ariel/rail.test.mjs
 */
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const STORE = '../../../src/osprey/interfaces/ariel/static/js/store.js';
const RAIL = '../../../src/osprey/interfaces/ariel/static/js/rail.js';

const MINE_KEY = 'ariel-mine-author';
const SAVED_KEY = 'ariel-saved-searches';

/** @type {typeof import('../../../src/osprey/interfaces/ariel/static/js/store.js')} */
let store;
/** @type {typeof import('../../../src/osprey/interfaces/ariel/static/js/rail.js')} */
let rail;
/** @type {HTMLElement} */
let root;

/** Fresh store + rail sharing one module graph, mounted into #ariel-rail. */
async function boot() {
  vi.resetModules();
  store = await import(STORE);
  rail = await import(RAIL);
  rail.initRail(root);
}

/**
 * @param {string} box
 * @returns {HTMLButtonElement}
 */
function row(box) {
  const el = root.querySelector(`[data-box="${box}"]`);
  if (!(el instanceof HTMLButtonElement)) throw new Error(`no row for ${box}`);
  return el;
}

/** @param {string} name */
function savedRow(name) {
  return row(`saved:${name}`);
}

/**
 * @param {HTMLInputElement} input
 * @param {string} value
 */
function submit(input, value) {
  input.value = value;
  const form = input.closest('form');
  if (!form) throw new Error('input is not inside a form');
  form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
}

function mineInput() {
  const el = root.querySelector('input[data-rail-field="mine"]');
  if (!(el instanceof HTMLInputElement)) throw new Error('no Mine name field');
  return el;
}

function saveInput() {
  const el = root.querySelector('input[data-rail-field="save"]');
  if (!(el instanceof HTMLInputElement)) throw new Error('no save-search name field');
  return el;
}

/** @param {HTMLElement} el */
function shown(el) {
  for (
    let node = /** @type {HTMLElement|null} */ (el);
    node && node !== root;
    node = node.parentElement
  ) {
    if (node.hidden) return false;
  }
  return true;
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute('data-osprey-storage-scope');
  document.body.innerHTML = '<nav id="ariel-rail"></nav>';
  root = /** @type {HTMLElement} */ (document.getElementById('ariel-rail'));
});

afterEach(() => {
  document.body.innerHTML = '';
});

describe('mailbox rows', () => {
  test('renders All entries, Drafts and Mine as nav rows with icons and names', async () => {
    await boot();
    for (const [box, name] of [
      ['all', 'All entries'],
      ['drafts', 'Drafts'],
      ['mine', 'Mine'],
    ]) {
      const btn = row(box);
      expect(btn.classList.contains('osprey-nav-row')).toBe(true);
      expect(btn.type).toBe('button');
      const glyph = btn.querySelector('svg.osprey-ic');
      expect(glyph).not.toBeNull();
      expect(glyph?.classList.contains('osprey-nav-icon')).toBe(true);
      // The icon strip hides labels, so the row itself must carry the name.
      expect(btn.getAttribute('aria-label')).toBe(name);
      expect(btn.getAttribute('title')).toBe(name);
      expect(btn.querySelector('.osprey-nav-label')?.textContent).toBe(name);
    }
  });

  test('a Saved searches group caption is rendered', async () => {
    await boot();
    const label = root.querySelector('.osprey-section-label');
    expect(label?.textContent).toContain('Saved searches');
  });

  test('the selected mailbox carries aria-current="page", only it', async () => {
    await boot();
    expect(row('all').getAttribute('aria-current')).toBe('page');
    expect(row('drafts').hasAttribute('aria-current')).toBe(false);
    store.setState({ mailbox: 'drafts' });
    expect(row('drafts').getAttribute('aria-current')).toBe('page');
    expect(row('all').hasAttribute('aria-current')).toBe(false);
  });

  test('clicking a row sets mailbox and rerank:now', async () => {
    await boot();
    store.setState({ rerank: 'when-stable' });
    const spy = vi.fn();
    store.subscribe(spy);
    row('drafts').click();
    expect(store.getState().mailbox).toBe('drafts');
    expect(store.getState().rerank).toBe('now');
    row('all').click();
    expect(store.getState().mailbox).toBe('all');
    const changed = spy.mock.calls.map((c) => [...c[2]]);
    expect(changed[0]).toEqual(expect.arrayContaining(['mailbox', 'rerank']));
  });
});

describe('Mine', () => {
  test('with no known name: expands the rail and asks inline instead of selecting', async () => {
    await boot();
    store.setState({ railCollapsed: true });
    row('mine').click();
    expect(store.getState().railCollapsed).toBe(false);
    expect(store.getState().mailbox).toBe('all');
    const input = mineInput();
    expect(shown(input)).toBe(true);
    expect(document.activeElement).toBe(input);
  });

  test('submitting a name remembers it and selects Mine', async () => {
    await boot();
    row('mine').click();
    submit(mineInput(), '  Jane Doe  ');
    expect(store.getState().mineAuthor).toBe('Jane Doe');
    expect(store.getState().mailbox).toBe('mine');
    expect(store.getState().rerank).toBe('now');
    expect(localStorage.getItem(MINE_KEY)).toBe('Jane Doe');
    expect(shown(mineInput())).toBe(false);
  });

  test('the name field suggests every author from the authors endpoint, narrowed by typing', async () => {
    const authors = ['Jane Doe', 'Janet Roe', 'Bob'];
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        field: 'authors',
        options: authors.map((a) => ({ value: a, label: a })),
      }),
    }));
    vi.stubGlobal('fetch', fetchMock);
    try {
      await boot();
      row('mine').click();
      const input = mineInput();
      expect(
        fetchMock,
        'nothing is fetched until the operator presses into the field'
      ).not.toHaveBeenCalled();
      const list = document.getElementById(/** @type {string} */ (input.getAttribute('list')));
      const offered = () => Array.from(list?.querySelectorAll('option') ?? []).map((o) => o.value);
      input.dispatchEvent(new Event('pointerdown'));
      await vi.waitFor(() => expect(offered()).toEqual(authors));
      expect(fetchMock.mock.calls[0]).toEqual(['/api/filter-options/authors']);
      input.value = 'jan';
      input.dispatchEvent(new Event('input'));
      expect(offered()).toEqual(['Jane Doe', 'Janet Roe']);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  test('submitting blank or Anonymous keeps asking', async () => {
    await boot();
    row('mine').click();
    submit(mineInput(), '   ');
    submit(mineInput(), 'Anonymous');
    expect(store.getState().mineAuthor).toBeNull();
    expect(store.getState().mailbox).toBe('all');
    expect(shown(mineInput())).toBe(true);
  });

  test('Escape in the name field closes the prompt without selecting', async () => {
    await boot();
    row('mine').click();
    mineInput().dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })
    );
    expect(shown(mineInput())).toBe(false);
    expect(store.getState().mailbox).toBe('all');
  });

  test('with a known name: selects Mine directly', async () => {
    localStorage.setItem(MINE_KEY, 'Jane');
    await boot();
    row('mine').click();
    expect(store.getState().mailbox).toBe('mine');
    expect(shown(mineInput())).toBe(false);
  });

  test('selecting Mine clears filter.author in the same write', async () => {
    localStorage.setItem(MINE_KEY, 'Jane');
    await boot();
    store.setState({ filter: { author: ['Bob'], source_system: ['ops'] } });
    const spy = vi.fn();
    store.subscribe(spy);
    row('mine').click();
    expect(spy).toHaveBeenCalledTimes(1);
    expect(store.getState()).toMatchObject({
      mailbox: 'mine',
      filter: { source_system: ['ops'] },
      rerank: 'now',
    });
  });

  test('submitting the Mine name clears filter.author in the same write', async () => {
    await boot();
    store.setState({ filter: { author: ['Bob'] } });
    row('mine').click();
    const spy = vi.fn();
    store.subscribe((_s, _p, changed) => spy([...changed]));
    submit(mineInput(), 'Jane');
    // One write for the name (rememberAuthor), one selecting Mine with the filter.
    expect(spy).toHaveBeenLastCalledWith(expect.arrayContaining(['mailbox', 'filter']));
    expect(store.getState().mailbox).toBe('mine');
    expect(store.getState().filter).toEqual({});
  });

  test('selecting another mailbox leaves filter.author alone', async () => {
    await boot();
    store.setState({ filter: { author: ['Bob'] } });
    row('drafts').click();
    expect(store.getState().filter).toEqual({ author: ['Bob'] });
  });

  test('mineAuthor is loaded from browser storage at init', async () => {
    localStorage.setItem(MINE_KEY, 'Jane');
    await boot();
    expect(store.getState().mineAuthor).toBe('Jane');
  });

  test('a stored blank or Anonymous name is not loaded', async () => {
    localStorage.setItem(MINE_KEY, 'Anonymous');
    await boot();
    expect(store.getState().mineAuthor).toBeNull();
  });

  test('Mine selected from elsewhere (hash) with no name opens the prompt', async () => {
    await boot();
    store.setState({ railCollapsed: true, mailbox: 'mine' });
    expect(store.getState().railCollapsed).toBe(false);
    expect(shown(mineInput())).toBe(true);
  });

  test('the Mine name key honours the storage scope', async () => {
    document.documentElement.setAttribute('data-osprey-storage-scope', 'acme');
    await boot();
    rail.rememberAuthor('Jane');
    expect(localStorage.getItem(`${MINE_KEY}--acme`)).toBe('Jane');
    expect(localStorage.getItem(MINE_KEY)).toBeNull();
  });
});

describe('rememberAuthor', () => {
  test('persists and writes mineAuthor', async () => {
    await boot();
    rail.rememberAuthor('Jane');
    expect(store.getState().mineAuthor).toBe('Jane');
    expect(localStorage.getItem(MINE_KEY)).toBe('Jane');
  });

  test.each([[''], ['   '], ['Anonymous'], ['anonymous'], [' Anonymous ']])(
    'ignores %j',
    async (name) => {
      await boot();
      rail.rememberAuthor('Jane');
      rail.rememberAuthor(name);
      expect(store.getState().mineAuthor).toBe('Jane');
      expect(localStorage.getItem(MINE_KEY)).toBe('Jane');
    }
  );

  test('ignores non-strings', async () => {
    await boot();
    rail.rememberAuthor(/** @type {any} */ (null));
    rail.rememberAuthor(/** @type {any} */ (undefined));
    expect(store.getState().mineAuthor).toBeNull();
  });

  test('works before initRail (compose host may call it first)', async () => {
    vi.resetModules();
    store = await import(STORE);
    rail = await import(RAIL);
    rail.rememberAuthor('Jane');
    expect(store.getState().mineAuthor).toBe('Jane');
  });

  test('survives storage that throws', async () => {
    await boot();
    const spy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    try {
      expect(() => rail.rememberAuthor('Jane')).not.toThrow();
      expect(store.getState().mineAuthor).toBe('Jane');
    } finally {
      spy.mockRestore();
    }
  });
});

describe('saved searches', () => {
  test('saveCurrentSearch stores query, mode, filter and knobs under a name and lists it', async () => {
    await boot();
    store.setState({
      query: 'kicker',
      mode: 'keyword',
      filter: { author: ['Bob'] },
      filters: { max_results: 5 },
    });
    rail.saveCurrentSearch('  Kickers  ');
    const btn = savedRow('Kickers');
    expect(btn.classList.contains('osprey-nav-row')).toBe(true);
    expect(btn.querySelector('svg.osprey-ic')).not.toBeNull();
    expect(btn.getAttribute('aria-label')).toBe('Kickers');
    const stored = JSON.parse(localStorage.getItem(SAVED_KEY) || 'null');
    expect(stored).toEqual([
      {
        name: 'Kickers',
        query: 'kicker',
        mode: 'keyword',
        filter: { author: ['Bob'] },
        filters: { max_results: 5 },
      },
    ]);
  });

  test('saving does not change what is shown', async () => {
    await boot();
    store.setState({ query: 'q', mailbox: 'drafts' });
    const spy = vi.fn();
    store.subscribe(spy);
    rail.saveCurrentSearch('one');
    expect(spy).not.toHaveBeenCalled();
  });

  test('a blank name is ignored', async () => {
    await boot();
    rail.saveCurrentSearch('   ');
    expect(localStorage.getItem(SAVED_KEY)).toBeNull();
    expect(root.querySelector('[data-box^="saved:"]')).toBeNull();
  });

  test('saving under an existing name replaces it in place', async () => {
    await boot();
    store.setState({ query: 'a' });
    rail.saveCurrentSearch('first');
    rail.saveCurrentSearch('second');
    store.setState({ query: 'b' });
    rail.saveCurrentSearch('first');
    const stored = JSON.parse(localStorage.getItem(SAVED_KEY) || '[]');
    expect(stored.map((/** @type {any} */ s) => [s.name, s.query])).toEqual([
      ['first', 'b'],
      ['second', 'a'],
    ]);
    expect(root.querySelectorAll('[data-box^="saved:"]').length).toBe(2);
  });

  test('saved filters are a copy, not the live state object', async () => {
    await boot();
    const filters = { max_results: 3 };
    store.setState({ filters });
    rail.saveCurrentSearch('s');
    filters.max_results = 9;
    const stored = JSON.parse(localStorage.getItem(SAVED_KEY) || '[]');
    expect(stored[0].filters).toEqual({ max_results: 3 });
  });

  test('a search saved under Mine carries the Mine name as its author', async () => {
    localStorage.setItem(MINE_KEY, 'Jane');
    await boot();
    row('mine').click();
    store.setState({ query: 'rf', filter: { source_system: ['ops'] } });
    rail.saveCurrentSearch('My RF');
    const stored = JSON.parse(localStorage.getItem(SAVED_KEY) || '[]');
    expect(stored[0].filter).toEqual({ author: ['Jane'], source_system: ['ops'] });
  });

  test('a P1 record is listed and runs with its filter migrated', async () => {
    localStorage.setItem(
      SAVED_KEY,
      JSON.stringify([
        {
          name: 'Old',
          query: 'beam',
          mode: 'keyword',
          filters: { author: 'Bob', start_date: '2026-01-01', max_results: 5 },
        },
      ])
    );
    await boot();
    savedRow('Old').click();
    const s = store.getState();
    expect(s.filter).toEqual({ author: ['Bob'], time: { start: '2026-01-01', end: null } });
    expect(s.filters).toEqual({ max_results: 5 });
  });

  test('saved searches are listed at init from storage', async () => {
    localStorage.setItem(
      SAVED_KEY,
      JSON.stringify([{ name: 'Mine-ish', query: 'x', mode: null, filters: {} }])
    );
    await boot();
    expect(savedRow('Mine-ish')).toBeTruthy();
  });

  test('corrupt storage yields an empty list, not a crash', async () => {
    localStorage.setItem(SAVED_KEY, '{not json');
    await boot();
    expect(root.querySelector('[data-box^="saved:"]')).toBeNull();
    localStorage.setItem(
      SAVED_KEY,
      JSON.stringify([{ nope: 1 }, 'str', { name: 'ok', query: 'q' }])
    );
    await boot();
    expect(root.querySelectorAll('[data-box^="saved:"]').length).toBe(1);
  });

  test('running a saved search restores query, mode, filter and knobs and selects its row', async () => {
    await boot();
    store.setState({
      query: 'kicker',
      mode: 'semantic',
      filter: { author: ['Bob'] },
      filters: { max_results: 4 },
    });
    rail.saveCurrentSearch('K');
    store.setState({
      query: 'other',
      mode: 'keyword',
      filter: {},
      filters: {},
      rerank: 'when-stable',
    });
    savedRow('K').click();
    const s = store.getState();
    expect(s.mailbox).toEqual({ saved: 'K' });
    expect(s.query).toBe('kicker');
    expect(s.mode).toBe('semantic');
    expect(s.filter).toEqual({ author: ['Bob'] });
    expect(s.filters).toEqual({ max_results: 4 });
    expect(s.rerank).toBe('now');
    expect(savedRow('K').getAttribute('aria-current')).toBe('page');
    expect(row('all').hasAttribute('aria-current')).toBe(false);
  });

  test('running a saved search writes once', async () => {
    await boot();
    store.setState({ query: 'kicker', filters: { a: 1 } });
    rail.saveCurrentSearch('K');
    store.setState({ query: '', filters: {} });
    const spy = vi.fn();
    store.subscribe(spy);
    savedRow('K').click();
    expect(spy).toHaveBeenCalledTimes(1);
  });

  test('remove deletes it from storage and the list', async () => {
    await boot();
    rail.saveCurrentSearch('gone');
    rail.saveCurrentSearch('kept');
    const remove = root.querySelector('[data-remove-saved="gone"]');
    expect(remove).toBeInstanceOf(HTMLButtonElement);
    expect(remove?.getAttribute('aria-label')).toContain('gone');
    /** @type {HTMLButtonElement} */ (remove).click();
    expect(root.querySelector('[data-box="saved:gone"]')).toBeNull();
    expect(savedRow('kept')).toBeTruthy();
    const stored = JSON.parse(localStorage.getItem(SAVED_KEY) || '[]');
    expect(stored.map((/** @type {any} */ s) => s.name)).toEqual(['kept']);
  });

  test('removing the selected saved search returns to All entries', async () => {
    await boot();
    rail.saveCurrentSearch('sel');
    savedRow('sel').click();
    /** @type {HTMLButtonElement} */ (root.querySelector('[data-remove-saved="sel"]')).click();
    expect(store.getState().mailbox).toBe('all');
    expect(store.getState().rerank).toBe('now');
  });

  test('a name with quotes and markup is rendered as text', async () => {
    await boot();
    const name = '"><img src=x onerror=alert(1)>';
    rail.saveCurrentSearch(name);
    expect(root.querySelector('img')).toBeNull();
    const rows = [...root.querySelectorAll('[data-box^="saved:"]')];
    expect(rows.length).toBe(1);
    expect(rows[0].getAttribute('data-box')).toBe(`saved:${name}`);
    expect(rows[0].querySelector('.osprey-nav-label')?.textContent).toBe(name);
  });

  test('a saved mailbox selected from elsewhere is not restored by the rail', async () => {
    await boot();
    store.setState({ filters: { max_results: 3 } });
    rail.saveCurrentSearch('K');
    store.setState({ filters: {} });
    const spy = vi.fn();
    store.subscribe(spy);
    // The router restores a hash-driven saved box in its own patch.
    store.setState({ mailbox: { saved: 'K' } });
    expect(spy).toHaveBeenCalledTimes(1);
    expect(store.getState().filters).toEqual({});
    expect(savedRow('K').getAttribute('aria-current')).toBe('page');
  });

  test('the saved-searches key honours the storage scope', async () => {
    document.documentElement.setAttribute('data-osprey-storage-scope', 'acme');
    await boot();
    rail.saveCurrentSearch('s');
    expect(localStorage.getItem(`${SAVED_KEY}--acme`)).not.toBeNull();
    expect(localStorage.getItem(SAVED_KEY)).toBeNull();
  });
});

describe('promptSaveSearch', () => {
  test('expands the rail and focuses the name field', async () => {
    await boot();
    store.setState({ railCollapsed: true });
    rail.promptSaveSearch();
    expect(store.getState().railCollapsed).toBe(false);
    const input = saveInput();
    expect(shown(input)).toBe(true);
    expect(document.activeElement).toBe(input);
  });

  test('submitting the field saves the current search and hides the field', async () => {
    await boot();
    store.setState({ query: 'beam dump', mode: 'keyword', filters: { a: 1 } });
    rail.promptSaveSearch();
    submit(saveInput(), 'Dumps');
    expect(savedRow('Dumps')).toBeTruthy();
    expect(shown(saveInput())).toBe(false);
    const stored = JSON.parse(localStorage.getItem(SAVED_KEY) || '[]');
    expect(stored[0]).toEqual({
      name: 'Dumps',
      query: 'beam dump',
      mode: 'keyword',
      filter: {},
      filters: { a: 1 },
    });
  });

  test('the rail offers its own save button that opens the field', async () => {
    await boot();
    const btn = root.querySelector('[data-rail-action="save-search"]');
    expect(btn).toBeInstanceOf(HTMLButtonElement);
    /** @type {HTMLButtonElement} */ (btn).click();
    expect(document.activeElement).toBe(saveInput());
  });

  test('Escape closes the field without saving', async () => {
    await boot();
    rail.promptSaveSearch();
    const input = saveInput();
    input.value = 'nope';
    input.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })
    );
    expect(shown(input)).toBe(false);
    expect(localStorage.getItem(SAVED_KEY)).toBeNull();
  });

  test('is a no-op without a mounted rail', async () => {
    vi.resetModules();
    store = await import(STORE);
    rail = await import(RAIL);
    expect(() => rail.promptSaveSearch()).not.toThrow();
  });
});

describe('initRail', () => {
  test('a second initRail does not double-subscribe', async () => {
    await boot();
    rail.initRail(root);
    expect(root.querySelectorAll('[data-box="all"]').length).toBe(1);
    store.setState({ query: 'q', filters: { a: 1 } });
    rail.saveCurrentSearch('K');
    store.setState({ filters: {} });
    const spy = vi.fn();
    store.subscribe(spy);
    store.setState({ mailbox: { saved: 'K' } });
    // One write from the test; the rail writes nothing for a saved box.
    expect(spy).toHaveBeenCalledTimes(1);
    expect(root.querySelectorAll('[aria-current="page"]').length).toBe(1);
  });
});
