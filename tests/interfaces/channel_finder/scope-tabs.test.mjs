// @ts-check
/**
 * The panel's scope switch (app.js, panel-toolbar.js): Curated · All PVs ·
 * Alarms · Archive · Feedback as the buttons of the toolbar seg `#cf-views` when
 * /api/info carries the directory descriptor (Alarms and Archive while their
 * service is configured), and Explore | Feedback otherwise; the graph paradigm
 * has no Feedback. An empty or unknown route lands on Curated. Nothing of it goes
 * to the hub's tile bar.
 *   npx vitest run tests/interfaces/channel_finder/scope-tabs.test.mjs
 *
 * app.js boots on import, so each case re-imports it into a fresh module registry
 * with the view modules replaced by spies, and removes the hashchange and
 * visibilitychange listeners the boot added, so one case's app never answers the
 * next case's events.
 */

import { test, expect, vi, afterEach, describe } from 'vitest';

const JS = '../../../src/osprey/interfaces/channel_finder/static/js';

const DESCRIPTOR = { label: 'All PVs', configured_services: [], canopy: true, ariel: true };

const WITH_DIRECTORY = { pipeline_type: 'hierarchical', directory: DESCRIPTOR };
const WITHOUT_DIRECTORY = { pipeline_type: 'hierarchical' };

/** The panel's view modules, by the file and the export names app.js imports. */
const VIEW_MODULES = {
  explore: ['explore.js', 'mountExplore', 'unmountExplore'],
  feedback: ['feedback.js', 'mountFeedback', 'unmountFeedback'],
  allpvs: ['allpvs.js', 'mountAllPvs', 'unmountAllPvs'],
  pv: ['pv-page.js', 'mountPvPage', 'unmountPvPage'],
  ioc: ['ioc-page.js', 'mountIocPage', 'unmountIocPage'],
  alarms: ['alarms.js', 'mountAlarms', 'unmountAlarms'],
  archive: ['archive.js', 'mountArchive', 'unmountArchive'],
};

/** @type {EventListener[]} */
let hashListeners = [];
/** @type {EventListener[]} */
let visibilityListeners = [];

afterEach(() => {
  for (const listener of hashListeners) window.removeEventListener('hashchange', listener);
  hashListeners = [];
  for (const listener of visibilityListeners) {
    document.removeEventListener('visibilitychange', listener);
  }
  visibilityListeners = [];
  for (const [file] of Object.values(VIEW_MODULES)) vi.doUnmock(`${JS}/${file}`);
  vi.doUnmock('/design-system/js/header-contrib.js');
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
  window.history.replaceState(null, '', '/');
});

/**
 * Boot app.js at `hash` on the toolbar index.html serves, against an /api/info
 * that answers `info`, or rejects when `info` is null.
 * @param {Record<string, any> | null} info
 * @param {string} [hash]
 * @param {any} [alarms] - the answer of the alarm route, or a function giving it
 */
async function boot(info, hash = '', alarms = {}) {
  document.body.innerHTML = `
    <div id="cf-panel">
      <header id="cf-toolbar">
        <button id="cf-sidebar-toggle" type="button" disabled></button>
        <div class="osprey-seg" id="cf-views" role="group" aria-label="View">
          <button type="button" data-view="explore" aria-pressed="true">Explore</button>
        </div>
        <span id="cf-result-count" hidden></span>
        <label id="cf-search-field"><input type="search" id="cf-search" disabled></label>
      </header>
      <div id="view-container"></div>
    </div>
    <button id="pipeline-badge"></button>
    <div id="pipeline-dropdown"></div>`;
  window.history.replaceState(null, '', `/${hash}`);

  /** @type {Record<string, { mount: import('vitest').Mock, unmount: import('vitest').Mock }>} */
  const views = {};
  for (const [view, [file, mountName, unmountName]] of Object.entries(VIEW_MODULES)) {
    views[view] = { mount: vi.fn(), unmount: vi.fn() };
    vi.doMock(`${JS}/${file}`, () => ({
      [mountName]: views[view].mount,
      [unmountName]: views[view].unmount,
    }));
  }

  /** Every contribution published to the hub's tile bar. */
  const contributions = /** @type {any[][]} */ ([]);
  vi.doMock('/design-system/js/header-contrib.js', () => ({
    contributeHeader: (/** @type {any[]} */ items) => contributions.push(items),
    onHeaderAction: vi.fn(),
  }));

  let answer = info;
  /** @type {string[]} */
  const alarmReads = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {any} */ url, /** @type {any} */ opts) => {
      const path = String(url);
      if (path.includes('/api/info') && answer === null) throw new TypeError('network');
      if (path.includes('/api/pipeline') && opts?.method === 'PUT') {
        // The answer after a switch carries `directory_after_switch`, or no descriptor.
        const { directory_after_switch: directory, ...rest } = answer ?? {};
        delete rest.directory;
        const { pipeline_type: type } = JSON.parse(opts.body);
        answer = directory
          ? { ...rest, directory, pipeline_type: type }
          : { ...rest, pipeline_type: type };
      }
      if (path.includes('/api/directory/alarms')) {
        alarmReads.push(path);
        const body = typeof alarms === 'function' ? await alarms() : alarms;
        return { ok: true, json: async () => body };
      }
      return { ok: true, json: async () => (path.includes('/api/info') ? answer : {}) };
    })
  );

  const add = window.addEventListener.bind(window);
  vi.spyOn(window, 'addEventListener').mockImplementation(
    (/** @type {any} */ type, /** @type {any} */ listener, /** @type {any} */ options) => {
      if (type === 'hashchange') hashListeners.push(listener);
      add(type, listener, options);
    }
  );
  const addToDocument = document.addEventListener.bind(document);
  vi.spyOn(document, 'addEventListener').mockImplementation(
    (/** @type {any} */ type, /** @type {any} */ listener, /** @type {any} */ options) => {
      if (type === 'visibilitychange') visibilityListeners.push(listener);
      addToDocument(type, listener, options);
    }
  );

  vi.resetModules();
  const { state } = await import(`${JS}/state.js`);
  await import(`${JS}/app.js`);
  await vi.waitFor(() => expect(hashListeners).toHaveLength(1));
  return { state, views, contributions, alarmReads };
}

/** Follow the current hash the way the browser does after a button sets it. */
function followHash() {
  for (const listener of hashListeners) listener(new Event('hashchange'));
}

/** @returns {HTMLButtonElement[]} */
function buttons() {
  return /** @type {HTMLButtonElement[]} */ ([
    ...document.querySelectorAll('#cf-views button[data-view]'),
  ]);
}

/**
 * A button's label: its own text, without a count pill.
 * @param {HTMLButtonElement} button
 */
function label(button) {
  return [...button.childNodes]
    .filter((node) => node.nodeType === Node.TEXT_NODE)
    .map((node) => node.textContent)
    .join('');
}

/** @returns {string[]} the buttons as `view:label`. */
function names() {
  return buttons().map((button) => `${button.dataset.view}:${label(button)}`);
}

/** @returns {string[]} the views of the pressed buttons. */
function pressed() {
  return buttons()
    .filter((button) => button.getAttribute('aria-pressed') === 'true')
    .map((button) => /** @type {string} */ (button.dataset.view));
}

/** @param {string} view */
function click(view) {
  const button = buttons().find((candidate) => candidate.dataset.view === view);
  if (!button) throw new Error(`no button for ${view}`);
  button.click();
  followHash();
}

/** @param {string} hash */
function go(hash) {
  window.history.replaceState(null, '', `/${hash}`);
  followHash();
}

const WITH = ['explore:Curated', 'allpvs:All PVs', 'feedback:Feedback'];
const WITHOUT = ['explore:Explore', 'feedback:Feedback'];
const GRAPH_WITH = ['explore:Curated', 'allpvs:All PVs'];
const GRAPH_WITHOUT = ['explore:Explore'];

describe('with the directory descriptor', () => {
  test('the seg reads Curated · All PVs · Feedback, Curated pressed', async () => {
    await boot(WITH_DIRECTORY);
    expect(names()).toEqual(WITH);
    expect(pressed()).toEqual(['explore']);
  });

  test('a button routes when clicked, and the pressed state follows', async () => {
    const { views } = await boot(WITH_DIRECTORY);
    click('allpvs');
    expect(location.hash).toBe('#/allpvs');
    expect(views.allpvs.mount).toHaveBeenCalledTimes(1);
    expect(pressed()).toEqual(['allpvs']);
    click('feedback');
    expect(views.feedback.mount).toHaveBeenCalledTimes(1);
    expect(pressed()).toEqual(['feedback']);
  });

  test.each(['', '#', '#/', '#constructor', '#/nowhere'])(
    'the hash %j lands on Curated, pressed',
    async (hash) => {
      const { views } = await boot(WITH_DIRECTORY, hash);
      expect(views.explore.mount).toHaveBeenCalledTimes(1);
      expect(pressed()).toEqual(['explore']);
    }
  );

  test.each(['#/pv/SR01%3AX', '#/ioc/ioc-sr01'])('%s presses All PVs', async (hash) => {
    await boot(WITH_DIRECTORY, hash);
    expect(pressed()).toEqual(['allpvs']);
  });
});

describe('the graph paradigm', () => {
  test.each([
    ['with', WITH_DIRECTORY, GRAPH_WITH],
    ['without', WITHOUT_DIRECTORY, GRAPH_WITHOUT],
  ])('has no Feedback button %s the descriptor', async (_, info, expected) => {
    await boot({ ...info, pipeline_type: 'graph' });
    expect(names()).toEqual(expected);
  });

  test('a #feedback route lands on Curated, Feedback never mounted', async () => {
    const { views } = await boot({ ...WITH_DIRECTORY, pipeline_type: 'graph' }, '#feedback');
    expect(views.feedback.mount).not.toHaveBeenCalled();
    expect(views.explore.mount).toHaveBeenCalledTimes(1);
    expect(pressed()).toEqual(['explore']);
  });
});

describe('without the directory descriptor', () => {
  test('the seg keeps Explore | Feedback', async () => {
    await boot(WITHOUT_DIRECTORY);
    expect(names()).toEqual(WITHOUT);
    expect(pressed()).toEqual(['explore']);
  });

  test('the buttons still route', async () => {
    const { views } = await boot(WITHOUT_DIRECTORY);
    click('feedback');
    expect(views.feedback.mount).toHaveBeenCalledTimes(1);
    expect(pressed()).toEqual(['feedback']);
  });

  test('a directory route or an unknown name falls back to Explore', async () => {
    const { views } = await boot(WITHOUT_DIRECTORY, '#feedback');
    for (const hash of ['#allpvs', '#/pv/A', '#constructor']) {
      go(hash);
      expect(views.allpvs.mount).not.toHaveBeenCalled();
      expect(views.pv.mount).not.toHaveBeenCalled();
      expect(pressed()).toEqual(['explore']);
    }
  });

  test('a failed /api/info leaves Explore | Feedback', async () => {
    await boot(null);
    expect(names()).toEqual(WITHOUT);
  });

  test('a stray directoryLabel event adds no button', async () => {
    const { state } = await boot(WITHOUT_DIRECTORY);
    state.emit('directoryLabel', 'Channel Finder');
    expect(names()).toEqual(WITHOUT);
  });
});

describe('the All PVs label (D20)', () => {
  const NAMED = { ...WITH_DIRECTORY, directory: { ...DESCRIPTOR, label: 'Channel Finder' } };

  test('comes from the descriptor', async () => {
    await boot(NAMED);
    expect(names()[1]).toBe('allpvs:Channel Finder');
  });

  test('follows a directoryLabel event', async () => {
    const { state } = await boot(NAMED);
    state.emit('directoryLabel', 'Archiver Appliance');
    expect(names()[1]).toBe('allpvs:Archiver Appliance');
  });

  test('the button stays pressed when the label changes on its own page', async () => {
    const { state } = await boot(NAMED, '#allpvs');
    state.emit('directoryLabel', 'Archiver Appliance');
    expect(pressed()).toEqual(['allpvs']);
  });

  test('an event keeps the buttons in place', async () => {
    const { state } = await boot(NAMED);
    const before = buttons();
    state.emit('directoryLabel', 'Channel Finder');
    state.emit('directoryLabel', 'Archiver Appliance');
    expect(buttons()).toEqual(before);
    buttons().forEach((button, index) => expect(button).toBe(before[index]));
  });

  test('is text, never markup', async () => {
    const label = '<img src=x onerror=alert(1)>';
    const { state } = await boot({ ...WITH_DIRECTORY, directory: { ...DESCRIPTOR, label } });
    expect(names()[1]).toBe(`allpvs:${label}`);
    expect(document.querySelector('#cf-views img')).toBeNull();
    state.emit('directoryLabel', '<b>x</b>');
    expect(names()[1]).toBe('allpvs:<b>x</b>');
    expect(document.querySelector('#cf-views b')).toBeNull();
  });
});

describe('a pipeline switch re-reads the descriptor', () => {
  /**
   * Switch the pipeline through the dropdown the panel builds.
   * @param {string} pipeline
   */
  async function switchTo(pipeline) {
    const item = /** @type {HTMLElement} */ (
      document.querySelector(`[data-pipeline="${pipeline}"]`)
    );
    item.click();
    await vi.waitFor(() =>
      expect(document.getElementById('pipeline-badge')?.textContent).toContain(
        pipeline.toUpperCase()
      )
    );
  }

  const PIPELINES = { available_pipelines: ['hierarchical', 'graph'] };

  test('the scope buttons appear when the answer starts carrying it', async () => {
    await boot({ ...WITHOUT_DIRECTORY, ...PIPELINES, directory_after_switch: DESCRIPTOR });
    await switchTo('graph');
    await vi.waitFor(() => expect(names()).toEqual(GRAPH_WITH));
    expect(pressed()).toEqual(['explore']);
  });

  test('they leave again when it stops', async () => {
    await boot({ ...WITH_DIRECTORY, ...PIPELINES });
    await switchTo('graph');
    await vi.waitFor(() => expect(names()).toEqual(GRAPH_WITHOUT));
  });

  test('a switch to graph on Feedback drops the button and lands on Explore', async () => {
    const { views } = await boot({ ...WITHOUT_DIRECTORY, ...PIPELINES }, '#feedback');
    expect(views.feedback.mount).toHaveBeenCalledTimes(1);
    await switchTo('graph');
    await vi.waitFor(() => expect(names()).toEqual(GRAPH_WITHOUT));
    expect(views.feedback.unmount).toHaveBeenCalled();
    expect(pressed()).toEqual(['explore']);
  });

  test('the new buttons route after the switch', async () => {
    const { views } = await boot({
      ...WITHOUT_DIRECTORY,
      ...PIPELINES,
      directory_after_switch: DESCRIPTOR,
    });
    await switchTo('graph');
    await vi.waitFor(() => expect(names()).toEqual(GRAPH_WITH));
    click('allpvs');
    expect(views.allpvs.mount).toHaveBeenCalledTimes(1);
    expect(pressed()).toEqual(['allpvs']);
  });
});

describe('state.scopeQuery across a tab switch', () => {
  test('All PVs to Curated and back', async () => {
    const { state, views } = await boot(WITH_DIRECTORY, '#allpvs');
    /** @type {string[]} */
    const seenByCurated = [];
    views.explore.mount.mockImplementation(() => seenByCurated.push(state.scopeQuery));
    /** @type {string[]} */
    const seenByAllPvs = [];
    views.allpvs.mount.mockImplementation(() => seenByAllPvs.push(state.scopeQuery));

    state.scopeQuery = 'beam current';
    click('explore');
    expect(seenByCurated).toEqual(['beam current']);

    state.scopeQuery = 'orbit';
    click('allpvs');
    expect(seenByAllPvs).toEqual(['orbit']);
    expect(state.scopeQuery).toBe('orbit');
  });
});

describe('the tile bar', () => {
  test.each([
    ['with', WITH_DIRECTORY],
    ['without', WITHOUT_DIRECTORY],
  ])('receives no nav contribution %s the descriptor', async (_, info) => {
    const { state, contributions } = await boot(info, '#allpvs');
    state.emit('directoryLabel', 'Channel Finder');
    click('feedback');
    expect(contributions.flat().filter((item) => item?.kind === 'nav')).toEqual([]);
  });
});

describe('the Alarms tab', () => {
  const ALARM_DIRECTORY = {
    ...WITH_DIRECTORY,
    directory: { ...DESCRIPTOR, configured_services: ['alarm_state'] },
  };
  const WITH_ALARMS = ['explore:Curated', 'allpvs:All PVs', 'alarms:Alarms', 'feedback:Feedback'];
  const SOURCE = { target: 'va', service: 'alarm_replay', label: 'alarm_replay', rehearsal: false };

  /**
   * An alarm route answer with `counts`.
   * @param {Record<string, any>} counts
   */
  const counted = (counts) => ({ outcome: 'found', source: SOURCE, caveats: [], counts, rows: [] });

  /** @returns {HTMLElement|null} */
  const pill = () => document.querySelector('#cf-views button[data-view="alarms"] .osprey-count');

  test('with an alarm-state service the seg reads Curated · All PVs · Alarms · Feedback', async () => {
    await boot(ALARM_DIRECTORY);
    expect(names()).toEqual(WITH_ALARMS);
  });

  test("without one the seg is today's, and #/alarms falls back to Curated", async () => {
    const { views } = await boot(WITH_DIRECTORY, '#/alarms');
    expect(names()).toEqual(WITH);
    expect(views.alarms.mount).not.toHaveBeenCalled();
    expect(pressed()).toEqual(['explore']);
  });

  test('#/alarms mounts the view and presses the tab', async () => {
    const { views } = await boot(ALARM_DIRECTORY, '#/alarms');
    expect(views.alarms.mount).toHaveBeenCalledTimes(1);
    expect(pressed()).toEqual(['alarms']);
    click('feedback');
    expect(views.alarms.unmount).toHaveBeenCalled();
  });

  test('the pill shows the active count in the worst tone', async () => {
    const { alarmReads } = await boot(
      ALARM_DIRECTORY,
      '',
      counted({ MAJOR: 2, INVALID: 1, MINOR: 1, OK: 7 })
    );
    await vi.waitFor(() => expect(pill()?.textContent).toBe('4'));
    expect(pill()?.dataset.severity).toBe('invalid');
    expect(names()).toEqual(WITH_ALARMS);
    expect(new URLSearchParams(alarmReads[0].split('?')[1]).get('limit')).toBe('0');
  });

  test('the pill follows the alarmCounts event, and disappears at zero', async () => {
    const { state } = await boot(ALARM_DIRECTORY, '', counted({ MINOR: 1 }));
    await vi.waitFor(() => expect(pill()?.textContent).toBe('1'));
    expect(pill()?.dataset.severity).toBe('minor');
    state.emit('alarmCounts', counted({ MAJOR: 1, OK: 2 }));
    expect(pill()?.textContent).toBe('1');
    expect(pill()?.dataset.severity).toBe('major');
    state.emit('alarmCounts', counted({ OK: 3 }));
    expect(pill()).toBeNull();
  });

  test('the button names its count apart from its label', async () => {
    const { state } = await boot(ALARM_DIRECTORY, '', counted({ MAJOR: 2 }));
    const button = () => document.querySelector('#cf-views button[data-view="alarms"]');
    await vi.waitFor(() => expect(pill()?.textContent).toBe('2'));
    expect(button()?.getAttribute('aria-label')).toBe('Alarms, 2 active');
    state.emit('alarmCounts', counted({ OK: 3 }));
    expect(button()?.hasAttribute('aria-label')).toBe(false);
  });

  test('the pill is text, never markup', async () => {
    const { state } = await boot(ALARM_DIRECTORY);
    state.emit('alarmCounts', counted({ '<img src=x onerror=alert(1)>': 2, MAJOR: '<b>1</b>' }));
    expect(document.querySelector('#cf-views img, #cf-views b')).toBeNull();
    const shown = pill();
    if (shown) expect(shown.children).toHaveLength(0);
  });

  test('a directoryLabel event keeps the pill', async () => {
    const { state } = await boot(ALARM_DIRECTORY, '', counted({ MAJOR: 1 }));
    await vi.waitFor(() => expect(pill()?.textContent).toBe('1'));
    state.emit('directoryLabel', 'Channel Finder');
    expect(pill()?.textContent).toBe('1');
    expect(names()[1]).toBe('allpvs:Channel Finder');
  });

  test('the counts are not read alone while the Alarms view reads the list', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    const { alarmReads } = await boot(ALARM_DIRECTORY, '', counted({ MAJOR: 1 }));
    await vi.waitFor(() => expect(alarmReads).toHaveLength(1));
    go('#/alarms');
    vi.advanceTimersByTime(15000 * 2);
    document.dispatchEvent(new Event('visibilitychange'));
    await Promise.resolve();
    expect(alarmReads).toHaveLength(1);
  });

  test('a panel opened on #/alarms does not read the counts alone', async () => {
    const { views, alarmReads } = await boot(ALARM_DIRECTORY, '#/alarms', counted({ MAJOR: 1 }));
    expect(views.alarms.mount).toHaveBeenCalledTimes(1);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(alarmReads).toEqual([]);
  });

  test('the counts are read again when the panel is shown on another view', async () => {
    const { alarmReads } = await boot(ALARM_DIRECTORY, '', counted({ MAJOR: 1 }));
    await vi.waitFor(() => expect(alarmReads).toHaveLength(1));
    document.dispatchEvent(new Event('visibilitychange'));
    await vi.waitFor(() => expect(alarmReads).toHaveLength(2));
    expect(new URLSearchParams(alarmReads[1].split('?')[1]).get('limit')).toBe('0');
  });

  /**
   * An alarm route whose first answer waits for `release`; later ones answer at once.
   * @param {any} later
   */
  function heldFirst(later) {
    /** @type {(value: any) => void} */
    let release = () => {};
    const held = new Promise((resolve) => {
      release = resolve;
    });
    let calls = 0;
    return {
      alarms: () => (calls++ === 0 ? held : later),
      release: (/** @type {any} */ v) => release(v),
    };
  }

  test('an older counts answer never overwrites a newer read', async () => {
    const { alarms, release } = heldFirst(counted({ MINOR: 1 }));
    const { alarmReads } = await boot(ALARM_DIRECTORY, '', alarms);
    await vi.waitFor(() => expect(alarmReads).toHaveLength(1));
    document.dispatchEvent(new Event('visibilitychange'));
    await vi.waitFor(() => expect(pill()?.textContent).toBe('1'));
    release(counted({ MAJOR: 5 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(pill()?.textContent).toBe('1');
    expect(pill()?.dataset.severity).toBe('minor');
  });

  test("an older counts answer never overwrites the Alarms view's poll", async () => {
    const { alarms, release } = heldFirst(counted({ MINOR: 1 }));
    const { state, alarmReads } = await boot(ALARM_DIRECTORY, '', alarms);
    await vi.waitFor(() => expect(alarmReads).toHaveLength(1));
    go('#/alarms');
    state.emit('alarmCounts', counted({ MINOR: 1 }));
    release(counted({ MAJOR: 5 }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(pill()?.textContent).toBe('1');
    expect(pill()?.dataset.severity).toBe('minor');
  });

  test('without an alarm-state service the alarm route is not read', async () => {
    const { alarmReads } = await boot(WITH_DIRECTORY);
    await Promise.resolve();
    expect(alarmReads).toEqual([]);
  });
});

describe('the Archive tab', () => {
  const BOTH = {
    ...WITH_DIRECTORY,
    directory: { ...DESCRIPTOR, configured_services: ['alarm_state', 'archive'] },
  };

  test('with both services the seg reads Curated · All PVs · Alarms · Archive · Feedback', async () => {
    await boot(BOTH);
    expect(names()).toEqual([
      'explore:Curated',
      'allpvs:All PVs',
      'alarms:Alarms',
      'archive:Archive',
      'feedback:Feedback',
    ]);
  });

  test('without an archive there is no tab, and #/archive falls back to Curated', async () => {
    const { views } = await boot(WITH_DIRECTORY, '#/archive');
    expect(names()).toEqual(WITH);
    expect(views.archive.mount).not.toHaveBeenCalled();
    expect(pressed()).toEqual(['explore']);
  });

  test('with only an archive the seg reads Curated · All PVs · Archive · Feedback', async () => {
    await boot({
      ...WITH_DIRECTORY,
      directory: { ...DESCRIPTOR, configured_services: ['archive'] },
    });
    expect(names()).toEqual([
      'explore:Curated',
      'allpvs:All PVs',
      'archive:Archive',
      'feedback:Feedback',
    ]);
  });

  test('the order is fixed, whatever order the descriptor lists the services in', async () => {
    await boot({
      ...WITH_DIRECTORY,
      directory: { ...DESCRIPTOR, configured_services: ['archive', 'alarm_state'] },
    });
    expect(names()).toEqual([
      'explore:Curated',
      'allpvs:All PVs',
      'alarms:Alarms',
      'archive:Archive',
      'feedback:Feedback',
    ]);
  });

  test('#/archive mounts the view and presses the tab', async () => {
    const { views } = await boot(BOTH, '#/archive');
    expect(views.archive.mount).toHaveBeenCalledTimes(1);
    expect(pressed()).toEqual(['archive']);
    click('feedback');
    expect(views.archive.unmount).toHaveBeenCalled();
  });
});
