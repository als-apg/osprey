// @ts-check
/**
 * Artifact tiles, one tile pinned to one artifact (artifact-tiles.js):
 *   npx vitest run tests/interfaces/web_terminal/artifact-tiles.test.mjs
 *
 * Pins the key namespace (`artifact:<id>` round-trips; nothing else is one),
 * the frame (the Artifacts panel's gallery, built only while it answers), the
 * viewer protocol (a tile's ready is answered with ITS artifact, a deleted one
 * with none; another frame's ready is not the tile's), the forwarded changes
 * and the bar they repaint, and the hydrate and release of tiles the dock
 * holds or no longer holds, and pane-window's frame gate.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const m = vi.hoisted(() => ({
  sendSelect: vi.fn(),
  sendChanged: vi.fn(),
  sendAction: vi.fn(),
  contributeLocal: vi.fn(),
  focusPanel: vi.fn(),
  setServiceMenuExtras: vi.fn(),
  fetchArtifact: vi.fn(async (/** @type {string} */ id) => ({ id, title: `read ${id}` })),
  /** Frames by key, as dock-iframe would manage them. @type {Map<string, HTMLIFrameElement>} */
  managed: new Map(),
  layout: /** @type {any} */ (null),
  detached: new Set(),
  /** Each tile's params, as the layout holder keeps them. @type {Map<string, any>} */
  params: new Map(),
  /** The active session's transcript. @type {string | null} */
  activeTranscript: null,
  /** @type {Set<(change: {reason: string}) => void>} */
  sessionListeners: new Set(),
}));

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-iframe-sync.js', () => ({
  sendArtifactSelectToIframe: m.sendSelect,
  sendArtifactChangedToIframe: m.sendChanged,
  sendArtifactActionToIframe: m.sendAction,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/tile-header-contrib.js', () => ({
  contributeLocal: m.contributeLocal,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js', () => ({
  managedIframe: (/** @type {string} */ key) => {
    const iframe = m.managed.get(key);
    return iframe ? { iframe, title: 'x' } : null;
  },
  focusPanel: m.focusPanel,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-sync.js', () => ({
  withEchoSuppressed: (/** @type {() => any} */ fn) => fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-view.js', () => ({
  DOCK_VIEW_EVENT: 'osprey-dock-view-changed',
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-menu-policy.js', () => ({
  setServiceMenuExtras: m.setServiceMenuExtras,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-iframe-factory.js', () => ({
  createPanelIframe: createFrame,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/session-model.js', () => ({
  activeRow: () => ({ transcript_id: m.activeTranscript }),
  sessionOfArtifact: () => null,
  subscribe: (/** @type {(change: {reason: string}) => void} */ fn) => {
    m.sessionListeners.add(fn);
    return () => m.sessionListeners.delete(fn);
  },
  switchTo: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js', () => ({
  getLayoutHolder: () => ({
    layout: m.layout,
    panelParams: (/** @type {string} */ key) => m.params.get(key),
    setPanelParams: (/** @type {string} */ key, /** @type {object} */ patch) => {
      m.params.set(key, { ...m.params.get(key), ...patch });
      return true;
    },
  }),
  isDetached: (/** @type {string} */ key) => m.detached.has(key),
  panelIdOf: (/** @type {string} */ key) => `iframe:${key}`,
}));
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifact-listing.js',
  () => ({
    ARTIFACTS_BASE: '/panel/artifacts',
    fetchArtifact: m.fetchArtifact,
  })
);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js', () => ({
  showConfirm: vi.fn(),
}));

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/artifact-tiles.js')} */
let mod;

const GALLERY = { id: 'artifacts', label: 'ARTIFACTS', configEndpoint: '/x' };
/** The Artifacts panel's state, as panel-manager holds it. */
const galleryState = { healthy: true, url: /** @type {string | null} */ ('/g/') };

/** The frame factory: a frame under the key, adopted by the dock adapter. */
const createFrame = vi.hoisted(() =>
  vi.fn(
    (
      /** @type {any} */ _panel,
      /** @type {string} */ key,
      /** @type {{iframe: HTMLIFrameElement | null}} */ state
    ) => {
      const iframe = document.createElement('iframe');
      document.body.append(iframe);
      state.iframe = iframe;
      m.managed.set(key, iframe);
    }
  )
);
const setMenuExtras = m.setServiceMenuExtras;

const PLOT = { id: 'a1', title: 'Beam size', filename: 'b.html', size_bytes: 10 };

/** A layout docking each key in a tile of its own. @param {...string} keys */
const docked = (...keys) => ({
  split: 'row',
  items: keys.map((key, i) => ({
    tile: `t${i}`,
    tabs: [key],
    showing: key,
    size: 1 / keys.length,
  })),
});

/** The window message a frame's viewer sends when it is ready. @param {Window | null} source */
function ready(source) {
  window.dispatchEvent(
    new MessageEvent('message', {
      origin: window.location.origin,
      source,
      data: { type: 'osprey-artifact-ready' },
    })
  );
}

/** The items the last bar contribution under `key` carried. @param {string} key */
const barOf = (key) => m.contributeLocal.mock.calls.filter(([k]) => k === key).at(-1)?.[1];

beforeEach(async () => {
  vi.resetModules();
  vi.clearAllMocks();
  m.managed.clear();
  m.detached.clear();
  m.layout = null;
  m.params.clear();
  m.activeTranscript = null;
  m.sessionListeners.clear();
  galleryState.healthy = true;
  document.body.replaceChildren();
  mod = await import(`${JS}/artifact-tiles.js`);
  mod.initArtifactTiles({
    panels: [GALLERY],
    panelState: { artifacts: galleryState },
    contentEl: document.body,
  });
});

afterEach(() => mod.resetArtifactTiles());

describe('the key namespace', () => {
  test('an artifact id round-trips through its tile key', () => {
    const key = mod.artifactTileKey('0a1b2c3d4e5f');
    expect(key).toBe('artifact:0a1b2c3d4e5f');
    expect(mod.isArtifactTileKey(key)).toBe(true);
    expect(mod.artifactIdOf(key)).toBe('0a1b2c3d4e5f');
  });

  test.each([['artifacts'], ['artifact:'], ['artifact:a/b'], ['iframe:artifact:a1'], [null], [7]])(
    '%s is no artifact tile key',
    (key) => {
      expect(mod.isArtifactTileKey(key)).toBe(false);
      expect(mod.artifactIdOf(key)).toBeNull();
    }
  );

  test('the dock keeps an artifact tile while the Artifacts panel is known and it is docked', () => {
    expect(mod.artifactTileStays('artifact:a1', new Set(['artifacts']))).toBe(true);
    expect(mod.artifactTileStays('artifact:a1', new Set(['ariel']))).toBe(false);
    m.detached.add('artifact:a1');
    expect(mod.artifactTileStays('artifact:a1', new Set(['artifacts']))).toBe(false);
    expect(mod.artifactTileStays('artifacts', new Set(['artifacts']))).toBe(false);
  });

  test('a held row with an id no key can carry gets no tile', () => {
    expect(mod.holdArtifactTile({ id: 'a/b', entry: null, title: 'x' })).toBeNull();
  });
});

describe('the frame and the viewer protocol', () => {
  test('a held artifact names its tab and fills its bar before any frame', () => {
    expect(mod.holdArtifactTile({ id: 'a1', entry: PLOT, title: '' })).toBe('artifact:a1');
    expect(mod.artifactTileTitle('artifact:a1')).toBe('Beam size');
    expect(mod.artifactTileTitle('ariel')).toBeNull();
    expect(barOf('artifact:a1')).toEqual(
      expect.arrayContaining([expect.objectContaining({ kind: 'crumb', leaf: 'Beam size' })])
    );
    expect(setMenuExtras).toHaveBeenCalledWith('artifact:a1', expect.any(Function));
    expect(m.fetchArtifact).not.toHaveBeenCalled();
  });

  test('a row with no entry names its tab with its own title and reads the entry', async () => {
    mod.holdArtifactTile({ id: 'a2', entry: null, title: 'Code' });
    expect(mod.artifactTileTitle('artifact:a2')).toBe('Code');
    await vi.waitFor(() => expect(m.fetchArtifact).toHaveBeenCalledWith('a2'));
    await vi.waitFor(() => expect(mod.artifactTileTitle('artifact:a2')).toBe('read a2'));
  });

  test('the frame is the gallery’s, built only while the Artifacts panel answers', () => {
    mod.holdArtifactTile({ id: 'a1', entry: PLOT, title: '' });
    galleryState.healthy = false;
    expect(mod.canLoadArtifactFrame('artifact:a1')).toBe(false);
    expect(mod.loadArtifactFrame('artifact:a1')).toBe(false);
    galleryState.healthy = true;
    expect(mod.canLoadArtifactFrame('artifact:a1')).toBe(true);
    expect(mod.loadArtifactFrame('artifact:a1', { dock: false })).toBe(true);
    expect(createFrame).toHaveBeenCalledExactlyOnceWith(
      { ...GALLERY, label: 'Beam size' },
      'artifact:a1',
      expect.objectContaining({ url: '/g/' }),
      document.body,
      { dock: false }
    );
    // One frame per tile.
    expect(mod.loadArtifactFrame('artifact:a1')).toBe(true);
    expect(createFrame).toHaveBeenCalledOnce();
    expect(mod.loadArtifactFrame('ariel')).toBe(false);
  });

  test('showing a tile frames it and focuses it', () => {
    mod.showArtifactTile('artifact:a1');
    expect(createFrame).toHaveBeenCalledOnce();
    expect(m.focusPanel).toHaveBeenCalledWith('artifact:a1');
  });

  test('a tile’s viewer is told its own artifact when ready; another frame’s ready is not its', () => {
    mod.holdArtifactTile({ id: 'a1', entry: PLOT, title: '' });
    mod.loadArtifactFrame('artifact:a1');
    mod.loadArtifactFrame('artifact:b2');
    const a1 = /** @type {HTMLIFrameElement} */ (m.managed.get('artifact:a1'));
    const b2 = /** @type {HTMLIFrameElement} */ (m.managed.get('artifact:b2'));
    ready(a1.contentWindow);
    expect(m.sendSelect).toHaveBeenCalledExactlyOnceWith(a1, 'a1');
    ready(b2.contentWindow);
    expect(m.sendSelect).toHaveBeenLastCalledWith(b2, 'b2');
    const other = document.createElement('iframe');
    document.body.append(other);
    ready(other.contentWindow);
    expect(m.sendSelect).toHaveBeenCalledTimes(2);
  });

  test('an update repaints the bar and reaches the viewer; a delete empties both', () => {
    mod.holdArtifactTile({ id: 'a1', entry: PLOT, title: '' });
    mod.loadArtifactFrame('artifact:a1');
    const frame = m.managed.get('artifact:a1');

    mod.onArtifactChanged('a1', 'updated', { id: 'a1', title: 'Beam size, fitted' });
    expect(barOf('artifact:a1')).toEqual(
      expect.arrayContaining([expect.objectContaining({ leaf: 'Beam size, fitted' })])
    );
    expect(m.sendChanged).toHaveBeenLastCalledWith(frame, 'a1', 'updated', {
      id: 'a1',
      title: 'Beam size, fitted',
    });

    mod.onArtifactChanged('a1', 'deleted');
    expect(m.sendChanged).toHaveBeenLastCalledWith(frame, 'a1', 'deleted', undefined);
    // The bar falls back to its disabled icons, the tile stays, and a reload
    // of its viewer shows the empty state.
    const bar = barOf('artifact:a1');
    expect(bar.some((/** @type {any} */ item) => item.kind === 'crumb')).toBe(false);
    expect(bar.every((/** @type {any} */ item) => item.disabled === true)).toBe(true);
    expect(/** @type {HTMLIFrameElement} */ (frame).isConnected).toBe(true);
    ready(/** @type {HTMLIFrameElement} */ (frame).contentWindow);
    expect(m.sendSelect).toHaveBeenLastCalledWith(frame, null);
    expect(setMenuExtras.mock.lastCall?.[1]()).toEqual([]);
  });

  test('a change to an artifact with no tile reaches nothing', () => {
    mod.onArtifactChanged('zz', 'deleted');
    expect(m.sendChanged).not.toHaveBeenCalled();
    expect(m.contributeLocal).not.toHaveBeenCalled();
  });

  test('the ⋯ rows send the viewer’s verbs to the tile’s own frame', () => {
    mod.holdArtifactTile({ id: 'a1', entry: PLOT, title: '' });
    mod.loadArtifactFrame('artifact:a1');
    const rows = setMenuExtras.mock.lastCall?.[1]();
    rows.find((/** @type {any} */ r) => r.label === 'Print…').run();
    expect(m.sendAction).toHaveBeenCalledWith(m.managed.get('artifact:a1'), 'a1', 'print');
  });
});

describe('an artifact from another session', () => {
  const T1 = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
  const T2 = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
  /** The bar's Info item. @param {string} key */
  const infoOf = (key) => barOf(key)?.find((/** @type {any} */ item) => item.id === 'info');
  /** @param {string} reason */
  const sessionChange = (reason) => {
    for (const fn of [...m.sessionListeners]) fn({ reason });
  };

  test('Info warns until the tile acknowledges it; the tile keeps that in its params', async () => {
    m.activeTranscript = T1;
    mod.holdArtifactTile({ id: 'a1', entry: { ...PLOT, owner: T2 }, title: '' });
    expect(infoOf('artifact:a1')).toMatchObject({ tone: 'warning' });

    const origin = await import(`${JS}/artifact-origin.js`);
    expect(origin.acknowledgeOrigin('artifact:a1', 'a1')).toBe(true);
    expect(m.params.get('artifact:a1')).toEqual({ [origin.ACK_PARAM]: 'a1' });
    expect(infoOf('artifact:a1')).not.toHaveProperty('tone');

    // Away to the owner and back: the params still say acknowledged.
    m.activeTranscript = T2;
    sessionChange('switched');
    expect(infoOf('artifact:a1')).not.toHaveProperty('tone');
    m.activeTranscript = T1;
    sessionChange('switched');
    expect(infoOf('artifact:a1')).not.toHaveProperty('tone');

    // A closed tile's params are gone: the bar warns again.
    m.params.delete('artifact:a1');
    document.dispatchEvent(new CustomEvent('osprey-dock-view-changed'));
    expect(infoOf('artifact:a1')).toMatchObject({ tone: 'warning' });
  });

  test('an artifact of the active session, or of none, never warns', () => {
    m.activeTranscript = T1;
    mod.holdArtifactTile({ id: 'a1', entry: { ...PLOT, owner: T1 }, title: '' });
    mod.holdArtifactTile({ id: 'a2', entry: { ...PLOT, id: 'a2', owner: null }, title: '' });
    expect(infoOf('artifact:a1')).not.toHaveProperty('tone');
    expect(infoOf('artifact:a2')).not.toHaveProperty('tone');
    // A new session with no transcript yet: an owned artifact is foreign.
    m.activeTranscript = null;
    sessionChange('new');
    expect(infoOf('artifact:a1')).toMatchObject({ tone: 'warning' });
    expect(infoOf('artifact:a2')).not.toHaveProperty('tone');
  });
});

describe('hydrate and release', () => {
  test('every docked artifact tile without a frame gets one; other keys are not this module’s', () => {
    m.layout = docked('artifact:a1', 'ariel', 'terminal');
    mod.hydrateArtifactTiles();
    expect(createFrame).toHaveBeenCalledOnce();
    expect(createFrame.mock.calls[0][1]).toBe('artifact:a1');
    mod.hydrateArtifactTiles();
    expect(createFrame).toHaveBeenCalledOnce();
  });

  test('a tile the dock no longer holds leaves the page on the next dock change, unless in its own window', () => {
    m.layout = docked('artifact:a1', 'artifact:b2');
    mod.hydrateArtifactTiles();
    const a1 = /** @type {HTMLIFrameElement} */ (m.managed.get('artifact:a1'));
    const b2 = /** @type {HTMLIFrameElement} */ (m.managed.get('artifact:b2'));
    m.layout = docked('terminal');
    m.detached.add('artifact:b2');
    document.dispatchEvent(new CustomEvent('osprey-dock-view-changed'));
    expect(a1.isConnected).toBe(false);
    expect(b2.isConnected).toBe(true);
    // Brought back (Undo, a session's layout), it is framed again.
    m.layout = docked('artifact:a1');
    mod.hydrateArtifactTiles();
    expect(createFrame).toHaveBeenCalledTimes(3);
    expect(m.managed.get('artifact:a1')?.isConnected).toBe(true);
  });

  test('pane-window’s frame gate and loader take artifact tiles here and panels there', () => {
    const panel = {
      canLoadFrame: vi.fn(/** @type {(id: string) => boolean} */ (() => true)),
      loadFrame: vi.fn(/** @type {(id: string) => void} */ (() => {})),
      other: 1,
    };
    const deps = mod.withArtifactFrames(panel);
    expect(deps.other).toBe(1);
    expect(deps.canLoadFrame('ariel')).toBe(true);
    deps.loadFrame('ariel');
    expect(panel.loadFrame).toHaveBeenCalledWith('ariel');
    galleryState.healthy = false;
    expect(deps.canLoadFrame('artifact:a1')).toBe(false);
    galleryState.healthy = true;
    deps.loadFrame('artifact:a1');
    expect(panel.loadFrame).toHaveBeenCalledOnce();
    expect(/** @type {any[]} */ (createFrame.mock.lastCall)?.[4]).toEqual({ dock: false });
  });
});
