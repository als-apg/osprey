// @ts-check
/**
 * The session an artifact comes from, and its acknowledgement on the tile
 * (artifact-origin.js):
 *   npx vitest run tests/interfaces/web_terminal/artifact-origin.test.mjs
 *
 * Pins when an artifact is foreign (its owner is not the active session's
 * transcript; never without an owner; always on a new session with no
 * transcript yet), that an acknowledgement is the tile's params (an artifact
 * tile's own id, the Artifacts tile's last 50 ids), what the Info card is
 * told about the owning session, and which changes say "look again".
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const m = vi.hoisted(() => ({
  /** @type {{transcript_id: string | null} | null} */
  active: null,
  /** @type {{transcript_id: string, title?: string}[]} */
  rows: [],
  /** Each tile's params, as the layout holder keeps them. @type {Map<string, any>} */
  params: new Map(),
  /** The keys the dock has a panel for. @type {Set<string>} */
  docked: new Set(),
  /** @type {Set<(change: {reason: string}) => void>} */
  sessionListeners: new Set(),
  switchTo: vi.fn(async () => true),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/session-model.js', () => ({
  activeRow: () => m.active,
  sessionOfArtifact: (/** @type {{owner?: string | null}} */ entry) =>
    m.rows.find((row) => row.transcript_id === entry.owner) ?? null,
  subscribe: (/** @type {(change: {reason: string}) => void} */ fn) => {
    m.sessionListeners.add(fn);
    return () => m.sessionListeners.delete(fn);
  },
  switchTo: m.switchTo,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js', () => ({
  getLayoutHolder: () => ({
    panelParams: (/** @type {string} */ key) => m.params.get(key),
    setPanelParams: (/** @type {string} */ key, /** @type {object} */ patch) => {
      if (!m.docked.has(key)) return false;
      m.params.set(key, { ...m.params.get(key), ...patch });
      return true;
    },
  }),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-view.js', () => ({
  DOCK_VIEW_EVENT: 'osprey-dock-view-changed',
}));

const origin =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/artifact-origin.js');

const T1 = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const T2 = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';

/** An artifact made in a session. @param {string} id @param {string | null} [owner] */
const art = (id, owner = T2) => ({ id, title: `Plot ${id}`, owner });

/** @param {string} reason */
const sessionChange = (reason) => {
  for (const fn of [...m.sessionListeners]) fn({ reason });
};

beforeEach(() => {
  m.active = { transcript_id: T1 };
  m.rows = [
    { transcript_id: T1, title: 'Orbit check' },
    { transcript_id: T2, title: 'Tune scan' },
  ];
  m.docked = new Set(['artifacts', 'artifact:a1']);
});

afterEach(() => {
  origin.resetArtifactOrigin();
  m.params.clear();
  m.sessionListeners.clear();
  m.switchTo.mockClear();
});

describe('foreign', () => {
  test("an artifact is foreign when its owner is not the active session's transcript", () => {
    expect(origin.isForeign(art('a1', T2))).toBe(true);
    expect(origin.isForeign(art('a1', T1))).toBe(false);
  });

  test('an artifact with no owner never is', () => {
    expect(origin.isForeign(art('a1', null))).toBe(false);
    expect(origin.isForeign(/** @type {any} */ ({ id: 'a1' }))).toBe(false);
    m.active = { transcript_id: null };
    expect(origin.isForeign(art('a1', null))).toBe(false);
  });

  test('on a new session with no transcript yet, every owned artifact is', () => {
    m.active = { transcript_id: null };
    expect(origin.isForeign(art('a1', T1))).toBe(true);
    expect(origin.isForeign(art('a2', T2))).toBe(true);
  });
});

describe('the acknowledgement is the tile’s', () => {
  test('an artifact tile stores its own id; the notice goes for that tile only', () => {
    const plot = art('a1');
    expect(origin.needsOriginNotice(plot, 'artifact:a1')).toBe(true);
    expect(origin.acknowledgeOrigin('artifact:a1', 'a1')).toBe(true);
    expect(m.params.get('artifact:a1')).toEqual({ [origin.ACK_PARAM]: 'a1' });
    expect(origin.needsOriginNotice(plot, 'artifact:a1')).toBe(false);
    // The Artifacts tile showing the same artifact has acknowledged nothing.
    expect(origin.needsOriginNotice(plot, 'artifacts')).toBe(true);
  });

  test('the Artifacts tile keeps the last 50 ids, a repeat moving to the end', () => {
    for (let i = 0; i < 55; i += 1) origin.acknowledgeOrigin('artifacts', `id${i}`);
    origin.acknowledgeOrigin('artifacts', 'id10');
    const kept = m.params.get('artifacts')[origin.ACK_PARAM];
    expect(kept).toHaveLength(origin.ACK_LIMIT);
    expect(kept[0]).toBe('id5');
    expect(kept.at(-1)).toBe('id10');
    expect(kept.filter((/** @type {string} */ id) => id === 'id10')).toHaveLength(1);
    expect(origin.needsOriginNotice(art('id4'), 'artifacts')).toBe(true);
    expect(origin.needsOriginNotice(art('id54'), 'artifacts')).toBe(false);
  });

  test('a tile the dock does not hold acknowledges nothing and says nothing', () => {
    const heard = vi.fn();
    origin.onOriginChange(heard);
    expect(origin.acknowledgeOrigin('artifact:zz', 'zz')).toBe(false);
    expect(heard).not.toHaveBeenCalled();
    expect(origin.needsOriginNotice(art('zz'), 'artifact:zz')).toBe(true);
  });

  test('an artifact of the active session needs no notice, acknowledged or not', () => {
    expect(origin.needsOriginNotice(art('a1', T1), 'artifact:a1')).toBe(false);
    expect(origin.needsOriginNotice(null, 'artifacts')).toBe(false);
  });
});

describe('what the Info card is told', () => {
  test('a foreign artifact: the owning session by title, a link that switches to it', () => {
    const told = origin.artifactOrigin(art('a1'), 'artifact:a1');
    expect(told).toMatchObject({ title: 'Tune scan', listed: true, notice: true });
    told?.open?.();
    expect(m.switchTo).toHaveBeenCalledWith(m.rows[1]);
    told?.acknowledge();
    expect(origin.artifactOrigin(art('a1'), 'artifact:a1')?.notice).toBe(false);
  });

  test('an artifact of the active session: its title, no link, no notice', () => {
    const told = origin.artifactOrigin(art('a1', T1), 'artifact:a1');
    expect(told).toMatchObject({ title: 'Orbit check', listed: true, open: null, notice: false });
  });

  test('an owner the rows do not list: Unknown session, no link; no owner: nothing', () => {
    const told = origin.artifactOrigin(
      art('a1', 'cccccccc-cccc-4ccc-8ccc-cccccccccccc'),
      'artifacts'
    );
    expect(told).toMatchObject({
      title: 'Unknown session',
      listed: false,
      open: null,
      notice: true,
    });
    expect(origin.artifactOrigin(art('a1', null), 'artifacts')).toBeNull();
  });

  test('a listed session with no title is named as the sidebar names it', () => {
    m.rows = [{ transcript_id: T2 }];
    expect(origin.artifactOrigin(art('a1'), 'artifacts')?.title).toBe('Untitled session');
  });
});

describe('look again', () => {
  test('on a session change, a settled dock change and an acknowledgement; not on a pick', () => {
    const heard = vi.fn();
    const off = origin.onOriginChange(heard);
    for (const reason of ['pointer', 'switched', 'new', 'rows', 'named']) sessionChange(reason);
    expect(heard).toHaveBeenCalledTimes(5);
    sessionChange('pick');
    expect(heard).toHaveBeenCalledTimes(5);
    document.dispatchEvent(new CustomEvent('osprey-dock-view-changed'));
    expect(heard).toHaveBeenCalledTimes(6);
    origin.acknowledgeOrigin('artifacts', 'a1');
    expect(heard).toHaveBeenCalledTimes(7);
    off();
    sessionChange('switched');
    expect(heard).toHaveBeenCalledTimes(7);
  });
});
