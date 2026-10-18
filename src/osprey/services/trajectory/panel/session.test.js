// @ts-check
import { afterEach, describe, expect, it, vi } from 'vitest';

import { LOST_SESSION_CODES, createSession, sessionControls } from './session.js';

/** @typedef {import('./session.js').SessionSnapshot} SessionSnapshot */
/** @typedef {import('./session.js').HolderView} HolderView */

/** @type {HolderView} */
const ALICE = { owner: 'alice', kind: 'panel' };
/** @type {HolderView} */
const BOB = { owner: 'bob', kind: 'panel' };
/** @type {HolderView} */
const ALICE_AGENT = { owner: 'alice', kind: 'agent' };

/**
 * @param {HolderView | null} holder
 * @param {string | null} [action]
 * @returns {SessionSnapshot}
 */
function snap(holder, action = null) {
  return { holder, action: action == null ? null : { name: action, stopping: false } };
}

/** @param {string} clientId */
function grant(clientId, owner = 'alice') {
  return { client_id: clientId, owner, target: 'va' };
}

/**
 * Feed one snapshot event.
 * @param {import('./session.js').Session} session
 * @param {SessionSnapshot} snapshot
 */
function sendSnapshot(session, snapshot) {
  session.onEvent('snapshot', { type: 'snapshot', snapshot });
}

/**
 * Feed one taken_over event.
 * @param {import('./session.js').Session} session
 * @param {HolderView} previous
 * @param {HolderView} by
 */
function sendTakenOver(session, previous, by) {
  session.onEvent('taken_over', { type: 'taken_over', previous, by });
}

/** A session that opened and has seen a snapshot confirming it. */
function confirmedSession() {
  const session = createSession();
  session.begin('open');
  session.granted(grant('id-1'));
  sendSnapshot(session, snap(ALICE));
  return session;
}

describe('sessionControls', () => {
  it('offers Open session when no one holds', () => {
    const c = sessionControls(snap(null), { holding: false });
    expect(c).toMatchObject({ open: true, takeOver: false, release: false, stop: false });
    expect(c.running).toBeNull();
  });

  it('offers Take over while someone holds and no action runs', () => {
    const c = sessionControls(snap(BOB), { holding: false });
    expect(c).toMatchObject({ open: false, takeOver: true, release: false, stop: false });
  });

  it('names the running action and its owner and offers Stop, not Take over', () => {
    const c = sessionControls(snap(BOB, 'setup'), { holding: false });
    expect(c).toMatchObject({ open: false, takeOver: false, release: false, stop: true });
    expect(c.running).toEqual({ name: 'setup', owner: 'bob', kind: 'panel', stopping: false });
  });

  it('reports a stopping action as stopping', () => {
    /** @type {SessionSnapshot} */
    const s = { holder: ALICE_AGENT, action: { name: 'apply', stopping: true } };
    expect(sessionControls(s, { holding: false }).running).toEqual({
      name: 'apply',
      owner: 'alice',
      kind: 'agent',
      stopping: true,
    });
  });

  it('offers Release to the holder, and Stop while its action runs', () => {
    expect(sessionControls(snap(ALICE), { holding: true })).toMatchObject({
      holding: true,
      open: false,
      takeOver: false,
      release: true,
      stop: false,
    });
    expect(sessionControls(snap(ALICE, 'shot'), { holding: true })).toMatchObject({
      release: true,
      stop: true,
      takeOver: false,
    });
  });

  it('offers neither Open nor Take over while such a request is in flight', () => {
    expect(sessionControls(snap(null), { holding: false, pending: true })).toMatchObject({
      open: false,
      pending: true,
    });
    expect(sessionControls(snap(BOB), { holding: false, pending: true })).toMatchObject({
      takeOver: false,
    });
  });

  it('offers nothing before the first snapshot', () => {
    expect(sessionControls(null, { holding: false })).toMatchObject({
      open: false,
      takeOver: false,
      release: false,
      stop: false,
    });
  });
});

describe('createSession: holding', () => {
  it('holds only while it has a client_id', () => {
    const session = createSession();
    expect(session.holding()).toBe(false);
    expect(session.clientId()).toBeNull();
    session.begin('open');
    expect(session.holding()).toBe(false);
    expect(session.pending()).toBe(true);
    session.granted(grant('id-1'));
    expect(session.holding()).toBe(true);
    expect(session.clientId()).toBe('id-1');
    expect(session.owner()).toBe('alice');
    expect(session.pending()).toBe(false);
  });

  it('holds nothing after a failed open', () => {
    const session = createSession();
    session.begin('open');
    session.failed();
    expect(session.holding()).toBe(false);
    expect(session.pending()).toBe(false);
  });

  it('drops the id on an explicit Release', () => {
    const session = confirmedSession();
    session.released();
    expect(session.clientId()).toBeNull();
  });

  it('reports holding changes through onChange', () => {
    const onChange = vi.fn();
    const session = createSession({ onChange });
    session.begin('open');
    session.granted(grant('id-1'));
    expect(onChange).toHaveBeenCalledTimes(1);
    session.refused('not_session_holder');
    expect(onChange).toHaveBeenCalledTimes(2);
    session.refused('not_session_holder');
    expect(onChange).toHaveBeenCalledTimes(2);
  });

  it('uses the latest snapshot for its controls', () => {
    const session = confirmedSession();
    expect(session.controls()).toMatchObject({ holding: true, release: true });
    sendSnapshot(session, snap(ALICE, 'shot'));
    expect(session.controls()).toMatchObject({ stop: true, running: { name: 'shot' } });
  });
});

describe('createSession: refusals', () => {
  it.each(LOST_SESSION_CODES)('drops the id on a %s refusal', (code) => {
    const session = confirmedSession();
    expect(session.refused(code)).toBe(true);
    expect(session.holding()).toBe(false);
  });

  it.each(['limits_violation', 'continuous_running', 'action_stopped', 'no_step'])(
    'keeps the id on a %s refusal',
    (code) => {
      const session = confirmedSession();
      expect(session.refused(code)).toBe(false);
      expect(session.clientId()).toBe('id-1');
    },
  );
});

describe('createSession: snapshots', () => {
  it('keeps the id when a stale holder-null snapshot arrives after the POST answer', () => {
    const session = createSession();
    session.begin('open');
    session.granted(grant('id-1'));
    sendSnapshot(session, snap(null));
    expect(session.clientId()).toBe('id-1');
  });

  it('keeps the id when a stale non-panel holder snapshot arrives before confirmation', () => {
    const session = createSession();
    session.begin('takeover');
    sendTakenOver(session, ALICE_AGENT, ALICE);
    session.granted(grant('id-1'));
    sendSnapshot(session, snap(ALICE_AGENT));
    expect(session.clientId()).toBe('id-1');
  });

  it('ignores snapshots that arrive while its own POST is in flight', () => {
    const session = createSession();
    session.begin('open');
    sendSnapshot(session, snap(ALICE));
    session.granted(grant('id-1'));
    sendSnapshot(session, snap(null)); // stale: the in-flight one did not confirm
    expect(session.clientId()).toBe('id-1');
  });

  it('drops the id on a holder-null snapshot once a later snapshot confirmed it', () => {
    const session = confirmedSession();
    sendSnapshot(session, snap(null));
    expect(session.holding()).toBe(false);
  });

  it('drops the id on a non-panel holder once confirmed', () => {
    const session = confirmedSession();
    sendSnapshot(session, snap(ALICE_AGENT));
    expect(session.holding()).toBe(false);
  });

  it('drops the id on another owner holding once confirmed', () => {
    const session = confirmedSession();
    sendSnapshot(session, snap(BOB));
    expect(session.holding()).toBe(false);
  });

  it('confirms afresh after each grant', () => {
    const session = confirmedSession();
    session.released();
    session.begin('open');
    session.granted(grant('id-2'));
    sendSnapshot(session, snap(null)); // stale: from before the second open
    expect(session.clientId()).toBe('id-2');
  });

  it('ignores malformed and unrelated events', () => {
    const session = confirmedSession();
    session.onEvent('snapshot', null);
    session.onEvent('snapshot', { type: 'snapshot' });
    session.onEvent('shot', { type: 'shot', shot: {} });
    session.onEvent('step', { type: 'step', step: null });
    expect(session.clientId()).toBe('id-1');
  });
});

describe('createSession: taken_over', () => {
  it('drops the id on a taken_over it did not start', () => {
    const session = confirmedSession();
    sendTakenOver(session, ALICE, BOB);
    expect(session.holding()).toBe(false);
  });

  it('keeps the id on its own taken_over arriving before the POST answer', () => {
    const session = createSession();
    session.begin('takeover');
    sendTakenOver(session, BOB, ALICE);
    session.granted(grant('id-1'));
    expect(session.clientId()).toBe('id-1');
    sendSnapshot(session, snap(ALICE));
    expect(session.clientId()).toBe('id-1');
  });

  it('keeps the id on its own taken_over arriving after the POST answer', () => {
    const session = createSession();
    session.begin('takeover');
    session.granted(grant('id-1'));
    sendTakenOver(session, BOB, ALICE);
    expect(session.clientId()).toBe('id-1');
  });

  it('drops the id on a second taken_over after its own', () => {
    const session = createSession();
    session.begin('takeover');
    session.granted(grant('id-1'));
    sendTakenOver(session, BOB, ALICE);
    sendTakenOver(session, ALICE, BOB);
    expect(session.holding()).toBe(false);
  });

  it('drops a previously held id when a take-over it started fails', () => {
    const session = confirmedSession();
    session.begin('takeover');
    sendTakenOver(session, ALICE, BOB);
    session.failed();
    expect(session.holding()).toBe(false);
  });
});

describe('two tabs of one owner', () => {
  it('only the tab with the client_id holds; the other is offered Take over', () => {
    const tabA = createSession();
    const tabB = createSession();
    tabA.begin('open');
    tabA.granted(grant('id-a'));
    for (const tab of [tabA, tabB]) sendSnapshot(tab, snap(ALICE));

    expect(tabA.controls()).toMatchObject({ holding: true, release: true, takeOver: false });
    expect(tabB.controls()).toMatchObject({ holding: false, takeOver: true, open: false });
  });

  it('a take-over by the second tab drops the first and keeps the second', () => {
    const tabA = createSession();
    const tabB = createSession();
    tabA.begin('open');
    tabA.granted(grant('id-a'));
    for (const tab of [tabA, tabB]) sendSnapshot(tab, snap(ALICE));

    tabB.begin('takeover');
    // The service publishes taken_over then a snapshot before answering.
    for (const tab of [tabA, tabB]) sendTakenOver(tab, ALICE, ALICE);
    for (const tab of [tabA, tabB]) sendSnapshot(tab, snap(ALICE));
    tabB.granted(grant('id-b'));

    expect(tabA.holding()).toBe(false);
    expect(tabA.controls()).toMatchObject({ takeOver: true });
    expect(tabB.clientId()).toBe('id-b');
    expect(tabB.controls()).toMatchObject({ holding: true, release: true });
  });

  it('the second tab keeps its id when its events trail its POST answer', () => {
    const tabA = confirmedSession();
    const tabB = createSession();
    sendSnapshot(tabB, snap(ALICE));

    tabB.begin('takeover');
    tabB.granted(grant('id-b'));
    sendTakenOver(tabA, ALICE, ALICE);
    sendTakenOver(tabB, ALICE, ALICE);
    sendSnapshot(tabB, snap(ALICE));

    expect(tabA.holding()).toBe(false);
    expect(tabB.clientId()).toBe('id-b');
  });
});

describe('page unload', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('never releases on unload: no unload listener, no request, id kept', () => {
    const add = vi.spyOn(window, 'addEventListener');
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const session = confirmedSession();

    for (const type of ['beforeunload', 'pagehide', 'unload', 'visibilitychange']) {
      window.dispatchEvent(new Event(type));
    }

    const types = add.mock.calls.map((call) => call[0]);
    expect(types).not.toContain('beforeunload');
    expect(types).not.toContain('pagehide');
    expect(types).not.toContain('unload');
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(session.clientId()).toBe('id-1');
  });
});
