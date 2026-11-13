// @ts-check
/**
 * The session tree's live artifact frames (sidebar-artifact-frames.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-artifact-frames.test.mjs
 *
 * The tree's own suite (sidebar-artifacts.test.mjs, "live updates") drives
 * these through a real tree; this file pins the module's edges: nothing is
 * followed while the panel is not served, a reconnect (not the first open)
 * re-fetches the summary and the focus, a delete clears the pick and the focus,
 * a focus is followed only for the active transcript, and every update and
 * delete reaches the artifact's own tile, shown or not.
 */

import { test, expect, beforeEach, vi } from 'vitest';

const model = vi.hoisted(() => ({
  activeRow: vi.fn(() => ({ transcript_id: 't1', session_key: 'k1' })),
  sessionOfArtifact: vi.fn(() => ({ transcript_id: 't1' })),
  isActiveTranscript: vi.fn((/** @type {string} */ t) => t === 't1'),
  refresh: vi.fn(),
  pick: vi.fn(() => 'a1'),
  setPick: vi.fn(),
  recordLastArtifact: vi.fn(),
}));
const store = vi.hoisted(() => ({
  fetchArtifact: vi.fn(async (/** @type {string} */ id) => ({ id, owner: 't1' })),
  fetchFocusedId: vi.fn(async () => /** @type {string | null} */ ('f1')),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/session-model.js', () => model);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifact-listing.js',
  () => store
);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/agent-surface.js', () => ({
  isTerminalSurface: () => false,
}));

const { createArtifactFrames } =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifact-frames.js');

/** @type {{focusId: string | null, shownId: string | null}} */
let state;
let served = true;
/** @type {Map<string, any>} */
let held;
const deps = () => {
  const data = /** @type {any} */ ({
    held: (/** @type {string} */ id) => held.get(id) ?? null,
    hold: (/** @type {any} */ e) => held.set(e.id, e),
    added: vi.fn(() => true),
    updated: vi.fn(() => true),
    deleted: vi.fn(),
  });
  return {
    isServed: () => served,
    data,
    state: () => state,
    inActiveSession: () => true,
    sendChanged: vi.fn(),
    tileChanged: vi.fn(),
    summarize: vi.fn(async () => {}),
    render: vi.fn(),
  };
};

beforeEach(() => {
  vi.clearAllMocks();
  state = { focusId: null, shownId: 'a1' };
  served = true;
  held = new Map();
});

test('nothing is followed while the panel is not served', () => {
  const d = deps();
  const frames = createArtifactFrames(d);
  served = false;
  frames.onFrame({ type: 'artifact', id: 'a2' });
  frames.onStreamOpen();
  expect(d.data.added).not.toHaveBeenCalled();
  expect(d.render).not.toHaveBeenCalled();
});

test('a new artifact lands in the active key too; a reconnect re-fetches, the first open does not', async () => {
  const d = deps();
  const frames = createArtifactFrames(d);
  frames.onFrame({ type: 'artifact', id: 'a2', owner: 't1' });
  expect(d.data.added).toHaveBeenCalledWith({ id: 'a2', owner: 't1' }, ['key:k1|']);
  expect(d.render).toHaveBeenCalledOnce();
  frames.onStreamOpen();
  expect(d.summarize).not.toHaveBeenCalled();
  frames.onStreamOpen();
  expect(d.summarize).toHaveBeenCalledWith(true);
  frames.reset();
  frames.onStreamOpen();
  expect(d.summarize).toHaveBeenCalledOnce();
});

test('an update and a delete of the shown artifact tell the viewer; a delete clears pick and focus', () => {
  const d = deps();
  const frames = createArtifactFrames(d);
  frames.onFrame({ type: 'artifact_updated', id: 'a1', status: 'done' });
  expect(d.sendChanged).toHaveBeenCalledWith('updated', { id: 'a1', status: 'done' });
  state.focusId = 'a1';
  frames.onFrame({ type: 'artifact_deleted', id: 'a1' });
  expect(d.sendChanged).toHaveBeenLastCalledWith('deleted');
  expect(model.setPick).toHaveBeenCalledWith(null);
  expect(state.focusId).toBeNull();
  expect(d.summarize).toHaveBeenCalledWith(true);
});

test('an update and a delete reach the artifact’s own tile, whether or not it is shown', () => {
  const d = deps();
  const frames = createArtifactFrames(d);
  frames.onFrame({ type: 'artifact_updated', id: 'a2', status: 'done' });
  expect(d.tileChanged).toHaveBeenCalledWith('a2', 'updated', { id: 'a2', status: 'done' });
  expect(d.sendChanged).not.toHaveBeenCalled();
  frames.onFrame({ type: 'artifact_deleted', id: 'a2' });
  expect(d.tileChanged).toHaveBeenLastCalledWith('a2', 'deleted');
  frames.onFrame({ type: 'artifact_updated', id: 'a1' });
  expect(d.tileChanged).toHaveBeenLastCalledWith('a1', 'updated', { id: 'a1' });
});

test('a focus is followed only for the active transcript; a manual one takes the pick', async () => {
  const d = deps();
  const frames = createArtifactFrames(d);
  frames.onFrame({ type: 'focus', id: 'f2', session: 'other' });
  await Promise.resolve();
  expect(store.fetchArtifact).not.toHaveBeenCalled();
  frames.onFrame({ type: 'focus', id: 'f2', session: 't1', automatic: false });
  await vi.waitFor(() => expect(model.setPick).toHaveBeenCalledWith('f2'));
  expect(state.focusId).toBe('f2');
  expect(model.recordLastArtifact).toHaveBeenCalledWith({ id: 'f2', owner: 't1' });
});

test('restoreFocus reads the persisted focus as automatic: no pick', async () => {
  const d = deps();
  const frames = createArtifactFrames(d);
  await frames.restoreFocus();
  expect(state.focusId).toBe('f1');
  expect(model.setPick).not.toHaveBeenCalled();
});
