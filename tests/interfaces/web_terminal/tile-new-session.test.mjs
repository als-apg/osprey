/**
 * The tile menu's "Open in new session" (tile-new-session.js): a new session
 * through switchTo, then the pane opened beside the Agent pane, only once the
 * switch resolved true.
 *
 *   npx vitest run tests/interfaces/web_terminal/tile-new-session.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const calls = vi.hoisted(() => /** @type {string[]} */ ([]));
const model = vi.hoisted(() => ({
  switchTo: vi.fn(async () => true),
}));
const placement = vi.hoisted(() => ({
  openPanelLeftOfAgent: vi.fn(),
  hasDockedTile: vi.fn(() => true),
}));
const tiles = vi.hoisted(() => ({
  artifactIdOf: vi.fn((/** @type {string} */ key) =>
    key.startsWith('artifact:') ? key.slice('artifact:'.length) : null
  ),
  artifactTileKey: vi.fn((/** @type {string} */ id) => `artifact:${id}`),
  holdArtifactTile: vi.fn(
    (/** @type {{id: string}} */ { id }) => /** @type {string | null} */ (`artifact:${id}`)
  ),
}));
const notice = vi.hoisted(() => ({ showUndoNotice: vi.fn() }));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/session-model.js', () => model);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/panel-placement.js',
  () => placement
);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/artifact-tiles.js', () => tiles);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/undo-notice.js', () => notice);

const { openInNewSession, openArtifactInNewSession, openInNewSessionRow, OPEN_FAILED_MESSAGE } =
  await import(`${JS}/tile-new-session.js`);

/** @type {ReturnType<typeof vi.spyOn>} */
let errors;

beforeEach(() => {
  vi.clearAllMocks();
  calls.length = 0;
  model.switchTo.mockImplementation(async () => {
    calls.push('switchTo');
    return true;
  });
  placement.openPanelLeftOfAgent.mockImplementation(() => {
    calls.push('openPanelLeftOfAgent');
  });
  placement.hasDockedTile.mockReturnValue(true);
  errors = vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => errors.mockRestore());

describe('the row', () => {
  test('is labelled Open in new session, with a glyph, and runs the action', async () => {
    const row = openInNewSessionRow('ariel');
    expect(row.label).toBe('Open in new session');
    expect(row.glyph).toBeTruthy();
    await /** @type {() => unknown} */ (row.run)();
    await vi.waitFor(() => expect(placement.openPanelLeftOfAgent).toHaveBeenCalledWith('ariel'));
  });
});

describe('openInNewSession', () => {
  test('starts a new session, then opens the service pane by its panel id', async () => {
    await expect(openInNewSession('ariel')).resolves.toBe(true);
    expect(model.switchTo).toHaveBeenCalledExactlyOnceWith({ new: true });
    expect(placement.openPanelLeftOfAgent).toHaveBeenCalledExactlyOnceWith('ariel');
    expect(calls).toEqual(['switchTo', 'openPanelLeftOfAgent']);
    expect(tiles.holdArtifactTile).not.toHaveBeenCalled();
    expect(notice.showUndoNotice).not.toHaveBeenCalled();
  });

  test('opens an artifact tile for the same artifact, held before it is placed', async () => {
    tiles.holdArtifactTile.mockImplementation(({ id }) => {
      calls.push('hold');
      return `artifact:${id}`;
    });
    await expect(openInNewSession('artifact:a1')).resolves.toBe(true);
    expect(tiles.holdArtifactTile).toHaveBeenCalledExactlyOnceWith({
      id: 'a1',
      entry: null,
      title: '',
    });
    expect(placement.openPanelLeftOfAgent).toHaveBeenCalledExactlyOnceWith('artifact:a1');
    expect(calls).toEqual(['switchTo', 'hold', 'openPanelLeftOfAgent']);
  });

  test('does nothing when the switch resolves false: the live turn was kept', async () => {
    model.switchTo.mockResolvedValueOnce(false);
    await expect(openInNewSession('ariel')).resolves.toBe(false);
    expect(placement.openPanelLeftOfAgent).not.toHaveBeenCalled();
    expect(tiles.holdArtifactTile).not.toHaveBeenCalled();
    expect(notice.showUndoNotice).not.toHaveBeenCalled();
  });

  test('a switch that throws changes nothing and is logged like the new-session button', async () => {
    const failure = new Error('socket closed');
    model.switchTo.mockRejectedValueOnce(failure);
    await expect(openInNewSession('artifact:a1')).resolves.toBe(false);
    expect(errors).toHaveBeenCalledWith('Failed to start new session:', failure);
    expect(placement.openPanelLeftOfAgent).not.toHaveBeenCalled();
    expect(tiles.holdArtifactTile).not.toHaveBeenCalled();
  });

  test('a pane that cannot be opened leaves the new session and says so', async () => {
    placement.hasDockedTile.mockReturnValue(false);
    const doc = document.implementation.createHTMLDocument('popup');
    await expect(openInNewSession('ariel', doc)).resolves.toBe(false);
    expect(notice.showUndoNotice).toHaveBeenCalledExactlyOnceWith({
      message: OPEN_FAILED_MESSAGE,
      doc,
    });
  });

  test('an open that throws is caught: a notice, no rejection', async () => {
    const failure = new Error('no dock');
    placement.openPanelLeftOfAgent.mockImplementation(() => {
      throw failure;
    });
    await expect(openInNewSession('ariel')).resolves.toBe(false);
    expect(errors).toHaveBeenCalledWith('Failed to open the pane in the new session:', failure);
    expect(notice.showUndoNotice).toHaveBeenCalledExactlyOnceWith({
      message: OPEN_FAILED_MESSAGE,
      doc: undefined,
    });
  });

  test('an artifact that cannot be held is caught the same way', async () => {
    tiles.holdArtifactTile.mockImplementation(() => {
      throw new Error('bad tile');
    });
    await expect(openInNewSession('artifact:a1')).resolves.toBe(false);
    expect(placement.openPanelLeftOfAgent).not.toHaveBeenCalled();
    expect(notice.showUndoNotice).toHaveBeenCalledTimes(1);
  });
});

describe('openArtifactInNewSession', () => {
  const entry = { id: 'a1', title: 'Beam plot', filename: 'beam.png' };

  test('holds the tile with the row’s entry and title, then opens its key', async () => {
    tiles.holdArtifactTile.mockImplementation(({ id }) => {
      calls.push('hold');
      return `artifact:${id}`;
    });
    await expect(openArtifactInNewSession({ id: 'a1', entry, title: 'Beam plot' })).resolves.toBe(
      true
    );
    expect(model.switchTo).toHaveBeenCalledExactlyOnceWith({ new: true });
    expect(tiles.holdArtifactTile).toHaveBeenCalledExactlyOnceWith({
      id: 'a1',
      entry,
      title: 'Beam plot',
    });
    expect(placement.openPanelLeftOfAgent).toHaveBeenCalledExactlyOnceWith('artifact:a1');
    expect(calls).toEqual(['switchTo', 'hold', 'openPanelLeftOfAgent']);
  });

  test('a declined switch holds and opens nothing', async () => {
    model.switchTo.mockResolvedValueOnce(false);
    await expect(openArtifactInNewSession({ id: 'a1', entry, title: 't' })).resolves.toBe(false);
    expect(tiles.holdArtifactTile).not.toHaveBeenCalled();
    expect(placement.openPanelLeftOfAgent).not.toHaveBeenCalled();
  });

  test('an id no tile can carry opens nothing and says so', async () => {
    tiles.holdArtifactTile.mockReturnValue(null);
    await expect(openArtifactInNewSession({ id: 'a b', entry: null, title: 't' })).resolves.toBe(
      false
    );
    expect(placement.openPanelLeftOfAgent).not.toHaveBeenCalled();
    expect(notice.showUndoNotice).toHaveBeenCalledTimes(1);
  });
});
