// @ts-check
/**
 * artifact-verbs.js: the one producer of the artifact and run rows.
 *
 * The artifact rows read the same in the sidebar and on the Artifacts tile
 * (the tile carries the viewer's verbs instead of Open and Download), and
 * Delete… asks in the hub for both. A run is named by its `exec_id`: held,
 * a part it lacks is disabled; not held, it is read from the store at the
 * pick.
 */
import { test, expect, describe, beforeEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const listing = vi.hoisted(() => ({
  ARTIFACTS_BASE: '/panel/artifacts',
  fetchArtifact: vi.fn(async () => null),
  heldRun: vi.fn(() => null),
  fetchRun: /** @type {import('vitest').Mock<(id: string) => Promise<any>>} */ (
    vi.fn(async () => null)
  ),
}));
const confirm = vi.hoisted(() => ({ showConfirm: vi.fn() }));
const controller = vi.hoisted(() => ({ showArtifact: vi.fn() }));
const paneHeader = vi.hoisted(() => ({ downloadArtifact: vi.fn() }));

vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifact-listing.js',
  () => listing
);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js', () => confirm);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js',
  () => controller
);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/artifacts-pane-header.js',
  () => paneHeader
);

const verbs = await import(`${JS}/artifact-verbs.js`);

const PLOT = { id: 'a1', title: 'BPM plot', filename: 'bpm.png' };
const RUN = {
  id: 'run1',
  exec_id: 'toolu_7',
  title: 'orbit scan',
  run_parts: [
    { id: 'out1', role: 'output' },
    { id: 'code1', role: 'code' },
  ],
};

/** @param {any[]} items */
const labels = (items) => items.map((i) => (i.divider ? '—' : i.label));
/** @param {any[]} items @param {string} label */
const row = (items, label) => items.find((i) => i.label === label);

beforeEach(() => {
  vi.clearAllMocks();
});

describe('an artifact', () => {
  test('a listed row: Open, Copy ID, Download, then Delete… behind a divider', () => {
    const open = vi.fn();
    const items = verbs.artifactMenuItems(PLOT, { open });
    expect(labels(items)).toEqual(['Open', 'Copy ID', 'Download', '—', 'Delete…']);
    row(items, 'Open').run();
    expect(open).toHaveBeenCalledOnce();
  });

  test('the shown artifact: the viewer’s verbs, no Open and no Download (the bar has it)', () => {
    const viewer = vi.fn();
    const items = verbs.artifactMenuItems(PLOT, { viewer });
    expect(labels(items)).toEqual(['Compose logbook entry', 'Print…', 'Copy ID', '—', 'Delete…']);
    row(items, 'Compose logbook entry').run();
    row(items, 'Print…').run();
    expect(viewer.mock.calls.map((c) => c[0])).toEqual(['logbook', 'print']);
  });

  test('Delete… asks in the hub, the same question from either place', () => {
    for (const where of [{ open: () => {} }, { viewer: () => {} }]) {
      row(verbs.artifactMenuItems(PLOT, where), 'Delete…').run();
    }
    expect(confirm.showConfirm).toHaveBeenCalledTimes(2);
    const [a, b] = confirm.showConfirm.mock.calls.map((c) => c[0]);
    expect(a.title).toBe('Delete artifact?');
    expect(JSON.stringify(b.body)).toEqual(JSON.stringify(a.body));
  });
});

describe('a run', () => {
  test('held: a part it lacks is disabled; Copy run ID is the exec id', () => {
    const writeText = vi.fn(async () => {});
    vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText } });
    const items = verbs.runMenuItems('toolu_7', RUN);
    expect(labels(items)).toEqual([
      'Open result',
      'Open approval preview',
      'Open code',
      'Copy run ID',
      '—',
      'Delete…',
    ]);
    expect(row(items, 'Open result').disabled).toBe(false);
    expect(row(items, 'Open approval preview').disabled).toBe(true);
    row(items, 'Copy run ID').run();
    expect(writeText).toHaveBeenCalledWith('toolu_7');
    vi.unstubAllGlobals();
  });

  test('held: Open result shows the output as the tree’s pick does, nothing fetched', async () => {
    row(verbs.runMenuItems('toolu_7', RUN), 'Open result').run();
    await vi.waitFor(() => expect(controller.showArtifact).toHaveBeenCalledWith('out1'));
    expect(listing.fetchRun).not.toHaveBeenCalled();
  });

  test('not held: every opener is offered and the run is read from the store at the pick', async () => {
    listing.fetchRun.mockResolvedValueOnce(RUN);
    const items = verbs.runMenuItems('toolu_7', null);
    expect(items.filter((/** @type {any} */ i) => i.disabled)).toEqual([]);
    row(items, 'Open code').run();
    await vi.waitFor(() => expect(controller.showArtifact).toHaveBeenCalledWith('code1'));
    expect(listing.fetchRun).toHaveBeenCalledWith('toolu_7');
  });

  test('not held and not in the store: a pick does nothing', async () => {
    const items = verbs.runMenuItems('toolu_gone', null);
    row(items, 'Open result').run();
    row(items, 'Delete…').run();
    await vi.waitFor(() => expect(listing.fetchRun).toHaveBeenCalledTimes(2));
    expect(controller.showArtifact).not.toHaveBeenCalled();
    expect(confirm.showConfirm).not.toHaveBeenCalled();
  });

  test('Delete… asks for the run and its parts', async () => {
    row(verbs.runMenuItems('toolu_7', RUN), 'Delete…').run();
    await vi.waitFor(() => expect(confirm.showConfirm).toHaveBeenCalledOnce());
    const spec = confirm.showConfirm.mock.calls[0][0];
    expect(spec.title).toBe('Delete run?');
    expect(JSON.stringify(spec.body)).toContain('and its parts (2)');
  });
});
