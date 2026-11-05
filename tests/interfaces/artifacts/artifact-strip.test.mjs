// @ts-check
/**
 * Unit tests for the content strip of the shown artifact (artifact-strip.js):
 *   npx vitest run tests/interfaces/artifacts/artifact-strip.test.mjs
 *
 * A plot_html artifact gets the strip over its frame: modes (Box zoom, the
 * default, and Pan), then view (Zoom in, Zoom out, Autoscale, Reset axes),
 * then output (Export PNG), each button posted to the frame as an
 * `osprey-artifact-command`. A table gets Copy table and Export CSV, a
 * markdown note Contents and Find in note, then Copy as Markdown; their
 * actions (table-export.js, note-tools.js) are mocked here and tested in
 * their own suites. The strip shows for a moment when it mounts, and over a
 * frame while the page reports the pointer. Every other type, a timeseries
 * chart included, gets none. The strip's own behaviour (radio modes, hover,
 * pinning) is content-strip.js's and is tested there.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

const actions = vi.hoisted(() => ({
  copyTable: vi.fn(async () => true),
  exportCsv: vi.fn(async () => true),
  copyMarkdown: vi.fn(async () => true),
  openContents: vi.fn(),
  openFind: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/artifacts/static/js/table-export.js', () => ({
  copyTable: actions.copyTable,
  exportCsv: actions.exportCsv,
}));
vi.mock('../../../src/osprey/interfaces/artifacts/static/js/note-tools.js', () => ({
  copyMarkdown: actions.copyMarkdown,
  openContents: actions.openContents,
  openFind: actions.openFind,
}));

import {
  mountArtifactStrip,
  PLOT_GROUPS,
} from '../../../src/osprey/interfaces/artifacts/static/js/artifact-strip.js';
import { PLOT_COMMANDS } from '../../../src/osprey/interfaces/artifacts/static/js/plot-commands.js';

const plot = { id: 'p1', title: 'Orbit at the bump', artifact_type: 'plot_html' };
const table = { id: 't1', title: 'Orbit table', artifact_type: 'table_html' };
const note = { id: 'n1', title: 'Dump and refill', artifact_type: 'markdown' };

/**
 * The preview pane's viewport holding one frame, in the document so the frame
 * has a window to post to.
 * @returns {{container: HTMLElement, viewport: HTMLElement, frame: HTMLIFrameElement}}
 */
function framed() {
  const container = document.createElement('div');
  container.innerHTML = '<div class="preview-viewport"><iframe></iframe></div>';
  document.body.append(container);
  return {
    container,
    viewport: /** @type {HTMLElement} */ (container.querySelector('.preview-viewport')),
    frame: /** @type {HTMLIFrameElement} */ (container.querySelector('iframe')),
  };
}

/**
 * The preview pane's viewport holding a markdown note's container.
 * @returns {{container: HTMLElement, viewport: HTMLElement, body: HTMLElement}}
 */
function noted() {
  const container = document.createElement('div');
  container.innerHTML =
    '<div class="preview-viewport"><div class="md-preview-container"></div></div>';
  document.body.append(container);
  return {
    container,
    viewport: /** @type {HTMLElement} */ (container.querySelector('.preview-viewport')),
    body: /** @type {HTMLElement} */ (container.querySelector('.md-preview-container')),
  };
}

/**
 * The labels of a strip's buttons, group by group.
 * @param {HTMLElement} root
 * @returns {(string | null)[][]}
 */
function groupsOf(root) {
  return Array.from(root.querySelectorAll('.osprey-content-strip-group'), (group) =>
    Array.from(group.querySelectorAll('button'), (b) => b.getAttribute('aria-label'))
  );
}

/**
 * @param {HTMLElement} root
 * @param {string} title
 * @returns {HTMLButtonElement}
 */
function button(root, title) {
  return /** @type {HTMLButtonElement} */ (root.querySelector(`button[aria-label="${title}"]`));
}

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
  vi.useRealTimers();
  document.body.replaceChildren();
});

describe('a plot', () => {
  test('gets the strip over its frame: modes, then view, then output', () => {
    const { container, viewport } = framed();

    expect(mountArtifactStrip(container, plot)).not.toBeNull();

    const strip = /** @type {HTMLElement} */ (viewport.querySelector('.osprey-content-strip'));
    expect(strip.parentElement).toBe(viewport);
    expect(strip.getAttribute('role')).toBe('toolbar');
    expect(strip.getAttribute('aria-label')).toBe('Plot controls');
    const groups = Array.from(strip.querySelectorAll('.osprey-content-strip-group'), (group) =>
      Array.from(group.querySelectorAll('button'), (b) => b.getAttribute('aria-label'))
    );
    expect(groups).toEqual([
      ['Box zoom', 'Pan'],
      ['Zoom in', 'Zoom out', 'Autoscale', 'Reset axes'],
      ['Export PNG'],
    ]);
  });

  test('Box zoom is the pressed default mode, Pan is not; the view and output items are no modes', () => {
    const { container, viewport } = framed();
    mountArtifactStrip(container, plot);

    expect(button(viewport, 'Box zoom').getAttribute('aria-pressed')).toBe('true');
    expect(button(viewport, 'Pan').getAttribute('aria-pressed')).toBe('false');
    expect(button(viewport, 'Zoom in').hasAttribute('aria-pressed')).toBe(false);
    expect(button(viewport, 'Box zoom').dataset.tipDetail).toBe(
      'Drag across the plot to zoom in. Double-click resets.'
    );
  });

  test('each button posts its command to the frame', () => {
    const { container, viewport, frame } = framed();
    const post = vi.spyOn(/** @type {Window} */ (frame.contentWindow), 'postMessage');
    mountArtifactStrip(container, plot);

    button(viewport, 'Pan').click();
    button(viewport, 'Zoom in').click();
    button(viewport, 'Reset axes').click();

    expect(post.mock.calls).toEqual([
      [{ type: 'osprey-artifact-command', command: 'pan' }, '*'],
      [{ type: 'osprey-artifact-command', command: 'zoom-in' }, '*'],
      [{ type: 'osprey-artifact-command', command: 'reset-axes' }, '*'],
    ]);
  });

  test('Export PNG names the image after the artifact', () => {
    const { container, viewport, frame } = framed();
    const post = vi.spyOn(/** @type {Window} */ (frame.contentWindow), 'postMessage');
    mountArtifactStrip(container, plot);

    button(viewport, 'Export PNG').click();

    expect(post).toHaveBeenCalledWith(
      { type: 'osprey-artifact-command', command: 'export-png', filename: 'Orbit at the bump' },
      '*'
    );
  });

  test('shows for a moment when it mounts', () => {
    vi.useFakeTimers();
    const { container, viewport } = framed();
    mountArtifactStrip(container, plot);
    const strip = /** @type {HTMLElement} */ (viewport.querySelector('.osprey-content-strip'));

    expect(strip.hasAttribute('data-visible')).toBe(true);
    vi.advanceTimersByTime(3000);
    expect(strip.hasAttribute('data-visible')).toBe(false);
  });

  test('shows while its plot page says the pointer moves over it, and hides when it leaves', () => {
    vi.useFakeTimers();
    const { container, viewport, frame } = framed();
    mountArtifactStrip(container, plot);
    const strip = /** @type {HTMLElement} */ (viewport.querySelector('.osprey-content-strip'));
    /** @param {any} data @param {any} source */
    const report = (data, source) =>
      window.dispatchEvent(new MessageEvent('message', { data, source }));
    vi.advanceTimersByTime(3000);
    expect(strip.hasAttribute('data-visible')).toBe(false);

    report({ type: 'osprey-artifact-pointer', over: true }, frame.contentWindow);
    expect(strip.hasAttribute('data-visible')).toBe(true);
    vi.advanceTimersByTime(2000);
    report({ type: 'osprey-artifact-pointer', over: true }, frame.contentWindow);
    vi.advanceTimersByTime(2000);
    expect(strip.hasAttribute('data-visible')).toBe(true);
    report({ type: 'osprey-artifact-pointer', over: false }, frame.contentWindow);
    expect(strip.hasAttribute('data-visible')).toBe(false);

    // Another frame, or another message from this one, changes nothing.
    const other = document.createElement('iframe');
    document.body.append(other);
    report({ type: 'osprey-artifact-pointer', over: true }, other.contentWindow);
    report({ type: 'osprey-artifact-ready' }, frame.contentWindow);
    expect(strip.hasAttribute('data-visible')).toBe(false);
  });

  test('every button is a command the plot page takes', () => {
    const ids = PLOT_GROUPS.flatMap((group) => group.items.map((item) => item.id));
    expect([...ids].sort()).toEqual([...PLOT_COMMANDS].sort());
  });
});

describe('a table', () => {
  test('gets Copy table and Export CSV over its frame', () => {
    const { container, viewport } = framed();

    mountArtifactStrip(container, table);

    const strip = /** @type {HTMLElement} */ (viewport.querySelector('.osprey-content-strip'));
    expect(strip.getAttribute('aria-label')).toBe('Table controls');
    expect(groupsOf(viewport)).toEqual([['Copy table', 'Export CSV']]);
  });

  test('Copy table copies it and shows a check for a moment; Export CSV downloads it', async () => {
    vi.useFakeTimers();
    const { container, viewport } = framed();
    mountArtifactStrip(container, table);
    const copy = button(viewport, 'Copy table');

    copy.click();
    await vi.advanceTimersByTimeAsync(0);
    expect(actions.copyTable).toHaveBeenCalledWith(table);
    expect(copy.title).toBe('Copied');
    await vi.advanceTimersByTimeAsync(1500);
    expect(copy.title).toBe('Copy table');

    button(viewport, 'Export CSV').click();
    expect(actions.exportCsv).toHaveBeenCalledWith(table);
  });

  test('a failed copy says so in the tooltip', async () => {
    actions.copyTable.mockResolvedValueOnce(false);
    const { container, viewport } = framed();
    mountArtifactStrip(container, table);
    const copy = button(viewport, 'Copy table');

    copy.click();
    await vi.waitFor(() => expect(copy.title).toBe('Copy failed'));
  });

  test('shows while its page says the pointer moves over it', () => {
    vi.useFakeTimers();
    const { container, viewport, frame } = framed();
    mountArtifactStrip(container, table);
    const strip = /** @type {HTMLElement} */ (viewport.querySelector('.osprey-content-strip'));
    vi.advanceTimersByTime(3000);

    window.dispatchEvent(
      new MessageEvent('message', {
        data: { type: 'osprey-artifact-pointer', over: true },
        source: frame.contentWindow,
      })
    );
    expect(strip.hasAttribute('data-visible')).toBe(true);
  });
});

describe('a markdown note', () => {
  test('gets Contents and Find in note, then Copy as Markdown', () => {
    const { container, viewport } = noted();

    mountArtifactStrip(container, note);

    const strip = /** @type {HTMLElement} */ (viewport.querySelector('.osprey-content-strip'));
    expect(strip.parentElement).toBe(viewport);
    expect(strip.getAttribute('aria-label')).toBe('Note controls');
    expect(groupsOf(viewport)).toEqual([['Contents', 'Find in note'], ['Copy as Markdown']]);
  });

  test('Contents and Find open under their buttons on the note; Copy as Markdown copies it', async () => {
    const { container, viewport, body } = noted();
    mountArtifactStrip(container, note);

    button(viewport, 'Contents').click();
    button(viewport, 'Find in note').click();
    button(viewport, 'Copy as Markdown').click();

    expect(actions.openContents).toHaveBeenCalledWith(button(viewport, 'Contents'), body);
    expect(actions.openFind).toHaveBeenCalledWith(button(viewport, 'Find in note'), body);
    expect(actions.copyMarkdown).toHaveBeenCalledWith(note);
    await vi.waitFor(() => expect(button(viewport, 'Copy as Markdown').title).toBe('Copied'));
  });
});

describe('no strip', () => {
  test.each([
    ['an image', { id: 'i', artifact_type: 'plot_png' }],
    ['an html page', { id: 'h', artifact_type: 'html' }],
    ['a dashboard', { id: 'd', artifact_type: 'dashboard_html' }],
    [
      'a plot with timeseries data (the chart keeps its own buttons)',
      { id: 't', artifact_type: 'plot_html', data_file: 'y.parquet', tool_source: 'archiver_read' },
    ],
  ])('for %s', (_label, artifact) => {
    const { container } = framed();

    expect(mountArtifactStrip(container, artifact)).toBeNull();
    expect(container.querySelector('.osprey-content-strip')).toBeNull();
  });

  test('for a plot whose viewport holds no frame', () => {
    const container = document.createElement('div');

    expect(mountArtifactStrip(container, plot)).toBeNull();
    expect(container.querySelector('.osprey-content-strip')).toBeNull();
  });
});
