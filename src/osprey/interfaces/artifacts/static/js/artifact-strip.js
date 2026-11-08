// @ts-check
/**
 * OSPREY Artifact Gallery — the content strip of the shown artifact.
 *
 * The controls that act on an artifact's view sit in one strip at the top
 * right of the content (content-strip.js; DESIGN.md §12): they show on hover
 * or focus, for about 3 s after the artifact opens, and while a mode other
 * than the default is pressed. Actions on the artifact as a whole (Info,
 * Download, the ⋯ verbs) stay in the hub's tile bar. Groups run modes, then
 * view, then output:
 *   - a plot (`plot_html`): Box zoom (the default mode) and Pan | Zoom in,
 *     Zoom out, Autoscale, Reset axes | Export PNG;
 *   - a table (`table_html`): Copy table, Export CSV (table-export.js);
 *   - a markdown note: Contents, Find in note | Copy as Markdown
 *     (note-tools.js).
 * Other types keep their own controls: a timeseries chart its buttons, an
 * extension viewer whatever it renders.
 *
 * Plot and table pages are sandboxed on an opaque origin, so the strip cannot
 * reach into them. A plot's buttons are posted to its frame as
 * `{type: 'osprey-artifact-command', command}`, which the snippet the server
 * injects into every plot page runs (plot-commands.js); that snippet also
 * hides Plotly's own modebar in a frame. And the strip's own hover rule never
 * fires over such a frame (the pointer goes straight to it, so its owner is
 * never `:hover`): the page reports the pointer instead (frame-pointer.js),
 * each report shows the strip for another 3 s, and a report that it left ends
 * that show. Over the strip itself, its own hover rule holds it.
 *
 * @module artifact-strip
 */

import { createContentStrip } from '/design-system/js/content-strip.js';
import { icon } from '/design-system/js/icons.js';
import { initTooltips, writeTitle } from '/design-system/js/tooltip.js';
import { copyMarkdown, openContents, openFind } from './note-tools.js';
import { copyTable, exportCsv } from './table-export.js';
import { hasTimeseriesData } from './types.js';

/**
 * @typedef {import('/design-system/js/content-strip.js').ContentStripGroup} ContentStripGroup
 * @typedef {import('/design-system/js/content-strip.js').ContentStripApi} ContentStripApi
 */

/** The message a served plot page takes its commands in (plot-commands.js). */
const COMMAND_MESSAGE = 'osprey-artifact-command';

/** The message a framed page posts while the pointer moves over it, and when it leaves. */
const POINTER_MESSAGE = 'osprey-artifact-pointer';

/** How long a copy button shows its result. */
const COPIED_MS = 1500;

/** @type {WeakMap<object, ContentStripApi>} each framed artifact's strip, by the frame's window */
const stripsByFrame = new WeakMap();
let followingPointer = false;

/** @type {ContentStripGroup[]} */
export const PLOT_GROUPS = [
  {
    id: 'modes',
    items: [
      {
        id: 'box-zoom',
        icon: 'box-zoom',
        title: 'Box zoom',
        detail: 'Drag across the plot to zoom in. Double-click resets.',
        mode: true,
        pressed: true,
      },
      {
        id: 'pan',
        icon: 'move',
        title: 'Pan',
        detail: 'Drag to move the view. Double-click resets.',
        mode: true,
      },
    ],
  },
  {
    id: 'view',
    items: [
      { id: 'zoom-in', icon: 'zoom-in', title: 'Zoom in' },
      { id: 'zoom-out', icon: 'zoom-out', title: 'Zoom out' },
      {
        id: 'autoscale',
        icon: 'autoscale',
        title: 'Autoscale',
        detail: 'Fit the axes to the data.',
      },
      {
        id: 'reset-axes',
        icon: 'reset-axes',
        title: 'Reset axes',
        detail: 'Back to the view the plot opened with.',
      },
    ],
  },
  {
    id: 'output',
    items: [
      {
        id: 'export-png',
        icon: 'image-down',
        title: 'Export PNG',
        detail: 'Download the plot as it shows now.',
      },
    ],
  },
];

/** @type {ContentStripGroup[]} */
const TABLE_GROUPS = [
  {
    id: 'output',
    items: [
      {
        id: 'copy-table',
        icon: 'copy',
        title: 'Copy table',
        detail: 'As tab-separated text, for the logbook or a spreadsheet.',
      },
      {
        id: 'export-csv',
        icon: 'download',
        title: 'Export CSV',
        detail: 'Download the table as a CSV file.',
      },
    ],
  },
];

/** @type {ContentStripGroup[]} */
const NOTE_GROUPS = [
  {
    id: 'view',
    items: [
      { id: 'contents', icon: 'list-tree', title: 'Contents', detail: 'Jump to a heading.' },
      { id: 'find', icon: 'text-search', title: 'Find in note' },
    ],
  },
  {
    id: 'output',
    items: [
      {
        id: 'copy-markdown',
        icon: 'copy',
        title: 'Copy as Markdown',
        detail: 'The source text of the note.',
      },
    ],
  },
];

/** Show a framed artifact's strip while its page says the pointer moves over it. */
function followPointer() {
  if (followingPointer) return;
  followingPointer = true;
  window.addEventListener('message', (event) => {
    const strip = event.source ? stripsByFrame.get(event.source) : undefined;
    const data = event.data;
    if (!strip || !data || data.type !== POINTER_MESSAGE) return;
    if (data.over === true) strip.show();
    else strip.hide();
  });
}

/**
 * A strip button by its item id.
 * @param {HTMLElement} host
 * @param {ContentStripGroup[]} groups the strip's groups
 * @param {string} id
 * @returns {HTMLElement | null}
 */
function stripButton(host, groups, id) {
  const title = groups.flatMap((group) => group.items).find((item) => item.id === id)?.title;
  if (!title) return null;
  return host.querySelector(`:scope > .osprey-content-strip button[aria-label="${title}"]`);
}

/**
 * Show a copy's result on its button for a moment: a check, or the copy icon
 * with a tooltip that says the copy failed.
 * @param {HTMLElement | null} button
 * @param {Promise<boolean>} copied
 */
async function confirmCopy(button, copied) {
  const ok = await copied;
  if (!button) return;
  const title = button.getAttribute('aria-label') ?? '';
  button.replaceChildren(icon(ok ? 'check' : 'copy'));
  writeTitle(button, ok ? 'Copied' : 'Copy failed');
  setTimeout(() => {
    button.replaceChildren(icon('copy'));
    writeTitle(button, title);
  }, COPIED_MS);
}

/**
 * Post a command to a plot page.
 * @param {HTMLIFrameElement} frame
 * @param {string} command
 * @param {any} a the artifact the frame shows
 */
function sendCommand(frame, command, a) {
  const message = { type: COMMAND_MESSAGE, command };
  if (command === 'export-png') Object.assign(message, { filename: a.title || 'plot' });
  // Intentional '*': the page runs on an opaque origin, which no target can
  // name; the frame's WindowProxy pins the receiver and the message is bare.
  frame.contentWindow?.postMessage(message, '*');
}

/**
 * The strip's host, its controls and what they do, for an artifact whose
 * viewport markup is in `container`; null for a type without a strip.
 * @param {HTMLElement} container
 * @param {any} a
 * @returns {{host: HTMLElement, frame: HTMLIFrameElement | null, label: string,
 *   groups: ContentStripGroup[], onAction: (id: string) => void} | null}
 */
function stripFor(container, a) {
  if (hasTimeseriesData(a)) return null;
  if (a.artifact_type === 'plot_html' || a.artifact_type === 'table_html') {
    const frame = container.querySelector('iframe');
    const host = frame?.parentElement;
    if (!frame || !host) return null;
    if (a.artifact_type === 'plot_html') {
      return {
        host,
        frame,
        label: 'Plot controls',
        groups: PLOT_GROUPS,
        onAction: (id) => sendCommand(frame, id, a),
      };
    }
    return {
      host,
      frame,
      label: 'Table controls',
      groups: TABLE_GROUPS,
      onAction: (id) => {
        if (id === 'export-csv') void exportCsv(a);
        else void confirmCopy(stripButton(host, TABLE_GROUPS, id), copyTable(a));
      },
    };
  }
  if (a.artifact_type === 'markdown') {
    const note = /** @type {HTMLElement | null} */ (
      container.querySelector('.md-preview-container')
    );
    const host = note?.parentElement;
    if (!note || !host) return null;
    return {
      host,
      frame: null,
      label: 'Note controls',
      groups: NOTE_GROUPS,
      onAction: (id) => {
        const button = stripButton(host, NOTE_GROUPS, id);
        if (!button) return;
        if (id === 'contents') openContents(button, note);
        else if (id === 'find') openFind(button, note);
        else void confirmCopy(button, copyMarkdown(a));
      },
    };
  }
  return null;
}

/**
 * Draw the content strip over the viewport `artifactViewportHtml(a)` built in
 * `container`, for the types that have one, and show it for a moment.
 * @param {HTMLElement} container the element the viewport markup was written into
 * @param {any} a the artifact
 * @returns {ContentStripApi | null} the strip, or null for a type without one
 */
export function mountArtifactStrip(container, a) {
  const spec = stripFor(container, a);
  if (!spec) return null;
  const { host, frame, label, groups, onAction } = spec;
  const strip = createContentStrip(host, { label, groups, onAction });
  const bar = host.querySelector(':scope > .osprey-content-strip');
  if (bar instanceof HTMLElement) initTooltips(bar);
  if (frame?.contentWindow) {
    stripsByFrame.set(frame.contentWindow, strip);
    followPointer();
  }
  strip.show();
  return strip;
}
