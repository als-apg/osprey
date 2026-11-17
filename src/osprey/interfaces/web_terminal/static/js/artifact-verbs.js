// @ts-check
/* OSPREY Web Terminal — the verbs on an artifact and on a run
 *
 * The one producer of the rows every place offers for an artifact or a script
 * run, so the labels and the actions behind them are the same wherever the
 * operator right-clicks:
 *
 *   Artifact   Open · Compose logbook entry · Print… · Copy ID · Download ·
 *              Delete…
 *   Run        Open result · Open approval preview · Open code · Copy run ID ·
 *              Delete…
 *
 * An artifact's rows: the sidebar's artifact row offers Open (the row's own
 * pick) and Download; the Artifacts tile's "This panel" rows are the shown
 * artifact's, so they carry the viewer's verbs (logbook, print) and no Download,
 * which the tile bar has as an icon. Delete… asks here, in the hub, for every
 * place: the viewer drops the artifact on the store's delete frame.
 *
 * A run is named by its `exec_id`, the Claude Code `tool_use_id` of the call
 * that ran it, so the sidebar's run head and the chat's run row ask for the
 * same menu. A run the tree holds is read at once and a part it lacks is
 * offered disabled; one it does not hold yet (a conversation replayed before
 * its runs were listed) is read from the store at the pick. A part opens as
 * the tree's own pick does (sidebar-artifacts.js `showArtifact`).
 *
 * sidebar-artifacts.js and artifacts-pane-header.js are imported lazily, at a
 * pick: both sit behind panel-menu-policy, which imports this module's callers.
 *
 * @module artifact-verbs
 */

import { withPrefix } from './api.js';
import { ARTIFACTS_BASE, fetchArtifact, fetchRun, heldRun } from './sidebar-artifact-listing.js';
import { showConfirm } from './posture-confirm.js';

/** @typedef {import('./panel-context-menu.js').MenuItem} MenuItem */
/** @typedef {import('./sidebar-artifact-listing.js').ArtifactEntry} ArtifactEntry */
/** @typedef {import('./sidebar-artifact-listing.js').RunEntry} RunEntry */
/** @typedef {'print' | 'logbook'} ViewerVerb */

/** @param {string} text */
export function copyText(text) {
  void navigator.clipboard?.writeText(text).catch(() => {});
}

/** @param {string} id */
const artifactUrl = (id) => withPrefix(`${ARTIFACTS_BASE}/api/artifacts/${encodeURIComponent(id)}`);

/**
 * Ask first, then delete. Deleting a run's head deletes the run and its
 * parts (the server's delete_with_group), and the question says so.
 * @param {RunEntry} entry  @param {boolean} isRun
 */
export function confirmDelete(entry, isRun) {
  const name = entry.title || entry.id;
  const parts = (entry.run_parts ?? []).filter((p) => p.id !== entry.id).length;
  const body = isRun
    ? [['This deletes the run ', { em: name }, ` and its parts (${parts}).`]]
    : [['This deletes ', { em: name }, '.']];
  showConfirm({
    title: isRun ? 'Delete run?' : 'Delete artifact?',
    body,
    live: null,
    confirmLabel: 'Delete',
    focus: 'cancel',
    onConfirm: async (ui) => {
      try {
        const res = await fetch(artifactUrl(entry.id), { method: 'DELETE' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        ui.done();
      } catch (err) {
        ui.error.textContent = `Delete failed: ${err instanceof Error ? err.message : err}`;
        ui.error.hidden = false;
      }
    },
  });
}

/**
 * Download an artifact's file. A run's part is held as {id, title} only, so
 * the filename is read here, at the pick, never while the menu is built.
 * @param {ArtifactEntry} entry
 */
async function download(entry) {
  const full = entry.filename ? entry : await fetchArtifact(entry.id);
  if (!full?.filename) return;
  (await import('./artifacts-pane-header.js')).downloadArtifact(full);
}

/** Show an artifact as the tree's pick does. @param {string} id */
async function showArtifact(id) {
  (await import('./sidebar-artifacts.js')).showArtifact(id);
}

/**
 * The rows for one artifact.
 * @param {ArtifactEntry} entry
 * @param {{open?: () => void, newSession?: () => void, viewer?: (verb: ViewerVerb) => void}} [where] -
 *   `open`: the place's own pick (a listed row); `newSession`: opens the
 *   artifact in a new session, offered where the artifact can be shown as a
 *   tile; `viewer`: the place shows the artifact, and these send it the
 *   viewer's verbs
 * @returns {MenuItem[]}
 */
export function artifactMenuItems(entry, { open, newSession, viewer } = {}) {
  /** @type {MenuItem[]} */
  const items = [];
  if (open) items.push({ label: 'Open', run: open });
  if (newSession) items.push({ label: 'Open in new session', glyph: '⊕', run: newSession });
  if (viewer) {
    items.push(
      { label: 'Compose logbook entry', run: () => viewer('logbook') },
      { label: 'Print…', run: () => viewer('print') }
    );
  }
  items.push({ label: 'Copy ID', run: () => copyText(entry.id) });
  if (!viewer) items.push({ label: 'Download', run: () => void download(entry) });
  items.push(
    { divider: true },
    { label: 'Delete…', danger: true, run: () => confirmDelete(entry, false) }
  );
  return items;
}

/** A run's part kinds, in the order a menu offers them. */
const OPENERS = /** @type {const} */ ([
  ['Open result', ['result', 'output']],
  ['Open approval preview', ['preview']],
  ['Open code', ['code']],
]);

/**
 * The first part of `run` with one of `roles`, the head itself excluded.
 * @param {RunEntry} run  @param {readonly string[]} roles
 */
function partOf(run, roles) {
  const parts = (run.run_parts ?? []).filter((p) => p.id !== run.id);
  for (const role of roles) {
    const part = parts.find((p) => p.role === role);
    if (part) return part;
  }
  return null;
}

/**
 * The run, held or read from the store now.
 * @param {string} execId  @param {RunEntry | null} held
 */
async function resolveRun(execId, held) {
  return held ?? heldRun(execId) ?? (await fetchRun(execId));
}

/**
 * The rows for one run.
 * @param {string} execId - the run's `exec_id`
 * @param {RunEntry | null} run - the run as held, or null when it is not held
 * @returns {MenuItem[]}
 */
export function runMenuItems(execId, run) {
  /** @type {MenuItem[]} */
  const openers = OPENERS.map(([label, roles]) => ({
    label,
    disabled: run !== null && !partOf(run, roles),
    run: () =>
      void resolveRun(execId, run).then((r) => {
        const part = r && partOf(r, roles);
        if (part) void showArtifact(part.id);
      }),
  }));
  return [
    ...openers,
    { label: 'Copy run ID', run: () => copyText(execId) },
    { divider: true },
    {
      label: 'Delete…',
      danger: true,
      run: () =>
        void resolveRun(execId, run).then((r) => {
          if (r) confirmDelete(r, true);
        }),
    },
  ];
}
