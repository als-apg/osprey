// @ts-check
/* OSPREY Web Terminal — the session tree's live artifact frames
 *
 * The artifacts panel's event stream, as the sidebar's session tree
 * (sidebar-artifacts.js) follows it:
 *
 *   - `artifact` counts once per run (`exec_id`, else id), only when
 *     `operator_visible`, by its listed form; a counted run only takes the new
 *     status. `artifact_updated` sets the status and re-places an entry whose
 *     owner changed. A frame whose owner is no known row refreshes the model.
 *   - `artifact_deleted` and a reconnect re-fetch the summary.
 *   - `focus` is followed only when its `session` names the active row's
 *     transcript. A non-automatic agent focus (or any focus on the terminal
 *     surface) takes the pick and records last-artifact; an automatic one only
 *     records the focus. A reload, a second tab or a reconnect reads the
 *     persisted focus as automatic ({@link ArtifactFrames.restoreFocus}).
 *
 * @module sidebar-artifact-frames
 */

import { isTerminalSurface } from './agent-surface.js';
import * as store from './sidebar-artifact-listing.js';
import * as model from './session-model.js';

/** @typedef {import('./sidebar-artifact-listing.js').ArtifactEntry} ArtifactEntry */

/**
 * @typedef {object} FrameDeps
 * @property {() => boolean} isServed  whether the artifacts panel is served
 * @property {ReturnType<typeof store.createTreeData>} data  the tree's listings
 * @property {() => {focusId: string | null, shownId: string | null}} state  the tree's state
 * @property {(entry: ArtifactEntry) => boolean} inActiveSession
 * @property {(change: 'updated' | 'deleted', entry?: ArtifactEntry) => void} sendChanged
 *   tell the viewer the shown artifact changed
 * @property {(all?: boolean) => Promise<void>} summarize  re-fetch the counts
 * @property {() => void} render
 */

/**
 * @typedef {object} ArtifactFrames
 * @property {(frame: any) => void} onFrame
 * @property {() => void} onStreamOpen
 * @property {() => Promise<void>} restoreFocus
 * @property {() => void} reset  forget the stream (a teardown)
 */

/**
 * @param {FrameDeps} deps
 * @returns {ArtifactFrames}
 */
export function createArtifactFrames({
  isServed,
  data,
  state,
  inActiveSession,
  sendChanged,
  summarize,
  render,
}) {
  let opened = false;

  /**
   * The agent focused an artifact.
   * @param {string} id
   * @param {boolean} automatic
   */
  async function onFocus(id, automatic) {
    if (!data.held(id)) {
      const entry = await store.fetchArtifact(id);
      if (!isServed() || entry?.id !== id) return;
      data.hold(entry);
    }
    state().focusId = id;
    if (!automatic || isTerminalSurface()) {
      model.setPick(id);
      if (!automatic) model.recordLastArtifact(data.held(id));
    }
    render();
  }

  async function restoreFocus() {
    const id = await store.fetchFocusedId();
    if (isServed() && id && id !== state().focusId) await onFocus(id, true);
  }

  /**
   * The extra listing a live entry also lands in: the active key's, when the
   * entry is the active session's.
   * @param {ArtifactEntry} entry
   */
  function extraListings(entry) {
    const active = model.activeRow();
    return active && inActiveSession(entry) ? [`key:${active.session_key}|`] : [];
  }

  /** Whether an entry's owner is a loaded row (no owner: nothing to load). @param {ArtifactEntry} entry */
  const knownOwner = (entry) => entry.owner == null || model.sessionOfArtifact(entry) !== null;

  /** @param {any} frame */
  function onFrame(frame) {
    if (!isServed() || !frame || typeof frame !== 'object' || typeof frame.id !== 'string') return;
    const entry = /** @type {ArtifactEntry} */ ({ ...frame });
    delete (/** @type {any} */ (entry).type);
    const now = state();
    if (frame.type === 'artifact') {
      if (!data.added(entry, extraListings(entry))) return;
      if (!knownOwner(entry)) model.refresh();
    } else if (frame.type === 'artifact_updated') {
      if (data.updated(entry, extraListings(entry)) && !knownOwner(entry)) model.refresh();
      if (frame.id === now.shownId) sendChanged('updated', entry);
    } else if (frame.type === 'artifact_deleted') {
      if (frame.id === now.shownId) sendChanged('deleted');
      data.deleted(frame.id);
      if (model.pick() === frame.id) model.setPick(null);
      if (now.focusId === frame.id) now.focusId = null;
      void summarize(true);
    } else {
      if (frame.type === 'focus' && model.isActiveTranscript(frame.session))
        void onFocus(frame.id, frame.automatic === true);
      return;
    }
    render();
  }

  function onStreamOpen() {
    if (!isServed()) return;
    if (opened) {
      void summarize(true);
      void restoreFocus();
    }
    opened = true;
  }

  return {
    onFrame,
    onStreamOpen,
    restoreFocus,
    reset: () => {
      opened = false;
    },
  };
}
