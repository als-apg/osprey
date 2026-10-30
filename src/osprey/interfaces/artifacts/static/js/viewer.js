// @ts-check
/**
 * OSPREY Artifact Gallery — the embedded viewer's channel to its host.
 *
 * Embedded in the Web Terminal hub, the gallery is a viewer of ONE artifact:
 * the hub's sidebar owns the artifact list and the hub renders the pane
 * header, so the hub tells this page which artifact to show and which action
 * to run on it, with two postMessage types:
 *
 *   - `{type: 'osprey-artifact-select', artifact_id: string | null}` — show
 *     that artifact (null: the empty state). A select never changes the
 *     agent's focus.
 *   - `{type: 'osprey-artifact-action', artifact_id: string, action}` — run
 *     the gallery's own code path for `action` (one of {@link VIEWER_ACTIONS})
 *     on that artifact.
 *
 * Both are accepted only from this page's own origin, which is the hub's: the
 * hub serves the gallery through its own proxy. A message of any other shape
 * is ignored. Once the listener is bound the page posts
 * `{type: 'osprey-artifact-ready'}` to its parent, so the hub sends its
 * current selection again to a frame that loaded after that selection was
 * made.
 *
 * The artifact a message names is the held entry when the gallery holds it,
 * else the server's copy. A select whose fetch is still in flight when a newer
 * select arrives is dropped, so a slow response never replaces what the hub
 * selected after it.
 *
 * `createViewer(deps)` takes every effect injected — the held list, the
 * single-entry fetch, the show and one handler per action — and does no DOM
 * work of its own; gallery.js binds it to state.js and the preview pane.
 *
 * @module viewer
 */

/**
 * The actions the hub may ask for. A message that names any other action is
 * ignored.
 * @type {ReadonlyArray<ViewerAction>}
 */
export const VIEWER_ACTIONS = ['pin', 'unpin', 'print', 'logbook', 'delete'];

/** @typedef {'pin'|'unpin'|'print'|'logbook'|'delete'} ViewerAction */

/**
 * @typedef {object} ViewerDeps
 * @property {() => any[]} getHeld - the artifacts the gallery holds now
 * @property {(id: string) => Promise<any|null>} fetchEntry - the server's copy of one artifact; null when the server has none
 * @property {(entry: any|null) => void} show - show that artifact in the viewer, or the empty state for null
 * @property {Record<ViewerAction, (entry: any) => void>} actions - the gallery's code path for each action
 * @property {Window} [win] - the window the viewer listens on and announces itself from (default: this page's)
 */

/**
 * Bind the viewer to its host: listen for the hub's select and action
 * messages, then tell the parent frame that the viewer is ready.
 * @param {ViewerDeps} deps
 * @returns {void}
 */
export function createViewer({ getHeld, fetchEntry, show, actions, win = window }) {
  // Bumped by every select; a fetch that answers an older select is dropped.
  let generation = 0;

  /**
   * Hand the entry for `id` to `use`: the held one at once, else the
   * server's copy when it arrives (null when the server has none, or the
   * request failed).
   * @param {string} id
   * @param {(entry: any|null) => void} use
   */
  function withEntry(id, use) {
    const held = getHeld().find((a) => a.id === id);
    if (held) {
      use(held);
      return;
    }
    fetchEntry(id)
      .catch(() => null)
      .then(use);
  }

  /** @param {string|null} id */
  function select(id) {
    const token = ++generation;
    if (id === null) {
      show(null);
      return;
    }
    withEntry(id, (entry) => {
      if (token === generation) show(entry);
    });
  }

  /**
   * @param {string} id
   * @param {ViewerAction} action
   */
  function act(id, action) {
    withEntry(id, (entry) => {
      if (entry) actions[action](entry);
    });
  }

  win.addEventListener('message', (e) => {
    if (e.origin !== win.location.origin) return;
    const data = e.data;
    if (!data || typeof data !== 'object') return;
    const id = data.artifact_id;
    if (data.type === 'osprey-artifact-select') {
      if (id === null || (typeof id === 'string' && id !== '')) select(id);
    } else if (data.type === 'osprey-artifact-action') {
      if (typeof id === 'string' && id !== '' && VIEWER_ACTIONS.includes(data.action)) {
        act(id, data.action);
      }
    }
  });

  if (win.parent !== win) {
    win.parent.postMessage({ type: 'osprey-artifact-ready' }, win.location.origin);
  }
}
