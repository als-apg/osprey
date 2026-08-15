// @ts-check
/* OSPREY Web Terminal — Rail Drag (drag a rail entry into the workspace)
 *
 * The precise half of the open-beside verb: dragging a rail entry into the
 * workspace opens (or moves) that panel as a NEW tile exactly where it is
 * dropped. The rail entry is a plain HTML5 drag source (panel-rail.js sets
 * `draggable` and forwards dragstart/dragend here via panel-manager's
 * closures); this module bridges that FOREIGN drag into dockview's own
 * external-drag hooks, verified against the vendored dockview-core build:
 *
 *   - api.onUnhandledDragOver fires for any non-dockview drag over a dockview
 *     drop target, carrying { nativeEvent, target, position, group } plus
 *     accept(). Accepting paints dockview's native drop overlay — the same
 *     highlight a tile drag shows, so the two gestures read identically.
 *     During dragover the HTML5 payload is unreadable (dataTransfer protected
 *     mode), so acceptance keys off dataTransfer.types alone.
 *   - api.onDidDrop fires on the real drop with { nativeEvent, position,
 *     group }; there the payload IS readable and resolves the panel id.
 *
 * One panel per tile stays an invariant: only split geometries are accepted —
 * tab, header-space and center targets are refused, mirroring
 * dock-workspace's wireStackingVeto for dockview's own drags (group-model
 * vetoes do not see foreign drags, so the refusal lives here).
 *
 * Placement itself is NOT this module's job: the accepted drop is handed to
 * the caller's onDropPanel(id, position) (panel-manager routes it into
 * dock-sync's dockPanelAt + the activation tail), keeping this module free of
 * panel state.
 *
 * Iframe shields: a drag crossing an overlay iframe would have its dragover
 * events swallowed by that separate browsing context, killing dockview's drop
 * detection. dockview's own drags shield via dock-workspace's onDragGesture;
 * a rail drag starts OUTSIDE dockview, so the same two shields are raised
 * here across the gesture — the `.dock-dragging` class on .main-container
 * (terminal.css) and the adapter's inline pointer-events toggle
 * (setIframePointerShield), which the class rule cannot reach.
 */

import { getDockApi } from './dock-workspace.js';
import { setIframePointerShield } from './dock-iframe.js';

/** DataTransfer type carrying the dragged panel's rail id. */
export const RAIL_DRAG_MIME = 'application/x-osprey-panel';

/** True once the dockview listeners are attached (idempotent guard). */
let wired = false;
/** Bounded retry so late DockviewApi arrival still wires (mirrors dock-sync). */
let wireAttempts = 0;
/** @type {((id: string, position: {referenceGroup?: any, direction: string} | null) => void) | null} */
let dropHandler = null;

/** @param {any} nativeEvent @returns {boolean} whether the drag carries our MIME */
function isRailDrag(nativeEvent) {
  const types = nativeEvent?.dataTransfer?.types;
  return !!types && Array.from(types).includes(RAIL_DRAG_MIME);
}

/**
 * Drop-overlay position → addPanel direction. The two vocabularies differ on
 * the vertical axis ('top'/'bottom' vs 'above'/'below'); addPanel throws
 * "invalid direction" on the untranslated values. 'center' is deliberately
 * absent — a center drop would stack, which the accept veto already refuses.
 * @type {Record<string, string>}
 */
const DIRECTION_BY_POSITION = {
  left: 'left',
  right: 'right',
  top: 'above',
  bottom: 'below',
};

/**
 * A drag-over geometry that would stack panels as tabs in one tile — the
 * foreign-drag twin of dock-workspace's isStackingDrop (that one reads
 * `kind`; the unhandled-drag-over event calls the same field `target`).
 * @param {any} e  an UnhandledDragOverEvent
 * @returns {boolean}
 */
function isStackingTarget(e) {
  return e.target === 'tab' || e.target === 'header_space'
    || (e.target === 'content' && e.position === 'center');
}

/**
 * Wire the dockview external-drag hooks and register the drop handler. Safe
 * to call repeatedly (idempotent); polls a bounded number of times while the
 * DockviewApi has not arrived yet, exactly like initDockSync.
 * @param {{ onDropPanel: (id: string, position: {referenceGroup?: any, direction: string} | null) => void }} options
 */
export function initRailDrag({ onDropPanel }) {
  dropHandler = onDropPanel;
  wire();
}

function wire() {
  if (wired) return;
  if (typeof document === 'undefined') return;
  const api = getDockApi();
  if (api) {
    api.onUnhandledDragOver((/** @type {any} */ e) => {
      if (!isRailDrag(e.nativeEvent)) return; // not ours (file drag, text, …)
      if (isStackingTarget(e)) return;        // one panel per tile — never stack
      e.accept();
    });
    api.onDidDrop((/** @type {any} */ e) => {
      const id = e?.nativeEvent?.dataTransfer?.getData?.(RAIL_DRAG_MIME);
      if (!id) return;                 // a dockview-internal or foreign drop
      const direction = DIRECTION_BY_POSITION[e.position];
      if (!direction) return;          // 'center' (stacking backstop) or unknown
      const position = e.group ? { referenceGroup: e.group, direction } : { direction };
      dropHandler?.(id, position);
    });
    wired = true;
    return;
  }
  if (wireAttempts++ > 30) return; // ~4.5s at 150ms — the shell is genuinely absent
  setTimeout(wire, 150);
}

// The gesture's natural terminators, mirroring dock-workspace.js's
// onDragGesture. HTML5 delivers dragend only to the drag's SOURCE element —
// a rail entry removed mid-drag (SSE remove_panel_from_rail, arrange prune, another
// client) can never fire it, and native drags suppress mouseup/pointerup
// until the pointer is released. Relying on the entry's own dragend alone
// therefore left the shields up FOREVER on a lost gesture: every panel
// iframe stayed pointer-events:none — rendering, but inert. These document
// listeners are the failsafe: capture-phase, self-disarming, armed per drag.
const DRAG_TERMINATORS = ['dragend', 'drop', 'mouseup', 'pointerup'];

/** The armed failsafe's listener, or null while no drag is in flight.
 *  @type {(() => void) | null} */
let failsafeLower = null;

function armFailsafe() {
  if (failsafeLower) return;
  const lower = () => railDragEnd();
  failsafeLower = lower;
  for (const type of DRAG_TERMINATORS) document.addEventListener(type, lower, true);
}

function disarmFailsafe() {
  if (!failsafeLower) return;
  for (const type of DRAG_TERMINATORS) document.removeEventListener(type, failsafeLower, true);
  failsafeLower = null;
}

/**
 * Begin a rail drag: stamp the payload and raise the iframe shields. Returns
 * false — cancelling the drag (panel-rail preventDefaults) — in simple mode
 * (locked layout) and in fallback mode (no dock to drop into).
 * @param {string} id  panel-manager rail id
 * @param {DataTransfer | null} dataTransfer
 * @returns {boolean} whether the drag may proceed
 */
export function railDragStart(id, dataTransfer) {
  if (document.documentElement.getAttribute('data-ui-mode') === 'simple') return false;
  if (!getDockApi()) return false;
  if (dataTransfer) {
    dataTransfer.setData(RAIL_DRAG_MIME, id);
    // text/plain fallback so a stray drop outside the app degrades harmlessly.
    dataTransfer.setData('text/plain', id);
    dataTransfer.effectAllowed = 'move';
  }
  shield(true);
  armFailsafe();
  return true;
}

/** End a rail drag (drop, cancel, escape, or failsafe): lower the iframe
 *  shields and disarm the document-level terminators. Idempotent. */
export function railDragEnd() {
  disarmFailsafe();
  shield(false);
}

/** @param {boolean} on */
function shield(on) {
  document.querySelector('.main-container')?.classList.toggle('dock-dragging', on);
  setIframePointerShield(on);
}
