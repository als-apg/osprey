// @ts-check
/* OSPREY Web Terminal — which panel the dock surfaces
 *
 * The sidebar's highlights are derived from the dock, never remembered. A tool
 * row is accented (`.active`, `aria-current="page"`) iff its panel is the one
 * the workspace's service tile shows: the tile the operator used last, else
 * the first service tile (dock-open.js's serviceTileTarget). Focusing the
 * Agent pane keeps that accent; a layout with no service tile has none.
 *
 * panel-manager.js applies the answer (its active-tab state, the rail, the
 * container stamp) after every whole-layout apply — a session's own layout
 * above all — and on every settled dock change this module follows, then
 * announces {@link DOCK_VIEW_EVENT} so the session tree re-derives its own
 * highlight (sidebar-artifacts.js: the artifact the Artifacts panel shows).
 */

import { getLayoutHolder } from './dock-apply.js';
import { serviceTileTarget } from './dock-open.js';
import { getDockApi } from './dock-workspace.js';
import { isPoppedOut, onPopoutChange } from './popout.js';

/** Dispatched on `document` each time the surfaced panel was derived again. */
export const DOCK_VIEW_EVENT = 'osprey-dock-view-changed';

let wired = false;

/**
 * The panel the dock surfaces, or null when it holds no service tile.
 * Undefined without a layout holder (the fallback host), where the caller's
 * own bookkeeping is all there is.
 * @param {(id: string) => boolean} isKnown - the page knows this panel
 * @returns {string | null | undefined}
 */
export function surfacedPanel(isKnown) {
  const holder = getLayoutHolder();
  if (!holder) return undefined;
  const key = serviceTileTarget(holder.layout, holder.context())?.showing ?? null;
  return key && isKnown(key) && !isPoppedOut(key) ? key : null;
}

/**
 * Apply the surfaced panel (panel-manager's `mark`) and announce it. A no-op
 * without a layout holder.
 * @param {(id: string) => boolean} isKnown
 * @param {(id: string | null) => void} mark
 * @returns {boolean | undefined} whether the dock holds a service tile;
 *   undefined without a layout holder
 */
export function deriveSurfaced(isKnown, mark) {
  const id = surfacedPanel(isKnown);
  if (id === undefined) return undefined;
  mark(id);
  document.dispatchEvent(new CustomEvent(DOCK_VIEW_EVENT));
  const holder = getLayoutHolder();
  return !!holder && serviceTileTarget(holder.layout, holder.context()) !== null;
}

/**
 * Call `onChange` on every settled dock change: a layout change (a tile opened,
 * closed, moved), a focus change, a panel leaving for or returning from its own
 * window. The holder's own rebuild is skipped: it settles through the
 * placeholders-applied seam. Wired once, and only with a dock.
 * @param {() => void} onChange
 */
export function followDock(onChange) {
  const api = getDockApi();
  if (wired || !api) return;
  wired = true;
  const follow = () => {
    if (!getLayoutHolder()?.applying) onChange();
  };
  api.onDidLayoutChange(follow);
  api.onDidActivePanelChange(follow);
  onPopoutChange(follow);
}
