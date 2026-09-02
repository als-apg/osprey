/* OSPREY Web Terminal — the bar drag gesture.
 *
 * One pointer gesture, four outcomes: reorder inside a bar, move to the other
 * bar, drag a new item in from the sheet, and drag one out to remove it. The
 * state machine is here; every change it decides on goes back through the
 * controller, so this file writes nothing and judges nothing.
 *
 * POINTER EVENTS, NOT HTML5 DRAG AND DROP. `dragleave.relatedTarget` is null in
 * Safari and `draggable` has to be set before the gesture starts, which makes a
 * drag out of the sheet and a drag between two hosts both awkward. Pointer
 * events behave the same everywhere and give the drop target the pointer
 * position directly.
 *
 * THE CAPTURE IS AN OPTIMISATION; THE DOCUMENT LISTENERS ARE THE CONTRACT.
 * `setPointerCapture` on the item keeps the gesture bound to it while the
 * pointer wanders over the shell, but a capture can be lost — the browser
 * releases it, or a reconcile re-parents the shell out from under it. So
 * pointermove/pointerup/pointercancel are bound to the DOCUMENT, which is what
 * guarantees the gesture always ends: a drag that never finished would leave
 * the page marked, the ghost floating and the next click swallowed.
 *
 * A REFUSED DROP IS NOT A REMOVAL. Releasing over a bar that will not take the
 * item leaves it exactly where it was, with the reason on the sheet. Only a
 * release over NEITHER bar removes, and a locked item refuses even that.
 */

import { BAR_CATALOG, BAR_HOSTS, barItemType } from './bar-catalog.js';
import { closeBarPopovers, docOf, hostElement, isLive } from './bar-host.js';

/** @typedef {import('./bar-catalog.js').BarHost} BarHost */
/** @typedef {import('./bar-customize.js').BarEditController} BarEditController */

/** How far the pointer travels before a press becomes a drag. */
const THRESHOLD = 4;

/** How far outside a bar's box still counts as being over it. */
const HOST_MARGIN = 8;

/** Where the item came from. A tile has no place in the layout yet. */
/**
 * @typedef {{kind: 'item', type: string, host: BarHost, index: number}
 *          | {kind: 'tile', type: string}} DragSource
 */

/**
 * @typedef {object} DragSession
 * @property {BarEditController} ctrl
 * @property {Document} owner
 * @property {DragSource} source
 * @property {HTMLElement} node - what the pointer went down on
 * @property {number} pointerId
 * @property {number} startX
 * @property {number} startY
 * @property {boolean} started - the threshold has been crossed
 * @property {HTMLElement | null} ghost
 * @property {BarHost | null} target - the host that will take the drop
 * @property {boolean} overBar - the pointer is over a bar, refusing or not
 * @property {number} index - where in `target` the item would land
 */

/** The gesture in flight, if any. @type {DragSession | null} */
let session = null;

/** Undo the pointerdown listener. @type {(() => void) | null} */
let stopListening = null;

/** Whether a drag just ended, so the click it produced can be ignored. */
let dragged = false;

/**
 * Whether the click now being handled is the tail of a drag. The options
 * popover asks before opening: a release that ended a drag is not a click on
 * the item.
 * @returns {boolean}
 */
export function dragJustEnded() {
  return dragged;
}

/* ---- arming ---- */

/**
 * Listen for drags. Armed when edit mode opens and disarmed when it closes —
 * outside edit mode an item is a control, and a press on it belongs to the
 * control.
 * @param {BarEditController} ctrl
 * @param {Document | Element} root
 */
export function armDrag(ctrl, root) {
  disarmDrag();
  const owner = docOf(root);
  /** @param {Event} event */
  const onDown = (event) => beginDrag(/** @type {PointerEvent} */ (event), ctrl, owner);
  owner.addEventListener('pointerdown', onDown);
  stopListening = () => owner.removeEventListener('pointerdown', onDown);
}

/** Stop listening, ending any gesture in flight without applying it. */
export function disarmDrag() {
  if (session) finish(false, null);
  stopListening?.();
  stopListening = null;
}

/* ---- the gesture ---- */

/**
 * @param {PointerEvent} event
 * @param {BarEditController} ctrl
 * @param {Document} owner
 */
function beginDrag(event, ctrl, owner) {
  if (session || event.button !== 0) return;
  const target = event.target instanceof Element ? event.target : null;
  if (!target) return;
  const source = sourceOf(target, ctrl);
  if (!source) return;

  session = {
    ctrl,
    owner,
    source: source.source,
    node: source.node,
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    started: false,
    ghost: null,
    target: null,
    overBar: false,
    index: 0,
  };
  try {
    source.node.setPointerCapture(event.pointerId);
  } catch {
    // No capture — the document listeners below are what actually end the drag.
  }
  owner.addEventListener('pointermove', onPointerMove);
  owner.addEventListener('pointerup', onPointerUp);
  owner.addEventListener('pointercancel', onPointerCancel);
  event.preventDefault();
}

/**
 * What the pointer went down on: a placed item, a sheet tile, or nothing this
 * module handles.
 * @param {Element} target
 * @param {BarEditController} ctrl
 * @returns {{source: DragSource, node: HTMLElement} | null}
 */
function sourceOf(target, ctrl) {
  // A press inside an open popover is that popover's own — its controls sit
  // over the item, and dragging the item out from under them is not the
  // gesture the operator is making.
  if (target.closest('.bar-pop, .bar-context-menu')) return null;
  const tile = /** @type {HTMLButtonElement | null} */ (target.closest('.bar-tile'));
  if (tile) {
    const type = tile.dataset.barTile ?? '';
    if (tile.disabled || !barItemType(type)) return null;
    return { source: { kind: 'tile', type }, node: tile };
  }
  const shell = /** @type {HTMLElement | null} */ (target.closest('.bar-item[data-bar-item]'));
  if (!shell || !isLive(shell)) return null;
  const at = ctrl.locate(shell);
  if (!at) return null;
  return { source: { kind: 'item', type: at.type, host: at.host, index: at.index }, node: shell };
}

/**
 * Whether an event belongs to the gesture in flight. The listeners are on the
 * document, so a SECOND pointer — another finger, a pen arriving while a mouse
 * button is down — reaches them too, and acting on it would aim and commit this
 * drag from coordinates the operator never dragged through. The capture would
 * have filtered them; standing in for the capture means filtering them here.
 * @param {Event} event
 * @returns {boolean}
 */
function ours(event) {
  return !!session && /** @type {PointerEvent} */ (event).pointerId === session.pointerId;
}

/** @param {Event} event */
function onPointerMove(event) {
  const pointer = /** @type {PointerEvent} */ (event);
  if (!ours(event)) return;
  if (!session) return;
  if (!session.started) {
    const far =
      Math.abs(pointer.clientX - session.startX) >= THRESHOLD ||
      Math.abs(pointer.clientY - session.startY) >= THRESHOLD;
    if (!far) return;
    start();
  }
  if (session.ghost) {
    session.ghost.style.transform = `translate(${pointer.clientX + 10}px, ${pointer.clientY + 10}px)`;
  }
  aim(pointer);
}

/** @param {Event} event */
function onPointerUp(event) {
  if (!ours(event)) return;
  finish(true, /** @type {PointerEvent} */ (event));
}

/** @param {Event} event */
function onPointerCancel(event) {
  if (!ours(event)) return;
  finish(false, null);
}

/**
 * The threshold has been crossed. An item about to move must not take an open
 * popover with it — the same invariant the reconcile keeps when it moves a
 * shell between hosts.
 */
function start() {
  if (!session) return;
  session.started = true;
  closeBarPopovers();
  session.owner.body?.classList.add('bar-dragging');
  session.node.classList.add('is-bar-dragging');
  const ghost = session.owner.createElement('div');
  ghost.className = 'bar-drag-ghost';
  ghost.dataset.type = session.source.type;
  ghost.textContent = BAR_CATALOG[session.source.type]?.label ?? session.source.type;
  session.owner.body?.append(ghost);
  session.ghost = ghost;
}

/**
 * Work out where the pointer is: which bar it is over, whether that bar will
 * take this item, and where in it the item would land. A bar that refuses says
 * so on the sheet as the pointer crosses it, rather than at the drop.
 * @param {PointerEvent} event
 */
function aim(event) {
  if (!session) return;
  const { ctrl, source } = session;
  session.target = null;
  session.overBar = false;
  clearMarkers(session.owner);

  for (const host of BAR_HOSTS) {
    const container = hostElement(host, session.owner);
    if (!container) continue;
    const box = container.getBoundingClientRect();
    if (event.clientY < box.top - HOST_MARGIN || event.clientY > box.bottom + HOST_MARGIN) continue;
    session.overBar = true;
    const from = source.kind === 'item' ? source.host : null;
    const refusal = ctrl.dropRefusal(source.type, host, from);
    if (refusal) {
      ctrl.notice(refusal);
      return;
    }
    const drop = dropAt(container, event.clientX, ctrl, host);
    session.target = host;
    session.index = drop.index;
    markAt(container, box, drop.x);
    ctrl.notice('');
    return;
  }
  ctrl.notice('');
}

/**
 * Where an item released at `clientX` would land in `container`, and where to
 * draw the marker for it.
 *
 * The index is the LAYOUT's, not the DOM's. The two are not the same list: the
 * overflow ladder folds an item by parking its shell in the pool, so a placed
 * item can legitimately render no shell in the host. Counting DOM children
 * would then insert before every folded item — a drop on a crowded bar quietly
 * writing an arrangement nobody made. So the visible shell the pointer is next
 * to is mapped back to its own position through the controller, and the
 * insertion point is stated in those terms.
 * @param {Element} container
 * @param {number} clientX
 * @param {BarEditController} ctrl
 * @param {BarHost} host
 * @returns {{index: number, x: number}}
 */
function dropAt(container, clientX, ctrl, host) {
  const shells = /** @type {HTMLElement[]} */ (
    Array.from(container.querySelectorAll('.bar-item[data-bar-item]'))
  );
  for (const shell of shells) {
    const box = shell.getBoundingClientRect();
    if (clientX < box.left + box.width / 2) {
      return { index: ctrl.locate(shell)?.index ?? 0, x: box.left };
    }
  }
  // Past everything visible. "After the last item I can see" is the honest
  // reading of the gesture, so a folded tail keeps its place behind the drop.
  const last = shells[shells.length - 1];
  const at = last ? ctrl.locate(last) : null;
  return {
    index: at ? at.index + 1 : ctrl.hostItems(host).length,
    x: last ? last.getBoundingClientRect().right : container.getBoundingClientRect().left,
  };
}

/**
 * Draw the insertion marker inside a host. It is a child of the host so it
 * scrolls and moves with the bar; `left` is relative to the host's own box.
 * @param {Element} container
 * @param {DOMRect} box
 * @param {number} x
 */
function markAt(container, box, x) {
  const owner = container.ownerDocument;
  const marker = owner.createElement('div');
  marker.className = 'bar-drop-marker';
  marker.style.left = `${Math.max(0, x - box.left)}px`;
  container.append(marker);
}

/** @param {Document} owner */
function clearMarkers(owner) {
  for (const marker of owner.querySelectorAll('.bar-drop-marker')) marker.remove();
}

/**
 * End the gesture and, when it was a real drag released over something,
 * apply it.
 * @param {boolean} commit
 * @param {PointerEvent | null} event
 */
function finish(commit, event) {
  const active = session;
  session = null;
  if (!active) return;

  active.owner.removeEventListener('pointermove', onPointerMove);
  active.owner.removeEventListener('pointerup', onPointerUp);
  active.owner.removeEventListener('pointercancel', onPointerCancel);
  try {
    active.node.releasePointerCapture(active.pointerId);
  } catch {
    // Already released, or never captured.
  }
  active.ghost?.remove();
  active.node.classList.remove('is-bar-dragging');
  active.owner.body?.classList.remove('bar-dragging');
  clearMarkers(active.owner);
  if (!active.started) return;

  dragged = true;
  setTimeout(() => {
    dragged = false;
  }, 0);
  if (!commit || !event) return;
  void apply(active);
}

/**
 * Turn a finished drag into one edit.
 * @param {DragSession} active
 * @returns {Promise<boolean>}
 */
function apply(active) {
  const { ctrl, source, target } = active;
  if (target) {
    if (source.kind === 'tile') return ctrl.addItem(source.type, target, active.index);
    return ctrl.moveItem(source.host, source.index, target, active.index);
  }
  // Released over a bar that refused it: the item stays, and the refusal is
  // already on the sheet.
  if (active.overBar || source.kind === 'tile') return Promise.resolve(false);
  return ctrl.removeAt(source.host, source.index);
}
