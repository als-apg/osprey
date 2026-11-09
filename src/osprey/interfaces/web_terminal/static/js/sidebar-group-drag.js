// @ts-check
/* OSPREY Web Terminal — dragging a whole sidebar group by its heading
 *
 * A group's heading (sidebar-groups.js paints it `draggable`) is a drag
 * handle for the group: dragged along the rail and dropped, the group, its
 * heading and its rows together, moves among the other groups, and the new
 * order is stored in the groups model (sidebar-groups.js placeGroup) like
 * every other change of it. A folded group drags as its heading, since that
 * is all of it there is to see. A group never splits or merges: a group drag
 * only ever changes the order of the groups.
 *
 * The headless Sessions group is never dragged and never a target, so
 * nothing moves above it. With the sidebar collapsed to icons, a heading is
 * a hairline and nothing drags.
 *
 * The drop line is the rows' own (terminal.css `.drop-before` /
 * `.drop-after`): above the group the dragged one would land in front of, or
 * under the last group for a drop at the end. The dragged group is
 * `.dragging`, dimmed like a dragged row. A drop that would leave the order
 * as it is draws no line and does nothing.
 *
 * A row drag is untouched: the listeners below act only while a heading is
 * being dragged, and the rows' own listeners (panel-rail.js) only while a row
 * is. The drag carries its own type, so the dock (dock-drop.js), which reads
 * only its own sources, ignores it.
 *
 * The keyboard's route to the same move is the group menu's Move up and
 * Move down (sidebar-groups.js).
 */

import { isHeadlessGroup, placeGroup } from './sidebar-groups.js';

/** DataTransfer type carrying the dragged group's id. */
export const GROUP_DRAG_MIME = 'application/x-osprey-sidebar-group';

const DROP_BEFORE = 'drop-before';
const DROP_AFTER = 'drop-after';

/** Rails already wired. @type {WeakSet<HTMLElement>} */
const wiredRails = new WeakSet();

/**
 * The group elements of a rail, top to bottom.
 * @param {HTMLElement} railEl
 * @returns {HTMLElement[]}
 */
function groupsOf(railEl) {
  return /** @type {HTMLElement[]} */ (
    Array.from(railEl.querySelectorAll(':scope > .sidebar-group[data-group]'))
  );
}

/** @param {HTMLElement} railEl */
function clearMarks(railEl) {
  for (const el of railEl.querySelectorAll(
    `:scope > .sidebar-group.${DROP_BEFORE}, :scope > .sidebar-group.${DROP_AFTER}`
  )) {
    el.classList.remove(DROP_BEFORE, DROP_AFTER);
  }
}

/**
 * Where a drop at the event's height puts the dragged group: in front of the
 * first other movable group whose middle lies below the pointer, or at the
 * end when there is none.
 * @param {HTMLElement} railEl
 * @param {HTMLElement} dragged
 * @param {DragEvent} e
 * @returns {{ before: HTMLElement | null, last: HTMLElement | null, noop: boolean }}
 */
function insertionPoint(railEl, dragged, e) {
  const movable = groupsOf(railEl).filter(
    (el) => !isHeadlessGroup(el.getAttribute('data-group') ?? '')
  );
  const others = movable.filter((el) => el !== dragged);
  const before =
    others.find((el) => {
      const r = el.getBoundingClientRect();
      return r.top + r.height / 2 > e.clientY;
    }) ?? null;
  const next = movable[movable.indexOf(dragged) + 1] ?? null;
  return { before, last: others.at(-1) ?? null, noop: before === next };
}

/**
 * Wire group dragging on a rail, once per rail element. The listeners sit on
 * the rail, so a render that moves or repaints a heading loses nothing.
 * @param {HTMLElement} railEl
 */
export function wireGroupDrag(railEl) {
  if (wiredRails.has(railEl)) return;
  wiredRails.add(railEl);

  /** The group being dragged, or null. @type {HTMLElement | null} */
  let dragged = null;

  const end = () => {
    dragged?.classList.remove('dragging');
    dragged = null;
    clearMarks(railEl);
  };

  railEl.addEventListener('dragstart', (e) => {
    const target = e.target instanceof Element ? e.target : null;
    const heading = target?.closest('.sidebar-group-head');
    if (!heading) return; // a row's drag: panel-rail.js handles it
    const groupEl = /** @type {HTMLElement | null} */ (heading.parentElement);
    const id = groupEl?.getAttribute('data-group') ?? '';
    const collapsed = document.documentElement.getAttribute('data-sidebar') === 'collapsed';
    if (!groupEl || groupEl.parentElement !== railEl || isHeadlessGroup(id) || collapsed) {
      e.preventDefault();
      return;
    }
    end();
    dragged = groupEl;
    groupEl.classList.add('dragging');
    if (e.dataTransfer) {
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData(GROUP_DRAG_MIME, id);
      e.dataTransfer.setDragImage?.(heading, 8, 8);
    }
  });

  railEl.addEventListener('dragover', (e) => {
    if (!dragged) return;
    e.preventDefault(); // accept the drop here
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';
    clearMarks(railEl);
    const point = insertionPoint(railEl, dragged, e);
    if (point.noop) return;
    if (point.before) point.before.classList.add(DROP_BEFORE);
    else point.last?.classList.add(DROP_AFTER);
  });

  railEl.addEventListener('dragleave', (e) => {
    const to = /** @type {Node | null} */ (e.relatedTarget);
    if (dragged && (!to || !railEl.contains(to))) clearMarks(railEl);
  });

  railEl.addEventListener('drop', (e) => {
    if (!dragged) return;
    e.preventDefault();
    const point = insertionPoint(railEl, dragged, e);
    const id = dragged.getAttribute('data-group') ?? '';
    end();
    if (!point.noop) placeGroup(id, point.before?.getAttribute('data-group') ?? null);
  });

  // dragend reaches the heading that started the drag, whatever the outcome
  // (a drop elsewhere, Escape); the heading stays in the rail for a whole drag.
  railEl.addEventListener('dragend', () => {
    if (dragged) end();
  });
}
