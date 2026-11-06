// @ts-check
/**
 * Paging for the session tree's scroll boxes. A box asks for its next page
 * when its end is near; a box given enough room to hold all it has never
 * scrolls, so it asks again after each render and each resize.
 */

/** How close to the end of a box (two rows) a scroll asks for the next page. */
const NEAR_END_PX = 52;

/**
 * @param {{
 *   isServed: () => boolean,
 *   loadMore: (treeId: string) => void,
 *   boxes: () => Iterable<Element>,
 * }} options
 */
export function createBoxPager({ isServed, loadMore, boxes }) {
  /** A scroll (or the box itself) near the end of a box loads its next page. @param {EventTarget | null} target */
  function pageBox(target) {
    const box = target instanceof HTMLElement ? target.closest('[data-session-box]') : null;
    if (!(box instanceof HTMLElement) || !isServed()) return;
    if (box.scrollHeight - box.scrollTop - box.clientHeight > NEAR_END_PX) return;
    loadMore(box.dataset.sessionBox ?? '');
  }

  let queued = false;
  /** Page every box the room is too big to scroll. */
  function fill() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      // A box with no layout (hidden, detached) has no room to fill.
      for (const box of boxes()) if (box.clientHeight > 0) pageBox(box);
    });
  }

  /**
   * Fill again whenever any of `elements` changes size, until `signal` aborts.
   * @param {Iterable<Element>} elements @param {AbortSignal} signal
   */
  function watch(elements, signal) {
    if (typeof ResizeObserver !== 'function') return;
    const observer = new ResizeObserver(fill);
    for (const element of elements) observer.observe(element);
    signal.addEventListener('abort', () => observer.disconnect());
  }

  /** @param {Event} event */
  const onScroll = (event) => pageBox(event.target);

  return { onScroll, fill, watch };
}
