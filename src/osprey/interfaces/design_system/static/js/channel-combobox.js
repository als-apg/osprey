// @ts-check
/**
 * Channel typeahead combobox for plan-form channel fields.
 *
 * DOM-touching companion to the pure channel-matcher module: it owns the
 * anchored suggestion popover (position:relative wrapper, `.open` class,
 * capture-phase document listener installed on open / removed on close) and
 * the ARIA combobox contract, while all ranking stays in channel-matcher.
 * Suggestions never block free-form entry — a value outside the catalog is
 * typed exactly as into a bare input. Hand-written ES module (mirrors
 * theme-manager.js) with JSDoc types so it type-checks under
 * `tsc --noEmit --strict`.
 *
 * Selection semantics (load-bearing, pinned by the tests):
 *   - no default highlight; aria-activedescendant moves only via
 *     ArrowUp/ArrowDown (wrapping), hover is visual-only + click-select
 *   - option rows preventDefault() on mousedown so the input keeps focus
 *   - Enter accepts only an arrow-armed item (then stops propagation);
 *     unarmed Enter closes the popup and propagates so the form's own
 *     Enter handling still runs
 *   - Escape closes; input re-queries the matcher behind a debounce
 *
 * @module channel-combobox
 */

import { matchChannels } from '/design-system/js/channel-matcher.js';
import { el, debounce } from '/design-system/js/dom.js';

/**
 * Trailing-edge delay between the last keystroke and a matcher query.
 * Exported so tests advance fake timers by exactly this amount.
 *
 * @type {number}
 */
export const DEFAULT_DEBOUNCE_MS = 150;

/**
 * Monotonic source for per-instance listbox/option ids, so several
 * comboboxes on one form never collide on aria-controls targets.
 *
 * @type {number}
 */
let comboboxSeq = 0;

/**
 * Tallest the popup may grow, and the gap it keeps from its input. Shared by
 * the initial style block and the per-open placement pass.
 *
 * @type {number}
 */
const MAX_POPUP_HEIGHT_PX = 240;
const POPUP_GAP_PX = 2;

/**
 * Below this many pixels of room, opening downward is not worth it and the
 * popup flips above the input instead.
 *
 * @type {number}
 */
const MIN_POPUP_HEIGHT_PX = 96;

/**
 * Row backgrounds for the arrow-armed option and for plain hover.
 *
 * Accent tints rather than a surface token: the popup already sits on
 * `--bg-elevated`, and in the `light` theme `--bg-secondary` is defined to the
 * very same colour, so painting the armed row with it left keyboard users no
 * visible cursor at all. A tint is an alpha overlay, so it reads against every
 * theme's popup surface by construction. Arming is also the stronger of the
 * two states -- hover explicitly never arms -- so it gets the stronger tint,
 * and the two states stay tellable apart when the pointer rests on a row that
 * is not the armed one.
 *
 * @type {string}
 */
const ARMED_BACKGROUND = 'var(--accent-tint-12)';
const HOVER_BACKGROUND = 'var(--accent-tint-06)';

/**
 * Whether the browser can promote the popup into the top layer.
 *
 * The popup is laid out `position: fixed` either way, which already escapes an
 * ancestor's `overflow: clip`/`hidden` (a fixed box's containing block is the
 * viewport, so those ancestors never clip it). The top layer additionally
 * makes it immune to ancestor stacking contexts and to the `transform`/
 * `filter`/`will-change` ancestors that would otherwise capture a fixed box's
 * containing block. Where `popover` is missing the component degrades to plain
 * fixed positioning rather than losing the popup.
 *
 * @type {boolean}
 */
// Probe the METHODS, not `'popover' in HTMLElement.prototype`: jsdom reflects
// the popover attribute without implementing the top layer, so the property
// test passes there and showPopover() then throws at the first keystroke.
const SUPPORTS_POPOVER =
  typeof HTMLElement !== 'undefined' &&
  !!HTMLElement.prototype &&
  typeof HTMLElement.prototype.showPopover === 'function' &&
  typeof HTMLElement.prototype.hidePopover === 'function';

/**
 * @typedef {Object} ChannelComboboxOptions
 * @property {number} [debounceMs] - override for DEFAULT_DEBOUNCE_MS
 * @property {(value: string) => void} [onSelect] - called after a suggestion
 *   is accepted (click or armed Enter), once the input value is updated
 */

/**
 * @typedef {Object} ChannelComboboxHandle
 * @property {() => void} close - close the popup (cancels any pending query)
 * @property {() => boolean} isOpen - whether the popup is currently open
 * @property {() => void} destroy - close, detach all listeners, unwrap the
 *   input back to its original parent and drop the combobox DOM
 */

/**
 * Append `text` to `container` with highlight spans over the matched ranges.
 * Built from text nodes + span elements (never innerHTML) so catalog values
 * containing markup render as literal text — the escapeHtml helper is
 * HTML-context-safe only, and building nodes sidesteps escaping entirely.
 *
 * @param {HTMLElement} container
 * @param {string} text
 * @param {Array<[number, number]>} spans - merged half-open [start,end) pairs
 * @returns {void}
 */
function appendHighlighted(container, text, spans) {
  let cursor = 0;
  for (const [start, end] of spans) {
    if (start > cursor) container.appendChild(document.createTextNode(text.slice(cursor, start)));
    const mark = el('span', 'channel-combobox-match');
    mark.style.color = 'var(--color-accent)';
    mark.style.fontWeight = '600';
    mark.textContent = text.slice(start, end);
    container.appendChild(mark);
    cursor = end;
  }
  if (cursor < text.length) container.appendChild(document.createTextNode(text.slice(cursor)));
}

/**
 * Attach a channel-suggestion combobox to a text input.
 *
 * Wraps the input in a position:relative `.channel-combobox` container and
 * floats a listbox popover at the input's viewport rect. The popup is a
 * top-layer `popover` laid out `position: fixed`, deliberately NOT an
 * absolutely-positioned child of the wrapper: several of the containers these
 * fields render into clip their overflow for rounded corners (`.channel-list`
 * with `overflow: hidden`, `.obj-table` with `overflow: clip`), which cropped
 * an in-flow popup down to a few pixels while every DOM-level signal —
 * aria-expanded, computed visibility, the option children — still read as a
 * healthy open popup. The component only ever *suggests*:
 * it rewrites `input.value` solely when the user accepts a suggestion, and a
 * programmatic accept re-dispatches a bubbling `input` event so form-level
 * listeners (e.g. schema-form model sync) observe the new value.
 *
 * Styling note: the design system has dropdown *tokens* but no shared
 * dropdown component classes, so the popover carries minimal inline styles
 * built from tokens (`--bg-elevated`, `--border-default`,
 * `--shadow-dropdown`, `--z-dropdown`, ...). The class hooks
 * (`channel-combobox`, `-list`, `-item`, `-match`, `-more`, `.open`) exist
 * for future promotion into a stylesheet.
 *
 * @param {HTMLInputElement} input - must already be in the DOM
 * @param {string[] | (() => string[])} catalog - channel address strings, or
 *   a getter re-read on every query so late-loading catalogs stay live
 * @param {ChannelComboboxOptions} [opts]
 * @returns {ChannelComboboxHandle}
 */
export function attachChannelCombobox(input, catalog, opts = {}) {
  const { debounceMs = DEFAULT_DEBOUNCE_MS, onSelect } = opts;
  const getCatalog = typeof catalog === 'function' ? catalog : () => catalog;

  const host = input.parentNode;
  if (!host) throw new Error('attachChannelCombobox: input must be attached to the DOM');

  const listId = `channel-combobox-list-${++comboboxSeq}`;
  const wrapper = el('div', 'channel-combobox');
  wrapper.style.position = 'relative';
  host.insertBefore(wrapper, input);
  wrapper.appendChild(input);

  const listbox = el('div', 'channel-combobox-list');
  listbox.id = listId;
  listbox.setAttribute('role', 'listbox');
  // No inline `display`: an inline declaration outranks the UA rule that hides
  // a closed popover (`[popover]:not(:popover-open) { display: none }`), so
  // setting it here would leave the popup permanently on screen. Visibility is
  // owned by showPopover()/hidePopover(), or by open()/close() in the
  // no-popover fallback.
  if (SUPPORTS_POPOVER) listbox.setAttribute('popover', 'manual');
  Object.assign(listbox.style, {
    // Fixed, not absolute: anchoring inside the wrapper meant any ancestor
    // that clips (`.channel-list` has `overflow: hidden` and `.obj-table`
    // `overflow: clip`, both for their rounded corners) cropped the popup to
    // an unusable sliver. Coordinates are written by place() on every open.
    position: 'fixed',
    // Neutralize the UA popover box, which otherwise centres itself in the
    // viewport with `inset: 0` and `margin: auto`.
    inset: 'auto',
    margin: '0',
    padding: '0',
    maxHeight: `${MAX_POPUP_HEIGHT_PX}px`,
    overflowY: 'auto',
    zIndex: 'var(--z-dropdown)',
    background: 'var(--bg-elevated)',
    border: '1px solid var(--border-default)',
    borderRadius: 'var(--radius-md)',
    boxShadow: 'var(--shadow-dropdown)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    color: 'var(--text-primary)',
  });
  if (!SUPPORTS_POPOVER) listbox.style.display = 'none';
  wrapper.appendChild(listbox);

  input.setAttribute('role', 'combobox');
  input.setAttribute('aria-expanded', 'false');
  input.setAttribute('aria-autocomplete', 'list');
  input.setAttribute('aria-controls', listId);

  /**
   * Currently rendered option rows in listbox order. Rebuilt on every query;
   * the "+N more" footer is deliberately NOT in here so arrow navigation and
   * Enter can never land on it.
   *
   * @type {Array<{ row: HTMLElement, value: string }>}
   */
  let options = [];
  /** Index of the arrow-armed option, -1 = none (the no-default-highlight rule). */
  let armed = -1;
  let isOpen = false;
  /**
   * Bumped on every user-intent close (Escape, select, outside click, blur)
   * so a debounced query scheduled before the close aborts instead of
   * reopening a popup the user just dismissed.
   */
  let generation = 0;
  /** True while the accept path re-dispatches its synthetic `input` event. */
  let suppressInput = false;

  /**
   * Paint a row's hover/armed background. Inline because there is no
   * stylesheet to hang :hover on (see the styling note above).
   *
   * @param {number} index
   * @param {boolean} on
   * @returns {void}
   */
  function paintRow(index, on) {
    const opt = options[index];
    if (opt) opt.row.style.background = on ? ARMED_BACKGROUND : 'transparent';
  }

  /**
   * Move the armed highlight. Owns the full contract: aria-selected +
   * background on the rows, aria-activedescendant on the input (removed when
   * nothing is armed), and keeping the armed row scrolled into view.
   *
   * @param {number} next - option index, or -1 to disarm
   * @returns {void}
   */
  function setArmed(next) {
    if (armed !== -1 && options[armed]) {
      options[armed].row.setAttribute('aria-selected', 'false');
      paintRow(armed, false);
    }
    armed = next;
    if (armed === -1) {
      input.removeAttribute('aria-activedescendant');
      return;
    }
    const { row } = options[armed];
    row.setAttribute('aria-selected', 'true');
    paintRow(armed, true);
    input.setAttribute('aria-activedescendant', row.id);
    if (typeof row.scrollIntoView === 'function') row.scrollIntoView({ block: 'nearest' });
  }

  /**
   * Close triggered by explicit user intent — additionally cancels any
   * pending debounced query via the generation counter, so the popup stays
   * dismissed even when a keystroke was still in flight.
   *
   * @returns {void}
   */
  function dismiss() {
    generation += 1;
    close();
  }

  /**
   * Outside-pointer dismissal. Installed capture-phase on the document only
   * while the popup is open, so a click anywhere (even inside widgets that
   * stop propagation) closes it; clicks inside the wrapper are the popup's
   * own business and are left alone.
   *
   * @param {MouseEvent} e
   * @returns {void}
   */
  function onDocMousedown(e) {
    if (!(e.target instanceof Node) || !wrapper.contains(e.target)) dismiss();
  }

  /**
   * Pin the popup to the input's current viewport rect.
   *
   * Runs on every open and again whenever the viewport moves under an open
   * popup, because a fixed box does not travel with the scroller its input
   * lives in. The popup flips above the input when the room below cannot seat
   * a usable list — these fields sit in a scrollable panel, so a field near
   * the bottom edge would otherwise open a popup off-screen, which reads to
   * the user exactly like the clipping bug this replaced.
   *
   * @returns {void}
   */
  function place() {
    const rect = input.getBoundingClientRect();
    const below = window.innerHeight - rect.bottom - POPUP_GAP_PX;
    const above = rect.top - POPUP_GAP_PX;
    const flip = below < MIN_POPUP_HEIGHT_PX && above > below;
    const room = Math.max(MIN_POPUP_HEIGHT_PX, Math.min(MAX_POPUP_HEIGHT_PX, flip ? above : below));

    listbox.style.left = `${rect.left}px`;
    listbox.style.width = `${rect.width}px`;
    listbox.style.maxHeight = `${room}px`;
    if (flip) {
      listbox.style.top = 'auto';
      listbox.style.bottom = `${window.innerHeight - rect.top + POPUP_GAP_PX}px`;
    } else {
      listbox.style.bottom = 'auto';
      listbox.style.top = `${rect.bottom + POPUP_GAP_PX}px`;
    }
  }

  /**
   * Re-place an open popup. Scroll is observed in the capture phase because
   * scroll events on an inner scroller do not bubble to window.
   *
   * @returns {void}
   */
  function onViewportChange() {
    if (isOpen) place();
  }

  /** @returns {void} */
  function open() {
    if (isOpen) return;
    // A debounced query can land after a form re-render detached this
    // combobox: showPopover() on a disconnected element throws, and a popup
    // for a field that is gone has nothing to point at anyway.
    if (!listbox.isConnected) return;
    isOpen = true;
    wrapper.classList.add('open');
    place();
    if (SUPPORTS_POPOVER) {
      if (!listbox.matches(':popover-open')) listbox.showPopover();
    } else listbox.style.display = 'block';
    input.setAttribute('aria-expanded', 'true');
    document.addEventListener('mousedown', onDocMousedown, true);
    window.addEventListener('scroll', onViewportChange, true);
    window.addEventListener('resize', onViewportChange);
  }

  /** @returns {void} */
  function close() {
    if (!isOpen) return;
    isOpen = false;
    setArmed(-1);
    wrapper.classList.remove('open');
    if (SUPPORTS_POPOVER) {
      // Removing a showing popover from the DOM already hides it, so a
      // close() on the way out of a re-render must not call hidePopover again.
      if (listbox.isConnected && listbox.matches(':popover-open')) listbox.hidePopover();
    } else listbox.style.display = 'none';
    input.setAttribute('aria-expanded', 'false');
    document.removeEventListener('mousedown', onDocMousedown, true);
    window.removeEventListener('scroll', onViewportChange, true);
    window.removeEventListener('resize', onViewportChange);
  }

  /**
   * Accept a suggestion: write it into the input, notify the outside world,
   * and dismiss. The synthetic bubbling `input` event exists so form-level
   * listeners see the programmatic value change exactly like a keystroke;
   * `suppressInput` keeps the combobox's own input handler from treating it
   * as typing and re-querying.
   *
   * @param {string} value
   * @returns {void}
   */
  function select(value) {
    input.value = value;
    suppressInput = true;
    try {
      input.dispatchEvent(new Event('input', { bubbles: true }));
    } finally {
      suppressInput = false;
    }
    if (onSelect) onSelect(value);
    dismiss();
  }

  /**
   * Build one option row: highlight-rendered label, mousedown-preventDefault
   * (keeps focus in the input), click-select, and visual-only hover — hover
   * never touches aria-activedescendant, which is reserved for arrow keys.
   *
   * @param {{ value: string, spans: Array<[number, number]> }} match
   * @param {number} index
   * @returns {HTMLElement}
   */
  function buildRow(match, index) {
    const row = el('div', 'channel-combobox-item');
    row.id = `${listId}-opt-${index}`;
    row.setAttribute('role', 'option');
    row.setAttribute('aria-selected', 'false');
    Object.assign(row.style, {
      padding: 'var(--space-1) var(--space-2)',
      cursor: 'pointer',
      whiteSpace: 'nowrap',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
    });
    appendHighlighted(row, match.value, match.spans);
    row.addEventListener('mousedown', (e) => e.preventDefault());
    row.addEventListener('click', () => select(match.value));
    row.addEventListener('mouseenter', () => {
      if (index !== armed) row.style.background = HOVER_BACKGROUND;
    });
    row.addEventListener('mouseleave', () => {
      if (index !== armed) row.style.background = 'transparent';
    });
    return row;
  }

  /**
   * Rebuild the listbox from a matcher result. Always disarms first (rows
   * are replaced wholesale, and a fresh result list must not inherit a
   * highlight — the no-default-highlight rule). A truncated pool gets an
   * aria-hidden "+N more" footer that is not an option.
   *
   * @param {{ results: Array<{ value: string, spans: Array<[number, number]> }>, total: number }} match
   * @returns {void}
   */
  function renderResults(match) {
    setArmed(-1);
    listbox.textContent = '';
    options = match.results.map((result, index) => {
      const row = buildRow(result, index);
      listbox.appendChild(row);
      return { row, value: result.value };
    });
    if (match.total > match.results.length) {
      const more = el('div', 'channel-combobox-more');
      more.setAttribute('aria-hidden', 'true');
      more.textContent = `${match.total - match.results.length} more matches — keep typing to narrow`;
      Object.assign(more.style, {
        padding: 'var(--space-1) var(--space-2)',
        color: 'var(--text-muted)',
        fontSize: 'var(--text-xs)',
      });
      more.addEventListener('mousedown', (e) => e.preventDefault());
      listbox.appendChild(more);
    }
  }

  /**
   * Run the matcher against the current input value and open/refresh or
   * close the popup accordingly. An empty field closes rather than browsing
   * the whole catalog: clearing a value is an edit, not a browse request.
   *
   * @returns {void}
   */
  function runQuery() {
    const value = input.value;
    if (value.trim() === '') {
      close();
      return;
    }
    const match = matchChannels(value, getCatalog());
    if (match.results.length === 0) {
      close();
      return;
    }
    renderResults(match);
    open();
  }

  const debouncedQuery = debounce(
    /** @param {number} gen */ (gen) => {
      if (gen === generation) runQuery();
    },
    debounceMs
  );

  /** @returns {void} */
  function onInput() {
    if (suppressInput) return;
    debouncedQuery(generation);
  }

  /**
   * Keyboard contract. Runs on the input itself (not the document): with the
   * popup closed every key falls through untouched, so free-form entry and
   * the surrounding form's own key handling are never affected.
   *
   * @param {KeyboardEvent} e
   * @returns {void}
   */
  function onKeydown(e) {
    if (!isOpen || e.isComposing) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setArmed(armed >= options.length - 1 ? 0 : armed + 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setArmed(armed <= 0 ? options.length - 1 : armed - 1);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      dismiss();
    } else if (e.key === 'Enter') {
      if (armed !== -1 && options[armed]) {
        e.preventDefault();
        e.stopPropagation();
        select(options[armed].value);
      } else {
        // Unarmed Enter: close, but let the event propagate so the form's
        // own Enter handling (submit, etc.) still fires.
        dismiss();
      }
    }
  }

  /** @returns {void} */
  function onBlur() {
    // Option mousedown preventDefault()s, so focus never leaves the input on
    // popup clicks — a real blur means the user moved on. Close behind them.
    dismiss();
  }

  input.addEventListener('input', onInput);
  input.addEventListener('keydown', onKeydown);
  input.addEventListener('blur', onBlur);

  return {
    close: dismiss,
    isOpen: () => isOpen,
    destroy() {
      dismiss();
      input.removeEventListener('input', onInput);
      input.removeEventListener('keydown', onKeydown);
      input.removeEventListener('blur', onBlur);
      input.removeAttribute('role');
      input.removeAttribute('aria-expanded');
      input.removeAttribute('aria-autocomplete');
      input.removeAttribute('aria-controls');
      input.removeAttribute('aria-activedescendant');
      const parent = wrapper.parentNode;
      if (parent) parent.insertBefore(input, wrapper);
      wrapper.remove();
    },
  };
}
