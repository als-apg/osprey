// @ts-check
/* OSPREY Web Terminal — Vertical Panel Rail (DOM renderer)
 *
 * A pure DOM-rendering module for the 74px left icon rail that replaces the
 * horizontal header tab strip. It builds one entry per enabled panel (icon,
 * label, health LED, active accent) plus a trailing `＋` add affordance, and
 * exposes small imperative mutators — set active, set health, enable, show/hide,
 * append — so `panel-manager.js`'s state machine can drive the rail without
 * owning any DOM specifics.
 *
 * This module holds NO panel state and issues NO fetches or POSTs. The caller
 * passes closures for the interactions (activate, close, add); everything else
 * is a one-shot mutation of the DOM the caller already owns. Styling is entirely
 * class/attribute driven (`.panel-rail-*`, `data-panel-id`, `data-icon`) — the
 * rail's CSS lands separately, this module only produces the markup it hooks.
 *
 * DOM contract (stable — the browser suite selects on it):
 *
 *   <nav class="panel-rail" role="tablist">
 *     <button class="panel-rail-button disabled" data-panel-id="artifacts"
 *             type="button" role="tab" aria-selected="false" title="WORKSPACE">
 *       <span class="panel-rail-led offline" aria-hidden="true"></span>
 *       <span class="panel-rail-icon" data-icon="artifacts" aria-hidden="true"></span>
 *       <span class="panel-rail-label">WORKSPACE</span>
 *       <span class="panel-rail-close" aria-hidden="true">×</span>   (only when onClose given)
 *     </button>
 *     ...
 *     <button class="panel-rail-add" type="button" aria-label="Add panel">＋</button>  (only when onAdd given)
 *   </nav>
 */

// ---- Types ----

/**
 * The minimal panel descriptor the rail renders — a structural subset of the
 * `Panel` records `panel-manager.js` holds and of the `/api/panels` payload
 * (`{ enabled, custom, labels, ... }`). Only id + label are needed to draw an
 * entry; health, visibility, and active state are applied by the mutators.
 * @typedef {object} RailPanel
 * @property {string} id
 * @property {string} label
 */

/**
 * Interaction closures + initial visibility, injected by the caller so the rail
 * stays a dumb view. Every callback is optional: omit `onClose` to render no
 * per-entry close affordance, omit `onAdd` to render no `＋` button.
 * @typedef {object} RailOptions
 * @property {(id: string) => void} [onActivate] - an entry was clicked
 * @property {(id: string) => void} [onClose]    - the entry's close "×" was clicked
 * @property {() => void} [onAdd]                 - the trailing `＋` was clicked
 * @property {Set<string>} [visible]              - ids to show; any entry whose id is
 *                                                  absent is built with `.panel-rail-hidden`
 */

const BUTTON_SELECTOR = '.panel-rail-button';
const ADD_SELECTOR = '.panel-rail-add';

// ---- Rendering ----

/**
 * Build one rail entry button for a panel. Children are assembled via DOM APIs
 * (never innerHTML) because `panel.label` originates from server JSON / SSE.
 *
 * Entries start `.disabled` (matching the tab strip's cold state); the caller
 * clears it via {@link setEntryEnabled} once the panel's backend is healthy.
 * The health LED starts `.offline`. `data-icon` carries the panel id so the
 * rail CSS can map known ids to glyphs and fall back generically for custom
 * panels.
 * @param {RailPanel} panel
 * @param {RailOptions} [options]
 * @returns {HTMLButtonElement}
 */
function buildRailButton(panel, options = {}) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'panel-rail-button disabled';
  btn.setAttribute('data-panel-id', panel.id);
  btn.setAttribute('role', 'tab');
  btn.setAttribute('aria-selected', 'false');
  btn.title = panel.label;
  btn.setAttribute('aria-label', panel.label);

  const led = document.createElement('span');
  led.className = 'panel-rail-led offline';
  led.setAttribute('aria-hidden', 'true');
  btn.appendChild(led);

  const icon = document.createElement('span');
  icon.className = 'panel-rail-icon';
  icon.setAttribute('data-icon', panel.id);
  icon.setAttribute('aria-hidden', 'true');
  btn.appendChild(icon);

  const label = document.createElement('span');
  label.className = 'panel-rail-label';
  label.textContent = panel.label;
  btn.appendChild(label);

  if (options.onActivate) {
    const onActivate = options.onActivate;
    btn.addEventListener('click', () => onActivate(panel.id));
  }

  // Per-entry close affordance — only when the caller wants one. Decorative
  // (aria-hidden) so it is not a control nested inside the button; the click
  // stops propagation so closing does not also activate the entry.
  if (options.onClose) {
    const onClose = options.onClose;
    const close = document.createElement('span');
    close.className = 'panel-rail-close';
    close.setAttribute('aria-hidden', 'true');
    close.title = `Close ${panel.label}`;
    close.textContent = '×';
    close.addEventListener('click', (e) => {
      e.stopPropagation();
      onClose(panel.id);
    });
    btn.appendChild(close);
  }

  if (options.visible && !options.visible.has(panel.id)) {
    btn.classList.add('panel-rail-hidden');
  }
  return btn;
}

/**
 * Build the `＋` add affordance that sits after the panel entries.
 * @param {() => void} onAdd
 * @returns {HTMLButtonElement}
 */
function buildAddButton(onAdd) {
  const add = document.createElement('button');
  add.type = 'button';
  add.className = 'panel-rail-add';
  add.setAttribute('aria-label', 'Add panel');
  add.textContent = '＋';
  add.addEventListener('click', () => onAdd());
  return add;
}

/**
 * Render the full rail into `railEl`, replacing any existing content. Entries
 * are appended in `panels` order, followed by the `＋` button when `onAdd` is
 * supplied. Marks `railEl` as `role="tablist"` for assistive tech.
 *
 * This is the destructive full render — the twin of the tab strip's
 * `renderTabs()`. Use {@link addEntry} for non-destructive runtime additions so
 * live entries keep their active / health / enabled state.
 * @param {HTMLElement} railEl
 * @param {RailPanel[]} panels
 * @param {RailOptions} [options]
 */
export function createRail(railEl, panels, options = {}) {
  railEl.classList.add('panel-rail');
  railEl.setAttribute('role', 'tablist');
  railEl.replaceChildren();
  for (const panel of panels) {
    railEl.appendChild(buildRailButton(panel, options));
  }
  if (options.onAdd) {
    railEl.appendChild(buildAddButton(options.onAdd));
  }
}

/**
 * Append a single entry without disturbing existing ones — the non-destructive
 * path for runtime panel registration. Inserts before the `＋` button when one
 * is present so the add affordance stays last; otherwise appends at the end.
 *
 * Idempotent by id: if an entry for `panel.id` already exists it is returned
 * unchanged (no duplicate node), mirroring the tab strip's re-register guard.
 * @param {HTMLElement} railEl
 * @param {RailPanel} panel
 * @param {RailOptions} [options] - `onAdd` is ignored here (the rail already owns its `＋`)
 * @returns {HTMLButtonElement}
 */
export function addEntry(railEl, panel, options = {}) {
  const existing = getEntry(railEl, panel.id);
  if (existing) return existing;

  const btn = buildRailButton(panel, options);
  const addBtn = railEl.querySelector(ADD_SELECTOR);
  if (addBtn) {
    railEl.insertBefore(btn, addBtn);
  } else {
    railEl.appendChild(btn);
  }
  return btn;
}

// ---- Mutators ----

/**
 * Return the entry button for a panel id, or null when absent.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @returns {HTMLButtonElement | null}
 */
export function getEntry(railEl, panelId) {
  return /** @type {HTMLButtonElement | null} */ (
    railEl.querySelector(`${BUTTON_SELECTOR}[data-panel-id="${panelId}"]`)
  );
}

/**
 * Mark exactly one entry active (the amber left-edge accent lives on
 * `.panel-rail-button.active` in CSS) and clear it from every other entry.
 * Passing an id with no matching entry clears the active state on all.
 * @param {HTMLElement} railEl
 * @param {string | null} panelId
 */
export function setActive(railEl, panelId) {
  for (const el of railEl.querySelectorAll(BUTTON_SELECTOR)) {
    const isActive = el.getAttribute('data-panel-id') === panelId;
    el.classList.toggle('active', isActive);
    el.setAttribute('aria-selected', isActive ? 'true' : 'false');
  }
}

/**
 * Update an entry's health LED. `.panel-rail-led` carries `healthy` or `offline`
 * so the rail CSS can color the dot. No-op when the entry is absent.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {boolean} healthy
 */
export function setHealth(railEl, panelId, healthy) {
  const entry = getEntry(railEl, panelId);
  const led = entry?.querySelector('.panel-rail-led');
  if (led) led.className = 'panel-rail-led ' + (healthy ? 'healthy' : 'offline');
}

/**
 * Enable or disable an entry by toggling `.disabled`. Entries render disabled;
 * the caller enables one once its backend first reports healthy. No-op when the
 * entry is absent.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {boolean} enabled
 */
export function setEntryEnabled(railEl, panelId, enabled) {
  getEntry(railEl, panelId)?.classList.toggle('disabled', !enabled);
}

/**
 * Show or hide an entry by toggling `.panel-rail-hidden` (a visibility change
 * driven by the server's visible set, not a health change). No-op when the
 * entry is absent.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {boolean} visible
 */
export function setEntryVisible(railEl, panelId, visible) {
  getEntry(railEl, panelId)?.classList.toggle('panel-rail-hidden', !visible);
}
