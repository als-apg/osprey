// @ts-check
/* OSPREY Web Terminal — Add-Panel Menu (the human "+")
 *
 * A popover, opened from the sidebar's "+ Add panel" row, that lets a human
 * add a panel — the browser-tab analogue the agent already has via MCP tools.
 * It offers up to three things:
 *
 *   1. "Layouts" — the deployment's configured presets, only where it has any.
 *   2. "Show panel" — a list of known-but-hidden panels. Clicking one reveals it
 *      (a POST /api/panel-visibility round-trip, driven by the parent).
 *   3. "New panel from URL" — a URL field, shown ONLY when the server reports
 *      web.allow_runtime_panels is enabled (POST /api/panels/register).
 *
 * This module is purely presentational: it owns the popover's body and the
 * input→request mapping. The popover itself — a body-level `.osprey-popover`
 * in index.html — is placed, opened and dismissed by the design system's
 * popover.js (outside click, Escape with focus back on the "+" row, resize),
 * beside the sidebar. It holds no panel state and issues no fetches — the
 * parent (panel-manager) passes closures for the actions and for the current
 * hidden-panel/allow-URL state, so the menu stays a dumb view over whatever the
 * panel manager knows.
 *
 * The one dock coupling: a "Show panel" pick docks the revealed panel BESIDE the
 * active dock group (dock-sync.dockPanelBesideActive) before the reveal POST, so
 * it lands where the operator is working rather than on the adapter's default
 * anchor. That call no-ops without a dockview shell, keeping the menu portable.
 */

import { closePopover, openPopover } from '/design-system/js/popover.js';
import { dockPanelBesideActive } from './dock-sync.js';

/**
 * @typedef {object} HiddenPanel
 * @property {string} id
 * @property {string} label
 */

/**
 * @typedef {object} RegisterResult
 * @property {boolean} ok
 * @property {string} [error]
 */

/**
 * @typedef {object} AddMenuOptions
 * @property {HTMLButtonElement} buttonEl            - the "+ Add panel" row
 * @property {HTMLElement} menuEl                    - the body-level `.osprey-popover`
 * @property {() => HiddenPanel[]} getHiddenPanels   - known-but-hidden panels, in tab order
 * @property {() => boolean} allowUrlPanels          - whether runtime URL registration is on
 * @property {(id: string) => void} onShowPanel      - reveal + focus a hidden panel
 * @property {(fields: {id: string, label: string, url: string}) => Promise<RegisterResult>} onRegisterUrl
 * @property {() => {name: string, panels: string[]}[]} getPresets - config-defined layouts, in config order
 * @property {(name: string) => void} onApplyPreset  - apply a named layout (those panels open, the rest close)
 */

/**
 * Derive a URL-safe panel id from a human label or a URL.
 *
 * Lowercases and slugs to ``[a-z0-9-]``; for a URL input the hostname is used
 * (so ``http://grafana.internal:3000`` → ``grafana-internal``) which reads
 * better than slugging the whole string. Always returns a non-empty id.
 * @param {string} input
 * @returns {string}
 */
export function derivePanelId(input) {
  let s = (input || '').trim().toLowerCase();
  if (/^https?:\/\//.test(s)) {
    try {
      s = new URL(s).hostname || s;
    } catch {
      /* not a parseable URL — slug the raw string below */
    }
  }
  const slug = s.replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  return slug || 'panel';
}

/** @param {string} url @returns {string} */
function hostnameOf(url) {
  try {
    return new URL(url).hostname;
  } catch {
    return '';
  }
}

/**
 * Wire up the add-panel menu. No-op (returns) if either element is absent so
 * a template without the "+" control degrades gracefully.
 * @param {AddMenuOptions} opts
 */
export function initPanelAddMenu(opts) {
  const { buttonEl, menuEl } = opts;
  if (!buttonEl || !menuEl) return;

  function closeMenu() {
    closePopover(menuEl);
  }

  /** Rebuild the menu body from the parent's current state. */
  function render() {
    menuEl.replaceChildren();

    // ---- Section 0: apply a layout (rendered ONLY when presets are configured) ----
    // Absent by default, so a deployment with no web.presets sees the popover
    // exactly as it was before this feature — the feature cannot clutter until a
    // facility opts in. Applying a preset is one click, then the menu closes,
    // matching the "Show panel" interaction grammar one altitude up.
    const presets = opts.getPresets();
    if (presets.length > 0) {
      menuEl.appendChild(heading('Layouts'));
      for (const preset of presets) {
        // textContent — preset.name is config-supplied JSON (never innerHTML).
        const item = menuItem(preset.name);
        item.addEventListener('click', () => {
          opts.onApplyPreset(preset.name);
          closeMenu();
        });
        menuEl.appendChild(item);
      }
      menuEl.appendChild(separator());
    }

    // ---- Section 1: reveal a hidden panel ----
    menuEl.appendChild(heading('Show panel'));
    const hidden = opts.getHiddenPanels();
    if (hidden.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'osprey-list-empty panel-add-empty';
      empty.textContent = 'All panels shown';
      menuEl.appendChild(empty);
    }
    for (const panel of hidden) {
      // textContent — panel.label is server/agent-supplied JSON (never innerHTML).
      const item = menuItem(panel.label);
      item.dataset.panelId = panel.id;
      item.addEventListener('click', () => {
        // Dock the placeholder beside the active group first, then reveal it —
        // the adapter adopts the pre-placed placeholder by id (no-op without a
        // dock shell). The reveal itself POSTs the visibility + focus.
        dockPanelBesideActive(panel.id, panel.label);
        opts.onShowPanel(panel.id);
        closeMenu();
      });
      menuEl.appendChild(item);
    }

    // ---- Section 2: add from URL (only when the server permits it) ----
    if (opts.allowUrlPanels()) {
      menuEl.appendChild(separator());
      menuEl.appendChild(buildUrlForm());
    }
  }

  /** @param {string} text @returns {HTMLElement} */
  function heading(text) {
    const h = document.createElement('div');
    h.className = 'osprey-popover-title panel-add-heading';
    h.textContent = text;
    return h;
  }

  /** @param {string} text @returns {HTMLButtonElement} */
  function menuItem(text) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'osprey-popover-item panel-add-item';
    item.textContent = text;
    return item;
  }

  /** @returns {HTMLElement} */
  function separator() {
    const sep = document.createElement('div');
    sep.className = 'osprey-popover-sep';
    sep.setAttribute('role', 'separator');
    return sep;
  }

  /** @returns {HTMLFormElement} */
  function buildUrlForm() {
    const form = document.createElement('form');
    form.className = 'panel-add-url';
    form.appendChild(heading('New panel from URL'));

    const urlInput = document.createElement('input');
    urlInput.className = 'panel-add-input';
    urlInput.name = 'url';
    urlInput.type = 'url';
    urlInput.placeholder = 'https://host:port';
    urlInput.required = true;
    urlInput.setAttribute('aria-label', 'Panel URL');
    form.appendChild(urlInput);

    const labelInput = document.createElement('input');
    labelInput.className = 'panel-add-input';
    labelInput.name = 'label';
    labelInput.type = 'text';
    labelInput.placeholder = 'Name (optional)';
    labelInput.setAttribute('aria-label', 'Panel name');
    form.appendChild(labelInput);

    const errorEl = document.createElement('div');
    errorEl.className = 'panel-add-url-error';
    errorEl.hidden = true;
    form.appendChild(errorEl);

    const submit = document.createElement('button');
    submit.type = 'submit';
    submit.className = 'osprey-btn primary panel-add-submit';
    submit.textContent = 'Add';
    form.appendChild(submit);

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const url = urlInput.value.trim();
      if (!url) return;
      const label = labelInput.value.trim() || hostnameOf(url) || url;
      const id = derivePanelId(labelInput.value.trim() || url);

      submit.disabled = true;
      errorEl.hidden = true;
      let res;
      try {
        res = await opts.onRegisterUrl({ id, label, url });
      } catch {
        res = { ok: false, error: 'Could not reach the server' };
      }
      submit.disabled = false;

      if (res.ok) {
        closeMenu();
      } else {
        errorEl.textContent = res.error || 'Could not add panel';
        errorEl.hidden = false;
      }
    });

    return form;
  }

  buttonEl.addEventListener('click', () => {
    if (!menuEl.hidden) {
      closeMenu();
      return;
    }
    render();
    openPopover(menuEl, buttonEl, {
      place: 'beside-right',
      edge: buttonEl.closest('#sidebar') ?? buttonEl,
    });
  });
}
