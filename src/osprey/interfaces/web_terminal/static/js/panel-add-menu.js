// @ts-check
/* OSPREY Web Terminal — Add-Panel Menu (the human "+")
 *
 * A small popover, opened from the "+" button in the header, that lets a human
 * add a panel — the browser-tab analogue the agent already has via MCP tools.
 * It offers two things:
 *
 *   1. "Show panel" — a list of known-but-hidden panels. Clicking one reveals it
 *      (a POST /api/panel-visibility round-trip, driven by the parent).
 *   2. "New panel from URL" — a URL field, shown ONLY when the server reports
 *      web.allow_runtime_panels is enabled (POST /api/panels/register).
 *
 * This module is purely presentational: it owns the popover DOM, open/close,
 * outside-click / Escape, and the input→request mapping. It holds no panel
 * state and issues no fetches — the parent (panel-manager) passes closures for
 * the two actions and for the current hidden-panel/allow-URL state, so the menu
 * stays a dumb view over whatever the panel manager knows.
 */

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
 * @property {HTMLElement} rootEl                    - wrapper, the position:relative anchor
 * @property {HTMLButtonElement} buttonEl            - the "+" toggle button
 * @property {HTMLElement} menuEl                    - the popover container
 * @property {() => HiddenPanel[]} getHiddenPanels   - known-but-hidden panels, in tab order
 * @property {() => boolean} allowUrlPanels          - whether runtime URL registration is on
 * @property {(id: string) => void} onShowPanel      - reveal + focus a hidden panel
 * @property {(fields: {id: string, label: string, url: string}) => Promise<RegisterResult>} onRegisterUrl
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
 * Wire up the add-panel menu. No-op (returns) if any required element is absent
 * so a template without the "+" control degrades gracefully.
 * @param {AddMenuOptions} opts
 */
export function initPanelAddMenu(opts) {
  const { rootEl, buttonEl, menuEl } = opts;
  if (!rootEl || !buttonEl || !menuEl) return;

  const isOpen = () => menuEl.classList.contains('open');

  /** @param {MouseEvent} e */
  function onDocClick(e) {
    if (e.target instanceof Node && !rootEl.contains(e.target)) closeMenu();
  }

  /** @param {KeyboardEvent} e */
  function onKeydown(e) {
    if (e.key === 'Escape') {
      closeMenu();
      buttonEl.focus();
    }
  }

  function openMenu() {
    render();
    menuEl.classList.add('open');
    buttonEl.setAttribute('aria-expanded', 'true');
    // Focus the first actionable element for keyboard users.
    const first = /** @type {HTMLElement | null} */ (
      menuEl.querySelector('.panel-add-item, .panel-add-input')
    );
    first?.focus();
    // Capture-phase so an outside click closes before it does anything else.
    document.addEventListener('click', onDocClick, true);
    document.addEventListener('keydown', onKeydown, true);
  }

  function closeMenu() {
    menuEl.classList.remove('open');
    buttonEl.setAttribute('aria-expanded', 'false');
    document.removeEventListener('click', onDocClick, true);
    document.removeEventListener('keydown', onKeydown, true);
  }

  /** Rebuild the menu body from the parent's current state. */
  function render() {
    menuEl.replaceChildren();

    // ---- Section 1: reveal a hidden panel ----
    const hidden = opts.getHiddenPanels();
    const section = document.createElement('div');
    section.className = 'panel-add-section';
    section.appendChild(heading('Show panel'));

    if (hidden.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'panel-add-empty';
      empty.textContent = 'All panels shown';
      section.appendChild(empty);
    } else {
      for (const panel of hidden) {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'panel-add-item';
        item.dataset.panelId = panel.id;
        // textContent — panel.label is server/agent-supplied JSON (never innerHTML).
        item.textContent = panel.label;
        item.addEventListener('click', () => {
          opts.onShowPanel(panel.id);
          closeMenu();
        });
        section.appendChild(item);
      }
    }
    menuEl.appendChild(section);

    // ---- Section 2: add from URL (only when the server permits it) ----
    if (opts.allowUrlPanels()) {
      const divider = document.createElement('div');
      divider.className = 'panel-add-divider';
      menuEl.appendChild(divider);
      menuEl.appendChild(buildUrlForm());
    }
  }

  /** @param {string} text @returns {HTMLElement} */
  function heading(text) {
    const h = document.createElement('div');
    h.className = 'panel-add-heading';
    h.textContent = text;
    return h;
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
    form.appendChild(urlInput);

    const labelInput = document.createElement('input');
    labelInput.className = 'panel-add-input';
    labelInput.name = 'label';
    labelInput.type = 'text';
    labelInput.placeholder = 'Name (optional)';
    form.appendChild(labelInput);

    const errorEl = document.createElement('div');
    errorEl.className = 'panel-add-url-error';
    errorEl.hidden = true;
    form.appendChild(errorEl);

    const submit = document.createElement('button');
    submit.type = 'submit';
    submit.className = 'panel-add-submit';
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

  buttonEl.addEventListener('click', (e) => {
    e.stopPropagation();
    if (isOpen()) closeMenu();
    else openMenu();
  });
}
