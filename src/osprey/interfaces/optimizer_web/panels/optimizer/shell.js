// @ts-check
/**
 * The panel's frame: the toolbar, the three zones (runs list, stage,
 * inspector) and the inspector drawer of a narrow tile.
 *
 * The toolbar names what the screen is about (its control target and
 * environment), carries the status chip (a button: its popover gives the
 * detail), the history toggle, and one slot for the primary action. A view
 * fills the slot through {@link Shell.setPrimary}; during a run the slot
 * stays empty whatever a view asks, and the chip takes the place where Start
 * was, so a second press there opens its detail and starts nothing. Pause
 * and Cancel live in the run header, never here.
 *
 * Up to {@link NARROW_PX} the runs list is an overlay (panel-sidebar.js) and
 * the inspector a drawer over the stage, opened from an edge tab; Escape
 * closes it. On a screen where the decision is in the inspector (set-up,
 * review, a past run) the drawer opens by itself when the screen changes.
 *
 * @module shell
 */

import { icon } from '/design-system/js/icons.js';
import { openInfoPopover } from '/design-system/js/info-popover.js';
import { initPanelSidebar } from '/design-system/js/panel-sidebar.js';
import { h, replaceChildren } from './hyperscript.js';
import { pauseWords } from './run-page-model.js';
import { RUN_SCREENS } from './store.js';
import { ENDINGS, SCREEN_STATUS, STOP_WORDS, endedChip, endingOf } from './words.js';

/** The widest tile with the narrow layout: runs list overlay, inspector drawer. */
const NARROW_PX = 1040;

/**
 * Every `ControlTarget`, in the operator's words. The service names the
 * target of a draft; the panel never picks one by default.
 */
export const TARGETS = Object.freeze({
  va: {
    label: 'Simulator',
    icon: 'monitor',
    tone: '',
    detail: 'Writes go to the virtual accelerator, not to the machine.',
  },
  standin: {
    label: 'Rehearsal',
    icon: 'layers',
    tone: 'info',
    detail: 'Writes go to the rehearsal stand-in, not to the machine.',
  },
  live: {
    label: 'Real machine',
    icon: 'radiation',
    tone: 'err',
    detail: 'Writes go to the accelerator itself.',
  },
});

/** The edge tab's word for what the inspector holds on each screen. */
const EDGE_LABELS = Object.freeze({
  empty: 'Guide',
  setup: 'Set up',
  'setup-issues': 'Set up',
  review: 'Review',
  'review-changed': 'Review',
  'review-no-owner': 'Review',
  'waiting-approval': 'Approval',
  running: 'Run',
  paused: 'Run',
  completed: 'Outcome',
  failed: 'Outcome',
  cancelled: 'Outcome',
  history: 'Details',
  result: 'Past run',
});

/** Screens whose inspector asks for attention: the edge tab shows a dot. */
const EDGE_DOT = new Set([
  'setup-issues',
  'review-changed',
  'review-no-owner',
  'waiting-approval',
  'failed',
  'cancelled',
]);

/** Screens whose decision is in the inspector: the drawer opens by itself. */
const DRAWER_OPENS = new Set([
  'setup',
  'setup-issues',
  'review',
  'review-changed',
  'review-no-owner',
  'result',
]);

/**
 * @typedef {import('./store.js').State} State
 * @typedef {import('./store.js').Screen} Screen
 * @typedef {{ tone: string, label: string }} ChipSpec
 * @typedef {{
 *   label: string,
 *   onClick: () => void,
 *   disabled?: boolean,
 *   why?: string|null,
 *   detail?: string|null,
 *   id?: string,
 *   key?: string,
 * }} PrimarySpec
 * @typedef {{
 *   title: string,
 *   desc?: string|null,
 *   tools?: Node|null,
 *   body: (Node|null|false)[]|Node,
 *   foot?: (Node|null|false)[]|Node|null,
 * }} InspectorParts
 */

/**
 * @param {number|null} done
 * @param {number|null} total
 */
function progressText(done, total) {
  return total == null ? `${done ?? 0}` : `${done ?? 0} of ${total}`;
}

/**
 * The status chip for a screen; null on a screen with no status to show.
 *
 * @param {State} state
 * @param {Screen} screen
 * @returns {ChipSpec|null}
 */
export function chipFor(state, screen) {
  if (state.service.ok === false) return { tone: 'err', label: 'Service unreachable' };
  if (screen.loading) return { tone: 'connecting', label: 'connecting…' };
  const f = screen.figures;
  const done = f?.done ?? 0;
  switch (screen.id) {
    case 'setup':
      return { tone: 'pending', label: 'Draft · not saved' };
    case 'setup-issues': {
      const n = state.setup?.issues.length ?? 0;
      return { tone: 'warn', label: `${n} problem${n === 1 ? '' : 's'}` };
    }
    case 'running':
    case 'paused': {
      if (screen.cancelling) return { tone: 'warn', label: 'Cancelling' };
      const { tone, label } = SCREEN_STATUS[screen.id];
      return { tone, label: `${label} · ${progressText(done, f?.total ?? null)}` };
    }
    case 'result': {
      const status = screen.past?.job?.status;
      const ending = endingOf(status);
      if (ending) return endedChip(ending);
      return status ? { tone: 'info', label: status } : null;
    }
    default: {
      const status = SCREEN_STATUS[screen.id];
      if (!status) return null;
      const ending = Object.values(ENDINGS).find((e) => e.kind === screen.id);
      return ending ? endedChip(ending, done) : { tone: status.tone, label: status.label };
    }
  }
}

/** @type {Record<string, string|((screen: Screen) => string)>} */
const CHIP_TEXT = {
  setup: 'Saving checks the draft and stores it. It writes nothing to the machine.',
  'setup-issues': 'The draft was not saved. The set-up names the reasons.',
  review: 'Start runs exactly this stored revision.',
  'review-changed': 'The draft changed while it was on screen. Look at it again before you start.',
  'review-no-owner':
    'This session names no account, so the write request may be refused. Start stays available.',
  'waiting-approval': 'Nothing has been written. The run waits for its write approval.',
  running: 'The run writes the knobs inside their search windows.',
  paused: (screen) => `The run is paused. ${pauseWords(screen).held}`,
  completed: 'The run has ended.',
  failed: `The run stopped with an error. ${STOP_WORDS.after}`,
  cancelled: `The run was cancelled. ${STOP_WORDS.after}`,
  result: 'A stored run.',
};

/**
 * What the status chip's popover says: a title, a sentence and the
 * identifiers behind the screen. Only fields the service sent are listed.
 *
 * @param {State} state
 * @param {Screen} screen
 * @returns {{ title: string, text: string, rows: { label: string, value: string, copy?: boolean }[] }}
 */
export function chipDetails(state, screen) {
  const chip = chipFor(state, screen);
  /** @type {{ label: string, value: string, copy?: boolean }[]} */
  const rows = [];
  /** @param {string} label @param {unknown} value @param {boolean} [copy] */
  const add = (label, value, copy = false) => {
    if (value != null && value !== '') rows.push({ label, value: String(value), copy });
  };
  const job = screen.job;
  add('Job', job?.job_id, true);
  if (screen.id.startsWith('review')) {
    add('Revision', state.approval.revision, true);
    if (state.ownerKnown === false) add('Starts as', 'no account');
    else add('Starts as', state.owner);
  }
  add('Environment', screen.subject.environment);
  add('Control target', screen.subject.target);
  if (screen.past?.runId) add('Stored run', screen.past.runId, true);
  add('Write approval', screen.authorization?.text);
  add('Error', job?.error_message);
  const link = state.connection;
  if (link && link.environment === screen.subject.environment) {
    /** @type {Record<string, string>} */
    const words = { true: 'connected', false: 'not connected', null: 'unknown' };
    const word = words[String(link.connected)];
    add('Machine', link.message ? `${word}: ${link.message}` : word);
  }
  if (state.service.ok === false) {
    return { title: 'Service unreachable', text: state.service.error || '', rows };
  }
  const text = CHIP_TEXT[screen.id];
  return {
    title: chip?.label || 'Optimizer',
    text: typeof text === 'function' ? text(screen) : text || '',
    rows,
  };
}

/**
 * The inspector's head, body and foot.
 *
 * @param {InspectorParts} parts
 * @param {() => void} onClose  the head's close button (shown in a narrow tile only)
 * @returns {HTMLElement[]}
 */
export function inspectorFrame({ title, desc = null, tools = null, body, foot = null }, onClose) {
  const head = h(
    'header',
    { class: 'osprey-detail-head opt-insp-head' },
    h('span', { class: 'osprey-card-title', text: title }),
    desc ? h('span', { class: 'osprey-detail-desc opt-insp-desc', text: desc }) : null,
    h('span', { class: 'opt-grow' }),
    tools,
    h(
      'button',
      {
        type: 'button',
        class: 'osprey-icon-btn opt-only-narrow',
        'aria-label': 'Close the inspector',
        title: 'Close',
        onclick: onClose,
      },
      icon('x')
    )
  );
  const list = (/** @type {(Node|null|false)[]|Node} */ nodes) =>
    /** @type {Node[]} */ ((Array.isArray(nodes) ? nodes : [nodes]).filter(Boolean));
  const parts = [head, h('div', { class: 'osprey-detail-body opt-insp-body' }, ...list(body))];
  if (foot) parts.push(h('footer', { class: 'osprey-detail-foot opt-insp-foot' }, ...list(foot)));
  return parts;
}

/**
 * @param {Document} doc
 * @param {string} id
 * @returns {HTMLElement}
 */
function byId(doc, id) {
  const node = doc.getElementById(id);
  if (!node) throw new Error(`optimizer panel: missing #${id}`);
  return node;
}

/**
 * Wire the frame in index.html.
 *
 * @param {{ actions: import('./actions.js').Actions, doc?: Document }} options
 */
export function createShell({ actions, doc = document }) {
  const els = {
    root: byId(doc, 'optimizer-root'),
    toggle: byId(doc, 'runs-toggle'),
    target: byId(doc, 'target-slot'),
    env: byId(doc, 'env-text'),
    end: byId(doc, 'toolbar-end'),
    chip: byId(doc, 'status-chip'),
    chipLabel: byId(doc, 'status-label'),
    history: byId(doc, 'history-btn'),
    slot: byId(doc, 'primary-slot'),
    banner: byId(doc, 'panel-banner'),
    sidebar: byId(doc, 'runs'),
    handle: byId(doc, 'runs-handle'),
    filterField: byId(doc, 'runs-filter-field'),
    filter: /** @type {HTMLInputElement} */ (byId(doc, 'run-filter')),
    newRun: byId(doc, 'new-run'),
    main: byId(doc, 'main'),
    stage: byId(doc, 'stage'),
    edge: byId(doc, 'edge-tab'),
    edgeLabel: byId(doc, 'edge-label'),
    edgeDot: byId(doc, 'edge-dot'),
    inspector: byId(doc, 'inspector'),
  };
  /** @type {{ state: State|null, screen: Screen|null }} */
  const last = { state: null, screen: null };
  /** @type {PrimarySpec|null} */
  let primary = null;
  let primaryKey = '';
  let identityKey = '';
  let narrow = false;
  let drawer = false;
  /** @type {import('/design-system/js/info-popover.js').InfoPopoverHandle|null} */
  let popover = null;

  const sidebar = initPanelSidebar({
    sidebar: els.sidebar,
    handle: els.handle,
    toggle: els.toggle,
    storageKey: 'osprey-optimizer-runs',
    defaultOpen: true,
    narrowQuery: `(max-width: ${NARROW_PX}px)`,
  });
  els.filterField.prepend(icon('search'));
  els.newRun.append(icon('plus'));
  els.history.append(icon('history'));

  function applyDrawer() {
    const open = narrow && drawer;
    els.main.dataset.drawer = open ? 'open' : 'closed';
    els.edge.setAttribute('aria-expanded', open ? 'true' : 'false');
    els.inspector.toggleAttribute('inert', narrow && !open);
  }

  /** @param {boolean} open */
  function setDrawer(open) {
    drawer = open;
    applyDrawer();
    if (!narrow) return;
    if (open) {
      const first = /** @type {HTMLElement|null} */ (
        els.inspector.querySelector(
          '.opt-insp-body button, .opt-insp-body input, .opt-insp-head button'
        )
      );
      first?.focus({ preventScroll: true });
    } else {
      els.edge.focus({ preventScroll: true });
    }
  }

  function measure() {
    const width = els.root.clientWidth;
    const next = width > 0 && width <= NARROW_PX;
    if (next === narrow) return;
    narrow = next;
    drawer = narrow && !!last.screen && DRAWER_OPENS.has(last.screen.id);
    applyDrawer();
  }

  function renderPrimary() {
    const running = !!last.screen && RUN_SCREENS.includes(last.screen.id);
    const spec = running ? null : primary;
    const key = spec ? (spec.key ?? `${spec.label}|${!!spec.disabled}|${spec.why ?? ''}`) : '';
    if (key === primaryKey) return;
    primaryKey = key;
    if (!spec) return replaceChildren(els.slot);
    replaceChildren(
      els.slot,
      h(
        'button',
        {
          type: 'button',
          class: 'osprey-btn primary',
          id: spec.id || 'primary-btn',
          disabled: !!spec.disabled,
          title: spec.disabled ? spec.why || spec.label : spec.label,
          'data-tip-detail': spec.disabled ? null : spec.detail || null,
          onclick: () => spec.onClick(),
        },
        h('span', { class: 'osprey-btn-label', text: spec.label })
      )
    );
  }

  /** @param {State} state @param {Screen} screen */
  function renderIdentity(state, screen) {
    const { environment, target } = screen.subject;
    const info = (state.environments.list || []).find((e) => e.name === environment);
    const key = JSON.stringify([environment, target, info?.display_name, info?.description]);
    if (key === identityKey) return;
    identityKey = key;
    const look = target ? TARGETS[/** @type {keyof typeof TARGETS} */ (target)] : null;
    replaceChildren(
      els.target,
      target
        ? h(
            'span',
            {
              class: `osprey-chip osprey-chip--plain${look?.tone ? ` osprey-chip--${look.tone}` : ''} opt-target`,
              'data-target': target,
              'data-tip': `Control target: ${look?.label ?? target}`,
              'data-tip-detail': `${look?.detail ?? 'A control target this panel does not know.'} (${target})`,
              tabindex: '0',
            },
            look ? icon(look.icon, { size: 12 }) : null,
            h('span', { class: 'osprey-chip-label', text: look?.label ?? target })
          )
        : null
    );
    els.env.textContent = environment ? info?.display_name || environment : '';
    els.env.setAttribute('data-tip', 'Environment');
    els.env.setAttribute(
      'data-tip-detail',
      environment ? [info?.description, `(${environment})`].filter(Boolean).join(' ') : ''
    );
  }

  /**
   * Draw the frame for `state` and its `screen`.
   *
   * @param {State} state
   * @param {Screen} screen
   */
  function update(state, screen) {
    const moved = last.screen?.id !== screen.id;
    last.state = state;
    last.screen = screen;
    els.root.dataset.screen = screen.id;
    renderIdentity(state, screen);
    const chip = chipFor(state, screen);
    els.chip.hidden = !chip;
    if (chip) {
      els.chip.className = `osprey-chip osprey-chip--${chip.tone}`;
      els.chipLabel.textContent = chip.label;
      els.chip.setAttribute('aria-label', `Optimizer status: ${chip.label}`);
      popover?.update(chipDetails(state, screen));
    }
    els.banner.hidden = state.service.ok !== false;
    els.banner.textContent = state.service.ok === false ? state.service.error || '' : '';
    els.history.setAttribute('aria-pressed', String(state.selection.kind === 'history'));
    const running = RUN_SCREENS.includes(screen.id);
    const order = running ? [els.history, els.chip, els.slot] : [els.chip, els.history, els.slot];
    if (order.some((node, i) => els.end.children[i] !== node)) els.end.replaceChildren(...order);
    renderPrimary();
    els.edgeLabel.textContent = EDGE_LABELS[screen.id];
    els.edgeDot.hidden = !EDGE_DOT.has(screen.id);
    els.edge.title = `Open the inspector: ${EDGE_LABELS[screen.id].toLowerCase()}`;
    if (els.filter.value !== state.filter) els.filter.value = state.filter;
    if (moved) {
      drawer = narrow && DRAWER_OPENS.has(screen.id);
      applyDrawer();
    }
  }

  els.chip.addEventListener('click', () => {
    if (last.state && last.screen)
      popover = openInfoPopover(els.chip, chipDetails(last.state, last.screen));
  });
  els.history.addEventListener('click', () =>
    actions.select(
      last.state?.selection.kind === 'history' ? { kind: 'present' } : { kind: 'history' }
    )
  );
  els.newRun.addEventListener('click', () => {
    if (narrow && sidebar.isOpen()) sidebar.close();
    actions.newRun();
  });
  els.filter.addEventListener('input', () => actions.setFilter(els.filter.value));
  els.edge.addEventListener('click', () => setDrawer(!drawer));
  doc.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !narrow || !drawer) return;
    if (doc.querySelector('.osprey-popover:not([hidden])')) return;
    setDrawer(false);
  });
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(measure).observe(els.root);
  measure();

  return {
    zones: { stage: els.stage, inspector: els.inspector },
    update,
    /**
     * Fill the primary slot (null empties it). The button is rebuilt only
     * when the spec's `key` (by default its label, state and reason) changes,
     * so a poll never rebuilds it under the cursor; give a `key` that names
     * whatever the click is bound to.
     *
     * @param {PrimarySpec|null} spec
     */
    setPrimary(spec) {
      primary = spec;
      renderPrimary();
    },
    /** @param {InspectorParts} parts */
    fillInspector(parts) {
      els.inspector.replaceChildren(...inspectorFrame(parts, () => setDrawer(false)));
    },
    setDrawer,
    /**
     * Open the drawer below `px` of the stage (a run header that must stay in sight).
     *
     * @param {number} px
     */
    setDrawerTop(px) {
      els.main.style.setProperty('--opt-drawer-top', `${Math.max(0, Math.round(px))}px`);
    },
    /** Close the runs list when it is an overlay (after a pick in it). */
    closeOverlay() {
      if (narrow && sidebar.isOpen()) sidebar.close();
    },
  };
}

/** @typedef {ReturnType<typeof createShell>} Shell */
