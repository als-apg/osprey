// @ts-check
/**
 * views — the trajectory panel's shell, built from the snapshot and the
 * stream's events: the layout and its area hosts, the header strip (machine
 * facts, session controls, notices), the status strip, the centre tabs and
 * what the hub's tile bar is handed for them.
 *
 * Every node is built through {@link h} — `createElement` plus
 * `textContent`/property assignment — so service strings (owners, BPM and
 * corrector names, refusal details) reach the DOM as text and are never
 * interpreted as markup.
 *
 * The decision functions (`sessionButtons`, `headerItems`, `noticeFromResult`,
 * `noticeFromEvent`, `shouldHeartbeat`, `describeEnd`) are pure; the `*View`
 * builders return a fresh element for a section, and the caller swaps it in.
 * {@link shellView} is built once: the area modules own what is inside their
 * hosts, and a new snapshot never rebuilds them.
 *
 * A control the operator cannot use right now is shown disabled with the
 * reason as its tooltip and `data-reason`, never hidden.
 *
 * This module performs no I/O.
 *
 * @module views
 */

/**
 * @typedef {import('./api.js').Moved} Moved
 * @typedef {import('./api.js').Result} Result
 * @typedef {import('./session.js').SessionControls} SessionControls
 */

/**
 * Create an element, apply props, append children.
 *
 * Props: `class` sets className, `text` sets textContent, anything else that
 * names an existing property is assigned as a property, the rest become
 * attributes. `null`, `undefined` and `false` values are skipped. String and
 * number children become text nodes.
 *
 * @param {string} tag
 * @param {Record<string, unknown> | null} [props]
 * @param {...(Node | string | number | null | undefined | false)} children
 * @returns {HTMLElement}
 */
export function h(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = String(value);
    else if (key === 'text') node.textContent = String(value);
    else if (key in node) /** @type {any} */ (node)[key] = value;
    else node.setAttribute(key, String(value));
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === 'object' ? child : document.createTextNode(String(child)));
  }
  return node;
}

/**
 * A control button named by `data-control` for the tests and the browser
 * driver. A non-null *reason* disables it and is shown as its tooltip and
 * `data-reason`.
 * @param {string} name
 * @param {string} label
 * @param {() => void} onclick
 * @param {string | null} [reason]  Why it cannot be used now.
 * @returns {HTMLButtonElement}
 */
export function control(name, label, onclick, reason = null) {
  return /** @type {HTMLButtonElement} */ (
    h('button', { type: 'button', 'data-control': name, disabled: reason != null, title: reason,
      'data-reason': reason, onclick }, label)
  );
}

/** What each `continuous_ended` reason means, in the operator's words. */
export const END_REASONS = Object.freeze({
  stopped: 'Stop was pressed.',
  lease_expired:
    'The lease expired: the panel stopped heartbeating (closed, crashed, or a hidden tab throttled).',
  max_shots: 'The shot cap was reached.',
  zero_step: 'The step came out all zero; there is nothing left to correct.',
  taken_over: 'The session was taken over.',
  released: 'The session was released.',
  shutdown: 'The service shut down.',
  failed: 'An action of the run failed.',
});

/**
 * @typedef {object} EndEvent
 * @property {string} end
 * @property {number} shots
 * @property {string | null} [step_reason]
 * @property {string | null} [step_status]
 */

/**
 * The headline and detail of a `continuous_ended` event.
 * @param {EndEvent} event
 * @returns {{ title: string, detail: string }}
 */
export function describeEnd(event) {
  const reason = /** @type {Record<string, string>} */ (END_REASONS)[event.end];
  const parts = [reason ?? `Ended: ${event.end}.`, `${event.shots} shot(s) taken.`];
  if (event.step_reason) {
    parts.push(`Last step: ${event.step_reason}${event.step_status ? ` — ${event.step_status}` : ''}.`);
  }
  return { title: `Continuous run ended: ${event.end}`, detail: parts.join(' ') };
}

/**
 * One message the panel lists for the operator.
 * @typedef {object} Notice
 * @property {'refused' | 'pending' | 'busy' | 'stopped' | 'moved' | 'ended' | 'info'} kind
 * @property {string} title
 * @property {string | null} [code]     The refusal code, when refused.
 * @property {string} [detail]
 * @property {Moved[] | null} [moved]   Settings written, with before-values.
 */

/**
 * The notice an API call's result calls for, or null when it needs none.
 * A refusal keeps its code and any `moved` list; an unknown outcome says to
 * read the state, never that the call was refused.
 * @param {string} label  The control that made the call (`apply`, `shot`, ...).
 * @param {Result} result
 * @returns {Notice | null}
 */
export function noticeFromResult(label, result) {
  switch (result.kind) {
    case 'refused': {
      const extras = Object.entries(result.extras)
        .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(', ') : String(value)}`)
        .join('; ');
      return {
        kind: 'refused',
        title: `${label} refused`,
        code: result.code,
        detail: extras ? `${result.detail} (${extras})` : result.detail,
        moved: result.moved,
      };
    }
    case 'pending':
      return { kind: 'pending', title: label, detail: result.message };
    case 'busy':
      return { kind: 'busy', title: label, detail: result.message };
    default:
      return null;
  }
}

/**
 * The notice a stream event calls for, or null when the snapshot that follows
 * says it all.
 * @param {string} type
 * @param {any} payload
 * @returns {Notice | null}
 */
export function noticeFromEvent(type, payload) {
  if (payload == null || typeof payload !== 'object') return null;
  if (type === 'stopped') {
    return {
      kind: 'stopped',
      title: `${payload.action} stopped by ${payload.stopper ?? 'unknown'}`,
      detail: 'Nothing is restored automatically; the before-values below put it back.',
      moved: payload.moved ?? [],
    };
  }
  if (type === 'apply') {
    return {
      kind: 'moved',
      title: payload.step_digest == null ? 'put back applied' : 'step applied',
      moved: payload.moved ?? [],
    };
  }
  if (type === 'continuous_ended') {
    const { title, detail } = describeEnd(payload);
    return { kind: 'ended', title, detail };
  }
  return null;
}

/**
 * Whether the continuous heartbeat should be running: a run is in progress and
 * this tab holds the session that started it.
 * @param {{ continuous: object | null } | null} snapshot
 * @param {boolean} holding
 */
export function shouldHeartbeat(snapshot, holding) {
  return holding && snapshot != null && snapshot.continuous != null;
}

/**
 * @param {{ owner: string, kind: string } | null | undefined} holder
 */
export function holderText(holder) {
  return holder ? `${holder.kind} of ${holder.owner}` : 'nobody';
}

/**
 * A table of text cells under a heading row.
 * @param {Record<string, unknown>} props  The table's own props.
 * @param {string[]} headings
 * @param {unknown[][]} rows  Each cell becomes text.
 */
function table(props, headings, rows) {
  return h('table', props,
    h('thead', null, h('tr', null, ...headings.map((t) => h('th', { text: t })))),
    h('tbody', null, ...rows.map((cells) =>
      h('tr', null, ...cells.map((cell) => h('td', { text: String(cell) }))))));
}

/**
 * A table of settings written, each with the value that puts it back.
 * @param {Moved[]} moved
 */
export function movedTable(moved) {
  return table({ class: 'moved', 'data-view': 'moved' }, ['Setting', 'Before', 'Outcome'],
    moved.map((m) => [m.name, m.before, m.outcome]));
}

/**
 * One notice: its title, the refusal code when there is one, its detail, and
 * any moved list with before-values.
 * @param {Notice} notice
 */
export function noticeView(notice) {
  return h('div', { class: `notice notice-${notice.kind}`, 'data-notice': notice.kind,
    role: notice.kind === 'ended' || notice.kind === 'refused' ? 'alert' : 'status' },
    h('strong', { text: notice.title }),
    notice.code ? h('code', { class: 'code', 'data-code': notice.code }, notice.code) : null,
    notice.detail ? h('span', { class: 'detail' }, ` ${notice.detail}`) : null,
    notice.moved && notice.moved.length ? movedTable(notice.moved) : null);
}

/**
 * The prominent banner of the last `continuous_ended`, and the notice list.
 * @param {Notice | null} ended
 * @param {Notice[]} notices  Newest first.
 */
export function noticesView(ended, notices) {
  return h('section', { class: 'notices', 'data-view': 'notices' },
    ended ? h('div', { class: 'ended-banner', 'data-view': 'ended' }, noticeView(ended)) : null,
    ...notices.map(noticeView));
}

/**
 * The keys the area modules share through the view store, with their initial
 * values: whether the Raw tab draws (`plotRaw`, set by the chain's Plot step),
 * whether the CM-step bars draw (`plotCmStep`), and the redraw request
 * (`redraw`, a counter bumped to ask the plots to redraw from the last data).
 */
export const VIEW_DEFAULTS = Object.freeze({ plotRaw: true, plotCmStep: true, redraw: 0 });

/**
 * @typedef {{plotRaw: boolean, plotCmStep: boolean, redraw: number}} ViewState
 * @typedef {keyof ViewState} ViewKey
 */

/**
 * @typedef {object} ViewStore
 * @property {<K extends ViewKey>(key: K) => ViewState[K]} get
 * @property {<K extends ViewKey>(key: K, value: ViewState[K]) => void} set
 *   Stores *value*; the key's listeners run when it changed.
 * @property {<K extends ViewKey>(key: K, fn: (value: ViewState[K]) => void) => () => void} on
 *   Listens to one key; answers the unsubscribe.
 * @property {() => void} requestRedraw  Bumps `redraw`.
 */

/**
 * The keyed store the areas share.
 * @param {Partial<ViewState>} [initial]
 * @returns {ViewStore}
 */
export function createView(initial = {}) {
  /** @type {ViewState} */
  const state = { ...VIEW_DEFAULTS, ...initial };
  /** @type {Map<ViewKey, Set<(value: any) => void>>} */
  const listeners = new Map();
  /** @type {ViewStore} */
  const store = {
    get: (key) => state[key],
    set(key, value) {
      if (Object.is(state[key], value)) return;
      state[key] = value;
      for (const fn of [...(listeners.get(key) ?? [])]) fn(value);
    },
    on(key, fn) {
      const set = listeners.get(key) ?? new Set();
      listeners.set(key, set);
      set.add(fn);
      return () => void set.delete(fn);
    },
    requestRedraw: () => store.set('redraw', state.redraw + 1),
  };
  return store;
}

/** The centre tabs: their `data-control` name, the tab they show and the manual's label. */
export const CENTRE_TABS = Object.freeze([
  Object.freeze({ name: 'tab-raw', tab: 'raw', label: 'Raw BPM Signal' }),
  Object.freeze({ name: 'tab-correction', tab: 'correction', label: 'Trajectory Correction' }),
]);

/** @typedef {'raw' | 'correction'} CentreTab */

/** The hub nav item the centre tabs are contributed as. */
export const CENTRE_TABS_ITEM = 'centre-tabs';

/**
 * The area hosts, by `data-area`. Each named module's `mount(host, ctx)` fills its host;
 * `header`, `notices`, `session` and `status` are the shell's own.
 */
export const AREAS = Object.freeze([
  'header', 'session', 'notices', 'status',
  'setup', 'chain',
  'raw-plots', 'correction-plots', 'correction-results',
  'plots', 'records', 'references', 'correction',
  'drawer',
]);

/** The plot hosts, by element id and `data-control`, per centre tab. */
export const PLOT_HOSTS = Object.freeze({
  raw: Object.freeze(['plot-raw-sum', 'plot-raw-x', 'plot-raw-y']),
  correction: Object.freeze(['plot-traj-x', 'plot-traj-y', 'plot-diff-x', 'plot-diff-y']),
});

/** @param {string} id */
function plotHost(id) {
  return h('div', { class: 'plot', id, 'data-control': id });
}

/** @param {string} area @param {Record<string, unknown>} [props] @param {...Node | null} children */
function areaHost(area, props = {}, ...children) {
  return h('div', { ...props, class: `area area-${area}${props.class ? ` ${props.class}` : ''}`, 'data-area': area },
    ...children);
}

/**
 * @typedef {object} Shell
 * @property {HTMLElement} root
 * @property {Record<string, HTMLElement>} areas  Every host in {@link AREAS}, by name.
 * @property {HTMLElement} tabStrip  The in-body centre tab strip.
 * @property {Record<CentreTab, HTMLElement[]>} tabPanels  What each centre tab shows: its plots and its controls.
 */

/**
 * The layout: header strip and status strip on top; left the setup tabs and
 * the Beam Injection chain; centre the Raw / Trajectory Correction tabs with
 * their plot hosts and the correction results under the dy plot; right the
 * per-tab controls with the SV spectrum host. Built once.
 * @param {(tab: CentreTab) => void} onTab
 * @returns {Shell}
 */
export function shellView(onTab) {
  /** @type {Record<string, HTMLElement>} */
  const areas = {};
  /** @param {string} name @param {Record<string, unknown>} [props] @param {...Node | null} children */
  const area = (name, props, ...children) => (areas[name] = areaHost(name, props, ...children));
  const tabStrip = h('nav', { class: 'centre-tabs', role: 'tablist', 'aria-label': 'Plots' },
    ...CENTRE_TABS.map(({ name, tab, label }) =>
      h('button', { type: 'button', role: 'tab', 'data-control': name, 'data-tab': tab,
        onclick: () => onTab(/** @type {CentreTab} */ (tab)) }, label)));
  const rawPlots = area('raw-plots', { class: 'tab-panel', 'data-tab-panel': 'raw' },
    ...PLOT_HOSTS.raw.map(plotHost));
  const correctionPlots = area('correction-plots', { class: 'tab-panel', 'data-tab-panel': 'correction' },
    h('h3', { text: 'Trajectories' }), plotHost('plot-traj-x'), plotHost('plot-traj-y'),
    h('h3', { text: 'Difference to Reference' }), plotHost('plot-diff-x'), plotHost('plot-diff-y'),
    area('correction-results'));
  const rawControls = h('div', { class: 'tab-panel', 'data-tab-panel': 'raw' }, area('plots'), area('records'));
  const correctionControls = h('div', { class: 'tab-panel', 'data-tab-panel': 'correction' },
    area('references'), area('correction'), plotHost('plot-sv'));
  const root = h('div', { class: 'traj-shell', 'data-view': 'shell' },
    h('section', { class: 'header-strip', 'data-view': 'header-strip' },
      area('header'), area('session', { class: 'session-controls' }), area('notices')),
    area('status', { class: 'status-strip' }),
    h('div', { class: 'traj-grid' },
      h('aside', { class: 'col col-left', 'data-column': 'left' }, area('setup'), area('chain')),
      h('section', { class: 'col col-centre', 'data-column': 'centre' }, tabStrip, rawPlots, correctionPlots),
      h('aside', { class: 'col col-right', 'data-column': 'right' }, rawControls, correctionControls)),
    area('drawer'));
  return { root, areas, tabStrip, tabPanels: { raw: [rawPlots, rawControls], correction: [correctionPlots, correctionControls] } };
}

/**
 * Show *tab*'s plots and controls, mark its strip button selected, and fire
 * `tab:show` on what became visible so the render adapter resizes its plots.
 * @param {Shell} shell
 * @param {CentreTab} tab
 */
export function showTab(shell, tab) {
  for (const [name, panels] of Object.entries(shell.tabPanels)) {
    for (const panel of panels) panel.hidden = name !== tab;
  }
  for (const button of shell.tabStrip.querySelectorAll('[data-tab]')) {
    button.setAttribute('aria-selected', String(button.getAttribute('data-tab') === tab));
  }
  for (const panel of shell.tabPanels[tab]) {
    panel.dispatchEvent(new CustomEvent('tab:show', { bubbles: true }));
  }
}

/**
 * One session control: its inventory name, its label, and why it is
 * disabled (null when usable).
 * @typedef {{name: 'session-open' | 'session-takeover' | 'session-end', label: string,
 *   reason: string | null}} SessionButton
 */

/**
 * The session controls, always all three, each with why it cannot be used now.
 * @param {any} snapshot
 * @param {SessionControls} controls
 * @returns {SessionButton[]}
 */
export function sessionButtons(snapshot, controls) {
  const holder = snapshot?.holder ?? null;
  /** @type {string | null} */
  let common = null;
  if (snapshot == null) common = 'Waiting for the service.';
  else if (controls.holding) common = 'This tab holds the session.';
  else if (controls.pending) common = 'A session request is in flight.';
  const running = controls.running;
  const takeOverReason = holder == null
    ? 'Nobody holds the session; open it.'
    : `Wait for ${running?.name ?? 'the action'} to end, or Stop it.`;
  return [
    { name: 'session-open', label: 'Open session',
      reason: common ?? (controls.open ? null : `Held by ${holderText(holder)}; take over instead.`) },
    { name: 'session-takeover', label: 'Take over',
      reason: common ?? (controls.takeOver ? null : takeOverReason) },
    { name: 'session-end', label: 'End session',
      reason: controls.release ? null : 'This tab holds no session.' },
  ];
}

/**
 * The session controls in the body.
 * @param {SessionButton[]} buttons
 * @param {(name: string) => void} onClick
 */
export function sessionView(buttons, onClick) {
  return h('div', { class: 'controls', 'data-view': 'session' },
    ...buttons.map((b) => control(b.name, b.label, () => onClick(b.name), b.reason)));
}

/**
 * What the hub's tile bar shows: the session controls, then the centre tabs
 * last (the item aimed at most, so it stays against the close button).
 * @param {SessionButton[]} buttons
 * @param {CentreTab} activeTab
 * @returns {import('/design-system/js/header-contrib.js').HeaderItem[]}
 */
export function headerItems(buttons, activeTab) {
  return [
    ...buttons.map((b) => ({ kind: /** @type {const} */ ('button'), id: b.name, label: b.label,
      disabled: b.reason != null, ...(b.reason != null ? { title: b.reason } : {}) })),
    { kind: 'nav', id: CENTRE_TABS_ITEM,
      items: CENTRE_TABS.map(({ name, tab, label }) => ({ id: name, label, active: tab === activeTab })) },
  ];
}

/** @param {unknown} value */
function shown(value) {
  return value == null ? '—' : String(value);
}

/**
 * @param {string} name
 * @param {string} label
 * @param {unknown} value
 */
function fact(name, label, value) {
  return h('div', { class: 'fact', 'data-fact': name },
    h('span', { class: 'fact-label', text: label }),
    h('span', { class: 'fact-value', 'data-control': name === 'stored-current' ? name : null, text: shown(value) }));
}

/**
 * The header facts: target, holder, writes posture, RF and stored current,
 * and a take-over when there was one.
 * @param {any} snapshot
 * @param {boolean} holding
 */
export function headerView(snapshot, holding) {
  const machine = snapshot?.machine ?? null;
  const current = machine?.stored_current;
  const writes = snapshot == null ? null
    : `${snapshot.posture}${snapshot.writes_enabled ? '' : ' (writes disabled)'}`;
  return h('div', { class: 'facts', 'data-view': 'header' },
    fact('target', 'Target', snapshot?.target),
    fact('holder', 'Holder', snapshot == null ? null : holderText(snapshot.holder) + (holding ? ' (this tab)' : '')),
    fact('writes', 'Writes', writes),
    fact('rf', 'RF', machine?.rf_on),
    fact('stored-current', 'Stored current', typeof current === 'number' ? `${current.toPrecision(4)} mA` : null),
    snapshot?.taken_over
      ? h('p', { class: 'notice notice-taken-over', role: 'status' },
          `Taken over by ${holderText(snapshot.taken_over.by)} from ${holderText(snapshot.taken_over.previous)}.`)
      : null);
}

/**
 * The status strip, always in the body: the stream's connection, the mode,
 * what runs and its stopping state, the continuous run, and Stop for every
 * admitted owner.
 * @param {any} snapshot
 * @param {string} connection
 * @param {SessionControls} controls
 * @param {() => void} onStop
 */
export function statusView(snapshot, connection, controls, onStop) {
  const running = controls.running;
  const run = snapshot?.continuous;
  /** @type {string | null} */
  let stopReason = null;
  if (snapshot == null || snapshot.caller == null) stopReason = 'Not admitted by the service.';
  else if (running == null && run == null) stopReason = 'Nothing is running.';
  return h('div', { class: 'status-line', 'data-view': 'status' },
    h('span', { class: `connection connection-${connection}`, 'data-connection': connection },
      `Stream: ${connection}`),
    h('span', { 'data-mode': snapshot?.mode ?? 'unknown' }, `Mode: ${snapshot?.mode ?? '—'}`),
    running
      ? h('span', { class: 'running', 'data-running': running.name },
          `Running: ${running.name}${running.owner ? ` (${running.kind} of ${running.owner})` : ''}` +
            (running.stopping ? ' — stopping' : ''))
      : h('span', { class: 'running' }, 'Nothing running'),
    run
      ? h('span', { 'data-view': 'continuous' }, `Continuous: ${run.shots} shot(s); lease expires in ` +
          `${run.lease_expires_in_s == null ? '—' : `${Math.round(run.lease_expires_in_s)} s`}`)
      : null,
    control('stop', running?.stopping ? 'Stop (stopping…)' : 'Stop', onStop, stopReason));
}
