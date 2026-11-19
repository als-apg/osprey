// @ts-check
/**
 * The runs list in the sidebar: the job of now (the current draft or run, or
 * the set-up of a new one), the other jobs the service holds open
 * (`GET jobs`), and the stored runs (`GET results`), newest first and
 * grouped by day. A click selects a row; the screen follows the selection.
 *
 * A stored run's entry names its environment, objective, start time and job:
 * it carries no status and no figures. Its status shows only once the
 * service has answered for its job (it keeps an ended job for an hour); the
 * list never guesses one.
 *
 * A stored run's start is the service host's wall clock as its directory
 * name writes it, and shows as written; the time an open job was made is an
 * instant, shown on the facility clock. Today and yesterday are the
 * facility's.
 *
 * @module runs-list
 */

import { facilityDayKey, facilityDayStart } from '/design-system/js/facility-time.js';
import { icon } from '/design-system/js/icons.js';
import { h } from './hyperscript.js';
import { TERMINAL_STATUSES } from './approval-model.js';
import { jobFigures } from './run-model.js';
import { currentScreen, presentScreen, runJob } from './store.js';
import { SCREEN_STATUS, endedDot, endingOf, instantText } from './words.js';

/**
 * @typedef {import('./store.js').State} State
 * @typedef {import('./store.js').Selection} Selection
 * @typedef {{ tone: string, label: string }} RowChip
 * @typedef {{
 *   key: string,
 *   title: string,
 *   meta: string,
 *   icon: string|null,
 *   chip: RowChip|null,
 *   dot: { tone: string, label: string }|null,
 *   aside: string|null,
 *   time: string|null,
 *   selected: boolean,
 *   selection: Selection,
 * }} Row
 * @typedef {{
 *   now: Row[],
 *   open: Row[],
 *   past: { label: string, rows: Row[] }[],
 *   total: number,
 *   shown: number,
 *   note: string|null,
 *   nowNote: string,
 * }} RunsModel
 */

/**
 * The chip of a job's row: the short word of its screen.
 *
 * @param {string} id  the screen of the job
 * @returns {RowChip|null}
 */
function screenChip(id) {
  const status = SCREEN_STATUS[id];
  return status ? { tone: status.tone, label: status.short } : null;
}

/**
 * A stored run's start, from its directory name (`YYYY-MM-DD_HH-MM-SS`, the
 * wall clock of the service host, perhaps with a suffix); null for another
 * shape. The text is read as written, never converted to another zone.
 *
 * @param {string|null|undefined} timestamp
 * @returns {{ day: string, clock: string }|null}
 */
export function runTime(timestamp) {
  const m = /^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-\d{2}/.exec(timestamp || '');
  return m ? { day: m[1], clock: `${m[2]}:${m[3]}` } : null;
}

/** The month of a date as written, as a day heading names it. */
const MONTHS = Object.freeze('Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split(' '));

/**
 * The day heading of a stored run: Today or Yesterday on the facility's
 * calendar, else the date as written (with its year when that is not the
 * facility's year now).
 *
 * @param {string} day  `YYYY-MM-DD`, as `runTime` reads it
 * @param {number} now  ms
 * @returns {string}
 */
export function dayLabel(day, now) {
  const today = facilityDayKey(now);
  if (day === today) return 'Today';
  if (day === facilityDayKey(facilityDayStart(now) - 1)) return 'Yesterday';
  const [y, m, d] = day.split('-').map(Number);
  const month = MONTHS[m - 1];
  if (!month) return day;
  return String(y) === today.slice(0, 4) ? `${d} ${month}` : `${d} ${month} ${y}`;
}

/**
 * The details of an environment, when the panel has read them.
 *
 * @param {State} state
 * @param {string|null|undefined} name
 */
function envDetails(state, name) {
  if (!name) return null;
  const own = state.envs[name]?.details;
  if (own) return own;
  const draft = state.approvalEnv.details;
  return draft && draft.name === name ? draft : null;
}

/**
 * An environment's display name; its identifier when it has none.
 *
 * @param {State} state
 * @param {string|null|undefined} name
 * @returns {string}
 */
export function environmentLabel(state, name) {
  if (!name) return '';
  const info = (state.environments.list || []).find((e) => e.name === name);
  return info?.display_name || name;
}

/**
 * An objective's display name; its identifier when the panel has not read
 * one.
 *
 * @param {State} state
 * @param {string|null|undefined} environment
 * @param {string|null|undefined} objective
 * @returns {string}
 */
export function objectiveLabel(state, environment, objective) {
  if (!objective) return '';
  return envDetails(state, environment)?.observables?.[objective]?.display_name || objective;
}

/**
 * The row of one job of now.
 *
 * @param {State} state
 * @param {any} job
 * @param {import('./store.js').ScreenId} id  the screen of the job
 * @param {boolean} isNow  the row picks "now" (else the job itself)
 * @returns {Row}
 */
function jobRow(state, job, id, isNow) {
  const { approval, setup, selection } = state;
  let config = null;
  if (job.job_id === state.activeJob?.job_id) config = state.activeConfig;
  else if (approval.jobId === job.job_id) config = approval.config;
  const environment = config?.environment_name || job.environment_name;
  const objective = objectiveLabel(
    state,
    environment,
    config?.objective_name || job.objective_name
  );
  const draft = id.startsWith('review');
  const chip = setup?.jobId === job.job_id ? { tone: 'pending', label: 'editing' } : screenChip(id);
  const knobs = config?.variables?.length;
  let aside = null;
  if (!draft) {
    const f = jobFigures(job, config);
    aside = f.total == null ? `${f.done}` : `${f.done} of ${f.total}`;
  }
  const parts =
    draft && knobs ? [objective, `${knobs} knob${knobs === 1 ? '' : 's'}`] : [objective];
  const picked = selection.kind === 'job' && selection.jobId === job.job_id;
  return {
    key: `job:${job.job_id}`,
    title: draft ? 'Draft' : 'Run',
    meta: parts.filter(Boolean).join(' · '),
    icon: draft ? 'file-text' : 'activity',
    chip: chip || { tone: 'info', label: job.status },
    dot: null,
    aside,
    time: null,
    selected: isNow ? selection.kind === 'present' || picked : picked,
    selection: isNow ? { kind: 'present' } : { kind: 'job', jobId: job.job_id },
  };
}

/**
 * The rows of now: the set-up of a new run, the active run, and the current
 * job when it is another one (a newer draft). The row that is "now" picks
 * now; the others pick their job, so a run that started elsewhere and a
 * newer draft are each one click away.
 *
 * @param {State} state
 * @returns {Row[]}
 */
function nowRows(state) {
  const { job, setup, selection } = state;
  /** @type {Row[]} */
  const rows = [];
  if (setup && !setup.jobId) {
    rows.push({
      key: 'setup',
      title: 'Draft',
      meta: setup.environment ? environmentLabel(state, setup.environment) : 'New run',
      icon: 'file-text',
      chip: { tone: 'pending', label: 'not saved' },
      dot: null,
      aside: null,
      time: null,
      selected: selection.kind === 'present',
      selection: { kind: 'present' },
    });
  }
  const run = runJob(state);
  if (run && run !== job) rows.push(jobRow(state, run, presentScreen(state), !setup));
  // The current job's row is "now" when it is the run or nothing runs, and
  // when it is the draft the open set-up edits (now is that set-up).
  const jobIsNow = setup ? setup.jobId === job?.job_id : !run || run === job;
  if (job) rows.push(jobRow(state, job, currentScreen(state), jobIsNow));
  return rows;
}

/**
 * The stored runs (`GET results` entries) the runs list shows: newest first,
 * narrowed by `state.filter` (objective, environment, start time or job id,
 * by display name or identifier). The history table uses the same list.
 *
 * @param {State} state
 * @returns {any[]}
 */
export function pastRuns(state) {
  // The run of now and a current job that has not ended are not past runs,
  // although the service lists their results directory from the start.
  const live = [runJob(state), state.job]
    .filter((j) => j && !TERMINAL_STATUSES.includes(j.status))
    .map((j) => j.job_id);
  const all = [...(state.results.list || [])]
    .filter((r) => !r.job_id || !live.includes(r.job_id))
    .sort((a, b) => String(b.timestamp).localeCompare(String(a.timestamp)));
  const query = state.filter.trim().toLowerCase();
  if (!query) return all;
  return all.filter((r) =>
    [
      objectiveLabel(state, r.environment, r.objective_name),
      r.objective_name,
      environmentLabel(state, r.environment),
      r.environment,
      r.timestamp,
      r.job_id,
    ]
      .join(' ')
      .toLowerCase()
      .includes(query)
  );
}

/**
 * What the runs list shows for `state`.
 *
 * @param {State} state
 * @param {number} now  ms, for the day headings
 * @returns {RunsModel}
 */
export function runsModel(state, now) {
  const current = state.job?.job_id ?? null;
  const running = runJob(state)?.job_id ?? null;
  const { selection } = state;
  const open = (state.openJobs.list || [])
    .filter((j) => j.job_id !== current && j.job_id !== running)
    .map((j) => ({
      key: `job:${j.job_id}`,
      title: j.status === 'created' ? 'Draft' : 'Run',
      meta: `${j.status} · ${String(j.job_id).slice(0, 8)}`,
      icon: j.status === 'created' ? 'file-text' : 'activity',
      chip: null,
      dot: null,
      aside: null,
      time: instantText(j.created_at, now) || null,
      selected: selection.kind === 'job' && selection.jobId === j.job_id,
      selection: /** @type {Selection} */ ({ kind: 'job', jobId: j.job_id }),
    }));
  const total = (state.results.list || []).length;
  const shownRuns = pastRuns(state);
  /** @type {{ label: string, rows: Row[] }[]} */
  const past = [];
  for (const r of shownRuns) {
    const title = objectiveLabel(state, r.environment, r.objective_name);
    const envLabel = environmentLabel(state, r.environment);
    const when = runTime(r.timestamp);
    const label = when ? dayLabel(when.day, now) : 'Undated';
    if (past[past.length - 1]?.label !== label) past.push({ label, rows: [] });
    const ending = endingOf(state.past[r.run_id]?.job?.status);
    past[past.length - 1].rows.push({
      key: `run:${r.run_id}`,
      title: title || envLabel,
      meta: envLabel,
      icon: null,
      chip: null,
      dot: ending ? endedDot(ending) : null,
      aside: null,
      time: when?.clock ?? null,
      selected: selection.kind === 'run' && selection.runId === r.run_id,
      selection: { kind: 'run', runId: r.run_id },
    });
  }
  let note = null;
  if (state.results.error) note = `Past runs could not be read: ${state.results.error}`;
  else if (!state.results.list) note = 'Reading past runs…';
  else if (!total) note = 'No stored run yet.';
  else if (!shownRuns.length) note = 'No past run matches.';
  /** @type {Record<string, string>} */
  const unknown = { null: 'Reading the optimizer…', false: 'The optimizer could not be read.' };
  const nowNote =
    state.service.ok === true ? 'No draft, nothing running.' : unknown[String(state.service.ok)];
  const shown = shownRuns.length;
  return { now: nowRows(state), open, past, total, shown, note, nowNote };
}

/** @typedef {{ node: Element|null, key: string|null }} Part */

/** @param {Element} node @param {string} name @param {string} value */
function setAttr(node, name, value) {
  if (node.getAttribute(name) !== value) node.setAttribute(name, value);
}

/** @param {Element} node @param {string} text */
function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

/**
 * Keep an optional part of a row as `key` says: none when `key` is null, else
 * built when it is missing and patched in place (`patch`; a part with no
 * patch is built again when `key` changes). It sits in `parent` before `before`.
 *
 * @param {Part} part
 * @param {string|null} key  what the part shows
 * @param {Element} parent
 * @param {Node|null} before
 * @param {() => Element} build
 * @param {(node: Element) => void} [patch]
 */
function keepPart(part, key, parent, before, build, patch) {
  if (part.key === key) return;
  part.key = key;
  if (key === null) {
    part.node?.remove();
    part.node = null;
  } else if (part.node && patch) {
    patch(part.node);
  } else {
    const next = build();
    patch?.(next);
    if (part.node) part.node.replaceWith(next);
    else parent.insertBefore(next, before);
    part.node = next;
  }
}

/**
 * One row of the list. `set` shows another state of the same row: it changes
 * only the attributes and text that differ, so the element and its children
 * stay (a click that straddles an update still lands).
 *
 * @param {Row} row
 * @param {(selection: Selection) => void} onSelect
 * @returns {{ el: HTMLElement, set: (row: Row) => void }}
 */
function rowButton(row, onSelect) {
  let current = row;
  const title = h('span', { class: 'opt-run-title' });
  const sub = h('span', { class: 'opt-run-sub' });
  const head = h('span', { class: 'opt-run-line' }, title);
  const meta = h('span', { class: 'opt-run-line opt-run-meta' }, sub);
  const el = h(
    'button',
    {
      type: 'button',
      class: 'osprey-list-row multiline opt-run',
      'data-row': row.key,
      onclick: () => onSelect(current.selection),
    },
    head,
    meta
  );
  /** @type {Record<'icon'|'dot'|'chip'|'aside'|'time', Part>} */
  const parts = {
    icon: { node: null, key: null },
    dot: { node: null, key: null },
    chip: { node: null, key: null },
    aside: { node: null, key: null },
    time: { node: null, key: null },
  };
  /** @param {Row} next */
  function set(next) {
    current = next;
    const { dot, chip } = next;
    setAttr(el, 'aria-selected', next.selected ? 'true' : 'false');
    setText(title, next.title);
    setText(sub, next.meta);
    // Parts after others first, so each finds the node it goes before.
    keepPart(
      parts.chip,
      chip && `${chip.tone}|${chip.label}`,
      head,
      null,
      () => h('span'),
      (node) => {
        if (!chip) return;
        setAttr(node, 'class', `osprey-chip osprey-chip--${chip.tone} opt-run-chip`);
        setAttr(node, 'title', chip.label);
        setText(node, chip.label);
      }
    );
    keepPart(
      parts.dot,
      dot && `${dot.tone}|${dot.label}`,
      head,
      title,
      () => h('span', { role: 'img' }),
      (node) => {
        if (!dot) return;
        setAttr(node, 'class', `osprey-dot ${dot.tone}`);
        setAttr(node, 'title', dot.label);
        setAttr(node, 'aria-label', dot.label);
      }
    );
    const iconName = next.icon;
    keepPart(parts.icon, iconName, head, parts.dot.node || title, () =>
      icon(/** @type {string} */ (iconName), { size: 14 })
    );
    keepPart(
      parts.time,
      next.time,
      meta,
      null,
      () => h('time'),
      (node) => setText(node, next.time || '')
    );
    keepPart(
      parts.aside,
      next.aside,
      meta,
      parts.time.node,
      () => h('span', { class: 'opt-run-aside' }),
      (node) => setText(node, next.aside || '')
    );
  }
  set(row);
  return { el, set };
}

/**
 * `key`, or `key#2`, `key#3`… when `taken` holds it already.
 *
 * @param {Map<string, unknown>} taken
 * @param {string} key
 */
function unique(taken, key) {
  let k = key;
  for (let n = 2; taken.has(k); n += 1) k = `${key}#${n}`;
  return k;
}

/**
 * Draw the runs list into `host` on each update; a list that did not change
 * is not redrawn. The list is updated in place: a row that stays keeps its
 * element (by its `data-row` key) and is not moved while its place holds, so
 * a click that straddles an update still selects it and the focus stays.
 * Headings, notes and buttons are kept while they show the same.
 *
 * @param {{
 *   host: HTMLElement,
 *   actions: import('./actions.js').Actions,
 *   onPicked?: () => void,
 *   now?: () => number,
 * }} options  `onPicked` runs after a row is clicked (closes an overlay)
 */
export function createRunsList({ host, actions, onPicked = () => {}, now = Date.now }) {
  let drawn = '';
  /** @type {Map<string, { el: HTMLElement, set: (row: Row) => void }>} */
  let rows = new Map();
  /** @type {Map<string, HTMLElement>} */
  let others = new Map();
  /** @param {Selection} selection */
  const select = (selection) => {
    actions.select(selection);
    onPicked();
  };
  return {
    /** @param {State} state */
    update(state) {
      // The display names of the stored runs' environments, read once each.
      for (const name of new Set((state.results.list || []).map((r) => r.environment))) {
        if (name && !state.envs[name]) void actions.loadEnvironment(name);
      }
      const model = runsModel(state, now());
      const key = JSON.stringify(model);
      if (key === drawn) return;
      drawn = key;
      const focused = /** @type {HTMLElement|null} */ (host.querySelector('[data-row]:focus'));
      const focusKey = focused?.dataset.row;
      /** @type {typeof rows} */
      const nextRows = new Map();
      /** @type {typeof others} */
      const nextOthers = new Map();
      /** @type {Element[]} */
      const nodes = [];
      /** @param {Row} row */
      const addRow = (row) => {
        const k = unique(nextRows, row.key);
        const held = rows.get(k);
        held?.set(row);
        const kept = held || rowButton(row, select);
        nextRows.set(k, kept);
        nodes.push(kept.el);
      };
      /** @param {string} what  what the node shows @param {() => HTMLElement} build */
      const addOther = (what, build) => {
        const k = unique(nextOthers, what);
        const node = others.get(k) || build();
        nextOthers.set(k, node);
        nodes.push(node);
      };
      const counted =
        model.shown === model.total ? `${model.total}` : `${model.shown} of ${model.total}`;
      addOther('label:now', () => h('div', { class: 'osprey-section-label', text: 'Now' }));
      model.now.forEach(addRow);
      if (!model.now.length) {
        addOther(`now-note:${model.nowNote}`, () =>
          h('div', { class: 'osprey-list-empty', text: model.nowNote })
        );
        addOther('new-run', () =>
          h(
            'button',
            {
              type: 'button',
              class: 'osprey-nav-row add',
              onclick: () => (actions.newRun(), onPicked()),
            },
            icon('plus'),
            h('span', { class: 'osprey-nav-label', text: 'New run' })
          )
        );
      }
      if (model.open.length) {
        addOther('label:open', () =>
          h('div', { class: 'osprey-section-label opt-past-label', text: 'Other open jobs' })
        );
      }
      model.open.forEach(addRow);
      const total = model.total ? counted : '';
      addOther(`label:past:${total}`, () =>
        h(
          'div',
          { class: 'osprey-section-label opt-past-label' },
          'Past runs',
          h('span', { class: 'osprey-aside', text: total }),
          h(
            'button',
            {
              type: 'button',
              class: 'osprey-mini',
              'aria-label': 'All past runs in a table',
              title: 'All past runs in a table',
              onclick: () => select({ kind: 'history' }),
            },
            icon('table')
          )
        )
      );
      const { note } = model;
      if (note) {
        addOther(`past-note:${note}`, () => h('div', { class: 'osprey-list-empty', text: note }));
      }
      for (const group of model.past) {
        addOther(`head:${group.label}`, () =>
          h('div', { class: 'osprey-list-head', text: group.label })
        );
        group.rows.forEach(addRow);
      }
      rows = nextRows;
      others = nextOthers;
      // Remove what goes, then put each node in its place: a node already
      // there is not touched, so a row that stays is not moved.
      const wanted = new Set(nodes);
      for (const child of Array.from(host.childNodes)) {
        if (!wanted.has(/** @type {Element} */ (child))) child.remove();
      }
      nodes.forEach((node, i) => {
        const at = host.childNodes[i] || null;
        if (at !== node) host.insertBefore(node, at);
      });
      if (focusKey) {
        const back = Array.from(rows.values()).find((r) => r.el.dataset.row === focusKey)?.el;
        if (back && back !== host.ownerDocument.activeElement) back.focus({ preventScroll: true });
      }
    },
  };
}
