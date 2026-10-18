// @ts-check
/**
 * views — what the trajectory panel shows, built from the snapshot and the
 * stream's events.
 *
 * Every node is built through {@link h} — `createElement` plus
 * `textContent`/property assignment — so service strings (owners, BPM and
 * corrector names, refusal details) reach the DOM as text and are never
 * interpreted as markup.
 *
 * The decision functions (`loopControls`, `noticeFromResult`,
 * `noticeFromEvent`, `shouldHeartbeat`, `describeEnd`) are pure; the
 * `*View` builders return a fresh element for a section, and the caller
 * swaps it in. Inputs the operator types into (`loopForm`) are built once and
 * kept, so a new snapshot never clears them.
 *
 * This module performs no I/O.
 *
 * @module views
 */

import { canPutBack } from './api.js';

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
 * A control button, named by `data-control` for the tests and the browser
 * driver.
 * @param {string} name
 * @param {string} label
 * @param {() => void} onclick
 * @param {boolean} [disabled]
 * @returns {HTMLButtonElement}
 */
function controlButton(name, label, onclick, disabled = false) {
  return /** @type {HTMLButtonElement} */ (
    h('button', { type: 'button', 'data-control': name, disabled, onclick }, label)
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
 * The snapshot fields the controls are decided from.
 * @typedef {object} LoopSnapshot
 * @property {string | null} caller
 * @property {'idle' | 'action' | 'continuous'} mode
 * @property {string} model_kind
 * @property {{ digest: string, all_zero: boolean } | null} step
 * @property {import('./api.js').LastAction | null} last_action
 * @property {object | null} continuous
 */

/**
 * @typedef {object} LoopControls
 * @property {boolean} stop  Shown to every admitted owner, holding or not.
 * @property {boolean} setup
 * @property {boolean} shot
 * @property {boolean} apply
 * @property {boolean} putBack
 * @property {boolean} continuous
 * @property {boolean} saveReference
 * @property {boolean} selectReference
 */

/**
 * Which loop controls are enabled. Writing controls need this tab to hold the
 * session and nothing to be running; the service still decides, and a refusal
 * is shown with its code.
 * @param {LoopSnapshot | null} snapshot
 * @param {{ holding: boolean }} tab
 * @returns {LoopControls}
 */
export function loopControls(snapshot, { holding }) {
  const admitted = snapshot != null && snapshot.caller != null;
  const free = snapshot != null && holding && snapshot.mode === 'idle';
  const model = snapshot != null && snapshot.model_kind !== 'none';
  return {
    stop: admitted,
    setup: free,
    shot: free && model,
    apply: free && snapshot.step != null && !snapshot.step.all_zero,
    putBack: free && canPutBack(snapshot.last_action),
    continuous: free && model,
    saveReference: admitted && model,
    selectReference: free && model,
  };
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
 * @param {string} label
 * @param {Node | string | number | null | undefined} value
 */
function fact(label, value) {
  return [h('dt', { text: label }), h('dd', null, value ?? '—')];
}

/** @param {readonly string[] | null | undefined} names */
function names(names) {
  return names && names.length ? names.join(', ') : 'none';
}

/**
 * @param {{ owner: string, kind: string } | null | undefined} holder
 */
export function holderText(holder) {
  return holder ? `${holder.kind} of ${holder.owner}` : 'nobody';
}

/**
 * The header: target, posture, limits, writing, model, holder and take-over,
 * and the stream's connection state.
 * @param {any} snapshot
 * @param {string} connection
 * @param {boolean} holding
 */
export function headerView(snapshot, connection, holding) {
  const status = h('p', { class: `connection connection-${connection}`, 'data-connection': connection },
    `Stream: ${connection}`);
  if (snapshot == null) return h('section', { class: 'header', 'data-view': 'header' }, status);
  const holder = holderText(snapshot.holder) + (holding ? ' (this tab)' : '');
  const takenOver = snapshot.taken_over
    ? h('p', { class: 'notice notice-taken-over', role: 'status' },
        `Taken over by ${holderText(snapshot.taken_over.by)} from ${holderText(snapshot.taken_over.previous)}.`)
    : null;
  return h('section', { class: 'header', 'data-view': 'header' },
    status,
    h('dl', { class: 'facts' },
      ...fact('Target', snapshot.target),
      ...fact('Operator', snapshot.caller ?? 'not identified'),
      ...fact('Posture', `${snapshot.posture}${snapshot.writes_enabled ? '' : ' (writes disabled)'}`),
      ...fact('Limits', snapshot.limits),
      ...fact('Writing', snapshot.writing_available ? 'available' : 'unavailable'),
      ...fact('Model', snapshot.model ? `${snapshot.model} (${snapshot.model_kind})` : 'none'),
      ...fact('Holder', holder)),
    takenOver);
}

/**
 * @typedef {object} SessionHandlers
 * @property {() => void} onOpen
 * @property {() => void} onTakeOver
 * @property {() => void} onRelease
 * @property {() => void} onStop
 */

/**
 * The action area: mode, running action and its stopping state, budgets, the
 * session controls, and Stop for every admitted owner.
 * @param {any} snapshot
 * @param {SessionControls} controls
 * @param {LoopControls} loop
 * @param {SessionHandlers} handlers
 */
export function actionView(snapshot, controls, loop, handlers) {
  const running = controls.running;
  const runningLine = running
    ? h('p', { class: 'running', 'data-running': running.name },
        `Running: ${running.name}${running.owner ? ` (${running.kind} of ${running.owner})` : ''}` +
          (running.stopping ? ' — stopping' : ''))
    : h('p', { class: 'running' }, 'Nothing running');
  const stragglers = snapshot?.action?.armed_stragglers?.length
    ? h('p', null, `Armed stragglers: ${names(snapshot.action.armed_stragglers)}`)
    : null;
  return h('section', { class: 'action', 'data-view': 'action' },
    h('h2', { text: 'Action' }),
    h('dl', { class: 'facts' },
      ...fact('Mode', snapshot?.mode),
      ...fact('Queue wait', snapshot ? `${snapshot.queue_wait_s} s` : null),
      ...fact('Action timeout', snapshot ? `${snapshot.action_timeout_s} s` : null),
      ...fact('Action budget', snapshot ? `${snapshot.action_budget_s} s` : null),
      ...fact('Stop budget', snapshot ? `${snapshot.stop_budget_s} s` : null)),
    runningLine,
    stragglers,
    h('div', { class: 'controls' },
      controls.open && controlButton('open', 'Open session', handlers.onOpen),
      controls.takeOver && controlButton('takeover', 'Take over', handlers.onTakeOver),
      controls.release && controlButton('release', 'Release', handlers.onRelease),
      loop.stop && controlButton('stop', running?.stopping ? 'Stop (stopping…)' : 'Stop', handlers.onStop),
      controls.pending && h('span', { class: 'pending' }, 'session request in flight…')));
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
 * The pending step: deltas, factor, reason and status, zeroed and rank
 * information, and every corrector the solve could not reach, named.
 * @param {any} step
 */
export function stepView(step) {
  if (step == null) return h('section', { class: 'step', 'data-view': 'step' }, h('h2', { text: 'Step' }),
    h('p', null, 'No step pending; take a shot.'));
  const deltas = Object.entries(step.deltas).sort(([a], [b]) => a.localeCompare(b));
  return h('section', { class: 'step', 'data-view': 'step' },
    h('h2', { text: 'Step' }),
    h('dl', { class: 'facts' },
      ...fact('Digest', h('code', { text: step.digest })),
      ...fact('Shot', step.shot_id),
      ...fact('Reference', step.reference ?? 'none'),
      ...fact('Factor', step.factor),
      ...fact('Reason', step.reason),
      ...fact('Status', step.status),
      ...fact('All zero', step.all_zero ? 'yes — apply refuses it' : 'no'),
      ...fact('Rank discarded', `${step.rank_discarded} (n_discard ${step.n_discard})`),
      ...fact('Zeroed H', names(step.zeroed_h)),
      ...fact('Zeroed V', names(step.zeroed_v)),
      ...fact('No response', names(step.no_response_names))),
    step.unsolved.length
      ? h('p', { class: 'notice notice-unsolved', 'data-view': 'unsolved', role: 'status' },
          `Not solved: ${names(step.unsolved)}`)
      : null,
    table({ class: 'deltas' }, ['Corrector', 'Delta'], deltas));
}

/**
 * The last shot, the last writing action and the continuous run, in brief.
 * @param {any} snapshot
 */
export function statusView(snapshot) {
  const shot = snapshot?.last_shot;
  const last = snapshot?.last_action;
  const run = snapshot?.continuous;
  return h('section', { class: 'status', 'data-view': 'status' },
    h('h2', { text: 'Last shot' }),
    shot
      ? h('dl', { class: 'facts' },
          ...fact('Shot', shot.shot_id),
          ...fact('Included', shot.shots_included),
          ...fact('Failed', shot.shots_failed.length
            ? shot.shots_failed.map((/** @type {any} */ f) => `#${f.index}: ${f.reason}`).join('; ') : 'none'),
          ...fact('Lost', names(shot.lost)),
          ...fact('Last BPM', shot.last_bpm),
          ...fact('Commands fired', names(shot.commands_fired)))
      : h('p', null, 'No shot yet.'),
    h('h2', { text: 'Last action' }),
    last
      ? h('div', { 'data-view': 'last-action' },
          h('p', null, `${last.name}${last.stopped ? ' (stopped)' : ''}, ${last.moved.length} setting(s) moved`),
          last.moved.length ? movedTable(last.moved) : null)
      : h('p', null, 'No writing action yet.'),
    h('h2', { text: 'Continuous' }),
    run
      ? h('p', { 'data-view': 'continuous' },
          `Running: ${run.shots} of ${run.max_shots} shot(s); lease expires in ` +
            `${run.lease_expires_in_s == null ? '—' : `${Math.round(run.lease_expires_in_s)} s`}`)
      : h('p', null, 'Not running.'));
}

/**
 * @typedef {object} LoopForm
 * @property {HTMLElement} root
 * @property {HTMLInputElement} bunchClean
 * @property {HTMLInputElement} nShots
 * @property {HTMLSelectElement} average
 * @property {HTMLInputElement} maxShots  Empty sets no cap.
 * @property {Record<'setup' | 'shot' | 'apply' | 'putBack' | 'continuous', HTMLButtonElement>} buttons
 */

/**
 * @typedef {object} LoopHandlers
 * @property {() => void} onSetup
 * @property {() => void} onShot
 * @property {() => void} onApply
 * @property {() => void} onPutBack
 * @property {() => void} onContinuous
 */

/**
 * Build the loop controls once; {@link setLoopEnabled} updates them.
 * @param {LoopHandlers} handlers
 * @returns {LoopForm}
 */
export function loopForm(handlers) {
  const input = (/** @type {Record<string, unknown>} */ props) =>
    /** @type {HTMLInputElement} */ (h('input', props));
  const bunchClean = input({ type: 'checkbox', name: 'bunch_clean' });
  const nShots = input({ type: 'number', name: 'n_shots', min: '1', step: '1', value: '1' });
  const maxShots = input({ type: 'number', name: 'max_shots', min: '1', step: '1', placeholder: 'no cap' });
  const average = /** @type {HTMLSelectElement} */ (h('select', { name: 'average' },
    h('option', { value: 'mean', text: 'mean' }), h('option', { value: 'nanmean', text: 'nanmean' })));
  const buttons = {
    setup: controlButton('setup', 'Setup', handlers.onSetup),
    shot: controlButton('shot', 'Shot', handlers.onShot),
    apply: controlButton('apply', 'Apply step', handlers.onApply),
    putBack: controlButton('put-back', 'Put back', handlers.onPutBack),
    continuous: controlButton('continuous', 'Start continuous', handlers.onContinuous),
  };
  const root = h('section', { class: 'loop', 'data-view': 'loop' },
    h('h2', { text: 'Loop' }),
    h('div', { class: 'controls' }, h('label', null, bunchClean, ' bunch clean'), buttons.setup),
    h('div', { class: 'controls' }, h('label', null, 'shots ', nShots), h('label', null, 'average ', average),
      buttons.shot),
    h('div', { class: 'controls' }, buttons.apply, buttons.putBack),
    h('div', { class: 'controls' }, h('label', null, 'max shots ', maxShots), buttons.continuous));
  return { root, bunchClean, nShots, average, maxShots, buttons };
}

/**
 * @param {LoopForm} form
 * @param {LoopControls} loop
 */
export function setLoopEnabled(form, loop) {
  for (const key of /** @type {(keyof LoopForm['buttons'])[]} */ (Object.keys(form.buttons))) {
    form.buttons[key].disabled = !loop[key];
  }
}

/**
 * The shot and average the form asks for.
 * @param {LoopForm} form
 * @returns {{ n_shots: number, average: string, max_shots: number | null }}
 */
export function loopArgs(form) {
  const max = form.maxShots.value.trim();
  return {
    n_shots: Number.parseInt(form.nShots.value, 10) || 1,
    average: form.average.value,
    max_shots: max === '' ? null : Number.parseInt(max, 10),
  };
}

/**
 * @typedef {object} ReferenceHandlers
 * @property {(name: string, shotId: string | undefined) => void} onSave
 * @property {(name: string | null) => void} onSelect
 * @property {() => void} onRefresh
 */

/**
 * References (list, save, select), the loaded reference, and the shots list.
 * @param {{ names: string[], current: string | null } | null} references
 * @param {any[] | null} shots  Newest first.
 * @param {any} snapshot
 * @param {LoopControls} loop
 * @param {ReferenceHandlers} handlers
 */
export function referencesView(references, shots, snapshot, loop, handlers) {
  const list = references?.names ?? [];
  const pick = /** @type {HTMLSelectElement} */ (h('select', { name: 'reference' },
    h('option', { value: '', text: '(no reference)' }),
    ...list.map((name) => h('option', { value: name, text: name, selected: name === references?.current }))));
  const nameInput = /** @type {HTMLInputElement} */ (h('input', { type: 'text', name: 'reference_name',
    placeholder: 'reference name' }));
  const shotPick = /** @type {HTMLSelectElement} */ (h('select', { name: 'reference_shot' },
    h('option', { value: '', text: "the step's shot, else the newest" }),
    ...(shots ?? []).map((s) => h('option', { value: s.shot_id, text: s.shot_id }))));
  const loaded = snapshot?.reference;
  return h('section', { class: 'references', 'data-view': 'references' },
    h('h2', { text: 'References' }),
    loaded
      ? h('p', { 'data-view': 'reference-loaded' }, `Selected: ${loaded.name} — ${loaded.matched.length} matched, ` +
          `missing ${names(loaded.missing_in_reference)}, unused ${names(loaded.unused_in_reference)}, ` +
          `${loaded.nan_count} NaN`)
      : h('p', null, 'No reference selected.'),
    h('div', { class: 'controls' }, pick,
      controlButton('select-reference', 'Select',
        () => handlers.onSelect(pick.value === '' ? null : pick.value), !loop.selectReference)),
    h('div', { class: 'controls' }, nameInput, shotPick,
      controlButton('save-reference', 'Save',
        () => handlers.onSave(nameInput.value.trim(), shotPick.value || undefined), !loop.saveReference)),
    controlButton('refresh', 'Refresh lists', handlers.onRefresh),
    h('h2', { text: 'Shots' }),
    shots && shots.length
      ? table({ class: 'shots', 'data-view': 'shots' }, ['Shot', 'Included', 'Failed', 'Last BPM', 'Lost'],
          shots.map((s) => [s.shot_id, s.shots_included, s.shots_failed.length, s.last_bpm ?? '—', s.lost.length]))
      : h('p', null, 'No shots stored.'));
}
