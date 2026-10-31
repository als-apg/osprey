// @ts-check
/* OSPREY Web Terminal — Control-Target Popover rows
 *
 * The card, the rows and the foot of the control-target popover
 * (control-target-popover.js), built from one roster payload. This module
 * builds DOM and nothing else: what the last gesture did and what a click
 * does come in through a {@link RowContext} the popover hands every call, so
 * the popover keeps the gestures, the confirms and the state between renders,
 * and this module keeps what a machine looks like.
 *
 * Every row is the design system's popover row (`.osprey-popover-item`) with a
 * status dot (`.osprey-dot`), its writes switch is an `.osprey-switch` inside
 * a `role=switch` button, and its Switch to is the primary `.osprey-btn`.
 */

import {
  LIFTED_BY_TURN_ON,
  SWITCH_APPLIED,
  contextRefusalCode,
  contextRefusalPhrase,
  contextWritable,
  descriptor,
  descriptorTone,
  displayName,
  identTitle,
  lockReason,
  reachException,
  reasonPhrase,
  statePhrase,
} from './control-target-facts.js';
import { isPending, kindAttr, stateWord } from './control-target-chip.js';

/**
 * What the popover hands every render: the facts about the last gestures that
 * only it holds, and what a click on a row's controls does.
 * @typedef {object} RowContext
 * @property {string|null} pendingTarget the row this browser's outstanding
 *   switch request named, which reads `switching…`
 * @property {string|null} confirmingTarget the row whose turn-on confirm is
 *   up, whose switch parks mid-track
 * @property {Map<string, string>} gestureNotes refusals the last click came
 *   back with, by row
 * @property {(row: any, state: any) => void} onTurnOff the switch of a row
 *   that is on (or unreadable) was clicked
 * @property {(row: any, state: any, reshow: boolean) => void} onTurnOn the
 *   switch of a row that is off was clicked; `reshow` is Shift
 * @property {(row: any, state: any, reshow: boolean) => void} onSwitchTo a
 *   row's Switch to was clicked; `reshow` is Shift
 * @property {(state: any) => void} onAllOff Turn all writes off was clicked
 */

/**
 * How recent a `last_switch` has to be to still be shown on its row.
 *
 * The route publishes the outcome and its age and leaves the call to the
 * renderer, because it is a question about what the operator is looking at
 * rather than about what happened: an outcome is news for about as long as
 * someone is still watching for it, and history afterwards. Past this the row
 * simply stops carrying it, so a popover opened an hour later describes the
 * machine rather than re-announcing a switch nobody is waiting on.
 */
export const OUTCOME_MAX_AGE_S = 60;

/* ---- derived facts (the pure ones live in control-target-facts.js) ---- */

/**
 * What the last switch did to THIS row, when it is still news.
 *
 * Matched on the target the outcome names, never on "the session moved": the
 * chip matches the request by `request_id` and hands the outcome on, and a
 * row that did not take part in it has nothing to report.
 * @param {any} row
 * @param {any} state
 * @param {RowContext} ctx
 * @returns {{status: string, text: string, title?: string}|null}
 */
function switchOutcome(row, state, ctx) {
  if (isPending() && ctx.pendingTarget === row.target) {
    return { status: 'pending', text: 'switching…' };
  }
  const last = state.last_switch;
  if (!last || last.target !== row.target) return null;
  if (typeof last.age_s === 'number' && last.age_s > OUTCOME_MAX_AGE_S) return null;
  // No age in the copy: `age_s` decides WHETHER the line renders (the
  // freshness window above), never what it says. A ticking counter would
  // restate the mechanism, and every re-render — the 5 s poll, every
  // gesture's re-read — would bump it in front of the operator.
  // The RECORD's word for an accepted switch is `applied` — never `success`,
  // which nothing on this path has ever published. The returned `status` is a
  // different vocabulary: it is the `data-status` value terminal.css keys the
  // outcome line on, the same split kindAttr makes between the route's words
  // and the stylesheet's.
  if (last.status === SWITCH_APPLIED) return { status: 'success', text: '✓ switched' };
  // refused / failed / expired render the operator phrase for the word the
  // gate (or the client's own deadline, for a request nothing ever answered)
  // put on them, with the gate's own sentence on the title where it sent one.
  return {
    status: last.status === 'expired' ? 'expired' : 'refused',
    text: `✗ ${reasonPhrase(last.reason || last.status)}`,
    title: typeof last.detail === 'string' && last.detail ? last.detail : undefined,
  };
}

/* ---- building blocks ---- */

/** @param {string} tag @param {string} [cls] @param {string} [text] */
export function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * A button, typed and with its `type` set. Every button here lives inside no
 * form, but an unset `type` is `submit` and a stray form anywhere above would
 * make each of these navigate.
 * @param {string} cls @param {string} text @returns {HTMLButtonElement}
 */
function button(cls, text) {
  const node = /** @type {HTMLButtonElement} */ (el('button', cls, text));
  node.type = 'button';
  return node;
}

/**
 * The card: the machine the agent stands on, under the popover's "The agent
 * is on" title. Name, consequence line, and the writes switch that both
 * states and changes the write state.
 * @param {any} row
 * @param {any} state
 * @param {RowContext} ctx
 */
export function renderCard(row, state, ctx) {
  const kind = kindAttr(row);
  const word = stateWord(row);
  const lock = lockReason(row, state);
  const card = el('div', 'ctc-card');
  card.dataset.target = row.target;
  card.dataset.targetKind = kind;
  card.dataset.state = word;

  const title = el('div', 'ctc-card-title');
  title.dataset.targetKind = kind;
  title.dataset.state = word;
  title.append(dot());
  title.append(el('span', 'ctc-name', displayName(row, kind)));
  const info = infoAffordance(row, kind);
  if (info) title.append(info);
  card.append(title);

  card.append(renderSwitch(row, state, word, lock, ctx));
  const phrase = writePhraseLine(row, word, lock);
  if (phrase) card.append(phrase);

  // The reach exception sits on the title line: the switch carries the write
  // state, so the card has no separate state line to put it on.
  const reach = reachException(row.reachability);
  if (reach) {
    const bad = el('span', 'ctc-reach-word', reach.text);
    bad.dataset.state = reach.state;
    bad.title = reach.title;
    title.append(bad);
  }

  for (const line of outcomeLines(row, state, ctx)) card.append(line);
  // Turning writes off on the machine the agent is ON only reaches it once
  // the connector is rebuilt, and that waits for the run in flight. Said out
  // loud, rather than leaving a button that appears to have done nothing.
  if (state.last_posture_realign?.state === 'pending') {
    const line = el('div', 'ctc-outcome', 'takes effect when the running execution finishes');
    line.dataset.status = 'realign';
    card.append(line);
  }
  return card;
}

/**
 * One other machine's row, on the design system's popover row: its dot and
 * name, what writing there means, its writes switch, and Switch to. The row
 * holds controls of its own, so it is a plain row rather than a menu item.
 * @param {any} row
 * @param {any} state
 * @param {RowContext} ctx
 */
export function renderRow(row, state, ctx) {
  const kind = kindAttr(row);
  const word = stateWord(row);
  const lock = lockReason(row, state);
  const node = el('div', 'ctc-row osprey-popover-item');
  node.dataset.target = row.target;
  node.dataset.targetKind = kind;
  node.dataset.state = word;

  const ident = el('div', 'ctc-ident');
  const name = el('div', 'ctc-name-line');
  name.dataset.targetKind = kind;
  name.dataset.state = word;
  name.append(dot());
  name.append(el('span', 'ctc-name', displayName(row, kind)));
  const info = infoAffordance(row, kind);
  if (info) name.append(info);
  const reach = reachException(row.reachability);
  if (reach) {
    const bad = el('span', 'ctc-reach-word', reach.text);
    bad.dataset.state = reach.state;
    bad.title = reach.title;
    name.append(bad);
  }
  ident.append(name);
  const phrase = writePhraseLine(row, word, lock);
  if (phrase) ident.append(phrase);
  for (const line of outcomeLines(row, state, ctx)) ident.append(line);
  node.append(ident);

  node.append(renderSwitch(row, state, word, lock, ctx));
  node.append(renderAction(row, state, ctx));
  return node;
}

/**
 * A machine's dot: the design system's status dot, shaped and coloured by
 * the `data-target-kind` and `data-state` of the line that holds it.
 * @returns {HTMLElement}
 */
function dot() {
  const node = el('span', 'osprey-dot ctc-dot');
  node.setAttribute('aria-hidden', 'true');
  return node;
}

/**
 * The ⓘ beside a name, and the tooltip behind it.
 *
 * Two facts in one place: what writing to this machine does (the consequence
 * line — the one that names hardware moving keeps its error tone), then the
 * machine vocabulary the name's `title` used to carry (the server's own
 * label, the endpoint, the measured reachability). A real element shown on
 * hover and keyboard focus rather than the browser's `title` bubble, which
 * is delayed, unstyled and mouse-only — the design system's shared info tip
 * (`.osprey-info` / `.osprey-tip`), with this popover's own classes kept for
 * the two line tones. Returns `null` for a row with nothing to say.
 * @param {any} row
 * @param {string} kind
 * @returns {HTMLElement|null}
 */
function infoAffordance(row, kind) {
  const what = descriptor(row, kind);
  const ident = identTitle(row);
  if (!what && !ident) return null;
  const info = button('osprey-info ctc-info', 'i');
  info.setAttribute('aria-label', `About ${displayName(row, kind)}`);
  const tip = el('span', 'osprey-tip ctc-tip');
  tip.setAttribute('role', 'tooltip');
  if (what) {
    const line = el('span', 'ctc-tip-what', what);
    const tone = descriptorTone(row, kind);
    if (tone) line.dataset.tone = tone;
    tip.append(line);
  }
  if (ident) tip.append(el('span', 'ctc-tip-ident', ident));
  info.append(tip);
  return info;
}

/**
 * The feedback lines a row or the card carries: the last switch's outcome
 * while it is news, and the reason the last click did nothing.
 * @param {any} row
 * @param {any} state
 * @param {RowContext} ctx
 * @returns {HTMLElement[]}
 */
function outcomeLines(row, state, ctx) {
  const lines = [];
  const outcome = switchOutcome(row, state, ctx);
  if (outcome) {
    const line = el('div', 'ctc-outcome', outcome.text);
    line.dataset.status = outcome.status;
    if (outcome.title) line.title = outcome.title;
    lines.push(line);
  }
  const gestureNote = ctx.gestureNotes.get(row.target);
  if (gestureNote) {
    const line = el('div', 'ctc-outcome', `✗ ${reasonPhrase(gestureNote)}`);
    line.dataset.status = 'refused';
    lines.push(line);
  }
  return lines;
}

/**
 * The switch that both states and changes a machine's writes.
 *
 * One widget instead of pill + verb: the knob's position is this login's
 * switch, the word beside it (`on` / `off` / `locked`, the row's `data-state`)
 * is the glance-and-screen-reader copy, and the click is the gesture. Off
 * applies on click, on confirms first — said on the hover title — and while
 * that confirm is up the knob parks mid-track (`data-pending`) rather than
 * claiming a state nothing has applied.
 *
 * Under every lock a switch that is on — or one nobody can read — keeps its
 * off control: no lock refuses a stop. Only a switch that is off and cannot
 * be turned on is disabled, with the reason on hover.
 * @param {any} row
 * @param {any} state
 * @param {string} word
 * @param {string|null} lock
 * @param {RowContext} ctx
 * @returns {HTMLElement}
 */
function renderSwitch(row, state, word, lock, ctx) {
  const name = displayName(row, kindAttr(row));
  // An unreadable switch is shown as one that may be on: the loud answer, and
  // the one that leaves the operator a way to stop it.
  const armed = row.switch !== 'off';
  const wrap = el('span', 'ctc-switchwrap');
  wrap.dataset.state = word;
  wrap.append(el('span', 'ctc-switch-word', 'Writes'));
  const toggle = button('ctc-toggle', '');
  toggle.setAttribute('role', 'switch');
  toggle.setAttribute('aria-checked', String(row.switch === 'on'));
  toggle.setAttribute('aria-label', `${statePhrase(word)} for ${name}`);
  if (ctx.confirmingTarget === row.target) toggle.dataset.pending = 'on';
  toggle.disabled = !armed && Boolean(lock);
  toggle.title = armed
    ? `Turn writes off for ${name} — applies now`
    : lock || `Turn writes on for ${name} — asks first`;
  toggle.addEventListener('click', (event) => {
    event.stopPropagation();
    if (armed) ctx.onTurnOff(row, state);
    else ctx.onTurnOn(row, state, event.shiftKey);
  });
  // The design system's track and knob; `aria-checked` on the button moves it.
  const track = el('span', 'osprey-switch');
  track.setAttribute('aria-hidden', 'true');
  toggle.append(track);
  wrap.append(toggle);
  const stateWordEl = el('span', 'ctc-switch-state', word);
  const why = word === 'locked' ? row.write_phrase || lock : armed ? null : lock;
  if (why) stateWordEl.title = why;
  wrap.append(stateWordEl);
  return wrap;
}

/**
 * The verdict's own sentence for a refused machine, at rest: what stopped the
 * write and who lifts it (C-PHRASE). Shown for a locked row — on and still
 * refused, or unreadable — and for an off row held by a lock no turn-on lifts;
 * never for a row that is merely off, whose switch already says so.
 * @param {any} row
 * @param {string} word
 * @param {string|null} lock
 * @returns {HTMLElement|null}
 */
function writePhraseLine(row, word, lock) {
  const held = word === 'off' && row.write_reason && !LIFTED_BY_TURN_ON.has(row.write_reason);
  const text = word === 'locked' || held ? row.write_phrase || lock : null;
  if (!text) return null;
  const line = el('div', 'ctc-write-phrase', text);
  line.dataset.state = word;
  return line;
}

/**
 * The Switch slot: the button, or the phrase for why there is no button.
 *
 * The refusal is keyed on the switch tool's own machine code, published by the
 * route — so the popover and the agent agree about the same refusal — and
 * rendered as {@link REASON_PHRASES}' operator phrase, with the route's
 * `reason_detail` sentence (falling back to the code) on the `title`.
 * @param {any} row
 * @param {any} state
 * @param {RowContext} ctx
 */
function renderAction(row, state, ctx) {
  const action = el('div', 'ctc-action');
  // One request is outstanding at a time, so while it is out no row offers a
  // second.
  if (isPending()) return action;
  if (row.available_now && contextWritable(state)) {
    const swap = button('ctc-switch osprey-btn primary', 'Switch to');
    swap.title = `Move all control reads and writes to ${displayName(row, kindAttr(row))}`;
    swap.addEventListener('click', (event) => {
      event.stopPropagation();
      ctx.onSwitchTo(row, state, event.shiftKey);
    });
    action.append(swap);
    return action;
  }
  // The row's own reason wins; the context supplies the fallback, and names
  // its holder itself — a two-word chip reading "another terminal" over an
  // agent's controls server would send the operator to a page that does not
  // exist.
  const code = row.reason || contextRefusalCode(state);
  const phrase = row.reason ? reasonPhrase(code) : contextRefusalPhrase(state);
  const reason = el('span', 'ctc-reason', phrase);
  const detail =
    typeof row.reason_detail === 'string' && row.reason_detail ? row.reason_detail : '';
  if (detail || code) reason.title = detail || String(code);
  action.append(reason);
  return action;
}

/**
 * The foot: all-off, and the popover's scope said once. `Turn all writes off`
 * only ever removes reach, so it applies on click; it is enabled exactly when
 * the read says some switch of this login's is on (`any_on`) — including a
 * machine no longer configured, which no row shows.
 *
 * The note beside it is the popover's scope, said once at rest: the switch is
 * the login's, so a stop recorded here reaches every session, notebook kernel
 * and hook of that login, not just the page it was made on.
 * @param {any} state
 * @param {RowContext} ctx
 */
export function renderFoot(state, ctx) {
  const foot = el('div', 'ctc-foot');
  const all = button('ctc-all-off osprey-btn', 'Turn all writes off');
  all.disabled = state.any_on !== true;
  all.addEventListener('click', (event) => {
    event.stopPropagation();
    ctx.onAllOff(state);
  });
  foot.append(all);
  foot.append(el('span', 'ctc-foot-note', 'Applies to every session of this login'));
  return foot;
}
