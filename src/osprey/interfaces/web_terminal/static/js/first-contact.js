// @ts-check
/* OSPREY Web Terminal — First Contact
 *
 * What a newcomer reads before they type anything: one sentence naming what
 * the agent can do in THIS deployment, and up to three questions they can send
 * as-is. The Simple view's empty state and the tour's first card both draw
 * from here, so a visitor cannot be told two different stories about the same
 * deployment depending on which surface they met first.
 *
 * Every fact is derived, never authored: the read phrase comes from the
 * control-target chip's kind for the machine this session stands on
 * ({@link module:control-target-facts.KIND_READ_PHRASES}), and the rest comes
 * from what the server published about this deployment ({@link setFacts}).
 * Nothing here is a claim the server cannot back — in particular a deployment
 * whose numbers are mock data says "read demo data", and a session standing on
 * no known machine gets no read phrase at all rather than a plausible guess.
 *
 * The functions above {@link setFacts} are pure and take their inputs
 * explicitly; the module state below them exists only so the surfaces and the
 * tour can call them with no arguments and get this session's answer.
 */

import { activeKind, subscribe } from './control-target-chip.js';
import { KIND_READ_PHRASES } from './control-target-facts.js';
import { focusTerminal, onSessionChange, pasteToTerminal } from './terminal.js';

/**
 * What the server knows about this deployment's first contact.
 * @typedef {Object} FirstContactFacts
 * @property {string[]} capabilities  capability phrases beyond reading values
 * @property {boolean} logbook        whether this deployment has a logbook
 */

/**
 * What the capability sentence opens with. Shared by the string form and the
 * rendered form, so the tour's card and the Simple block cannot drift apart.
 */
const SENTENCE_LEAD = 'Here the agent can ';

/** The prompt offered only while the chip names a machine to read from. */
const PROMPT_READ = 'What can you read right now?';

/**
 * The prompt every deployment offers. What the agent is ALLOWED to do is
 * per-session posture the page cannot see, so the honest move is to let the
 * agent answer it rather than to phrase it here.
 */
const PROMPT_ALLOWED = 'What are you allowed to do in this session?';

/** The prompt offered only where a logbook exists to have a today. */
const PROMPT_LOGBOOK = 'What happened in the logbook today?';

/** "a, b, and c" @param {string[]} items */
export const listPhrase = (items) => {
  if (items.length <= 1) return items.join('');
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(', ')}, and ${items[items.length - 1]}`;
};

/**
 * The text between phrase `index - 1` and phrase `index` of an `n`-item list —
 * the same grammar as `listPhrase`, so the emphasised paragraph and the plain
 * sentence read identically.
 * @param {number} index
 * @param {number} n
 * @returns {string}
 */
const joiner = (index, n) => {
  if (n === 2) return ' and ';
  return index === n - 1 ? ', and ' : ', ';
};

/**
 * Everything the agent can do here, read phrase first.
 *
 * The read phrase leads because it is the one the machine kind decides, and a
 * kind the chip has no answer for contributes nothing: an unknown machine is
 * exactly the case where naming where the numbers come from would be a guess.
 * @param {string|null} kind  {@link module:control-target-chip.activeKind}'s answer
 * @param {FirstContactFacts} facts
 * @returns {string[]}
 */
export function capabilityPhrases(kind, facts) {
  const read = kind ? KIND_READ_PHRASES[kind] : null;
  return read ? [read, ...facts.capabilities] : [...facts.capabilities];
}

/**
 * The one sentence both views and the tour's first card put in front of a
 * newcomer, or `""` when there is nothing to claim — a deployment that
 * published no capabilities on a session standing on no known machine says
 * nothing rather than an empty promise.
 * @param {string|null} [kind]
 * @param {FirstContactFacts} [facts]
 * @returns {string}
 */
export function capabilitySentence(kind = activeKind(), facts = current) {
  const phrases = capabilityPhrases(kind, facts);
  return phrases.length ? `${SENTENCE_LEAD}${listPhrase(phrases)}.` : '';
}

/**
 * The questions offered as chips, in reading order: what is there to see, what
 * may be done, what already happened. Each one is dropped when the deployment
 * cannot answer it, so a chip never opens a door onto nothing.
 * @param {string|null} [kind]
 * @param {FirstContactFacts} [facts]
 * @returns {string[]}
 */
export function starterPrompts(kind = activeKind(), facts = current) {
  const prompts = [];
  if (kind) prompts.push(PROMPT_READ);
  prompts.push(PROMPT_ALLOWED);
  if (facts.logbook) prompts.push(PROMPT_LOGBOOK);
  return prompts;
}

/* ---- insert seam -------------------------------------------------------- */

/** The Simple view's one prompt input. */
const SIMPLE_INPUT = '#operator-container .op-input-area textarea';

/**
 * Put one prompt where the operator is typing, and leave the cursor there.
 *
 * The one seam the Simple empty state and the tour's chips share, because a
 * chip that inserts somewhere the reader cannot see is worse than no chip: the
 * Simple view hides the terminal, so a paste there reaches nobody, and the
 * tour runs in both views. Routing on the rendered mode here keeps that
 * impossible to reintroduce from a new caller.
 *
 * Never sends. The text lands with no trailing newline and focus moves into
 * the same input, so the operator's next Enter is their own decision — a chip
 * that submitted would be the agent acting on a click that only said "ask me
 * this".
 *
 * `append` is for the callers that CONTRIBUTE to a prompt rather than propose
 * one: a panel handing over the addresses it just found is an ingredient of
 * whatever the operator is already writing, and replacing that half-written
 * sentence would destroy the only copy of it. A chip, which offers a whole
 * question, still replaces. Nothing is appended in the terminal view: the
 * terminal has no value to read back, and pasting is already additive there.
 * @param {string} text
 * @param {{append?: boolean}} [options]  `append`: add below existing text
 *   instead of replacing it
 */
export function insertPrompt(text, { append = false } = {}) {
  if (document.documentElement.getAttribute('data-ui-mode') === 'simple') {
    const input = /** @type {HTMLTextAreaElement|null} */ (document.querySelector(SIMPLE_INPUT));
    // Disabled means a turn is streaming: the chat owns the input until it is
    // done, and overwriting a value it is about to clear would be a no-op the
    // operator watched happen.
    if (!input || input.disabled) return;
    // An input holding only whitespace has nothing to preserve, so appending to
    // it would just open the prompt with a blank line. The trailing run is
    // collapsed to the single newline that separates the two, rather than
    // stacked on: the operator's half-typed line and the contribution belong
    // next to each other, not a screen apart.
    input.value =
      append && input.value.trim() ? `${input.value.replace(/\s*$/, '\n')}${text}` : text;
    // What the chat's autoResize listens for; typing raises the same event.
    input.dispatchEvent(new Event('input'));
    input.focus();
    return;
  }
  pasteToTerminal(text);
  focusTerminal();
}

/* ---- the Simple view's empty state -------------------------------------- */

/**
 * The capability sentence as nodes, each phrase emphasised and the connective
 * text plain.
 *
 * Built from the same phrase list {@link capabilitySentence} joins, and with
 * the same joiner, so the block's text content IS that sentence — the emphasis
 * is the only difference between them. Finding the runs to embolden by
 * searching the finished string would put a second, weaker copy of the grammar
 * here, and the two would disagree the first time a phrase gained a comma.
 * @param {string[]} phrases  a non-empty {@link capabilityPhrases} result
 * @returns {HTMLParagraphElement}
 */
function introParagraph(phrases) {
  const intro = document.createElement('p');
  intro.className = 'op-empty-intro';
  intro.append(document.createTextNode(SENTENCE_LEAD));
  phrases.forEach((phrase, index) => {
    if (index > 0) {
      intro.append(document.createTextNode(joiner(index, phrases.length)));
    }
    const run = document.createElement('strong');
    run.textContent = phrase;
    intro.append(run);
  });
  intro.append(document.createTextNode('.'));
  return intro;
}

/**
 * One row of prompt chips, each inserting its text through {@link insertPrompt}.
 *
 * The one chip row for both surfaces — the tour's card and the Simple empty
 * state — because the same prompts offered in two places that LOOK like two
 * different affordances is how an operator learns to distrust both.
 * @param {string[]} prompts
 * @returns {HTMLDivElement}
 */
export function chipRow(prompts) {
  const chips = document.createElement('div');
  chips.className = 'tour-chips';
  for (const text of prompts) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'tour-chip';
    chip.textContent = text;
    chip.addEventListener('click', () => insertPrompt(text));
    chips.append(chip);
  }
  return chips;
}

/**
 * Fill (or refill) the empty-state block: the sentence, then the prompts.
 *
 * Both halves read ONE {@link activeKind} answer, so a switch landing between
 * them cannot leave a sentence about one machine above a prompt about another.
 * Rebuilding in place rather than replacing the block keeps the caller's handle
 * valid across a switch.
 * @param {HTMLElement} el
 * @returns {HTMLElement} the same element
 */
export function renderEmptyStateContent(el) {
  const kind = activeKind();
  const facts = current;
  el.replaceChildren();

  const phrases = capabilityPhrases(kind, facts);
  // No sentence at all rather than a lead-in with nothing after it: a
  // deployment that published nothing has nothing to promise.
  if (phrases.length) el.append(introParagraph(phrases));

  el.append(chipRow(starterPrompts(kind, facts)));
  return el;
}

/**
 * The block the Simple view shows in place of an empty message list: what the
 * agent can do here, and the questions that start a conversation about it.
 * @returns {HTMLElement}
 */
export function buildEmptyState() {
  const block = document.createElement('div');
  block.className = 'op-empty';
  return renderEmptyStateContent(block);
}

/* ---- module state ------------------------------------------------------- */

/**
 * This deployment's facts, as last published. Empty until {@link setFacts},
 * which is also what a failed read settles on: no capabilities and no logbook
 * still render the one prompt that needs neither.
 * @type {FirstContactFacts}
 */
let current = { capabilities: [], logbook: false };

/**
 * Record what `GET /api/panels` said about this deployment.
 *
 * Takes the server's `tour` object, or `null` when the read failed — the null
 * call is not a no-op, it is how first contact learns there is nothing more
 * coming and renders what it can.
 * @param {{capabilities?: unknown, logbook?: unknown}|null|undefined} tour
 */
export function setFacts(tour) {
  const capabilities = Array.isArray(tour?.capabilities) ? tour.capabilities.map(String) : [];
  current = { capabilities, logbook: Boolean(tour?.logbook) };
  factsSet = true;
  evaluate();
}

/* ---- the settled moment ------------------------------------------------- */

/*
 * First contact is the copy a newcomer reads once, and a sentence that
 * rewrites itself while they read it is worse than one that arrives a beat
 * late: they would watch the deployment appear to change its mind about what
 * it can do. So the surfaces do not render as each fact lands. They wait for
 * the moment all three inputs are in — the server's facts, a session to speak
 * about, and a chip render carrying that session's machine — and render once.
 *
 * The three flags below are that moment, and each is a distinct question. A
 * chip callback that arrives BEFORE the session is known describes the
 * previous session's machine, so it is ignored rather than counted; only the
 * first callback after the session settles the kind. Everything is
 * synchronous: no timers, no deferral, so a surface built by the first true
 * evaluation is complete on the frame it appears.
 */

/** Whether the server's facts have arrived — a failed read counts. */
let factsSet = false;

/** Whether a session id has been announced. */
let sessionSeen = false;

/** Whether the chip has rendered since that session was announced. */
let chipRenderedSinceSession = false;

/** Whether the moment has passed; it never un-passes. */
let settled = false;

/**
 * The kind rendered against, so the 5 s idle poll — which calls back with the
 * same state indefinitely — repaints nothing. Only a genuine switch does.
 * @type {string|null}
 */
let lastKind = null;

/** @type {(() => void)[]} */
let settledListeners = [];

/** @type {((kind: string|null) => void)[]} */
const kindListeners = [];

/**
 * Be called once, when there is enough known to render first contact.
 *
 * A listener registered after the moment has passed fires immediately: a
 * surface built late (a view the operator only now switched to) must not be
 * the one that misses it.
 * @param {() => void} fn
 */
export function onSettled(fn) {
  if (settled) {
    fn();
    return;
  }
  settledListeners.push(fn);
}

/**
 * Be told when the machine this session stands on becomes a different one,
 * and only then — the one thing that makes the rendered sentence wrong.
 * @param {(kind: string|null) => void} fn
 */
export function onKindChange(fn) {
  kindListeners.push(fn);
}

/**
 * Decide what this input changed: the moment itself, once, or a later switch.
 * Called at the end of {@link setFacts} and of every chip callback.
 */
function evaluate() {
  if (!settled) {
    if (!factsSet || !sessionSeen || !chipRenderedSinceSession) return;
    settled = true;
    lastKind = activeKind();
    const pending = settledListeners;
    settledListeners = [];
    pending.forEach((fn) => fn());
    return;
  }
  const kind = activeKind();
  if (kind === lastKind) return;
  lastKind = kind;
  kindListeners.forEach((fn) => fn(kind));
}

onSessionChange(() => {
  sessionSeen = true;
  // Deliberately does not evaluate: a session with no chip render behind it
  // has no machine to speak about yet, and the next chip callback evaluates.
  chipRenderedSinceSession = false;
});

subscribe(() => {
  if (!sessionSeen) return;
  chipRenderedSinceSession = true;
  evaluate();
});
