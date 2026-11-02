// @ts-check
/**
 * OSPREY Web Terminal — operator chat: question card.
 *
 * The card for an AskUserQuestion: one question per page, options as radio rows
 * (checkboxes for multi-select), and an "Other…" row for free text. Once
 * settled it folds to a trace: one line that opens to each question and its
 * answer. A replay draws the same trace from the call and its result.
 *
 * @module chat-question
 */

import { chatIcon } from './chat-icons.js';
import { button, elem } from './chat-dom.js';
import { deliver } from './chat-permission.js';

/** @typedef {import('./chat-icons.js').ChatIconName} ChatIconName */
/** @typedef {import('./chat-dom.js').WireEvent} WireEvent */
/** @typedef {import('./chat-permission.js').CardHooks} CardHooks */
/** @typedef {import('./chat-permission.js').PermissionCard} PermissionCard */

/** Numbers the prompts, so each question card can name its group by its prompt. */
let questionSeq = 0;

/**
 * @typedef {{ label: string, description: string }} QuestionOption
 */

/**
 * @typedef {object} QuestionSpec
 * @property {string} question
 * @property {string} header
 * @property {boolean} multiSelect
 * @property {QuestionOption[]} options
 */

/**
 * @typedef {object} QuestionPick
 * @property {Set<number>} picked - indices of the chosen options
 * @property {boolean} otherOn - whether the "Other…" answer counts
 * @property {string} other - the free text typed into "Other…"
 */

/**
 * The shape of an AskUserQuestion's `questions`, cleaned for display. A missing
 * or empty list becomes one free-text question, so the card always has a
 * page to show.
 * @param {unknown} raw
 * @returns {QuestionSpec[]}
 */
function normaliseQuestions(raw) {
  const specs = (Array.isArray(raw) ? raw : []).map((item) => {
    const q = /** @type {WireEvent} */ (item && typeof item === 'object' ? item : {});
    const options = (Array.isArray(q.options) ? q.options : [])
      .map((o) => ({
        label: String(o?.label ?? ''),
        description: String(o?.description ?? ''),
      }))
      .filter((o) => o.label !== '');
    return {
      question: String(q.question ?? ''),
      header: String(q.header ?? ''),
      multiSelect: Boolean(q.multiSelect),
      options,
    };
  });
  return specs.length > 0
    ? specs
    : [{ question: 'The agent has a question.', header: '', multiSelect: false, options: [] }];
}

/**
 * How a question settled, as its trace names it.
 * @typedef {'answered' | 'answered-elsewhere' | 'skipped' | 'expired' | 'cancelled'} QuestionOutcome
 */

/** @type {Readonly<Record<QuestionOutcome, { state: string, icon: ChatIconName, text: string }>>} */
const OUTCOMES = Object.freeze({
  answered: { state: 'answered', icon: 'check', text: 'Answered' },
  'answered-elsewhere': { state: 'answered', icon: 'check', text: 'Answered in another tab' },
  skipped: { state: 'skipped', icon: 'x', text: 'Skipped' },
  expired: { state: 'skipped', icon: 'x', text: 'No answer · skipped' },
  cancelled: { state: 'cancelled', icon: 'circle', text: 'Cancelled' },
});

/**
 * Fill `el` with a settled question's trace: a line naming the outcome and
 * what was asked, which opens to every question with its answer.
 * @param {HTMLElement} el - the card, whose `data-state` styles the line
 * @param {QuestionSpec[]} questions
 * @param {QuestionOutcome} outcome
 * @param {Record<string, string> | null} answers - by exact question text;
 *   null when nothing was answered
 */
function fillTrace(el, questions, outcome, answers) {
  const { state, icon, text } = OUTCOMES[outcome];
  el.dataset.state = state;
  const subject =
    questions.length === 1
      ? questions[0].header || questions[0].question
      : `${questions.length} questions`;
  const head = button('op-qtrace-head');
  head.setAttribute('aria-expanded', 'false');
  const status = chatIcon(icon);
  status.classList.add('op-qtrace-status');
  const chevron = chatIcon('chevron');
  chevron.classList.add('op-qtrace-chevron');
  head.append(status, elem('span', 'op-qtrace-label', `${text} · ${subject}`), chevron);

  const body = elem('div', 'op-qtrace-body');
  body.hidden = true;
  for (const q of questions) {
    const item = elem('div', 'op-qtrace-item');
    if (q.header) item.append(elem('span', 'op-card-chip', q.header));
    item.append(elem('p', 'op-qtrace-q', q.question));
    const value = answers?.[q.question] ?? '';
    const reply = elem('p', 'op-qtrace-a', value || 'No answer');
    if (!value) reply.dataset.empty = 'true';
    item.append(reply);
    body.append(item);
  }
  head.addEventListener('click', () => {
    const open = body.hidden;
    body.hidden = !open;
    head.setAttribute('aria-expanded', String(open));
  });
  el.replaceChildren(head, body);
}

/**
 * A settled question as history draws it: the trace its card folded to live.
 * @param {unknown} rawQuestions - an AskUserQuestion's `input.questions`
 * @param {{ outcome: QuestionOutcome, answers: Record<string, string> | null }} settled
 * @returns {HTMLElement}
 */
function createQuestionTrace(rawQuestions, { outcome, answers }) {
  const el = elem('section', 'op-card op-question');
  el.setAttribute('role', 'group');
  el.setAttribute('aria-label', 'The agent asked');
  fillTrace(el, normaliseQuestions(rawQuestions), outcome, answers);
  return el;
}

/**
 * The answers in an AskUserQuestion's result text, keyed by question. The CLI
 * writes them as `"<question>"="<answer>"`, comma-joined, before a closing
 * sentence. Each answer runs to the next question's opening or to that
 * sentence, so a quote inside an answer does not cut it short.
 * @param {string} text - the call's result text
 * @param {unknown} rawQuestions - the call's `input.questions`
 * @returns {Record<string, string>}
 */
export function answersFromResult(text, rawQuestions) {
  const asked = normaliseQuestions(rawQuestions).map((q) => q.question);
  /** @type {Record<string, string>} */
  const answers = {};
  for (const question of asked) {
    const marker = `"${question}"="`;
    const at = text.indexOf(marker);
    if (at < 0) continue;
    const from = at + marker.length;
    const ends = [
      text.indexOf('". You can now continue', from),
      ...asked
        .filter((other) => other !== question)
        .map((other) => text.indexOf(`", "${other}"="`, from)),
    ].filter((i) => i >= 0);
    const end = ends.length > 0 ? Math.min(...ends) : text.lastIndexOf('"');
    if (end > from) answers[question] = text.slice(from, end);
  }
  return answers;
}

/**
 * The bookkeeping that redraws replayed AskUserQuestion calls. History holds no
 * question card, only the call and its result, so the call's questions are kept
 * until its result arrives, which then becomes the trace the card settled to:
 * answered with the result's answers, or skipped when the call ended in an error.
 * @returns {{ ask: (event: WireEvent) => void, answer: (event: WireEvent) => HTMLElement | null }}
 */
export function createQuestionReplay() {
  /** @type {Map<string, unknown>} */
  const asked = new Map();
  return {
    ask(event) {
      if (event.tool_use_id) asked.set(String(event.tool_use_id), event.input?.questions);
    },
    answer(event) {
      const id = String(event.tool_use_id ?? '');
      if (!asked.has(id)) return null;
      const questions = asked.get(id);
      asked.delete(id);
      const skipped = Boolean(event.is_error);
      return createQuestionTrace(questions, {
        outcome: skipped ? 'skipped' : 'answered',
        answers: skipped ? null : answersFromResult(String(event.preview ?? ''), questions),
      });
    },
  };
}

/**
 * The question card for an AskUserQuestion: one question per page, options as
 * radio rows (checkboxes for multi-select), and an "Other…" row for free text.
 * Keys 1–9 pick an option on the current page, the arrow keys turn pages,
 * Enter takes the primary action and Escape skips. Submit answers every
 * question, keyed by its exact text; Skip denies the request.
 * @param {WireEvent} request - a `permission_request` event of kind `question`
 * @param {CardHooks} hooks
 * @returns {PermissionCard}
 */
export function createQuestionCard(request, hooks) {
  const questions = normaliseQuestions(request.input?.questions);
  /** @type {QuestionPick[]} */
  const picks = questions.map(() => ({ picked: new Set(), otherOn: false, other: '' }));
  let page = 0;
  let state = 'pending';
  const isPending = () => state === 'pending' || state === 'sending';
  const isLast = () => page === questions.length - 1;

  const el = elem('section', 'op-card op-question');
  el.dataset.state = 'pending';
  el.tabIndex = -1;
  el.setAttribute('role', 'group');

  const head = elem('div', 'op-card-head');
  const icon = chatIcon('question');
  icon.classList.add('op-card-icon');
  const chip = elem('span', 'op-card-chip');
  const pager = elem('div', 'op-card-pager');
  const prev = button('op-pager');
  prev.setAttribute('aria-label', 'Previous question');
  const prevIcon = chatIcon('chevron');
  prevIcon.classList.add('op-pager-prev');
  prev.append(prevIcon);
  const position = elem('span', 'op-card-pos');
  const next = button('op-pager');
  next.setAttribute('aria-label', 'Next question');
  next.append(chatIcon('chevron'));
  pager.append(prev, position, next);
  head.append(icon, chip, pager);

  const prompt = elem('p', 'op-question-text');
  prompt.id = `op-question-title-${++questionSeq}`;
  el.setAttribute('aria-labelledby', prompt.id);
  const list = elem('div', 'op-options');
  const error = elem('p', 'op-card-error');
  error.hidden = true;
  const skip = button('op-btn', 'Skip');
  const primary = button('op-btn op-btn-primary', 'Continue');
  const actions = elem('div', 'op-card-actions');
  actions.append(skip, primary);
  el.append(head, prompt, list, error, actions);

  /** @returns {QuestionPick} */
  const current = () => /** @type {QuestionPick} */ (picks[page]);

  /**
   * What one question's answer reads as: the chosen labels, then the typed
   * text, comma-joined. Empty when nothing is chosen.
   * @param {number} index
   * @returns {string}
   */
  function valueOf(index) {
    const q = questions[index];
    const pick = picks[index];
    const labels = [...pick.picked].sort((a, b) => a - b).map((n) => q.options[n].label);
    const other = pick.otherOn ? pick.other.trim() : '';
    if (other) labels.push(other);
    return labels.join(', ');
  }
  const answered = (/** @type {number} */ index) => valueOf(index) !== '';

  /** Sync every control to the current picks and pending state. No rebuild: typing must keep focus. */
  function refresh() {
    const q = questions[page];
    const pick = current();
    const busy = state === 'sending';
    for (const row of list.querySelectorAll('[data-option]')) {
      const option = row.getAttribute('data-option');
      const on = option === 'other' ? pick.otherOn : pick.picked.has(Number(option));
      if (option === 'other') row.setAttribute('data-checked', String(on));
      else row.setAttribute('aria-checked', String(on));
      row.classList.toggle('on', on);
      const mark = row.querySelector('.op-option-mark');
      if (mark) mark.replaceChildren(...(on && q.multiSelect ? [chatIcon('check')] : []));
      if (row instanceof HTMLButtonElement) row.disabled = busy;
    }
    const input = list.querySelector('input.op-other-input');
    if (input instanceof HTMLInputElement) input.disabled = busy;

    position.textContent = `${page + 1} / ${questions.length}`;
    pager.hidden = questions.length < 2;
    prev.disabled = busy || page === 0;
    next.disabled = busy || isLast();
    primary.textContent = isLast() ? 'Submit' : 'Continue';
    const complete = questions.every((_, i) => answered(i));
    primary.disabled = busy || (isLast() ? !complete : !answered(page));
    skip.disabled = busy;
  }

  /** Rebuild the current page's prompt and rows. */
  function render() {
    const q = questions[page];
    chip.textContent = q.header;
    chip.hidden = !q.header;
    prompt.textContent = q.question;
    /** @type {HTMLElement[]} */ const rows = q.options.map((option, index) => {
      const row = button('op-option');
      row.dataset.option = String(index);
      row.setAttribute('role', q.multiSelect ? 'checkbox' : 'radio');
      const body = elem('span', 'op-option-body');
      body.append(elem('span', 'op-option-label', option.label));
      if (option.description) body.append(elem('span', 'op-option-desc', option.description));
      row.append(
        elem('span', 'op-option-key', String(index + 1)),
        elem('span', 'op-option-mark'),
        body
      );
      row.addEventListener('click', () => choose(index));
      return row;
    });
    rows.push(otherRow(q));
    list.replaceChildren(...rows);
    refresh();
  }

  /** @param {QuestionSpec} q */
  function otherRow(q) {
    const row = elem('div', 'op-option op-option-other');
    row.dataset.option = 'other';
    row.dataset.multi = String(q.multiSelect);
    const key = q.options.length + 1 <= 9 ? String(q.options.length + 1) : '';
    const input = /** @type {HTMLInputElement} */ (document.createElement('input'));
    input.type = 'text';
    input.className = 'op-other-input';
    input.placeholder = 'Other…';
    input.setAttribute('aria-label', 'Other answer');
    input.value = current().other;
    input.addEventListener('input', () => {
      const pick = current();
      pick.other = input.value;
      pick.otherOn = input.value.trim() !== '';
      if (!q.multiSelect && pick.otherOn) pick.picked.clear();
      refresh();
    });
    const body = elem('span', 'op-option-body');
    body.append(input);
    row.append(elem('span', 'op-option-key', key), elem('span', 'op-option-mark'), body);
    row.addEventListener('click', (event) => {
      if (event.target !== input) chooseOther();
    });
    return row;
  }

  /** @param {number} index */
  function choose(index) {
    if (state !== 'pending') return;
    const q = questions[page];
    const pick = current();
    if (q.multiSelect) {
      if (pick.picked.has(index)) pick.picked.delete(index);
      else pick.picked.add(index);
    } else {
      pick.picked.clear();
      pick.picked.add(index);
      pick.otherOn = false;
      pick.other = '';
      const input = list.querySelector('input.op-other-input');
      if (input instanceof HTMLInputElement) input.value = '';
    }
    refresh();
  }

  function chooseOther() {
    if (state !== 'pending') return;
    const q = questions[page];
    const pick = current();
    if (q.multiSelect) {
      pick.otherOn = !pick.otherOn;
    } else {
      pick.picked.clear();
      pick.otherOn = true;
    }
    refresh();
    const input = list.querySelector('input.op-other-input');
    if (pick.otherOn && input instanceof HTMLInputElement) input.focus();
  }

  /** @param {number} next */
  function goTo(next) {
    if (state !== 'pending') return;
    const target = Math.min(Math.max(next, 0), questions.length - 1);
    if (target === page) return;
    page = target;
    render();
    el.focus({ preventScroll: true });
  }

  /** @returns {Record<string, string>} */
  function answersFor() {
    /** @type {Record<string, string>} */
    const answers = {};
    questions.forEach((q, i) => {
      answers[q.question] = valueOf(i);
    });
    return answers;
  }

  /**
   * Fold the card to its trace, holding what was asked and, when answered, the answers.
   * @param {boolean} allowed
   */
  function foldResolved(allowed) {
    state = allowed ? 'answered' : 'skipped';
    fillTrace(el, questions, allowed ? 'answered' : 'skipped', allowed ? answersFor() : null);
  }

  /** @param {boolean} allowed */
  async function send(allowed) {
    if (state !== 'pending') return;
    state = 'sending';
    el.dataset.state = 'sending';
    error.hidden = true;
    refresh();
    const body = allowed ? { allow: true, answers: answersFor() } : { allow: false };
    const outcome = await deliver(hooks, body);
    if (state !== 'sending') return; // the turn ended while the answer was in flight
    if (outcome === 'answered') {
      foldResolved(allowed);
    } else if (outcome === 'stale') {
      cancel();
    } else {
      state = 'pending';
      el.dataset.state = 'pending';
      error.textContent = 'Could not send that. Try again.';
      error.hidden = false;
      refresh();
    }
    hooks.onSettled();
  }

  function cancel() {
    if (!isPending()) return;
    state = 'cancelled';
    fillTrace(el, questions, 'cancelled', null);
  }

  prev.addEventListener('click', () => goTo(page - 1));
  next.addEventListener('click', () => goTo(page + 1));
  skip.addEventListener('click', () => void send(false));
  primary.addEventListener('click', () => {
    if (!isLast()) goTo(page + 1);
    else void send(true);
  });
  el.addEventListener('keydown', (event) => {
    if (state !== 'pending') return;
    const target = /** @type {HTMLElement} */ (event.target);
    if (event.key === 'Escape') {
      event.preventDefault();
      void send(false);
      return;
    }
    // A focused button keeps its own Enter (a click); anywhere else Enter takes the primary action.
    if (event.key === 'Enter' && !event.shiftKey && !(target instanceof HTMLButtonElement)) {
      event.preventDefault();
      primary.click();
      return;
    }
    if (target instanceof HTMLInputElement) return;
    if (/^[1-9]$/.test(event.key)) {
      const n = Number(event.key) - 1;
      const count = questions[page].options.length;
      if (n < count) {
        event.preventDefault();
        choose(n);
      } else if (n === count) {
        event.preventDefault();
        chooseOther();
      }
      return;
    }
    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      goTo(page - 1);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      goTo(page + 1);
    }
  });

  render();

  return {
    el,
    get pending() {
      return isPending();
    },
    cancel,
    focus: () => el.focus(),
    resolved(answer) {
      // Answered in another tab, or declined by the server's deadline; this
      // tab's own answer in flight folds itself.
      if (state !== 'pending') return;
      /** @type {QuestionOutcome} */
      const outcome = answer.expired ? 'expired' : answer.allow ? 'answered-elsewhere' : 'skipped';
      state = OUTCOMES[outcome].state;
      fillTrace(el, questions, outcome, answer.allow ? (answer.answers ?? null) : null);
      hooks.onSettled();
    },
  };
}
