// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode operator chat: question card.
 *
 * The card for an AskUserQuestion: one question per page, options as radio rows
 * (checkboxes for multi-select), and an "Other…" row for free text. Once
 * answered it folds to one line; a skip denies the request.
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
      row.append(elem('span', 'op-option-key', String(index + 1)), elem('span', 'op-option-mark'), body);
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
   * Fold the card to its resolved state, listing what was answered.
   * @param {boolean} allowed
   */
  function foldResolved(allowed) {
    if (!allowed) {
      fold('skipped', 'x', 'Skipped');
      return;
    }
    fold('answered', 'check', 'Answered');
    const list2 = elem('ul', 'op-card-answers');
    questions.forEach((q, i) => {
      const item = document.createElement('li');
      const label = document.createElement('strong');
      label.textContent = q.header || q.question;
      item.append(label, document.createTextNode(`: ${valueOf(i)}`));
      list2.append(item);
    });
    el.append(list2);
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
    fold('cancelled', 'circle', 'Cancelled');
  }

  /**
   * @param {string} tone
   * @param {ChatIconName} iconName
   * @param {string} text
   */
  function fold(tone, iconName, text) {
    state = tone;
    el.dataset.state = tone;
    const line = elem('div', 'op-card-line');
    line.append(chatIcon(iconName), elem('span', 'op-card-line-text', text));
    el.replaceChildren(line);
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
      if (answer.expired) fold('skipped', 'x', 'No answer · skipped');
      else if (answer.allow) fold('answered', 'check', 'Answered in another tab');
      else fold('skipped', 'x', 'Skipped');
      hooks.onSettled();
    },
  };
}
