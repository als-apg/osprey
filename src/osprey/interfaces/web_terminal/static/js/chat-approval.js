// @ts-check
/**
 * OSPREY Web Terminal — operator chat: approval card.
 *
 * The card for a tool call the operator must allow or deny, or for a plan the
 * agent wants to carry out. Filed under the call's own row, it leaves the log
 * once answered and the row shows the answer; otherwise it folds to one line
 * stating the outcome.
 *
 * @module chat-approval
 */

import { chatIcon } from './chat-icons.js';
import { button, elem } from './chat-dom.js';
import { inputDetail, inputView } from './chat-format.js';
import { CATEGORY_ICON, summarizeInput, toolCategory } from './chat-tool-vocab.js';
import { deliver } from './chat-permission.js';
import { KIND_BANNERS, KIND_WORDS, machineKind } from './control-target-facts.js';

/** @typedef {import('./chat-icons.js').ChatIconName} ChatIconName */
/** @typedef {import('./chat-dom.js').WireEvent} WireEvent */
/** @typedef {import('./chat-permission.js').CardHooks} CardHooks */
/** @typedef {import('./chat-permission.js').PermissionCard} PermissionCard */
/** @typedef {import('./chat-tool-cards.js').ToolRow} ToolRow */
/** @typedef {import('./chat-tool-cards.js').Decision} Decision */

/**
 * The line naming the machine a write lands on, from the frame's `target`
 * (C-FRAME): its key and what a write there does, under the same
 * `data-target-kind` the chip sets, so the stylesheet colours both alike.
 * `null` when the call writes nothing or cannot be placed. A kind outside the
 * three reads as the real machine, and then its words are the real machine's
 * too rather than whatever the frame said.
 * @param {any} target
 * @returns {HTMLElement | null}
 */
function targetLine(target) {
  if (!target || typeof target !== 'object') return null;
  const kind = machineKind(target);
  const published = String(target.machine_kind || '')
    .trim()
    .toLowerCase();
  const sent = typeof target.kind_words === 'string' ? target.kind_words.trim() : '';
  const words = published === kind && sent ? sent : (KIND_BANNERS[kind] ?? KIND_WORDS[kind]);
  const line = elem('p', 'op-card-target');
  line.dataset.targetKind = kind;
  const key = String(target.key || '').trim();
  if (key)
    line.append(elem('span', 'op-card-target-key', key), elem('span', 'op-card-target-sep', '·'));
  line.append(elem('span', 'op-card-target-words', words));
  return line;
}

/** Numbers the card titles, so each card can name its group by its title. */
let cardSeq = 0;

/**
 * The approval card for a tool call the operator must allow or deny, or — for
 * kind `plan` — the plan the agent wants to carry out. Enter allows and Escape
 * denies while the card has focus. Where the request offers it, "Allow for this
 * session" also lets the tool run without asking until the session ends. A card
 * filed under its tool row leaves the log once answered, and the row shows the
 * answer; any other card folds to one line stating the outcome.
 * @param {WireEvent} request - a `permission_request` event of kind `tool` or `plan`
 * @param {CardHooks & { renderText?: (el: HTMLElement, text: string) => void, row?: ToolRow }} hooks
 *   - `row`: the tool row of the call, when it has one; the card is filed under it
 * @returns {PermissionCard}
 */
export function createApprovalCard(request, hooks) {
  const isPlan = request.kind === 'plan';
  const toolName = String(request.tool_name || request.tool_name_raw || 'this tool');
  const el = elem('section', `op-card op-approval${isPlan ? ' op-plan' : ''}`);
  el.dataset.state = 'pending';
  el.tabIndex = -1;
  el.setAttribute('role', 'group');

  const head = elem('div', 'op-card-head');
  const toolIcon = chatIcon(
    isPlan ? 'read' : CATEGORY_ICON[toolCategory(request.tool_name_raw ?? request.tool_name)]
  );
  toolIcon.classList.add('op-card-icon');
  const titleText = isPlan ? 'Ready to start on this plan?' : `Allow ${toolName}?`;
  const title = elem('span', 'op-card-title', titleText);
  title.id = `op-card-title-${++cardSeq}`;
  el.setAttribute('aria-labelledby', title.id);
  head.append(toolIcon, title);

  const body = elem('div', 'op-card-body');
  const reason = typeof request.reason === 'string' ? request.reason.trim() : '';
  if (reason && !isPlan) body.append(elem('p', 'op-card-reason', reason));
  const target = isPlan ? null : targetLine(request.target);
  if (target) body.append(target);

  if (isPlan) {
    const plan = elem('div', 'op-card-plan osprey-md-rendered');
    const text = String(request.input?.plan ?? '');
    if (hooks.renderText) hooks.renderText(plan, text);
    else plan.textContent = text;
    body.append(plan);
  } else {
    const shown = summarizeInput(request.input);
    const summary = elem('div', 'op-card-summary');
    summary.append(elem('code', 'op-card-input', shown));
    body.append(summary);
    const rest = inputDetail(request.input, shown);
    if (rest !== null) {
      const detail = elem('div', 'op-card-detail');
      detail.hidden = true;
      detail.append(inputView(rest));
      const toggle = button('op-card-toggle', 'Details');
      toggle.setAttribute('aria-expanded', 'false');
      toggle.addEventListener('click', () => {
        const open = detail.hidden;
        detail.hidden = !open;
        toggle.setAttribute('aria-expanded', String(open));
        toggle.textContent = open ? 'Hide' : 'Details';
      });
      summary.append(toggle);
      body.append(detail);
    }
  }

  const error = elem('p', 'op-card-error');
  error.hidden = true;
  const deny = button('op-btn', isPlan ? 'Keep planning' : 'Deny');
  const allow = button('op-btn op-btn-primary', isPlan ? 'Approve plan' : 'Allow');
  const session =
    request.can_allow_session && !isPlan ? button('op-btn', 'Allow for this session') : null;
  const actions = elem('div', 'op-card-actions');
  actions.append(deny, ...(session ? [session] : []), allow);
  body.append(error, actions);
  el.append(head, body);
  // Filed under its own row, the card is the row's to show until it is answered.
  const placement = hooks.row ? hooks.row.file(el) : null;
  let state = 'pending';
  const isPending = () => state === 'pending' || state === 'sending';

  /**
   * Fold the card to its one-line resolved state.
   * @param {string} tone - the state, also the `data-state` CSS hook
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

  function cancel() {
    if (!isPending()) return;
    if (placement) {
      state = 'cancelled';
      placement.drop();
      return;
    }
    fold('cancelled', 'circle', `Cancelled · ${isPlan ? 'plan' : toolName}`);
  }

  /**
   * The answer is in. A card filed under its row hands it to the row; any other
   * card folds to its outcome line.
   * @param {boolean} allowed
   * @param {boolean} forSession
   */
  function foldOutcome(allowed, forSession) {
    if (placement) {
      state = allowed ? 'allowed' : 'denied';
      const decision = allowed ? (forSession ? 'session' : 'allowed') : 'denied';
      placement.decide(decision);
      return;
    }
    if (isPlan) {
      if (allowed) fold('allowed', 'check', 'Plan approved');
      else fold('denied', 'x', 'Kept planning');
    } else if (allowed) {
      fold(
        'allowed',
        'check',
        `${forSession ? 'Allowed for this session' : 'Allowed'} · ${toolName}`
      );
    } else {
      fold('denied', 'x', `Denied · ${toolName}`);
    }
  }

  /** Nobody answered within the server's deadline, so it declined the ask. */
  function foldExpired() {
    if (placement) {
      state = 'denied';
      placement.decide('expired');
      return;
    }
    if (isPlan) fold('denied', 'x', 'No answer · kept planning');
    else fold('denied', 'x', `No answer · denied · ${toolName}`);
  }

  /**
   * @param {boolean} allowed
   * @param {boolean} [forSession]
   */
  async function decide(allowed, forSession = false) {
    if (state !== 'pending') return;
    state = 'sending';
    el.dataset.state = 'sending';
    for (const control of [allow, deny, session]) if (control) control.disabled = true;
    error.hidden = true;

    const outcome = await deliver(hooks, {
      allow: allowed,
      ...(forSession ? { for_session: true } : {}),
    });
    if (state !== 'sending') return; // the turn ended while the answer was in flight
    if (outcome === 'answered') {
      foldOutcome(allowed, forSession);
    } else if (outcome === 'stale') {
      cancel();
    } else {
      state = 'pending';
      el.dataset.state = 'pending';
      for (const control of [allow, deny, session]) if (control) control.disabled = false;
      error.textContent = 'Could not send that. Try again.';
      error.hidden = false;
    }
    hooks.onSettled();
  }

  allow.addEventListener('click', () => void decide(true));
  deny.addEventListener('click', () => void decide(false));
  session?.addEventListener('click', () => void decide(true, true));
  el.addEventListener('keydown', (event) => {
    if (state !== 'pending') return;
    // Enter on a focused button is that button's: Deny must not allow, and Details must not decide.
    if (event.key === 'Enter' && !(event.target instanceof HTMLButtonElement)) {
      event.preventDefault();
      void decide(true);
    } else if (event.key === 'Escape') {
      event.preventDefault();
      void decide(false);
    }
  });

  return {
    el,
    get pending() {
      return isPending();
    },
    cancel,
    focus: () => allow.focus(),
    resolved(answer) {
      // Answered in another tab, or declined by the server's deadline; an
      // answer of this tab's own still in flight folds the card itself.
      if (state !== 'pending') return;
      if (answer.expired) foldExpired();
      else foldOutcome(Boolean(answer.allow), Boolean(answer.for_session));
      hooks.onSettled();
    },
  };
}
