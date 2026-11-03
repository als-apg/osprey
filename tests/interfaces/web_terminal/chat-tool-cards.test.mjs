/**
 * Unit tests for chat-tool-cards.js -- a write the hooks refused shows its
 * phrase in the conversation.
 *
 * A refused write's tool_result frame carries `refusal: {reason, phrase}`
 * (C-FRAME). The phrase says who can lift the refusal, so it is drawn as a
 * line the operator reads without expanding anything: under the call's row,
 * outside the row's collapsed body, and under the run row when the call is
 * folded into a run, whose list is closed. A tool error without `refusal`
 * keeps the collapsed form.
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-tool-cards.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach } from 'vitest';

import { qs } from '../_support/dom.mjs';

import { refusalLine } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-tool-cards.js';
import { createTurnActions } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-tool-runs.js';
import { createChatRenderer } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-render.js';

const SWITCH_OFF = 'Writes are off; turn it on from the chip in the web terminal.';
const TIER_NEVER =
  "Writes are refused because this login's write tier does not allow them; ask the deployment admin.";

/** A write call to the control system. */
const write = (/** @type {string} */ id) => ({
  type: 'tool_use',
  tool_use_id: id,
  tool_name: 'channel_write',
  tool_name_raw: 'mcp__controls__channel_write',
  input: { channel: 'SR:C01:QF:SP', value: 1.5 },
});

/** The hooks' refusal of a call, as the frame carries it. */
const refused = (/** @type {string} */ id, reason = 'switch_off', phrase = SWITCH_OFF) => ({
  type: 'tool_result',
  tool_use_id: id,
  is_error: true,
  preview: phrase,
  truncated: false,
  refusal: { reason, phrase },
});

/** An ordinary tool error: no refusal. */
const failed = (/** @type {string} */ id) => ({
  type: 'tool_result',
  tool_use_id: id,
  is_error: true,
  preview: 'NameError: name x is not defined',
  truncated: false,
  refusal: null,
});

/** Whether `el` is in view: neither it nor anything holding it is hidden. */
const visible = (/** @type {Element} */ el) => el.closest('[hidden]') === null;

/** The refusal lines in `root`. */
const refusalLines = (/** @type {ParentNode} */ root) =>
  /** @type {HTMLElement[]} */ ([...root.querySelectorAll('.op-tool-refusal')]);

/** @type {HTMLElement} */
let container;

beforeEach(() => {
  container = document.createElement('div');
  document.body.appendChild(container);
});

afterEach(() => {
  container.remove();
});

/** A turn's action rows, placed in `container`. */
const turnActions = () => createTurnActions(container);

/**
 * A write that folds into a run with the calls of its kind: a Python block,
 * which the hooks refuse the same way when it would write.
 */
const script = (/** @type {string} */ id) => ({
  type: 'tool_use',
  tool_use_id: id,
  tool_name: 'execute',
  tool_name_raw: 'mcp__python__execute',
  input: { code: `caput('SR:C01:QF:SP', 1.5)  # ${id}` },
});

describe('the action rows', () => {
  test('a refused write shows its phrase as a visible line, under its folded row', () => {
    const actions = turnActions();
    const chip = actions.addTool(write('w1'));
    chip.finish(refused('w1'));

    const [line] = refusalLines(container);
    expect(line.textContent).toBe(SWITCH_OFF);
    expect(line.dataset.reason).toBe('switch_off');
    expect(line.getAttribute('role')).toBe('note');
    expect(visible(line)).toBe(true);

    const row = qs(container, '.op-tool');
    expect(row.classList.contains('open')).toBe(false);
    expect(line.parentElement).toBe(row);
    expect(qs(row, '.op-row-body').contains(line)).toBe(false);
    expect(refusalLines(container)).toHaveLength(1);
  });

  test('a refused write still reads as a failed call', () => {
    const actions = turnActions();
    actions.addTool(write('w1')).finish(refused('w1'));
    actions.addTool(script('s1')).finish(refused('s1'));
    actions.addTool(script('s2')).finish({ preview: 'ok' });

    expect(qs(container, '.op-tool').dataset.state).toBe('error');
    // A run's folded row counts it among its failures.
    expect(qs(container, '.op-run-failed').textContent).toContain('1 failed');
  });

  test('an ordinary tool error keeps the collapsed form and draws no line', () => {
    const actions = turnActions();
    actions.addTool(write('w1')).finish(failed('w1'));

    expect(refusalLines(container)).toHaveLength(0);
    const chip = qs(container, '.op-tool');
    expect(chip.dataset.state).toBe('error');
    const pre = qs(container, '.op-tool-pre.op-tool-error');
    expect(pre.textContent).toContain('NameError');
    expect(visible(pre)).toBe(false);
  });

  test('a write refused inside a subagent shows its phrase outside the subagent trace', () => {
    const actions = turnActions();
    const agent = actions.addAgent(
      {
        type: 'tool_use',
        tool_use_id: 'a1',
        tool_name: 'Agent',
        tool_name_raw: 'Agent',
        input: { subagent_type: 'operator', description: 'set a quad' },
      },
      (el, text) => {
        el.textContent = text;
      }
    );
    agent.addTool(write('w1')).finish(refused('w1', 'tier_never', TIER_NEVER));

    const [line] = refusalLines(container);
    expect(line.textContent).toBe(TIER_NEVER);
    expect(visible(line)).toBe(true);
    expect(line.parentElement).toBe(qs(container, '.op-agent'));
    expect(qs(container, '.op-agent-trace').contains(line)).toBe(false);
  });

  test('refused writes folded into a run show under the run, one line each in call order', () => {
    const actions = turnActions();
    const first = actions.addTool(script('s1'));
    const second = actions.addTool(script('s2'));
    const third = actions.addTool(script('s3'));
    // Results land out of order; the lines keep the calls' order.
    third.finish(refused('s3', 'tier_never', TIER_NEVER));
    second.finish(failed('s2'));
    first.finish(refused('s1'));

    const run = qs(container, '.op-run');
    expect(qs(run, '.op-row-body').hidden).toBe(true);
    expect(qs(run, '.op-run-failed').textContent).toBe('3 failed');
    const lines = refusalLines(container);
    expect(lines.map((line) => line.textContent)).toEqual([SWITCH_OFF, TIER_NEVER]);
    for (const line of lines) {
      expect(line.parentElement).toBe(run);
      expect(visible(line)).toBe(true);
    }
  });

  test('a refused call taken out of its run for a card keeps its line under its own row', () => {
    const actions = turnActions();
    const first = actions.addTool(script('s1'));
    actions.addTool(script('s2'));
    const third = actions.addTool(script('s3'));
    first.finish(refused('s1'));
    third.file(document.createElement('div'));

    const [line] = refusalLines(container);
    expect(line.textContent).toBe(SWITCH_OFF);
    expect(visible(line)).toBe(true);
    // s1 and s2 keep their run; the line stays under it.
    expect(line.parentElement).toBe(qs(container, '.op-run'));
  });
});

describe('refusalLine', () => {
  test('is null with no refusal, an empty phrase, or a result that did not fail', () => {
    expect(refusalLine({ is_error: true, refusal: null })).toBeNull();
    expect(
      refusalLine({ is_error: true, refusal: { reason: 'switch_off', phrase: '  ' } })
    ).toBeNull();
    expect(
      refusalLine({ is_error: false, refusal: { reason: 'switch_off', phrase: SWITCH_OFF } })
    ).toBeNull();
  });

  test('draws the phrase as text, never as markup', () => {
    const line = refusalLine({
      is_error: true,
      refusal: { reason: 'switch_off', phrase: '<b>off</b>' },
    });
    expect(line?.textContent).toBe('<b>off</b>');
    expect(line?.querySelector('b')).toBeNull();
  });
});

describe('the chat log', () => {
  test('a pre-card refusal (no approval asked) shows its phrase in the conversation, live and after the turn moves on', () => {
    const r = createChatRenderer(container);
    r.handleEvent(write('w1'));
    r.handleEvent(refused('w1'));
    r.handleEvent({ type: 'text', content: 'The write was refused.' });

    expect(container.querySelector('.op-approval')).toBeNull();
    const [line] = refusalLines(container);
    expect(line.textContent).toBe(SWITCH_OFF);
    expect(visible(line)).toBe(true);
  });

  test('a replayed refused write shows the same line', () => {
    const r = createChatRenderer(container);
    r.replay([
      { role: 'user', content: 'set the quad' },
      {
        role: 'assistant',
        content: '',
        events: [
          write('w1'),
          { ...refused('w1', 'tier_never', TIER_NEVER), decision: 'denied' },
          { type: 'text', content: 'Refused.' },
          { type: 'result' },
        ],
      },
    ]);

    const [line] = refusalLines(container);
    expect(line.textContent).toBe(TIER_NEVER);
    expect(visible(line)).toBe(true);
  });

  test('a refusal for a call with no row still reaches the conversation', () => {
    const r = createChatRenderer(container);
    r.handleEvent({ type: 'text', content: 'Trying.' });
    r.handleEvent(refused('unknown'));

    const [line] = refusalLines(container);
    expect(line.textContent).toBe(SWITCH_OFF);
    expect(visible(line)).toBe(true);
  });

  test('an ordinary tool error in the log draws no refusal line', () => {
    const r = createChatRenderer(container);
    r.handleEvent(write('w1'));
    r.handleEvent(failed('w1'));
    r.handleEvent({ type: 'text', content: 'It failed.' });

    expect(refusalLines(container)).toHaveLength(0);
    expect(qs(container, '.op-tool').dataset.state).toBe('error');
  });
});
