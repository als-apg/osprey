/**
 * Unit tests for the chat's task bar (chat-tasks.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-tasks.test.mjs
 *
 * The bar reads the chat's frames from the bus, so the tests publish the same
 * frames the renderer does, live or replayed, and read the bar the operator
 * sees above the composer.
 */

import { test, expect, describe, beforeEach, afterEach } from 'vitest';

import { initChatTasks } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-tasks.js';
import { emitChat } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js';

/** @type {() => void} */
let dispose;
/** @type {HTMLElement} */
let container;

beforeEach(() => {
  container = document.createElement('div');
  const inputArea = document.createElement('div');
  container.append(document.createElement('div'), inputArea);
  document.body.append(container);
  dispose = initChatTasks(/** @type {any} */ ({ inputArea }));
});

afterEach(() => {
  dispose();
  container.remove();
});

/** @param {Record<string, any>} event */
const frame = (event) => emitChat('frame', { event, replay: false });

/**
 * @param {string} callId
 * @param {string} name
 * @param {Record<string, any>} input
 * @param {Record<string, any>} [extra]
 */
const use = (callId, name, input, extra = {}) =>
  frame({ type: 'tool_use', tool_use_id: callId, tool_name_raw: name, input, ...extra });

/**
 * @param {string} callId
 * @param {string} preview
 * @param {boolean} [isError]
 */
const result = (callId, preview, isError = false) =>
  frame({ type: 'tool_result', tool_use_id: callId, preview, is_error: isError });

/**
 * Create a task and land its result, reporting *id*.
 * @param {string} callId
 * @param {string} id
 * @param {string} subject
 * @param {Record<string, any>} [extra]
 */
function created(callId, id, subject, extra = {}) {
  use(callId, 'TaskCreate', { subject, description: 'd', ...extra });
  result(callId, `Task #${id} created successfully: ${subject}`);
}

const bar = () => /** @type {HTMLElement} */ (container.querySelector('.op-taskbar'));
const head = () => /** @type {HTMLButtonElement} */ (bar().querySelector('.op-taskbar-head'));
const text = (/** @type {string} */ sel) => bar().querySelector(sel)?.textContent;
const rows = () =>
  [...bar().querySelectorAll('.op-taskbar-item')].map((li) => [
    /** @type {HTMLElement} */ (li).dataset.status,
    li.textContent,
  ]);

describe('placement', () => {
  test('sits just above the composer and stays hidden with no tasks', () => {
    expect(bar().nextElementSibling).toBe(container.lastElementChild);
    expect(bar().hidden).toBe(true);
  });
});

describe('TaskCreate and TaskUpdate', () => {
  test('a task appears once its create call succeeds, keyed by the id the result names', () => {
    use('c1', 'TaskCreate', { subject: 'Find the correctors' });
    expect(bar().hidden).toBe(true);

    result('c1', 'Task #7 created successfully: Find the correctors');
    created('c2', '8', 'Read them');
    use('u1', 'TaskUpdate', { taskId: '7', status: 'completed' });
    result('u1', 'Updated task #7 status');

    expect(bar().hidden).toBe(false);
    expect(text('.op-taskbar-count')).toBe('1/2');
    expect(text('.op-taskbar-summary')).toBe('Next: Read them');
    expect(rows()).toEqual([
      ['completed', 'Find the correctors'],
      ['pending', 'Read them'],
    ]);
  });

  test.each([
    ['taskId', { taskId: '2' }],
    ['task_id', { task_id: '2' }],
    ['only the id its result reports', {}],
  ])('an update that names its task by %s finds it', (_case, naming) => {
    created('c1', '1', 'First');
    created('c2', '2', 'Second');
    use('u1', 'TaskUpdate', { ...naming, status: 'completed' });
    result('u1', 'Updated task #2 status');

    expect(rows()).toEqual([
      ['pending', 'First'],
      ['completed', 'Second'],
    ]);
  });

  test('the task in hand is named by what the agent is doing', () => {
    created('c1', '1', 'Plot setpoint vs readback', {
      activeForm: 'Plotting setpoint vs readback',
    });
    use('u1', 'TaskUpdate', { taskId: '1', status: 'in_progress' });
    result('u1', 'Updated task #1 status');

    expect(bar().dataset.state).toBe('running');
    expect(text('.op-taskbar-summary')).toBe('Plotting setpoint vs readback');
  });

  test('a failed call changes nothing', () => {
    use('c1', 'TaskCreate', { subject: 'Never made' });
    result('c1', 'Error: no task list', true);
    created('c2', '1', 'Made');
    use('u1', 'TaskUpdate', { taskId: '1', status: 'completed' });
    result('u1', 'Error: task not found', true);

    expect(rows()).toEqual([['pending', 'Made']]);
  });

  test('an update that names no known task is ignored', () => {
    created('c1', '1', 'Only task');
    use('u1', 'TaskUpdate', { taskId: '9', status: 'completed' });
    result('u1', 'Updated task #9 status');

    expect(text('.op-taskbar-count')).toBe('0/1');
  });

  test('a deleted task leaves the list', () => {
    created('c1', '1', 'Keep');
    created('c2', '2', 'Drop');
    use('u1', 'TaskUpdate', { taskId: '2', status: 'deleted' });
    result('u1', 'Updated task #2 status');

    expect(rows()).toEqual([['pending', 'Keep']]);
  });

  test("a subagent's task calls are its own and stay out of the bar", () => {
    use('c1', 'TaskCreate', { subject: 'Inner step' }, { parent_tool_use_id: 'agent-1' });
    frame({
      type: 'tool_result',
      tool_use_id: 'c1',
      preview: 'Task #1 created successfully: Inner step',
      parent_tool_use_id: 'agent-1',
    });

    expect(bar().hidden).toBe(true);
  });
});

describe('TodoWrite', () => {
  test('each call replaces the whole list', () => {
    use('t1', 'TodoWrite', {
      todos: [
        { content: 'Read the file', activeForm: 'Reading the file', status: 'in_progress' },
        { content: 'Summarise it', status: 'pending' },
      ],
    });
    expect(text('.op-taskbar-summary')).toBe('Reading the file');

    use('t2', 'TodoWrite', { todos: [{ content: 'Summarise it', status: 'completed' }] });
    expect(rows()).toEqual([['completed', 'Summarise it']]);
    expect(bar().dataset.state).toBe('done');
    expect(text('.op-taskbar-summary')).toBe('All done');
  });
});

describe('lifetime', () => {
  test('a finished list stays up until the next prompt, then goes', () => {
    created('c1', '1', 'Only task');
    use('u1', 'TaskUpdate', { taskId: '1', status: 'completed' });
    result('u1', 'Updated task #1 status');
    emitChat('turn-end', {});
    expect(bar().hidden).toBe(false);

    emitChat('user-message', { text: 'next question', replay: false });
    expect(bar().hidden).toBe(true);
  });

  test('an unfinished list carries over to the next prompt', () => {
    created('c1', '1', 'Still open');
    emitChat('user-message', { text: 'keep going', replay: false });

    expect(rows()).toEqual([['pending', 'Still open']]);
  });

  test('clearing the log clears the list', () => {
    created('c1', '1', 'Old conversation');
    emitChat('reset', {});

    expect(bar().hidden).toBe(true);
  });
});

describe('folding', () => {
  test('folded by default; a click opens the checklist and it stays open across updates', () => {
    created('c1', '1', 'First');
    const list = /** @type {HTMLElement} */ (bar().querySelector('.op-taskbar-list'));
    expect(list.hidden).toBe(true);
    expect(head().getAttribute('aria-expanded')).toBe('false');

    head().click();
    created('c2', '2', 'Second');

    expect(list.hidden).toBe(false);
    expect(head().getAttribute('aria-expanded')).toBe('true');
    expect(head().getAttribute('aria-controls')).toBe(list.id);
    expect(rows()).toHaveLength(2);
  });
});
