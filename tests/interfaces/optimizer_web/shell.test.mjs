/**
 * Tests for the panel's frame (shell.js) on its real index.html markup: the
 * status chip and its detail for each screen, the toolbar's identity, the
 * primary slot (empty during a run), and the history toggle.
 */

import { readFileSync } from 'node:fs';
import { cwd } from 'node:process';

import { describe, test, expect, beforeEach, vi } from 'vitest';

import { envDetails, runningJob, storedConfig } from './_fixtures.mjs';
import { approvalEnvKey } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-model.js';
import {
  chipDetails,
  chipFor,
  createShell,
  inspectorFrame,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/shell.js';
import {
  deriveScreen,
  initialState,
  reduce,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';
import { pauseWords } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-page-model.js';

/** @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').StoreEvent} StoreEvent */

const INDEX = `${cwd()}/src/osprey/interfaces/optimizer_web/panels/optimizer/index.html`;

/** @param {StoreEvent[]} events */
const run = (events) => events.reduce(reduce, initialState());

/** @type {StoreEvent[]} */
const DRAFT = [
  { type: 'session', owner: 'op1', ownerKnown: true },
  { type: 'poll/job', job: { job_id: 'j1', status: 'created' }, at: 0 },
  { type: 'poll/config', jobId: 'j1', revision: 'rev-abcdef', config: storedConfig() },
  { type: 'approvalEnv/reset', key: approvalEnvKey(storedConfig()) },
  {
    type: 'approvalEnv/loaded',
    env: {
      key: approvalEnvKey(storedConfig()),
      details: envDetails(),
      models: [],
      readbacks: {},
      error: null,
    },
  },
];

/** @param {Record<string, unknown>} overrides @returns {StoreEvent[]} */
const RUN = (overrides) => [
  { type: 'poll/job', job: runningJob({ job_id: 'j1', ...overrides }), at: 0 },
  { type: 'poll/config', jobId: 'j1', revision: 'rev-1', config: storedConfig() },
];

describe('chipFor', () => {
  /**
   * Each screen: its events, the chip's tone, and the count its label must
   * carry ('' for none; null for no chip).
   *
   * @type {[string, StoreEvent[], string|null, string|null][]}
   */
  const CASES = [
    ['loading', [], 'connecting', ''],
    ['empty', [{ type: 'poll/job', job: null, at: 0 }], null, null],
    ['review', DRAFT, 'info', ''],
    [
      'review-changed',
      [...DRAFT, { type: 'poll/config', jobId: 'j1', revision: 'r2', config: storedConfig() }],
      'stale',
      '',
    ],
    [
      'review-no-owner',
      [...DRAFT, { type: 'session', owner: null, ownerKnown: false }],
      'warn',
      '',
    ],
    [
      'setup',
      [
        { type: 'poll/job', job: null, at: 0 },
        { type: 'setup/open', jobId: null, environment: '', target: '' },
      ],
      'pending',
      '',
    ],
    [
      'setup-issues',
      [
        { type: 'poll/job', job: null, at: 0 },
        { type: 'setup/open', jobId: null, environment: '', target: '' },
        { type: 'setup/refused', issues: ['a', 'b'] },
      ],
      'warn',
      '2',
    ],
    ['waiting', RUN({ write_authorization: { state: 'awaiting' } }), 'pending', ''],
    ['running', RUN({}), 'running', '3 of 10'],
    ['cancelling', RUN({ status: 'cancelling' }), 'warn', ''],
    ['paused', RUN({ status: 'paused' }), 'pending', '3 of 10'],
    ['completed', RUN({ status: 'completed' }), 'done', ''],
    ['failed', RUN({ status: 'error' }), 'failed', '3'],
    ['failed before any evaluation', RUN({ status: 'error', data: [] }), 'failed', ''],
    ['cancelled', RUN({ status: 'cancelled' }), 'warn', '3'],
    [
      'history',
      [
        { type: 'poll/job', job: null, at: 0 },
        { type: 'select', selection: { kind: 'history' } },
      ],
      null,
      null,
    ],
    ['unreachable', [{ type: 'poll/failed', error: 'down' }], 'err', ''],
  ];

  /** @param {StoreEvent[]} events */
  const chipOf = (events) => {
    const state = run(events);
    return chipFor(state, deriveScreen(state));
  };

  test.each(CASES)('%s', (_name, events, tone, count) => {
    const chip = chipOf(events);
    if (count === null) return expect(chip).toBeNull();
    expect(chip?.tone).toBe(tone);
    const digits = chip?.label.match(/\d+( of \d+)?/)?.[0] ?? '';
    expect(digits).toBe(count);
  });

  test('each screen has a label of its own', () => {
    const labels = CASES.map(([, events]) => chipOf(events)?.label).filter(Boolean);
    expect(new Set(labels).size).toBe(labels.length);
  });

  test('a stored run shows the status of its job only once the service answered', () => {
    const runId = 'injector/2026-11-16_21-15-04';
    const unknown = run([
      { type: 'poll/job', job: null, at: 0 },
      { type: 'select', selection: { kind: 'run', runId } },
    ]);
    expect(chipFor(unknown, deriveScreen(unknown))).toBeNull();
    const known = reduce(unknown, {
      type: 'past/loaded',
      runId,
      details: null,
      job: { job_id: 'x', status: 'cancelled' },
      error: null,
    });
    // The same chip as a cancelled run of now that recorded no evaluation.
    const present = chipOf(RUN({ status: 'cancelled', data: [] }));
    expect(chipFor(known, deriveScreen(known))).toEqual(present);
  });
});

describe('chipDetails', () => {
  test('a draft: its job, its revision and whose writes the run carries', () => {
    const state = run(DRAFT);
    const rows = Object.fromEntries(
      chipDetails(state, deriveScreen(state)).rows.map((r) => [r.label, r.value])
    );
    expect(rows).toMatchObject({
      Job: 'j1',
      Revision: 'rev-abcdef',
      'Starts as': 'op1',
      Environment: 'injector',
      'Control target': 'va',
    });
  });

  test('a session that names no account says so', () => {
    const state = run([...DRAFT, { type: 'session', owner: null, ownerKnown: false }]);
    const rows = chipDetails(state, deriveScreen(state)).rows;
    expect(rows.find((r) => r.label === 'Starts as')?.value).toBe('no account');
  });

  test('a run: the write approval and the machine connection, when read', () => {
    const state = run([
      ...RUN({ write_authorization: { state: 'awaiting', owner: 'op1' } }),
      {
        type: 'connection',
        connection: {
          environment: 'injector',
          target: 'va',
          connected: false,
          message: 'IOC down',
        },
      },
    ]);
    const screen = deriveScreen(state);
    const details = chipDetails(state, screen);
    expect(details.title).toBe(chipFor(state, screen)?.label);
    const rows = Object.fromEntries(details.rows.map((r) => [r.label, r.value]));
    expect(rows['Write approval']).toContain('op1 approves');
    expect(rows.Machine).toBe('not connected: IOC down');
    expect(rows.Revision).toBeUndefined();
  });

  test('a paused run says what the pause does at the point the run is at', () => {
    const state = run(RUN({ status: 'paused' }));
    const screen = deriveScreen(state);
    expect(chipDetails(state, screen).text).toContain(pauseWords(screen).held);
  });

  test('an unreachable service names its error', () => {
    const state = run([{ type: 'poll/failed', error: 'optimizer service unreachable' }]);
    expect(chipDetails(state, deriveScreen(state))).toMatchObject({
      title: 'Service unreachable',
      text: 'optimizer service unreachable',
    });
  });
});

test('inspectorFrame: a head with its title and a close button, a body and a foot', () => {
  const onClose = vi.fn();
  const [head, body, foot] = inspectorFrame(
    {
      title: 'Review and start',
      desc: 'revision 1',
      body: [document.createElement('p'), null],
      foot: document.createElement('button'),
    },
    onClose
  );
  expect(head.querySelector('.osprey-card-title')?.textContent).toBe('Review and start');
  /** @type {HTMLElement} */ (head.querySelector('[aria-label="Close the inspector"]')).click();
  expect(onClose).toHaveBeenCalledTimes(1);
  expect(body.children).toHaveLength(1);
  expect(foot.tagName).toBe('FOOTER');
  expect(inspectorFrame({ title: 't', body: [] }, onClose)).toHaveLength(2);
});

describe('createShell', () => {
  beforeEach(() => {
    const html = readFileSync(INDEX, 'utf8');
    document.body.innerHTML = html
      .slice(html.indexOf('<body>') + 6, html.indexOf('</body>'))
      .replace(/<script[^>]*><\/script>/g, '');
  });

  function mount() {
    const actions = /** @type {any} */ ({ select: vi.fn(), newRun: vi.fn(), setFilter: vi.fn() });
    const shell = createShell({ actions });
    /** @param {StoreEvent[]} events */
    const show = (events) => {
      const state = run(events);
      shell.update(state, deriveScreen(state));
      return state;
    };
    const $ = (/** @type {string} */ sel) =>
      /** @type {HTMLElement} */ (document.querySelector(sel));
    return { actions, shell, show, $ };
  }

  test("the toolbar names the screen's control target and environment", () => {
    const { show, $ } = mount();
    show([
      { type: 'environments', list: [{ name: 'injector', display_name: 'Injector' }], error: null },
      ...DRAFT,
    ]);
    expect($('#target-slot').textContent).toBe('Simulator');
    expect($('#env-text').textContent).toBe('Injector');
    show([
      ...DRAFT,
      {
        type: 'poll/config',
        jobId: 'j1',
        revision: 'r9',
        config: storedConfig({ control_target: 'live' }),
      },
    ]);
    expect($('#target-slot .opt-target')?.classList.contains('osprey-chip--err')).toBe(true);
    show([{ type: 'poll/job', job: null, at: 0 }]);
    expect($('#target-slot').textContent).toBe('');
    expect($('#env-text').textContent).toBe('');
  });

  test('the chip shows the screen status; no status hides it', () => {
    const { show, $ } = mount();
    show(DRAFT);
    expect($('#status-chip').hidden).toBe(false);
    expect($('#status-label').textContent).toBe('Ready to start');
    expect($('#status-chip').className).toContain('osprey-chip--info');
    show([{ type: 'poll/job', job: null, at: 0 }]);
    expect($('#status-chip').hidden).toBe(true);
  });

  test('the primary slot holds one button; its click runs the spec; it is rebuilt only for a new key', () => {
    const { shell, show, $ } = mount();
    show(DRAFT);
    const onClick = vi.fn();
    shell.setPrimary({ label: 'Start run', id: 'go', key: 'k1', onClick });
    const button = $('#go');
    button.click();
    expect(onClick).toHaveBeenCalledTimes(1);
    shell.setPrimary({ label: 'Start run', id: 'go', key: 'k1', onClick: vi.fn() });
    expect($('#go')).toBe(button);
    shell.setPrimary({
      label: 'Start run',
      id: 'go',
      key: 'k2',
      disabled: true,
      why: 'Review first',
      onClick,
    });
    expect($('#go')).not.toBe(button);
    expect(/** @type {HTMLButtonElement} */ ($('#go')).disabled).toBe(true);
    shell.setPrimary(null);
    expect($('#primary-slot').children).toHaveLength(0);
  });

  test('during a run the primary slot stays empty and the chip takes its place', () => {
    const { shell, show, $ } = mount();
    show(RUN({}));
    shell.setPrimary({ label: 'Start run', id: 'go', onClick: vi.fn() });
    expect($('#go')).toBeNull();
    const end = Array.from($('#toolbar-end').children);
    expect(end.indexOf($('#history-btn'))).toBeLessThan(end.indexOf($('#status-chip')));
    show(RUN({ status: 'completed' }));
    expect($('#go')).not.toBeNull();
    const after = Array.from($('#toolbar-end').children);
    expect(after.indexOf($('#status-chip'))).toBeLessThan(after.indexOf($('#history-btn')));
  });

  test('the history button toggles between the past runs and the present', () => {
    const { actions, show, $ } = mount();
    show([{ type: 'poll/job', job: null, at: 0 }]);
    $('#history-btn').click();
    expect(actions.select).toHaveBeenLastCalledWith({ kind: 'history' });
    show([
      { type: 'poll/job', job: null, at: 0 },
      { type: 'select', selection: { kind: 'history' } },
    ]);
    expect($('#history-btn').getAttribute('aria-pressed')).toBe('true');
    $('#history-btn').click();
    expect(actions.select).toHaveBeenLastCalledWith({ kind: 'present' });
  });

  test('the new-run button and the filter reach their actions', () => {
    const { actions, $ } = mount();
    $('#new-run').click();
    expect(actions.newRun).toHaveBeenCalledTimes(1);
    const filter = /** @type {HTMLInputElement} */ ($('#run-filter'));
    filter.value = 'charge';
    filter.dispatchEvent(new Event('input'));
    expect(actions.setFilter).toHaveBeenCalledWith('charge');
  });

  test('a failed read shows its reason in the banner', () => {
    const { show, $ } = mount();
    show([{ type: 'poll/failed', error: 'optimizer service unreachable' }]);
    expect($('#panel-banner').hidden).toBe(false);
    expect($('#panel-banner').textContent).toBe('optimizer service unreachable');
    show([{ type: 'poll/job', job: null, at: 0 }]);
    expect($('#panel-banner').hidden).toBe(true);
  });

  test('the edge tab names what the inspector holds and flags a screen that asks for attention', () => {
    const { show, $ } = mount();
    show([...DRAFT, { type: 'session', owner: null, ownerKnown: false }]);
    expect($('#edge-label').textContent).toBe('Review');
    expect($('#edge-dot').hidden).toBe(false);
    show(RUN({}));
    expect($('#edge-label').textContent).toBe('Run');
    expect($('#edge-dot').hidden).toBe(true);
  });
});
