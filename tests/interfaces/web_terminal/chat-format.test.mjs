/**
 * Unit tests for chat-format.js -- how a tool's input and result read in the chat chat.
 *
 * A tool's input is drawn as key and value rows, for the call's row and for its approval
 * card. Its result is written as text, for the row's RESULT. Both follow one set of rules
 * for a field's shape, so a field reads the same in either place. The tool-row and card
 * cases drive the real renderers, the way a live stream and a reloaded page do.
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-format.test.mjs
 */

import { describe, test, expect, vi } from 'vitest';

import { qs } from '../_support/dom.mjs';

import {
  formatToolResult,
  inputDetail,
  inputView,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-format.js';
import {
  runPhrase,
  summarizeInput,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-tool-vocab.js';
import { activityLabel } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-activity.js';
import { createApprovalCard } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-approval.js';
import {
  createAgentChip,
  createToolChip,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-tool-cards.js';

/** Text well past the server's 300-character string cut, as the artifact-save result carries. */
const LONG =
  'Analysis of the orbit response for the sector, with the fit residuals and the per-BPM notes. '.repeat(
    4
  );

/** The artifact-save result the screenshot showed, whole, as the server's preview carries it once peeled. */
const ARTIFACT_SAVE = {
  status: 'success',
  artifact_id: 'e8ab25455333',
  title: `Orbit fit ${LONG}`,
  artifact_type: 'json',
  size_bytes: 763,
  pinned: false,
  category: 'code_output',
  summary: { description: LONG, status: 'success' },
};

/**
 * The same result as the server's preview cuts it: each long string is kept to 300 characters
 * and says how many it dropped.
 */
function shortenedArtifactSave() {
  const cut = (/** @type {string} */ text) => `${text.slice(0, 300)}… +${text.length - 300} chars`;
  return {
    ...ARTIFACT_SAVE,
    title: cut(ARTIFACT_SAVE.title),
    summary: { ...ARTIFACT_SAVE.summary, description: cut(LONG) },
  };
}

/**
 * A tool call event as the stream carries it.
 * @param {Record<string, unknown>} input
 */
function call(input) {
  return {
    type: 'tool_use',
    tool_name: 'Execute',
    tool_name_raw: 'mcp__python__execute',
    tool_use_id: 't1',
    input,
    parent_tool_use_id: null,
  };
}

/**
 * A permission request for a tool call, as the stream carries it.
 * @param {Record<string, unknown>} input
 */
function approvalRequest(input) {
  return {
    type: 'permission_request',
    request_id: 'req-1',
    kind: 'tool',
    tool_name: 'Execute',
    tool_name_raw: 'mcp__python__execute',
    tool_use_id: 't1',
    input,
    reason: null,
    can_allow_session: false,
  };
}

/**
 * The value shown beside a key in a key and value list.
 * @param {HTMLElement} root
 * @param {string} key
 * @returns {string | undefined}
 */
function valueOf(root, key) {
  const term = [...root.querySelectorAll('dt')].find((dt) => dt.textContent === key);
  return term?.nextElementSibling?.textContent ?? undefined;
}

describe('formatToolResult', () => {
  // Every preview here is shaped as the server sends it: tool_preview.py has
  // already peeled the MCP envelope and cut the result to compact JSON.
  test.each([
    [
      'a flat result as aligned key/value lines',
      '{"status":"success","panel_id":"artifacts"}',
      'status    success\npanel_id  artifacts',
    ],
    [
      'a nested record as indented key and value lines under its key',
      '{"status":"ok","summary":{"description":"d","status":"ok"}}',
      'status  ok\nsummary\n  description  d\n  status       ok',
    ],
    [
      'a list of plain values on one line',
      '{"channels": ["SR:BPM1", "SR:BPM2"], "count": 2}',
      'channels  SR:BPM1, SR:BPM2\ncount     2',
    ],
    [
      'a multi-line value as a block indented under its key',
      '{"code": "print(1)\\nprint(2)"}',
      'code\n  print(1)\n  print(2)',
    ],
    [
      'a list that mixes kinds as indented JSON',
      '{"items": [1, {"a": 2}]}',
      'items\n  [\n    1,\n    {\n      "a": 2\n    }\n  ]',
    ],
    [
      'a record nested two deep as indented JSON',
      '{"outer": {"inner": {"x": 1}}}',
      'outer\n  {\n    "inner": {\n      "x": 1\n    }\n  }',
    ],
    [
      'a result key beside another key as key and value lines, not as an envelope',
      '{"result":"ok","warnings":[]}',
      'result    ok\nwarnings  (empty)',
    ],
    ['JSON that does not parse as it came', '{"status": "succ', '{"status": "succ'],
    ['a bare number as it came', '42.0', '42.0'],
    ['non-JSON text as it came', 'Error: channel not found', 'Error: channel not found'],
  ])('shows %s', (_shape, preview, shown) => {
    expect(formatToolResult(preview)).toBe(shown);
  });

  test.each([
    ['whole', ARTIFACT_SAVE, LONG],
    ['as the server shortens it', shortenedArtifactSave(), '… +'],
  ])(
    'shows the artifact-save result (%s) as key and value lines, with its nested summary',
    (_how, result, description) => {
      const lines = formatToolResult(JSON.stringify(result)).split('\n');
      expect(lines.find((line) => /^artifact_id\s+e8ab25455333$/.test(line))).toBeDefined();
      expect(lines).toContain('summary');
      expect(lines.find((line) => /^ {2}status\s+success$/.test(line))).toBeDefined();
      expect(lines.find((line) => line.startsWith('  description'))).toContain(description);
    }
  );
});

describe('inputView', () => {
  test.each([
    [
      'a multi-line code value, with its real line breaks and no escaped quotes',
      { code: 'print("first")\nprint("second")' },
      'print("first")\nprint("second")',
    ],
    ['a single-line command', { command: 'ls -la' }, 'ls -la'],
  ])('shows %s as a block', (_shape, input, text) => {
    const view = inputView(input);
    expect(qs(view, 'pre.op-kv-block').textContent).toBe(text);
    expect(view.textContent).not.toContain('\\"');
  });

  test('shows strings unquoted, and numbers, flags and null as their plain values', () => {
    const view = inputView({ file_path: '/tmp/a.txt', limit: 3, dry_run: false, note: null });
    const values = [...view.querySelectorAll('dd')].map((dd) => dd.textContent);
    expect(values).toEqual(['/tmp/a.txt', '3', 'false', 'null']);
  });

  test('shows a nested record as indented key and value rows', () => {
    const view = inputView({ summary: { description: 'Sweep the sector', status: 'ok' } });
    const nested = qs(view, 'dd dl.op-kv-nested');
    expect([...nested.querySelectorAll('dt')].map((dt) => dt.textContent)).toEqual([
      'description',
      'status',
    ]);
    expect([...nested.querySelectorAll('dd')].map((dd) => dd.textContent)).toEqual([
      'Sweep the sector',
      'ok',
    ]);
  });

  test('shows a list of plain values on one line', () => {
    expect(qs(inputView({ channels: ['SR:BPM1', 'SR:BPM2'] }), 'dd .op-kv-text').textContent).toBe(
      'SR:BPM1, SR:BPM2'
    );
  });

  test.each([
    ['a list that mixes kinds', [1, { a: 2 }]],
    ['a record nested two deep', { inner: { x: 1 } }],
  ])('falls back to indented JSON for %s', (_shape, value) => {
    expect(qs(inputView({ value }), 'pre.op-kv-json').textContent).toBe(
      JSON.stringify(value, null, 2)
    );
  });

  /** The machine kinds the chip holds from GET, keyed by target (C-KIND). */
  const KINDS = { va: 'simulator', live: 'real', standin: 'rehearsal' };

  test.each([
    ['va', 'Simulator'],
    ['live', 'Live'],
    ['standin', 'Rehearsal'],
    ['ioc2', 'ioc2'],
    ['constructor', 'constructor'],
  ])('reads approved_target %s as a Target row labelled %s', (target, label) => {
    expect(valueOf(inputView({ approved_target: target }, KINDS), 'Target')).toBe(label);
  });

  test('a non-live standin GET calls the real machine never reads "Rehearsal"', () => {
    // The kind comes from GET, never from the target's name: a `standin` no
    // live server vouched for is the real machine.
    expect(valueOf(inputView({ approved_target: 'standin' }, { standin: 'real' }), 'Target')).toBe(
      'Live'
    );
  });

  test.each([['standin'], ['va'], ['live']])(
    'with no kind handed in, approved_target %s reads as the id given, never a guessed kind',
    (target) => {
      expect(valueOf(inputView({ approved_target: target }), 'Target')).toBe(target);
    }
  );

  test('a kind outside the vocabulary is not a word to look up', () => {
    expect(valueOf(inputView({ approved_target: 'x' }, { x: 'constructor' }), 'Target')).toBe('x');
  });

  test.each([[{}], [undefined]])('shows an empty input (%o) as no input', (input) => {
    expect(inputView(input).textContent).toBe('(no input)');
  });
});

describe('inputDetail', () => {
  test('leaves out the field the summary line states in full', () => {
    const input = { code: 'x = 1', description: 'Set x' };
    expect(inputDetail(input, summarizeInput(input))).toEqual({ code: 'x = 1' });
  });

  test('keeps a field the summary cuts short, so no text the call sent is lost', () => {
    const input = {
      code: 'x = 1',
      description: `Sweep the sector ${'with the residuals '.repeat(20)}`,
    };
    const shown = summarizeInput(input);
    expect(shown.endsWith('…')).toBe(true);
    expect(inputDetail(input, shown)).toEqual(input);
  });

  test('keeps a multi-line field under Details, since the summary only flattens it', () => {
    const input = { code: 'x = 1\ny = 2' };
    const shown = summarizeInput(input);
    expect(shown).toBe('x = 1 y = 2');
    expect(inputDetail(input, shown)).toEqual(input);
  });

  test('is null when the summary states every field', () => {
    expect(inputDetail({ description: 'Set x' }, 'Set x')).toBeNull();
  });
});

describe('the approval card', () => {
  const hooks = () => ({ answer: vi.fn(async () => {}), onSettled: vi.fn() });

  test('states the summary once, and shows the rest under Details with the target row', () => {
    const card = createApprovalCard(
      approvalRequest({
        code: 'print("hi")\nprint(2)',
        description: 'Print two lines',
        approved_target: 'va',
      }),
      hooks()
    );
    expect(qs(card.el, '.op-card-input').textContent).toBe('Print two lines');
    const detail = qs(card.el, '.op-card-detail');
    expect(detail.hidden).toBe(true);
    expect(detail.textContent).not.toContain('Print two lines');
    expect(qs(detail, 'pre.op-kv-block').textContent).toBe('print("hi")\nprint(2)');
    // The card hands inputView no kinds yet, so the target reads by its id
    // rather than by a kind guessed from the name.
    expect(valueOf(detail, 'Target')).toBe('va');

    qs(card.el, '.op-card-toggle').click();
    expect(detail.hidden).toBe(false);
  });

  test('has no Details control when the summary states every field', () => {
    const card = createApprovalCard(approvalRequest({ description: 'Read the orbit' }), hooks());
    expect(card.el.querySelector('.op-card-toggle')).toBeNull();
  });

  test('shows a multi-line code under Details with its line breaks, past a flattened summary', () => {
    const card = createApprovalCard(approvalRequest({ code: 'print(1)\nprint(2)' }), hooks());
    expect(qs(card.el, '.op-card-input').textContent).toBe('print(1) print(2)');
    const detail = qs(card.el, '.op-card-detail');
    expect(qs(detail, 'pre.op-kv-block').textContent).toBe('print(1)\nprint(2)');
  });
});

describe('a tool row', () => {
  test.each([
    ['a shortened result carries a quiet note, not an ellipsis line', true],
    ['a result that was not shortened carries no note', false],
  ])('%s', (_case, truncated) => {
    const chip = createToolChip(call({ code: 'print(1)' }));
    chip.finish({ is_error: false, preview: 'x'.repeat(40), truncated });
    expect(qs(chip.el, '.op-tool-result > .op-tool-section-head .op-tool-note').hidden).toBe(
      !truncated
    );
    expect(qs(chip.el, '.op-tool-result .op-tool-pre').textContent).toBe('x'.repeat(40));
  });

  test('opens to an Input and a Result box, and a failed call marks its Result box', () => {
    const chip = createToolChip(call({ code: 'print(1)' }));
    const titles = [...chip.el.querySelectorAll('.op-tool-section-head .op-tool-label')];
    expect(titles.map((t) => t.textContent)).toEqual(['Input', 'Result']);
    const result = qs(chip.el, '.op-tool-result');
    expect(result.dataset.state).toBeUndefined();

    chip.finish({ is_error: true, preview: 'Error: boom', truncated: false });

    expect(result.dataset.state).toBe('error');
    expect(qs(result, '.op-tool-section-body .op-tool-pre').textContent).toBe('Error: boom');
  });

  test('shows the call input and the artifact-save result as key and value lines', () => {
    const chip = createToolChip(call({ code: 'print("a")\nprint("b")' }));
    chip.finish({
      is_error: false,
      preview: JSON.stringify(shortenedArtifactSave()),
      truncated: false,
    });
    const inputBox = qs(chip.el, '.op-tool-section .op-tool-pre');
    expect(qs(inputBox, 'pre.op-kv-block').textContent).toBe('print("a")\nprint("b")');
    const lines = qs(chip.el, '.op-tool-result .op-tool-pre').textContent.split('\n');
    expect(lines).toContain('summary');
    expect(lines.find((line) => /^ {2}status\s+success$/.test(line))).toBeDefined();
  });

  test('recognises a background agent by its launch mark in the preview', () => {
    const agent = createAgentChip(
      {
        type: 'tool_use',
        tool_name: 'Agent',
        tool_name_raw: 'Agent',
        tool_use_id: 'a1',
        input: { subagent_type: 'scout', description: 'Survey' },
        parent_tool_use_id: null,
      },
      (/** @type {HTMLElement} */ el, /** @type {string} */ text) => {
        el.textContent = text;
      },
      () => {}
    );
    agent.finish({ is_error: false, preview: 'Async agent launched successfully.\nagentId: 7' });
    expect(agent.el.dataset.state).toBe('background');
  });
});

describe('a skill', () => {
  /** @param {Record<string, unknown>} input */
  const skill = (input) => ({
    type: 'tool_use',
    tool_name: 'Skill',
    tool_name_raw: 'Skill',
    tool_use_id: 's1',
    input,
  });
  /** @param {ReturnType<typeof createToolChip>} chip */
  const shown = (chip) => ({
    label: qs(chip.el, '.op-row-label').textContent,
    detail: qs(chip.el, '.op-row-detail').textContent,
    kind: chip.el.dataset.kind,
    glyph: chip.el.querySelector('.op-row-glyph path')?.getAttribute('d'),
  });

  test.each([
    ['the agent chose it', { skill: 'diagnose', args: 'SR03 orbit' }, 'Used skill'],
    ['the operator ran it', { skill: 'diagnose', args: 'SR03 orbit', by: 'operator' }, 'Ran skill'],
  ])('reads as its own row, named by the skill, when %s', (_case, input, label) => {
    expect(shown(createToolChip(skill(input)))).toEqual({
      label,
      detail: 'diagnose',
      kind: 'skill',
      glyph: 'M12 7v14',
    });
  });

  test('the status line names the skill once the call says which one', () => {
    expect(activityLabel({ type: 'tool_start', tool_name: 'Skill', tool_name_raw: 'Skill' })).toBe(
      'Loading a skill…'
    );
    expect(activityLabel(skill({ skill: 'diagnose' }))).toBe('Using skill diagnose…');
  });

  test('a run of skills is counted as skills', () => {
    expect(runPhrase('skill', ['skill', 'skill'])).toBe('Used 2 skills');
  });

  test('a run of skills the operator ran says so; one the agent chose does not', () => {
    expect(runPhrase('skill', ['skill', 'skill'], true)).toBe('Ran 2 skills');
    expect(runPhrase('skill', ['skill'], true)).toBe('Ran 1 skill');
    expect(runPhrase('skill', ['skill', 'skill'], false)).toBe('Used 2 skills');
  });

  test('a long skill name is cut in the status line', () => {
    const label = activityLabel(skill({ skill: 'a'.repeat(200) }));
    expect(label).toBe(`Using skill ${'a'.repeat(60)}…`);
    expect(activityLabel(skill({ skill: 'b'.repeat(60) }))).toBe(`Using skill ${'b'.repeat(60)}…`);
  });
});

describe('a tool named like an Object.prototype member', () => {
  test.each(['constructor', '__proto__'])(
    '%s labels as an unmapped tool, and folds as a run, without throwing',
    (name) => {
      const event = { type: 'tool_use', tool_name: name, tool_name_raw: name };
      expect(activityLabel(event)).toBe(`Using ${name}…`);
      expect(runPhrase('other', [name.toLowerCase(), name.toLowerCase()])).toBe(
        'Made 2 tool calls'
      );
    }
  );
});
