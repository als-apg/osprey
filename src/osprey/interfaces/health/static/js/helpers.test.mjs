// @ts-check
/**
 * Unit tests for the System Health dashboard's pure helpers (helpers.js):
 *   npx vitest run src/osprey/interfaces/health/static/js/helpers.test.mjs
 */

import { test, expect, describe } from 'vitest';

import { STATUS_RANK, esc, el, worstStatus, fmtName, fmtMs, msCls, byCategory } from './helpers.js';
import { fmtName as sharedFmtName } from '/design-system/js/check-name.js';
import {
  STATUS_RANK as sharedStatusRank,
  worstStatus as sharedWorstStatus,
} from '/design-system/js/check-status.js';

describe('esc', () => {
  test('escapes HTML metacharacters so server data cannot inject markup', () => {
    expect(esc('<img src=x onerror=alert(1)>')).toBe('&lt;img src=x onerror=alert(1)&gt;');
    expect(esc('a & b')).toBe('a &amp; b');
  });

  test('null and undefined become the empty string', () => {
    expect(esc(null)).toBe('');
    expect(esc(undefined)).toBe('');
  });

  test('non-string values are stringified before escaping', () => {
    expect(esc(42)).toBe('42');
    expect(esc(0)).toBe('0');
  });

  test('plain text is returned unchanged', () => {
    expect(esc('Beam Current')).toBe('Beam Current');
  });
});

describe('el', () => {
  test('creates an element with the given tag', () => {
    expect(el('span').tagName).toBe('SPAN');
  });

  test('the "class" key sets className; "text" sets textContent', () => {
    const node = el('div', { class: 'dot s-ok', text: 'hi' });
    expect(node.className).toBe('dot s-ok');
    expect(node.textContent).toBe('hi');
  });

  test('the "text" key is never parsed as HTML', () => {
    const node = el('div', { text: '<b>x</b>' });
    expect(node.textContent).toBe('<b>x</b>');
    expect(node.querySelector('b')).toBe(null);
  });

  test('other keys become attributes readable via getAttribute', () => {
    const node = el('a', { title: 'A title', 'data-s': 'error' });
    expect(node.getAttribute('title')).toBe('A title');
    expect(node.getAttribute('data-s')).toBe('error');
  });

  test('children are appended in order and falsy entries skipped', () => {
    const a = el('span', { text: 'first' });
    const b = el('span', { text: 'second' });
    const parent = el('div', null, [null, a, undefined, b]);
    expect(parent.children.length).toBe(2);
    expect(parent.children[0]).toBe(a);
    expect(parent.children[1]).toBe(b);
  });

  test('attrs may be null and children omitted', () => {
    const node = el('li', null);
    expect(node.tagName).toBe('LI');
    expect(node.children.length).toBe(0);
  });
});

describe('worstStatus', () => {
  test('reduces to the single most-severe status present', () => {
    expect(worstStatus([{ status: 'ok' }, { status: 'warning' }, { status: 'ok' }])).toBe(
      'warning'
    );
    expect(worstStatus([{ status: 'warning' }, { status: 'error' }])).toBe('error');
    expect(worstStatus([{ status: 'ok' }, { status: 'skip' }])).toBe('skip');
  });

  test('an empty list is ok', () => {
    expect(worstStatus([])).toBe('ok');
  });

  test('priority order is error > warning > skip > ok', () => {
    expect(STATUS_RANK.error).toBeGreaterThan(STATUS_RANK.warning);
    expect(STATUS_RANK.warning).toBeGreaterThan(STATUS_RANK.skip);
    expect(STATUS_RANK.skip).toBeGreaterThan(STATUS_RANK.ok);
    // error must win regardless of position in the list.
    expect(worstStatus([{ status: 'error' }, { status: 'warning' }, { status: 'skip' }])).toBe(
      'error'
    );
    expect(worstStatus([{ status: 'skip' }, { status: 'warning' }, { status: 'error' }])).toBe(
      'error'
    );
  });

  test('an unknown status is treated as ok-like (priority 0)', () => {
    expect(worstStatus([{ status: 'ok' }, { status: 'mystery' }])).toBe('ok');
    expect(worstStatus([{ status: 'mystery' }, { status: 'warning' }])).toBe('warning');
  });

  // The web terminal's health bar item ranks the same statuses and cannot
  // import this bundle, so the order and its reducer live in the design system
  // and this module re-exports them. Asserting identity — not behaviour — is
  // what makes a future second copy a failure here rather than a quiet
  // divergence on two surfaces.
  test("are the design system's own, not copies of them", () => {
    expect(worstStatus).toBe(sharedWorstStatus);
    expect(STATUS_RANK).toBe(sharedStatusRank);
  });
});

describe('fmtName', () => {
  test('strips the leading "<category>." prefix and title-cases the rest', () => {
    expect(fmtName('control_system.beam_current', 'control_system')).toBe('Beam Current');
  });

  test('a name with no dot is title-cased as-is', () => {
    expect(fmtName('disk_space', 'file_system')).toBe('Disk Space');
  });

  // A check name is free-form, so a dot in it means nothing on its own: only
  // the row's OWN category is a prefix. Stripping to the first dot regardless
  // is how a name that merely spells one loses its leading word.
  test('the prefix is stripped only when it is the row category', () => {
    expect(fmtName('a.b_c', 'a')).toBe('B C');
    expect(fmtName('a.b_c', 'x')).toBe('A.B C');
  });

  test('a single word is capitalized', () => {
    expect(fmtName('latency', 'network')).toBe('Latency');
  });

  // The web terminal's health bar item lists the same check names and cannot
  // import this bundle, so the rule lives in the design system and this module
  // re-exports it. Asserting identity — not behaviour — is what makes a future
  // second copy a failure here rather than a quiet divergence on two surfaces.
  test("is the design system's function itself, not a copy of it", () => {
    expect(fmtName).toBe(sharedFmtName);
  });
});

describe('fmtMs', () => {
  test('sub-second latencies render as rounded milliseconds', () => {
    expect(fmtMs(42)).toBe('42ms');
    expect(fmtMs(42.6)).toBe('43ms');
    expect(fmtMs(999)).toBe('999ms');
  });

  test('one second and above renders as fixed-1 seconds', () => {
    expect(fmtMs(1000)).toBe('1.0s');
    expect(fmtMs(2500)).toBe('2.5s');
  });

  test('zero, negative, or missing latency renders empty (per to_dict omission)', () => {
    expect(fmtMs(0)).toBe('');
    expect(fmtMs(-5)).toBe('');
    expect(fmtMs(undefined)).toBe('');
  });
});

describe('msCls', () => {
  test('buckets latency into fast / medium / slow classes', () => {
    expect(msCls(50)).toBe('ms-f');
    expect(msCls(99)).toBe('ms-f');
    expect(msCls(100)).toBe('ms-m');
    expect(msCls(499)).toBe('ms-m');
    expect(msCls(500)).toBe('ms-s');
    expect(msCls(5000)).toBe('ms-s');
  });

  test('zero, negative, or missing latency yields no class', () => {
    expect(msCls(0)).toBe('');
    expect(msCls(-1)).toBe('');
    expect(msCls(undefined)).toBe('');
  });
});

describe('byCategory', () => {
  test('groups results by category, preserving first-seen order', () => {
    const results = [
      { category: 'file_system', name: 'a' },
      { category: 'providers', name: 'b' },
      { category: 'file_system', name: 'c' },
    ];
    const grouped = byCategory(results);
    expect([...grouped.keys()]).toEqual(['file_system', 'providers']);
    expect(grouped.get('file_system')?.map((r) => r.name)).toEqual(['a', 'c']);
    expect(grouped.get('providers')?.map((r) => r.name)).toEqual(['b']);
  });

  test('an empty result list yields an empty map', () => {
    expect(byCategory([]).size).toBe(0);
  });
});

// -- layer-stack view helpers ----------------------------------------------------

import {
  SEVERITY,
  sevOf,
  worstSev,
  commonMethod,
  tagOf,
  fmtAge,
  fmtClock,
  fmtDur,
  parseHash,
  formatHash,
  headline,
  ty,
  argTree,
  inlineSpans,
  mdBlocks,
  firstLine,
  mcpFact,
  mcpSummary,
  pingText,
} from './helpers.js';

describe('severity labels', () => {
  test('wire status maps to the EPICS-style label', () => {
    expect(SEVERITY).toEqual({ ok: 'OK', warning: 'MINOR', error: 'MAJOR', skip: 'DISABLED' });
    expect(sevOf('warning')).toBe('MINOR');
    expect(sevOf('bogus')).toBe('DISABLED');
    expect(sevOf(undefined)).toBe('DISABLED');
  });

  test('worstSev ranks MAJOR > MINOR > OK > DISABLED', () => {
    expect(worstSev(['OK', 'MINOR', 'DISABLED'])).toBe('MINOR');
    expect(worstSev(['OK', 'MAJOR', 'MINOR'])).toBe('MAJOR');
    expect(worstSev(['DISABLED', 'OK'])).toBe('OK');
    expect(worstSev([])).toBe('DISABLED');
  });
});

describe('commonMethod', () => {
  test('rows differing only in the trailing path share the prefix', () => {
    expect(commonMethod(['dir stat /data/a', 'dir stat /data/b'])).toBe('dir stat');
  });

  test('identical methods are shared whole', () => {
    expect(commonMethod(['http GET /health', 'http GET /health'])).toBe('http GET /health');
  });

  test('different methods or a single row share nothing', () => {
    expect(commonMethod(['bolt query', 'tcp connect'])).toBeNull();
    expect(commonMethod(['dir stat'])).toBeNull();
    expect(commonMethod(['a b c d', 'a x y z'])).toBeNull();
    expect(commonMethod(['', ''])).toBeNull();
  });
});

describe('tagOf', () => {
  test('annotations decide when present', () => {
    expect(tagOf({ name: 'channel_write', annotations: { readOnlyHint: true } })).toEqual({
      tag: '',
      derived: false,
    });
    expect(tagOf({ name: 'x', annotations: { destructiveHint: true } })).toEqual({
      tag: 'write · gated',
      derived: false,
    });
    expect(tagOf({ name: 'x', annotations: { readOnlyHint: false } })).toEqual({
      tag: 'write',
      derived: false,
    });
  });

  test('otherwise the name decides, and says it was derived', () => {
    expect(tagOf({ name: 'channel_write' })).toEqual({ tag: 'write · gated', derived: true });
    expect(tagOf({ name: 'execute_file' }).tag).toBe('write · gated');
    expect(tagOf({ name: 'artifact_create' }).tag).toBe('write');
    expect(tagOf({ name: 'channel_read' })).toEqual({ tag: '', derived: true });
    expect(tagOf({ name: 'setup_info', annotations: null }).tag).toBe('');
  });
});

describe('time formatting', () => {
  test('fmtAge buckets seconds, minutes, hours and days', () => {
    expect(fmtAge(1)).toBe('just now');
    expect(fmtAge(-5)).toBe('just now');
    expect(fmtAge(14)).toBe('14 s ago');
    expect(fmtAge(125)).toBe('2 m ago');
    expect(fmtAge(7300)).toBe('2 h ago');
    expect(fmtAge(200000)).toBe('2 d ago');
  });

  test('fmtClock prints UTC wall-clock time', () => {
    expect(fmtClock(Date.UTC(2026, 10, 6, 6, 31, 7) / 1000)).toBe('06:31:07Z');
    expect(fmtClock(null)).toBe('--:--:--Z');
  });

  test('fmtDur switches to seconds at one second', () => {
    expect(fmtDur(166)).toBe('166 ms');
    expect(fmtDur(4590)).toBe('4.6 s');
    expect(fmtDur(null)).toBe('—');
  });
});

describe('hash', () => {
  test('parse decodes each part; format is its inverse', () => {
    expect(parseHash('#mcp-servers/controls/channel_write')).toEqual([
      'mcp-servers',
      'controls',
      'channel_write',
    ]);
    expect(parseHash('#a%20b/c')).toEqual(['a b', 'c']);
    expect(parseHash('')).toEqual([]);
    expect(formatHash(['a b', 'c'])).toBe('#a%20b/c');
    expect(formatHash([])).toBe('');
    expect(parseHash(formatHash(['telemetry-store', 'not used/x']))).toEqual([
      'telemetry-store',
      'not used/x',
    ]);
  });
});

describe('headline', () => {
  const layers = [
    {
      components: [
        { id: 'a', name: 'alpha', status: 'ok' },
        { id: 'b', name: 'beta', status: 'warning', issue: '3 unreadable' },
      ],
    },
    { components: [{ id: 'c', name: 'gamma', status: 'error', issue: null }] },
  ];
  test('counts rows by severity and lists faulty components, MAJOR first', () => {
    const hl = headline(layers, [
      { status: 'warning' },
      { status: 'warning' },
      { status: 'error' },
      { status: 'ok' },
    ]);
    expect(hl.minor).toBe(2);
    expect(hl.major).toBe(1);
    expect(hl.faults.map((f) => f.id)).toEqual(['c', 'b']);
    expect(hl.faults[1].issue).toBe('3 unreadable');
    expect(hl.faults[0].issue).toBe('MAJOR');
  });
});

describe('schema rendering', () => {
  const root = {
    type: 'object',
    required: ['writes'],
    properties: {
      writes: { type: 'array', items: { $ref: '#/$defs/W' }, description: 'pairs\nof values' },
      mode: { enum: ['a', 'b'], default: 'a' },
      n: { anyOf: [{ type: 'integer' }, { type: 'null' }], default: null },
    },
    $defs: {
      W: {
        type: 'object',
        required: ['channel'],
        properties: { channel: { type: 'string' }, value: { type: 'number' } },
      },
    },
  };

  test('ty renders enums, unions, arrays and objects on one line', () => {
    expect(ty({ type: 'string' })).toBe('string');
    expect(ty({ enum: ['a', 'b'] })).toBe('"a" | "b"');
    expect(ty({ anyOf: [{ type: 'integer' }, { type: 'null' }] })).toBe('integer | null');
    expect(ty({ type: ['string', 'null'] })).toBe('string | null');
    expect(ty({ type: 'array', items: { type: 'string' } })).toBe('array<string>');
    expect(ty(root.properties.writes, root)).toBe('array<object{2}>');
    expect(ty({ const: 3 })).toBe('3');
    expect(ty({})).toBe('any');
  });

  test('argTree resolves refs, marks required, folds descriptions and nests objects', () => {
    const tree = argTree(root);
    expect(tree.map((a) => a.name)).toEqual(['writes', 'mode', 'n']);
    expect(tree[0].required).toBe(true);
    expect(tree[0].desc).toBe('pairs of values');
    expect(tree[0].children.map((a) => [a.name, a.required, a.type])).toEqual([
      ['channel', true, 'string'],
      ['value', false, 'number'],
    ]);
    expect(tree[1].def).toBe('"a"');
    expect(tree[2].def).toBe('null');
    expect(argTree({})).toEqual([]);
  });
});

describe('markdown-lite', () => {
  test('inlineSpans splits single and double backtick code', () => {
    expect(inlineSpans('use `a` or ``b c``')).toEqual([
      { code: false, text: 'use ' },
      { code: true, text: 'a' },
      { code: false, text: ' or ' },
      { code: true, text: 'b c' },
    ]);
  });

  test('mdBlocks yields paragraphs and bullet lists, never HTML', () => {
    const b = mdBlocks('First line\ncontinues.\n\nArgs:\n- `x`: one\n  more\n- y <b>');
    expect(b[0]).toEqual({ kind: 'p', spans: [{ code: false, text: 'First line continues.' }] });
    expect(b[1].kind).toBe('ul');
    if (b[1].kind === 'ul') {
      expect(b[1].lead).toEqual([{ code: false, text: 'Args:' }]);
      expect(b[1].items.length).toBe(2);
      expect(b[1].items[0]).toEqual([
        { code: true, text: 'x' },
        { code: false, text: ': one more' },
      ]);
      expect(b[1].items[1][0].text).toBe('y <b>');
    }
    expect(mdBlocks(null)).toEqual([]);
  });

  test('firstLine', () => {
    expect(firstLine('  a\nb')).toBe('a');
    expect(firstLine(undefined)).toBe('');
  });
});

describe('MCP summaries', () => {
  test('mcpFact before and after introspection', () => {
    expect(mcpFact({ configured: ['a', 'b'], answered: null, tools: null })).toBe('2 servers');
    expect(mcpFact({ configured: ['a'], answered: null })).toBe('1 server');
    expect(mcpFact({ configured: ['a', 'b'], answered: 2, tools: 9 })).toBe(
      '2 / 2 answered · 9 tools'
    );
    expect(mcpFact({ configured: ['a', 'b', 'c'], stdio: 2, answered: 2, tools: 9 })).toBe(
      '2 / 2 answered · 9 tools'
    );
    expect(mcpFact(null)).toBe('');
  });

  test('mcpSummary counts stdio answers and tools, http excluded', () => {
    const s = mcpSummary({
      probed_at: 5,
      servers: [
        { name: 'a', transport: 'stdio', tools: [1, 2] },
        { name: 'b', transport: 'stdio', tools: null },
        { name: 'c', transport: 'http', tools: null },
      ],
    });
    expect(s).toEqual({
      configured: ['a', 'b', 'c'],
      answered: 1,
      stdio: 2,
      tools: 2,
      probed_at: 5,
    });
  });
});

describe('pingText', () => {
  test('a completion shows model, latency and tokens', () => {
    expect(
      pingText({
        status: 'ok',
        kind: 'completion',
        model: 'm',
        latency_ms: 880.4,
        input_tokens: 14,
        output_tokens: 1,
      })
    ).toEqual({ value: 'm · 880 ms · 14 in / 1 out', method: '1-token completion' });
  });

  test('an embedding shows input tokens only', () => {
    expect(
      pingText({ status: 'ok', kind: 'embedding', model: 'e', latency_ms: 9, input_tokens: 4 })
    ).toEqual({ value: 'e · 9 ms · 4 in', method: '1 embedding' });
  });

  test('a failed ping shows its message', () => {
    expect(pingText({ status: 'skip', kind: 'completion', message: 'API key not set' }).value).toBe(
      'API key not set'
    );
  });
});
