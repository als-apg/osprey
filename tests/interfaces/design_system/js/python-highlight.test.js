/**
 * Unit tests for the design-system Python syntax highlighter.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/python-highlight.test.js
 *
 * Two invariants matter more than any single grammar rule and are asserted
 * over every fixture below:
 *
 *   losslessness  concatenating the tokens reproduces the input exactly, so
 *                 nothing can be dropped or reordered on the way to the screen;
 *   no markup     the painter never parses HTML, so untrusted plan source
 *                 cannot introduce elements.
 */

import { test, expect, describe } from 'vitest';

import {
  tokenizePython,
  highlightPythonInto,
} from '/design-system/js/python-highlight.js';

/** Fixtures reused by the losslessness sweep. */
const SOURCES = [
  '',
  'x = 1\n',
  '# just a comment',
  'def run(detector, *, steps=10):\n    """Doc."""\n    return [s for s in range(steps)]\n',
  '@stage\nasync def scan(md=None):\n    yield from bp.count([det], num=3)\n',
  "s = f'{a!r:>{w}} and {b}'  # f-string\n",
  'n = 0xFF + 0b1010 + 1_000.5e-3 + 2j\n',
  'raw = r"C:\\path\\"  # trailing escape\n',
  'unterminated = "oops\nnext = 1\n',
  'm = a @ b\n',
  'text = """multi\nline @notadecorator #nothash\n"""\n',
];

/**
 * Collect the text of every token of a given kind.
 *
 * @param {string} src
 * @param {string} kind
 * @returns {string[]}
 */
function textsOfKind(src, kind) {
  return tokenizePython(src)
    .filter((t) => t.kind === kind)
    .map((t) => t.text);
}

describe('tokenizePython losslessness', () => {
  test.each(SOURCES)('round-trips %j', (src) => {
    expect(tokenizePython(src).map((t) => t.text).join('')).toBe(src);
  });

  test('emits no empty tokens', () => {
    for (const src of SOURCES) {
      for (const token of tokenizePython(src)) {
        expect(token.text.length).toBeGreaterThan(0);
      }
    }
  });
});

describe('keywords and definitions', () => {
  test('classifies reserved words', () => {
    expect(textsOfKind('if x: return None', 'keyword')).toEqual(['if', 'return']);
  });

  test('names the thing being defined', () => {
    expect(textsOfKind('def run(a):\n  pass\n', 'def')).toEqual(['run']);
    expect(textsOfKind('class Motor:\n  pass\n', 'def')).toEqual(['Motor']);
  });

  test('soft keywords stay plain — they are ordinary identifiers here', () => {
    expect(textsOfKind('match = 1\ncase = 2\n', 'keyword')).toEqual([]);
  });

  test('a builtin reached through a dot is an attribute, not a builtin', () => {
    expect(textsOfKind('record.type', 'builtin')).toEqual([]);
    expect(textsOfKind('type(record)', 'builtin')).toEqual(['type']);
  });

  test('self and cls read as language furniture', () => {
    expect(textsOfKind('def go(self, cls):\n  pass\n', 'constant')).toEqual(['self', 'cls']);
  });
});

describe('strings', () => {
  test('handles single, double and triple quotes', () => {
    expect(textsOfKind('a = "x"; b = \'y\'', 'string')).toEqual(['"x"', "'y'"]);
    expect(textsOfKind('d = """a\nb"""\n', 'string')).toEqual(['"""a\nb"""']);
  });

  test('prefixed literals are strings, not identifiers', () => {
    expect(textsOfKind('f"{x}"', 'string')).toEqual(['f"{x}"']);
    expect(textsOfKind('rb\'\\x00\'', 'string')).toEqual(["rb'\\x00'"]);
  });

  test('identifiers that merely start with prefix letters stay identifiers', () => {
    expect(textsOfKind('buf = format(u)', 'string')).toEqual([]);
    expect(textsOfKind('buf = format(u)', 'builtin')).toEqual(['format']);
  });

  test('an escaped quote does not end the literal', () => {
    expect(textsOfKind('a = "he said \\"hi\\""', 'string')).toEqual(['"he said \\"hi\\""']);
  });

  test('an unterminated single-quoted string stops at the newline', () => {
    // Otherwise one stray quote would paint the rest of the file as a string.
    expect(textsOfKind('a = "oops\nb = 1\n', 'string')).toEqual(['"oops']);
  });

  test('# and @ inside a string are not comment or decorator', () => {
    expect(textsOfKind('a = "# @x"', 'comment')).toEqual([]);
    expect(textsOfKind('a = "# @x"', 'decorator')).toEqual([]);
  });
});

describe('comments, numbers and decorators', () => {
  test('a comment runs to end of line only', () => {
    expect(textsOfKind('x = 1  # note\ny = 2\n', 'comment')).toEqual(['# note']);
  });

  test('numeric literal forms', () => {
    expect(textsOfKind('0xFF 0b1010 0o17 1_000 1.5e-3 2j .5', 'number')).toEqual([
      '0xFF',
      '0b1010',
      '0o17',
      '1_000',
      '1.5e-3',
      '2j',
      '.5',
    ]);
  });

  test('a decorator is only a decorator at the head of a line', () => {
    expect(textsOfKind('@run_decorator\ndef f():\n  pass\n', 'decorator')).toEqual([
      '@run_decorator',
    ]);
    // Matrix multiplication must not be mistaken for one.
    expect(textsOfKind('m = a @ b\n', 'decorator')).toEqual([]);
  });

  test('a dotted decorator keeps its path', () => {
    expect(textsOfKind('  @bp.stage\n', 'decorator')).toEqual(['@bp.stage']);
  });
});

describe('highlightPythonInto', () => {
  test('marks the container and preserves the exact text', () => {
    const el = document.createElement('pre');
    const src = 'def f():\n    return "x"  # done\n';
    highlightPythonInto(el, src);

    expect(el.classList.contains('code-python')).toBe(true);
    expect(el.textContent).toBe(src);
    expect(el.querySelectorAll('span.tok').length).toBeGreaterThan(0);
  });

  test('source that looks like markup produces text, never elements', () => {
    const el = document.createElement('pre');
    const src = 'x = "<img src=q onerror=alert(1)>"\n# <script>alert(2)</script>\n';
    highlightPythonInto(el, src);

    expect(el.querySelector('img')).toBeNull();
    expect(el.querySelector('script')).toBeNull();
    // Every element created is one of the highlighter's own spans.
    for (const node of el.querySelectorAll('*')) {
      expect(node.tagName).toBe('SPAN');
      expect(node.className.startsWith('tok ')).toBe(true);
    }
    expect(el.textContent).toBe(src);
  });

  test('re-painting replaces the previous render', () => {
    const el = document.createElement('pre');
    highlightPythonInto(el, 'first = 1\n');
    highlightPythonInto(el, 'second = 2\n');
    expect(el.textContent).toBe('second = 2\n');
  });
});
