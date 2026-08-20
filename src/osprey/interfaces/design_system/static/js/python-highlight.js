// @ts-check
/**
 * Read-only Python syntax highlighting for the design-system front-end.
 *
 * Pairs with css/code.css: {@link tokenizePython} splits Python source into
 * classified runs, and {@link highlightPythonInto} paints those runs into an
 * element as `<span class="tok tok-KIND">` nodes. Hand-written ES module
 * (mirrors dom.js / highlight.js) with JSDoc types so it type-checks under
 * `tsc --noEmit --strict`.
 *
 * Two properties matter more than grammar completeness here:
 *
 * 1. **No markup sink.** The painter builds nodes with `createElement` and
 *    `textContent` — never `innerHTML`. Source shown in a panel is untrusted
 *    (it is whatever the control system's plan library happens to contain), so
 *    highlighting it must not open a markup sink where there wasn't one. This
 *    is what lets the Bluesky plan bundle keep its "no innerHTML anywhere"
 *    contract (see panels/bluesky/hyperscript.js) while gaining colour.
 *
 * 2. **Lossless.** Concatenating every token's `text` reproduces the input
 *    byte-for-byte, so nothing can be silently dropped or reordered on the way
 *    to the screen. The tests assert this over every fixture.
 *
 * Deliberately NOT a full Python parser. It is a single-pass lexer that gets
 * the constructs an operator actually reads — comments, strings, numbers,
 * keywords, decorators, definition names — and falls back to unstyled text for
 * everything else. Interpolations inside an f-string stay part of the string
 * run rather than being re-lexed; that matches how most editors render them at
 * a glance and keeps the scanner one pass.
 *
 * @module python-highlight
 */

/** Reserved words. Soft keywords (`match`, `case`, `type`) are deliberately
 *  omitted: they are ordinary identifiers outside their statement context, and
 *  a lexer with no parser behind it would colour those false positives. */
const KEYWORDS = new Set(
  (
    'and as assert async await break class continue def del elif else except finally for ' +
    'from global if import in is lambda nonlocal not or pass raise return try while with yield'
  ).split(' '),
);

/** Singletons and the conventional first arguments — names that read as
 *  language furniture rather than as ordinary variables. */
const CONSTANTS = new Set(['True', 'False', 'None', 'NotImplemented', 'Ellipsis', 'self', 'cls']);

/** Built-in callables and the common exception types. A partial list on
 *  purpose: an unlisted name simply renders as plain text, which is the right
 *  failure mode. */
const BUILTINS = new Set(
  (
    'abs all any bool bytes callable classmethod dict dir enumerate filter float format ' +
    'frozenset getattr hasattr hash id input int isinstance issubclass iter len list map max ' +
    'min next object open ord print property range repr reversed round set setattr sorted ' +
    'staticmethod str sum super tuple type vars zip ' +
    'AttributeError Exception IndexError KeyError NotImplementedError OSError RuntimeError ' +
    'StopIteration TypeError ValueError'
  ).split(' '),
);

/** Letters that may prefix a string literal (`r`, `b`, `u`, `f` and pairs). */
const STRING_PREFIX = /[rbufRBUF]/;

/** Identifier start: any Unicode letter or underscore (Python 3 allows both). */
const IDENT_START = /[\p{L}\p{Nl}_]/u;

/** Identifier continuation: adds digits and combining marks. */
const IDENT_PART = /[\p{L}\p{Nl}\p{Nd}\p{Mn}\p{Mc}\p{Pc}_]/u;

/**
 * A classified run of source text.
 *
 * @typedef {object} PythonToken
 * @property {'plain'|'comment'|'string'|'number'|'keyword'|'constant'|'builtin'|'decorator'|'def'} kind
 * @property {string} text
 */

/**
 * Scan a string literal starting at `start`, prefix included.
 *
 * @param {string} src
 * @param {number} start
 * @returns {number} index just past the literal, or -1 if none starts here
 */
function readString(src, start) {
  let i = start;
  // Up to two prefix letters (rb, fr, ...). They only count as a prefix if a
  // quote follows, so ordinary identifiers like `buf` or `format` fall through.
  let prefix = 0;
  while (prefix < 2 && STRING_PREFIX.test(src[i + prefix] ?? '')) prefix++;
  i += prefix;

  const quote = src[i];
  if (quote !== '"' && quote !== "'") return -1;

  const delim = src.startsWith(quote.repeat(3), i) ? quote.repeat(3) : quote;
  i += delim.length;

  while (i < src.length) {
    const ch = src[i];
    // A backslash escapes the next character even in raw strings: `r"\""` does
    // not end at the middle quote, so scanning must skip the pair either way.
    if (ch === '\\') {
      i += 2;
      continue;
    }
    // An unterminated single-quoted string ends at the newline rather than
    // running on: one stray quote must not paint the rest of the file.
    if (delim.length === 1 && ch === '\n') return i;
    if (src.startsWith(delim, i)) return i + delim.length;
    i++;
  }
  return src.length;
}

/**
 * Scan a numeric literal starting at `start`.
 *
 * @param {string} src
 * @param {number} start
 * @returns {number} index just past the literal, or -1 if none starts here
 */
function readNumber(src, start) {
  let i = start;
  const digit = /[0-9]/;
  if (!digit.test(src[i] ?? '') && !(src[i] === '.' && digit.test(src[i + 1] ?? ''))) return -1;

  if (src[i] === '0' && /[xXoObB]/.test(src[i + 1] ?? '')) {
    i += 2;
    while (i < src.length && /[0-9a-fA-F_]/.test(src[i])) i++;
    return i;
  }

  while (i < src.length && /[0-9_]/.test(src[i])) i++;
  if (src[i] === '.') {
    i++;
    while (i < src.length && /[0-9_]/.test(src[i])) i++;
  }
  if (/[eE]/.test(src[i] ?? '')) {
    // Only consume the exponent if digits actually follow, so `1.e` keeps its
    // trailing name separate instead of swallowing it into the number.
    let j = i + 1;
    if (/[+-]/.test(src[j] ?? '')) j++;
    if (digit.test(src[j] ?? '')) {
      i = j;
      while (i < src.length && /[0-9_]/.test(src[i])) i++;
    }
  }
  if (/[jJ]/.test(src[i] ?? '')) i++;
  return i;
}

/**
 * Split Python source into classified runs.
 *
 * Concatenating the returned `text` values reproduces `source` exactly.
 *
 * @param {string} source
 * @returns {PythonToken[]}
 */
export function tokenizePython(source) {
  /** @type {PythonToken[]} */
  const tokens = [];
  let plain = '';

  /** Emit any buffered unstyled text. */
  const flush = () => {
    if (plain) {
      tokens.push({ kind: 'plain', text: plain });
      plain = '';
    }
  };

  /**
   * @param {PythonToken['kind']} kind
   * @param {string} text
   */
  const push = (kind, text) => {
    flush();
    tokens.push({ kind, text });
  };

  let i = 0;
  let lineHasContent = false; // has this line produced anything but whitespace?
  let pendingDef = false; // the previous keyword was `def` or `class`

  while (i < source.length) {
    const ch = source[i];

    if (ch === '\n') {
      plain += ch;
      i++;
      lineHasContent = false;
      continue;
    }
    if (ch === ' ' || ch === '\t' || ch === '\r' || ch === '\f') {
      plain += ch;
      i++;
      continue;
    }

    if (ch === '#') {
      const nl = source.indexOf('\n', i);
      const end = nl === -1 ? source.length : nl;
      push('comment', source.slice(i, end));
      i = end;
      lineHasContent = true;
      continue;
    }

    // A decorator only at the head of a line; `@` elsewhere is matrix-multiply.
    if (ch === '@' && !lineHasContent) {
      let j = i + 1;
      while (j < source.length && (IDENT_PART.test(source[j]) || source[j] === '.')) j++;
      if (j > i + 1) {
        push('decorator', source.slice(i, j));
        i = j;
        lineHasContent = true;
        pendingDef = false;
        continue;
      }
    }

    // Strings before identifiers: a prefixed literal like f"..." starts with
    // an identifier character.
    const strEnd = readString(source, i);
    if (strEnd !== -1) {
      push('string', source.slice(i, strEnd));
      i = strEnd;
      lineHasContent = true;
      pendingDef = false;
      continue;
    }

    const numEnd = readNumber(source, i);
    if (numEnd !== -1) {
      push('number', source.slice(i, numEnd));
      i = numEnd;
      lineHasContent = true;
      pendingDef = false;
      continue;
    }

    if (IDENT_START.test(ch)) {
      let j = i + 1;
      while (j < source.length && IDENT_PART.test(source[j])) j++;
      const word = source.slice(i, j);
      // After a dot the name is an attribute, not the language's own — so
      // `record.type` does not light up as the `type` builtin.
      const isAttribute = source[i - 1] === '.';

      /** @type {PythonToken['kind']} */
      let kind = 'plain';
      if (pendingDef) kind = 'def';
      else if (isAttribute) kind = 'plain';
      else if (KEYWORDS.has(word)) kind = 'keyword';
      else if (CONSTANTS.has(word)) kind = 'constant';
      else if (BUILTINS.has(word)) kind = 'builtin';

      if (kind === 'plain') plain += word;
      else push(kind, word);

      pendingDef = kind === 'keyword' && (word === 'def' || word === 'class');
      i = j;
      lineHasContent = true;
      continue;
    }

    plain += ch;
    i++;
    lineHasContent = true;
    pendingDef = false;
  }

  flush();
  return tokens;
}

/**
 * Replace `el`'s contents with syntax-highlighted `source`.
 *
 * Every run becomes either a bare text node (unstyled) or a `<span>` whose
 * text is set through `textContent`. No markup is ever parsed, so the source
 * cannot introduce elements regardless of what it contains. The container gets
 * `code-python`, which css/code.css scopes all token colours to.
 *
 * @param {HTMLElement} el - target element; its contents are replaced
 * @param {string} source - Python source text
 * @returns {void}
 */
export function highlightPythonInto(el, source) {
  el.textContent = '';
  el.classList.add('code-python');

  const frag = document.createDocumentFragment();
  for (const token of tokenizePython(source)) {
    if (token.kind === 'plain') {
      frag.appendChild(document.createTextNode(token.text));
      continue;
    }
    const span = document.createElement('span');
    span.className = `tok tok-${token.kind}`;
    span.textContent = token.text;
    frag.appendChild(span);
  }
  el.appendChild(frag);
}
