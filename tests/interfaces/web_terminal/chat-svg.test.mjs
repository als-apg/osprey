// @ts-check
// @vitest-environment jsdom
/**
 * SVG callouts in agent replies (chat-svg.js), rendered by the vendored
 * libraries through chat-markdown.js:
 *   npx vitest run tests/interfaces/web_terminal/chat-svg.test.mjs
 *
 * The drawing is model output placed in the page, so these tests are the
 * sanitiser pins: each tries a way an SVG can run code, embed HTML or reach
 * outside the page. They run on jsdom with the real marked and DOMPurify
 * (see agent-links-vendored.test.mjs for why happy-dom proves nothing for a
 * sanitiser, and for the manifest pin on these same builds).
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, beforeAll, beforeEach, describe, expect, test, vi } from 'vitest';

import { renderMarkdownInto } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-markdown.js';
import { MAX_SVG_SOURCE } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-svg.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');

/**
 * Evaluate a UMD build as a classic `<script>` would, and take its global back.
 * @param {string} name - the global the build installs
 * @param {string} file - the build, relative to the repo root
 * @returns {any}
 */
function loadClassicScript(name, file) {
  const source = readFileSync(path.join(REPO_ROOT, file), 'utf8');
  new Function('exports', 'module', 'define', 'require', source)();
  const g = /** @type {any} */ (globalThis);
  const lib = g[name];
  delete g[name];
  return lib;
}

/** @type {{ marked: any, DOMPurify: any }} */
let real;
/** @type {HTMLElement} */
let el;

beforeAll(() => {
  real = {
    marked: loadClassicScript('marked', 'node_modules/marked/marked.min.js'),
    DOMPurify: loadClassicScript('DOMPurify', 'node_modules/dompurify/dist/purify.min.js'),
  };
});

beforeEach(() => {
  vi.stubGlobal('marked', real.marked);
  vi.stubGlobal('DOMPurify', real.DOMPurify);
  el = document.createElement('div');
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** @param {string} svg */
const reply = (svg) => `Top results:\n\n\`\`\`svg\n${svg}\n\`\`\`\n\nartifact_id: 28bfaa39e90e`;

/** @param {string} svg */
function drawn(svg) {
  renderMarkdownInto(el, reply(svg), { math: true, svg: true });
  return el.querySelector('.op-svg-art svg');
}

const BAR =
  '<svg viewBox="0 0 300 40" xmlns="http://www.w3.org/2000/svg">' +
  '<rect x="0" y="10" width="120" height="20" fill="var(--color-accent)"/>' +
  '<text x="130" y="25" fill="currentColor">HCM 01: 2.4 A</text></svg>';

describe('a complete svg block', () => {
  test('is drawn in a box in place of its code block, with its source one click away', () => {
    const svg = drawn(BAR);

    expect(svg).not.toBeNull();
    expect(el.querySelector('pre > code.language-svg')).toBeNull();
    expect(svg?.querySelector('text')?.textContent).toBe('HCM 01: 2.4 A');
    expect(svg?.querySelector('rect')?.getAttribute('fill')).toBe('var(--color-accent)');
    expect(el.querySelector('.op-svg-source > summary')?.textContent).toBe('Source');
    expect(el.querySelector('.op-svg-source code')?.textContent).toBe(BAR);
    // The prose around it is still the reply.
    expect(el.textContent).toContain('Top results:');
  });

  test('takes its width from its viewBox, so it is not stretched to the reply', () => {
    expect(drawn(BAR)?.getAttribute('width')).toBe('300');
  });

  test('keeps a reference to its own gradient', () => {
    const svg = drawn(
      '<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg"><defs>' +
        '<linearGradient id="g"><stop offset="0" stop-color="red"/></linearGradient></defs>' +
        '<rect width="10" height="10" fill="url(#g)"/></svg>'
    );
    expect(svg?.querySelector('rect')?.getAttribute('fill')).toBe('url(#g)');
  });
});

describe('what a drawing may not carry', () => {
  test.each([
    ['a script element', '<script>alert(1)</script>', 'script'],
    ['embedded HTML', '<foreignObject><div>hi</div></foreignObject>', 'foreignObject'],
    ['a link', '<a href="javascript:alert(1)"><text>x</text></a>', 'a'],
    ['an outside image', '<image href="http://evil.example/x.png" width="5" height="5"/>', 'image'],
    ['a reused outside node', '<use href="http://evil.example/s.svg#x"/>', 'use'],
    ['a style sheet', '<style>rect{fill:url(http://evil.example/x)}</style>', 'style'],
    ['an animation', '<animate attributeName="x" to="5" dur="1s"/>', 'animate'],
  ])('%s is removed', (_case, inner, tag) => {
    const svg = drawn(
      `<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg">${inner}<rect width="1" height="1"/></svg>`
    );
    expect(svg).not.toBeNull();
    expect(svg?.getElementsByTagName(tag)).toHaveLength(0);
    expect(svg?.querySelector('rect')).not.toBeNull();
  });

  test('an event handler is removed', () => {
    const svg = drawn(
      '<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg" onload="alert(1)">' +
        '<rect width="1" height="1" onclick="alert(2)"/></svg>'
    );
    expect(svg?.hasAttribute('onload')).toBe(false);
    expect(svg?.querySelector('rect')?.hasAttribute('onclick')).toBe(false);
  });

  test('an attribute that loads from outside the page is removed', () => {
    const svg = drawn(
      '<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg">' +
        '<rect width="1" height="1" fill="url(http://evil.example/p)" style="background:url(//evil.example/b)"/></svg>'
    );
    const rect = svg?.querySelector('rect');
    expect(rect?.hasAttribute('fill')).toBe(false);
    expect(rect?.hasAttribute('style')).toBe(false);
  });

  test.each([
    ['a css-escaped url()', 'filter', '\\75rl(https://evil.example/f.svg#f)'],
    ['a long css escape', 'mask', '\\000075rl(https://evil.example/m.svg#m)'],
    [
      'an escape beside an inside reference',
      'clip-path',
      'url(#c) \\75rl(https://evil.example/c.svg#c)',
    ],
    ['an outside url after an inside one', 'filter', 'url(#f) url(https://evil.example/f.svg#f)'],
    ['an image-set()', 'mask', 'image-set("https://evil.example/c.png" 1x), auto'],
    ['a src()', 'filter', 'src("https://evil.example/c.png"), auto'],
  ])('%s in an attribute is removed', (_case, name, value) => {
    const svg = drawn(
      `<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg">` +
        `<rect width="1" height="1" ${name}='${value}'/></svg>`
    );
    const rect = svg?.querySelector('rect');
    expect(rect).not.toBeNull();
    expect(rect?.hasAttribute(name)).toBe(false);
    expect(svg?.outerHTML).not.toContain('evil.example');
  });

  test("a reference to the drawing's own parts is kept in a filter, a mask and a clip", () => {
    const svg = drawn(
      '<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg"><defs>' +
        '<filter id="f"><feGaussianBlur stdDeviation="1"/></filter>' +
        '<clipPath id="c"><rect width="5" height="5"/></clipPath></defs>' +
        '<rect width="1" height="1" filter="url(#f)" clip-path="url(&quot;#c&quot;)"/></svg>'
    );
    const rect = svg?.querySelector('rect[filter]');
    expect(rect?.getAttribute('filter')).toBe('url(#f)');
    expect(rect?.getAttribute('clip-path')).toBe('url("#c")');
  });

  test('any attribute holding a backslash is removed', () => {
    const svg = drawn(
      '<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg">' +
        '<text x="1" y="5" font-family="a\\62 c">x</text></svg>'
    );
    expect(svg?.querySelector('text')?.hasAttribute('font-family')).toBe(false);
    expect(svg?.querySelector('text')?.getAttribute('x')).toBe('1');
  });
});

describe('a drawing reads in every theme', () => {
  /** @param {string} inner */
  const one = (inner) =>
    drawn(`<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg">${inner}</svg>`);

  test.each([
    ['a hex fill', '<text fill="#000">x</text>', 'text', 'fill'],
    ['a named fill', '<text fill="white">x</text>', 'text', 'fill'],
    ['a hex stroke', '<line x2="5" stroke="#111"/>', 'line', 'stroke'],
    [
      'a token with a hard fallback',
      '<rect width="1" height="1" fill="var(--color-accent, #000)"/>',
      'rect',
      'fill',
    ],
    [
      'a token not in the palette',
      '<rect width="1" height="1" fill="var(--bg-terminal)"/>',
      'rect',
      'fill',
    ],
    [
      'a gradient stop in hex',
      '<linearGradient id="g"><stop stop-color="#000"/></linearGradient>',
      'stop',
      'stop-color',
    ],
  ])('%s is dropped, so the part draws in the theme colour', (_case, inner, tag, attr) => {
    expect(one(inner)?.querySelector(tag)?.hasAttribute(attr)).toBe(false);
  });

  test.each([
    ['a palette token', 'var(--text-muted)'],
    ['the text colour', 'currentColor'],
    ['none', 'none'],
    ['its own gradient', 'url(#g)'],
  ])('%s is kept', (_case, value) => {
    expect(
      one(`<rect width="1" height="1" fill="${value}"/>`)
        ?.querySelector('rect')
        ?.getAttribute('fill')
    ).toBe(value);
  });

  test('a colour set by a style element or a style attribute is dropped with it', () => {
    const svg = one('<style>text{fill:#000}</style><text style="fill:#000">x</text>');
    expect(svg?.getElementsByTagName('style')).toHaveLength(0);
    expect(svg?.querySelector('text')?.hasAttribute('style')).toBe(false);
  });
});

describe('a block that is not one complete drawing stays code', () => {
  test.each([
    [
      'cut off before its end',
      '<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg"><rect width="1"',
    ],
    ['not svg at its root', '<div xmlns="http://www.w3.org/1999/xhtml">hi</div>'],
    [
      'too long for a callout',
      `<svg xmlns="http://www.w3.org/2000/svg"><!--${'x'.repeat(MAX_SVG_SOURCE)}--></svg>`,
    ],
  ])('%s', (_case, svg) => {
    expect(drawn(svg)).toBeNull();
    expect(el.querySelector('pre > code.language-svg')).not.toBeNull();
  });

  test('where drawings are not asked for (the scaffold Preview), the block stays code', () => {
    renderMarkdownInto(el, reply(BAR), { math: true });
    expect(el.querySelector('svg')).toBeNull();
    expect(el.querySelector('pre > code.language-svg')).not.toBeNull();
  });
});

describe('while a drawing streams', () => {
  const OPEN =
    'Quick look:\n\n```svg\n<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg"><rect';

  /** @param {string} text */
  function streamed(text) {
    renderMarkdownInto(el, text, { math: true, svg: true });
    return {
      placeholder: el.querySelector('.op-svg-pending')?.textContent ?? null,
      // Code the operator sees in the reply, not the drawing's own Source toggle.
      code: [...el.querySelectorAll('pre')].some((pre) => !pre.closest('.op-svg-source')),
      drawing: el.querySelector('.op-svg-art svg') !== null,
    };
  }

  test.each([
    ['its fence is open', OPEN],
    ['the fence is still naming its language', 'Quick look:\n\n```sv'],
    ['only the fence has arrived', 'Quick look:\n\n```'],
  ])('a placeholder stands in when %s, and no code shows', (_case, text) => {
    expect(streamed(text)).toEqual({ placeholder: 'Drawing…', code: false, drawing: false });
    expect(el.textContent).toContain('Quick look:');
  });

  test('once the fence closes, the drawing replaces the placeholder', () => {
    expect(streamed(reply(BAR))).toEqual({ placeholder: null, code: false, drawing: true });
  });

  test('another language streams as code, as before', () => {
    expect(streamed('Here:\n\n```python\nprint(1)')).toEqual({
      placeholder: null,
      code: true,
      drawing: false,
    });
  });

  test('a closed drawing followed by an open code block shows both as they are', () => {
    const state = streamed(`${reply(BAR)}\n\n\`\`\`python\nprint(1)`);
    expect(state).toEqual({ placeholder: null, code: true, drawing: true });
  });
});
