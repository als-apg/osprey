// @ts-check
// @vitest-environment jsdom
/**
 * An entry body that declares Markdown (entry-markdown.js), driven with the
 * real `marked` and DOMPurify builds the page loads:
 *   npx vitest run tests/interfaces/ariel/entry-markdown.test.mjs
 *
 * The ARIEL page has no Content-Security-Policy that would stop a script, so
 * the sanitiser is the only barrier: hostile entry text (`<script>`, an
 * `onerror` attribute, a `javascript:` or `data:` link, a remote picture, a
 * picture of another entry, attributes this page's click handlers read) must
 * come out inert, while tables, nested lists, the entry's own pictures and its
 * facility ids come out as structure.
 *
 * jsdom, not the suite's happy-dom: DOMPurify does not sanitise under
 * happy-dom (see security_render_markdown.test.mjs).
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, beforeAll, beforeEach, describe, expect, test, vi } from 'vitest';

import { renderEntryMarkdown } from '../../../src/osprey/interfaces/ariel/static/js/entry-markdown.js';
import { renderEntryText } from '../../../src/osprey/interfaces/ariel/static/js/entries-helpers.js';
import { renderEntryTextHtml } from '../../../src/osprey/interfaces/ariel/static/js/reader-facility.js';
import { renderEntryDetail } from '../../../src/osprey/interfaces/ariel/static/js/reader.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
/** @type {Readonly<Record<string, string>>} */
const BUILDS = Object.freeze({
  marked: 'node_modules/marked/marked.min.js',
  DOMPurify: 'node_modules/dompurify/dist/purify.min.js',
});

/**
 * Evaluate a UMD build as a classic <script> would, then take its global back.
 * @param {string} name
 * @returns {any}
 */
function loadClassicScript(name) {
  const source = readFileSync(path.join(REPO_ROOT, BUILDS[name]), 'utf8');
  new Function('exports', 'module', 'define', 'require', source)();
  const g = /** @type {any} */ (globalThis);
  const lib = g[name];
  delete g[name];
  return lib;
}

const RECORD = { kind: 'channel', unit: 'A', description: 'Main current' };
const RENDITION = '/attachments/att-0123456789ab/rendition';
const RENDITION_SRC = '/api/attachments/att-0123456789ab/rendition';

/** The entry's own attachments: one picture the page can show, one file it cannot. */
const ATTACHMENTS = [
  {
    attachment_id: 'att-0123456789ab',
    filename: 'beam profile.png',
    viewable: true,
    display_url: RENDITION,
    url: 'https://logbook.example.org/files/scan-17.png',
  },
  {
    attachment_id: 'att-pdf',
    filename: 'report.pdf',
    viewable: false,
    display_url: '/attachments/att-pdf',
  },
];

const REFERENCES = [
  { start: 0, end: 6, target: 'SR:DEV', records: [RECORD] },
  { start: 0, end: 4, target: 'GONE', records: [] },
  { start: 0, end: 11, target: 'SR:DEV:FULL', records: [RECORD] },
];

/** @type {{ marked: any, DOMPurify: any }} */
let real;

beforeAll(() => {
  real = {
    marked: loadClassicScript('marked'),
    DOMPurify: loadClassicScript('DOMPurify')(window),
  };
});

beforeEach(() => {
  const g = /** @type {any} */ (globalThis);
  g.marked = real.marked;
  g.DOMPurify = real.DOMPurify;
});

afterEach(() => {
  const g = /** @type {any} */ (globalThis);
  delete g.marked;
  delete g.DOMPurify;
  delete g.__arielPwned;
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

/**
 * Render a body and parse the result the way the reader's `innerHTML` does.
 * @param {string} text
 * @param {{references?: any[], attachments?: any[]}} [entry]
 * @returns {HTMLElement}
 */
function render(text, entry = {}) {
  const html = renderEntryMarkdown(text, entry);
  expect(html).not.toBeNull();
  const div = document.createElement('div');
  div.innerHTML = /** @type {string} */ (html);
  document.body.append(div);
  return div;
}

/**
 * Nothing under `root` can run script or reach this page's handlers.
 * @param {ParentNode} root
 */
function expectInert(root) {
  for (const tag of ['script', 'iframe', 'style', 'form', 'input', 'button', 'svg', 'object']) {
    expect(root.querySelector(tag), tag).toBeNull();
  }
  for (const el of root.querySelectorAll('*')) {
    for (const attr of Array.from(el.attributes)) {
      expect(attr.name.startsWith('on'), `${el.tagName} carries ${attr.name}`).toBe(false);
      expect(attr.name === 'style' || attr.name === 'id', `${el.tagName} ${attr.name}`).toBe(false);
      if (attr.name === 'href' || attr.name === 'src') {
        expect(attr.value.trim().toLowerCase()).not.toMatch(/^(javascript|data|vbscript):/);
      }
    }
  }
  expect(/** @type {any} */ (globalThis).__arielPwned).toBeUndefined();
}

describe('hostile entry text', () => {
  test('a script element does not survive', () => {
    const body = render('# Shift\n\n<script>globalThis.__arielPwned = 1</script>\n\nbeam back');
    expectInert(body);
    expect(body.querySelector('h1')?.textContent).toBe('Shift');
    expect(body.textContent).toContain('beam back');
    expect(body.textContent).not.toContain('__arielPwned');
  });

  test('an onerror attribute does not survive, on a raw picture or any element', () => {
    const body = render(
      'see <img src="x" onerror="globalThis.__arielPwned=1"> and ' +
        '<b onmouseover="globalThis.__arielPwned=1">bold</b>'
    );
    expectInert(body);
    expect(body.querySelector('img')).toBeNull();
    expect(body.querySelector('b')?.textContent).toBe('bold');
  });

  test.each([
    ['a javascript: link', '[click](javascript:globalThis.__arielPwned=1)'],
    ['a raw javascript: anchor', '<a href="JavaScript:globalThis.__arielPwned=1">click</a>'],
    ['a data: link', '[click](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)'],
    ['a protocol-relative link', '<a href="//evil.example/x">click</a>'],
    ['a relative link', '[click](/api/entries)'],
  ])('%s keeps its text and loses its address', (_name, text) => {
    const body = render(text);
    expectInert(body);
    const link = body.querySelector('a');
    expect(link?.textContent).toBe('click');
    expect(link?.hasAttribute('href')).toBe(false);
    expect(link?.hasAttribute('target')).toBe(false);
  });

  test('an http(s) link opens in a new tab and names no opener', () => {
    const link = render('[the wiki](https://wiki.example.org/page "Wiki")').querySelector('a');
    expect(link?.getAttribute('href')).toBe('https://wiki.example.org/page');
    expect(link?.getAttribute('target')).toBe('_blank');
    expect(link?.getAttribute('rel')).toBe('noopener noreferrer');
    expect(link?.getAttribute('title')).toBe('Wiki');
  });

  test.each([
    ['a remote picture', '![tracker](https://evil.example/pixel.png)'],
    [
      'a remote picture named like an attachment',
      '![tracker](https://evil.example/beam%20profile.png)',
    ],
    ['a protocol-relative picture', '![tracker](//evil.example/beam%20profile.png)'],
    ['a data: picture', '![tracker](data:image/png;base64,iVBORw0KGgo=)'],
    ['a picture of another entry by route', '![tracker](/attachments/att-of-another-entry)'],
    ['a picture of another entry by API path', '![tracker](/api/attachments/att-other/rendition)'],
    ['a picture of another entry by name', '![tracker](other-entry.png)'],
    ['an attachment the page cannot show', '![tracker](report.pdf)'],
    ['a raw remote picture', '<img src="https://evil.example/pixel.png" alt="tracker">'],
  ])('%s is removed and its alt text kept', (_name, text) => {
    const html = /** @type {string} */ (renderEntryMarkdown(text, { attachments: ATTACHMENTS }));
    expect(html).not.toContain('evil.example');
    expect(html).not.toContain('<img');
    const body = render(text, { attachments: ATTACHMENTS });
    expectInert(body);
    expect(body.textContent?.trim()).toBe('tracker');
  });

  test('the text cannot carry what the page styles or its click handlers read', () => {
    const body = render(
      '<span id="entry-subject" class="btn btn-primary" name="subject" style="position:fixed"' +
        ' data-follow-up="e-1" data-thread-entry="e-2" data-reader-action="ask"' +
        ' data-lightbox-url="https://evil.example/pixel.png">click</span>\n\n' +
        '<form action="https://evil.example"><input name="auth_password"><button>Go</button></form>\n\n' +
        '<style>.ariel-shell { display: none }</style>\n\n' +
        '<iframe src="https://evil.example"></iframe>\n\n' +
        '<svg onload="globalThis.__arielPwned=1"><circle r="4"/></svg>'
    );
    expectInert(body);
    const span = body.querySelector('span');
    expect(span?.textContent).toBe('click');
    expect(span?.getAttributeNames()).toEqual([]);
    expect(body.querySelector('[data-lightbox-url]')).toBeNull();
  });

  test('an attachment name or record text cannot leave its attribute', () => {
    const body = render('![shot](x.png) and SR:DEV', {
      attachments: [
        {
          attachment_id: 'a"><script>globalThis.__arielPwned=1</script>',
          filename: 'x.png',
          viewable: true,
          display_url: RENDITION,
        },
        {
          attachment_id: 'att-2',
          filename: 'y" onerror="globalThis.__arielPwned=1.png',
          viewable: true,
          display_url: RENDITION,
        },
      ],
      references: [
        {
          target: 'SR:DEV',
          records: [{ kind: '"><img src=x onerror=globalThis.__arielPwned=1>', unit: 'A' }],
        },
      ],
    });
    expectInert(body);
    const img = body.querySelector('img');
    expect(img?.getAttribute('data-attachment-id')).toBe(
      'a"><script>globalThis.__arielPwned=1</script>'
    );
    expect(body.querySelectorAll('img')).toHaveLength(1);
    expect(body.querySelector('.entry-facility-link')?.getAttribute('title')).toBe(
      '"><img src=x onerror=globalThis.__arielPwned=1> · A'
    );
  });
});

describe('structure', () => {
  test('a table is a table', () => {
    const body = render('| Magnet | Current |\n| :-- | --: |\n| QF1 | 12.5 |\n| QD1 | 11.0 |');
    expect(body.querySelectorAll('table thead th')).toHaveLength(2);
    expect(body.querySelectorAll('table tbody tr')).toHaveLength(2);
    expect(body.querySelector('tbody td')?.textContent).toBe('QF1');
    expect(body.querySelectorAll('th')[1].getAttribute('align')).toBe('right');
  });

  test('a nested list keeps its levels', () => {
    const body = render('- vacuum\n  - valve closed\n  - pump on\n- RF\n\n1. first\n2. second');
    expect(body.querySelectorAll(':scope > ul > li')).toHaveLength(2);
    const inner = Array.from(body.querySelectorAll('ul > li > ul > li')).map((li) =>
      li.textContent?.trim()
    );
    expect(inner).toEqual(['valve closed', 'pump on']);
    expect(body.querySelectorAll('ol > li')).toHaveLength(2);
  });

  test('headings, emphasis, code and quotes come out as elements', () => {
    const body = render('## Summary\n\n**beam** was *lost*\n\n> as reported\n\n```\ncaget X\n```');
    expect(body.querySelector('h2')?.textContent).toBe('Summary');
    expect(body.querySelector('strong')?.textContent).toBe('beam');
    expect(body.querySelector('em')?.textContent).toBe('lost');
    expect(body.querySelector('blockquote')?.textContent?.trim()).toBe('as reported');
    expect(body.querySelector('pre code')?.textContent).toBe('caget X\n');
  });
});

describe('inline pictures', () => {
  test('a picture naming an attachment of the entry shows that attachment', () => {
    const body = render('before\n\n![Beam profile](beam%20profile.png)\n\nafter', {
      attachments: ATTACHMENTS,
    });
    const img = /** @type {HTMLImageElement} */ (body.querySelector('img'));
    expect(img.getAttribute('src')).toBe(RENDITION_SRC);
    expect(img.getAttribute('alt')).toBe('Beam profile');
    expect(img.className).toBe('entry-inline-image');
    expect(img.dataset.lightboxUrl).toBe(RENDITION);
    expect(img.dataset.lightboxName).toBe('beam profile.png');
    expect(img.dataset.attachmentId).toBe('att-0123456789ab');
    expect(img.closest('p')?.previousElementSibling?.textContent).toBe('before');
  });

  test.each([
    ['a relative path ending in the file name', '![](media/beam%20profile.png)'],
    ['the last path part of the attachment source url', '![](scan-17.png)'],
    ['an address with a query', '![](beam%20profile.png?width=400)'],
  ])('%s names the attachment', (_name, text) => {
    const img = render(text, { attachments: ATTACHMENTS }).querySelector('img');
    expect(img?.getAttribute('src')).toBe(RENDITION_SRC);
    expect(img?.getAttribute('alt')).toBe('beam profile.png');
  });

  test('an attachment whose display_url is not safe is not shown', () => {
    const body = render('![shot](x.png)', {
      attachments: [{ filename: 'x.png', viewable: true, display_url: 'javascript:alert(1)' }],
    });
    expect(body.querySelector('img')).toBeNull();
    expect(body.textContent?.trim()).toBe('shot');
  });

  test('a click on the picture opens the lightbox through the reader delegation', () => {
    const container = document.createElement('div');
    document.body.append(container);
    renderEntryDetail(container, {
      entry_id: 'e-1',
      raw_text: 'Scan\n\n![Beam profile](beam%20profile.png)',
      metadata: { body_format: 'markdown' },
      attachments: ATTACHMENTS,
    });
    const text = container.querySelector('.entry-detail-text');
    expect(text?.className).toBe('entry-detail-text entry-detail-text--rendered');
    /** @type {HTMLElement} */ (container.querySelector('img.entry-inline-image')).click();
    const lightbox = document.getElementById('image-lightbox');
    expect(lightbox?.querySelector('img')?.getAttribute('src')).toBe(RENDITION_SRC);
    expect(
      lightbox?.querySelector('[data-use-in-workspace]')?.getAttribute('data-attachment-id')
    ).toBe('att-0123456789ab');
  });
});

describe('facility references', () => {
  test('an id inside a table cell is marked', () => {
    const body = render('| Device | State |\n| --- | --- |\n| SR:DEV | tripped |\n| GONE | ? |', {
      references: REFERENCES,
    });
    const mark = body.querySelector('td .entry-facility-link');
    expect(mark?.textContent).toBe('SR:DEV');
    expect(mark?.getAttribute('title')).toBe('channel · A · Main current');
    const dangling = body.querySelector('td .entry-facility-link--dangling');
    expect(dangling?.textContent).toBe('GONE');
    expect(dangling?.getAttribute('title')).toBe('GONE · not in the facility file');
  });

  test('the longest id wins and an id inside a longer word is not marked', () => {
    const body = render('SR:DEV:FULL then SR:DEV2 and xSR:DEV, GONE_2 but (GONE).', {
      references: REFERENCES,
    });
    const marks = Array.from(
      body.querySelectorAll('.entry-facility-link, .entry-facility-link--dangling')
    ).map((el) => el.textContent);
    expect(marks).toEqual(['SR:DEV:FULL', 'GONE']);
  });

  test('an id in a link address is left alone, so the link works; its text is marked', () => {
    const body = render('[SR:DEV page](https://wiki.example.org/SR:DEV)', {
      references: REFERENCES,
    });
    const link = body.querySelector('a');
    expect(link?.getAttribute('href')).toBe('https://wiki.example.org/SR:DEV');
    expect(link?.querySelector('.entry-facility-link')?.textContent).toBe('SR:DEV');
    expect(body.querySelectorAll('.entry-facility-link')).toHaveLength(1);
  });

  test('an id in code is marked, as the server counts it wherever it stands', () => {
    const body = render('run `caget SR:DEV`\n\n```\ncaput SR:DEV 1\n```', {
      references: REFERENCES,
    });
    expect(body.querySelector('p code .entry-facility-link')?.textContent).toBe('SR:DEV');
    expect(body.querySelector('pre code .entry-facility-link')?.textContent).toBe('SR:DEV');
    expect(body.querySelector('pre code')?.textContent).toBe('caput SR:DEV 1\n');
  });

  test('an id that holds pattern characters is matched as written', () => {
    const body = render('see BL[1].X+Y(2) here', {
      references: [{ target: 'BL[1].X+Y(2)', records: [RECORD] }],
    });
    expect(body.querySelector('.entry-facility-link')?.textContent).toBe('BL[1].X+Y(2)');
  });

  test.each(['SR01-BPM1', 'SR01C/BPM-1:X^2', 'A$B|C{1}?*\\D'])(
    'the id %j builds a valid pattern and is matched as written',
    (target) => {
      const html = renderEntryMarkdown(`before\n\n\`\`\`\n${target}\n\`\`\`\n`, {
        references: [{ target, records: [RECORD] }],
      });
      expect(html).not.toBeNull();
      const body = document.createElement('div');
      body.innerHTML = /** @type {string} */ (html);
      expect(body.querySelector('.entry-facility-link')?.textContent).toBe(target);
    }
  );

  test('an entry that quotes a hyphenated id is rendered, not shown as plain text', () => {
    const rawText = 'Orbit\n\n| Monitor |\n| --- |\n| SR01-BPM1 |';
    const start = rawText.indexOf('SR01-BPM1');
    const { bodyHtml, rendered } = renderEntryTextHtml({
      raw_text: rawText,
      facility_references: [{ start, end: start + 9, target: 'SR01-BPM1', records: [RECORD] }],
      metadata: { body_format: 'markdown' },
    });
    expect(rendered).toBe(true);
    const body = document.createElement('div');
    body.innerHTML = bodyHtml;
    expect(body.querySelector('td .entry-facility-link')?.textContent).toBe('SR01-BPM1');
  });
});

describe('the format switch (renderEntryTextHtml)', () => {
  const rawText = 'Scan\n\n| Device |\n| --- |\n| SR:DEV |';

  test.each(['markdown', 'Markdown', 'MARKDOWN'])('body_format %j renders the body', (format) => {
    const start = rawText.indexOf('SR:DEV');
    const parser = vi.spyOn(real.marked, 'Marked');
    const { titleHtml, bodyHtml, rendered } = renderEntryTextHtml({
      raw_text: rawText,
      facility_references: [{ start, end: start + 6, target: 'SR:DEV', records: [RECORD] }],
      metadata: { body_format: format },
    });
    expect(rendered).toBe(true);
    expect(parser).toHaveBeenCalledTimes(1);
    expect(titleHtml).toBe('Scan');
    expect(bodyHtml).toContain('<table>');
    expect(bodyHtml).toContain('<span class="entry-facility-link"');
  });

  test.each([undefined, null, '', 'html', 'plain', 'markdown ', 'md', 7, ['markdown']])(
    'body_format %j keeps the stored text, with both libraries loaded',
    (format) => {
      const parser = vi.spyOn(real.marked, 'Marked');
      const sanitize = vi.spyOn(real.DOMPurify, 'sanitize');
      const { bodyHtml, rendered } = renderEntryTextHtml({
        raw_text: rawText,
        metadata: { body_format: format },
      });
      expect(rendered).toBe(false);
      expect(bodyHtml).toBe('| Device |\n| --- |\n| SR:DEV |');
      expect(parser).not.toHaveBeenCalled();
      expect(sanitize).not.toHaveBeenCalled();
    }
  );

  test.each([
    ['marked is not loaded', () => delete (/** @type {any} */ (globalThis).marked)],
    ['DOMPurify is not loaded', () => delete (/** @type {any} */ (globalThis).DOMPurify)],
    [
      'DOMPurify has no hooks',
      () => {
        /** @type {any} */ (globalThis).DOMPurify = { sanitize: real.DOMPurify.sanitize };
      },
    ],
    [
      'the sanitiser throws',
      () => {
        vi.spyOn(real.DOMPurify, 'sanitize').mockImplementation(() => {
          throw new Error('boom');
        });
      },
    ],
  ])('a Markdown entry shows as escaped plain text when %s', (_name, breakIt) => {
    breakIt();
    const hostile = 'Scan\n\n<img src=x onerror=alert(1)> SR:DEV';
    expect(renderEntryMarkdown('<b>x</b>')).toBeNull();
    const start = hostile.indexOf('SR:DEV');
    const entry = {
      raw_text: hostile,
      facility_references: [{ start, end: start + 6, target: 'SR:DEV', records: [RECORD] }],
      metadata: { body_format: 'markdown' },
    };
    const { bodyHtml, rendered } = renderEntryTextHtml(entry);
    expect(rendered).toBe(false);
    expect(bodyHtml).toBe(renderEntryText(hostile, entry.facility_references, 6, hostile.length));
    expect(bodyHtml).toContain('<span class="entry-facility-link"');
    expect(bodyHtml).toContain('&lt;img src=x onerror=alert(1)&gt;');
  });

  test('the hook is taken off DOMPurify again', () => {
    const removeHook = vi.spyOn(real.DOMPurify, 'removeHook');
    renderEntryMarkdown('[a](https://example.org)');
    expect(removeHook).toHaveBeenCalledTimes(1);
    const div = document.createElement('div');
    div.innerHTML = real.DOMPurify.sanitize('<a href="/relative">a</a>');
    expect(div.querySelector('a')?.getAttribute('href')).toBe('/relative');
  });
});
