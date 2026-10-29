// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode chat: SVG callouts in agent replies.
 *
 * A fenced ```svg block in an agent reply is drawn as the picture it holds: a
 * quick look the agent made for the operator (a few top results, a small
 * schematic), set in a box with its source one click away. The full result is
 * the artifact the reply links to; the callout is a glance at it.
 *
 * The SVG is model output, so it is untrusted, and it is placed in the page
 * itself (not as an `<img>`) so it can draw in the theme's colours. Before it
 * is placed it must:
 * - parse as one complete SVG document, so a block still streaming, or a
 *   broken one, stays a code block;
 * - pass DOMPurify's SVG profile, with every element that runs code, embeds
 *   HTML, animates or loads another resource removed outright, and no `href`
 *   at all;
 * - keep no attribute whose value reaches outside the drawing: a value may
 *   hold `url(#id)` references to the drawing's own parts (a gradient stays),
 *   no other `url()`, `src()` or `image-set()`, and no backslash, because a
 *   CSS escape (`\75rl(…)`) spells a function name the checks cannot see;
 * - keep no colour outside the theme's palette ({@link PALETTE}) and no
 *   `style` attribute, so every colour it draws in is one the theme sets and
 *   it reads in light and dark alike. A part whose colour was dropped, or
 *   that never had one, draws in the reply's text colour (operator.css sets
 *   the drawing's fill to `currentColor`; SVG's own default is black).
 *
 * @module chat-svg
 */

/** Elements a callout may not hold: code, embedded HTML, links, loads, animation. */
const FORBID_TAGS = Object.freeze([
  'script',
  'style',
  'foreignObject',
  'a',
  'image',
  'use',
  'feImage',
  'animate',
  'animateMotion',
  'animateTransform',
  'set',
  'iframe',
]);

/** Attributes a callout may not carry: every link and resource reference, and inline styles. */
const FORBID_ATTR = Object.freeze(['href', 'xlink:href', 'src', 'style']);

/** The attributes that set a colour. */
const COLOUR_ATTRS = Object.freeze([
  'fill',
  'stroke',
  'stop-color',
  'flood-color',
  'lighting-color',
  'color',
]);

/** The theme tokens a drawing may colour with: the agent is given this list. */
export const PALETTE = Object.freeze([
  'text-primary',
  'text-secondary',
  'text-muted',
  'border-default',
  'bg-elevated',
  'color-accent',
  'color-accent-secondary',
  'color-success',
  'color-warning',
  'color-error',
]);

/** A colour value the theme sets: a palette token, the text colour, none, or the drawing's own gradient. */
const THEME_COLOUR = new RegExp(
  `^(?:none|currentColor|var\\(--(?:${PALETTE.join('|')})\\)|url\\(#[\\w-]+\\))$`,
  'i'
);

/** A `url()` that points at the drawing's own `#id`, the one form a value may use. */
const OWN_URL = /url\(\s*(['"]?)#[^)'"\s]*\1\s*\)/gi;

/** A CSS function that can load a resource. */
const LOADING_FUNCTION = /(?:url|src|image|image-set|cross-fade|element)\s*\(/i;

/**
 * Whether *value* may stay on an attribute: no backslash, and no resource
 * loaded from anywhere but the drawing's own `url(#id)` parts.
 * @param {string} value
 * @returns {boolean}
 */
function staysInsideDrawing(value) {
  return !value.includes('\\') && !LOADING_FUNCTION.test(value.replace(OWN_URL, ''));
}

/** Longer sources stay code: a callout is a glance, not a document. */
export const MAX_SVG_SOURCE = 50_000;

/**
 * The cleaned `<svg>` element for *source*, or null when it is not one
 * complete, safe drawing.
 * @param {string} source - the fenced block's text
 * @param {any} DOMPurify - the page's sanitiser
 * @returns {SVGSVGElement | null}
 */
export function sanitizeSvg(source, DOMPurify) {
  if (!source.trim() || source.length > MAX_SVG_SOURCE) return null;
  const parsed = new DOMParser().parseFromString(source, 'image/svg+xml');
  if (
    parsed.getElementsByTagName('parsererror').length > 0 ||
    parsed.documentElement?.localName !== 'svg'
  ) {
    return null;
  }
  const fragment = DOMPurify.sanitize(source, {
    USE_PROFILES: { svg: true, svgFilters: true },
    FORBID_TAGS: [...FORBID_TAGS],
    FORBID_ATTR: [...FORBID_ATTR],
    RETURN_DOM_FRAGMENT: true,
  });
  const svg = /** @type {SVGSVGElement | null} */ (fragment.querySelector('svg'));
  if (!svg) return null;
  for (const node of [svg, ...svg.querySelectorAll('*')]) {
    for (const attr of [...node.attributes]) {
      if (!staysInsideDrawing(attr.value)) node.removeAttribute(attr.name);
      else if (COLOUR_ATTRS.includes(attr.name) && !THEME_COLOUR.test(attr.value.trim())) {
        node.removeAttribute(attr.name);
      }
    }
  }
  // Drawn no wider than the agent drew it, and never wider than the reply.
  const box = svg
    .getAttribute('viewBox')
    ?.trim()
    .split(/[\s,]+/);
  if (!svg.hasAttribute('width') && box?.length === 4 && Number(box[2]) > 0) {
    svg.setAttribute('width', box[2]);
  }
  if (!svg.hasAttribute('role')) svg.setAttribute('role', 'img');
  return svg;
}

/** A fence line: up to three spaces, three or more backticks or tildes, then its info string. */
const FENCE_LINE = /^ {0,3}(`{3,}|~{3,})([^\n]*)$/gm;

/**
 * Where a fenced `svg` block that is still streaming opens in *text*, or -1
 * when every `svg` block in it is closed. A fence on the last line whose info
 * string so far could still become `svg` counts as one, so the code does not
 * flash up while the word is arriving.
 * @param {string} text - a reply's markdown so far
 * @returns {number}
 */
export function openSvgFenceAt(text) {
  /** @type {{ marker: string, svg: boolean, at: number } | null} */
  let open = null;
  for (const m of text.matchAll(FENCE_LINE)) {
    const [line, marker, info] = m;
    const at = m.index ?? 0;
    if (open === null) {
      const lang = info.trim().split(/\s+/)[0] ?? '';
      const last = at + line.length === text.length;
      open = { marker, at, svg: lang === 'svg' || (last && 'svg'.startsWith(lang)) };
    } else if (
      marker[0] === open.marker[0] &&
      marker.length >= open.marker.length &&
      !info.trim()
    ) {
      open = null;
    }
  }
  return open?.svg ? open.at : -1;
}

/**
 * The box an `svg` block takes while it streams: the callout's frame, with a
 * line saying the drawing is on its way.
 * @returns {HTMLElement}
 */
export function createSvgPlaceholder() {
  const figure = document.createElement('figure');
  figure.className = 'op-svg op-svg-pending';
  const label = document.createElement('span');
  label.className = 'op-svg-pending-label op-shimmer';
  label.textContent = 'Drawing…';
  figure.append(label);
  return figure;
}

/**
 * Replace every complete fenced `svg` code block under *root* with its drawing.
 * A block that does not pass {@link sanitizeSvg} is left as code.
 * @param {HTMLElement} root - a rendered, already sanitised reply
 * @param {any} DOMPurify
 * @returns {void}
 */
export function mountSvgCallouts(root, DOMPurify) {
  for (const code of root.querySelectorAll('pre > code.language-svg')) {
    const pre = /** @type {HTMLElement} */ (code.parentElement);
    const source = (code.textContent ?? '').trim();
    const svg = sanitizeSvg(source, DOMPurify);
    if (!svg) continue;

    const figure = document.createElement('figure');
    figure.className = 'op-svg';
    const art = document.createElement('div');
    art.className = 'op-svg-art';
    art.append(svg);

    const details = document.createElement('details');
    details.className = 'op-svg-source';
    const summary = document.createElement('summary');
    summary.textContent = 'Source';
    const sourcePre = document.createElement('pre');
    const sourceCode = document.createElement('code');
    sourceCode.className = 'language-xml';
    sourceCode.textContent = source;
    sourcePre.append(sourceCode);
    details.append(summary, sourcePre);

    figure.append(art, details);
    pre.replaceWith(figure);
  }
}
