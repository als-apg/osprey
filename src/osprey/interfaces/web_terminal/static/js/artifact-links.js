// @ts-check
/**
 * Artifact ids in the chat's agent answers, made into controls that
 * show the artifact in the workspace.
 *
 * Invariants:
 * - Only the phrase `artifact_id: <hex id>` (or `artifact_id=<hex id>`) becomes
 *   a control, and it reads exactly as the agent wrote it. The id may also sit
 *   in an inline code span right after `artifact_id: `.
 * - Text inside a link, a button or a code block is left alone.
 * - Linking a body twice changes nothing more: a streamed answer is linked
 *   again after every render.
 * - A click asks the gallery to focus the artifact, the same request the
 *   agent's `artifact_focus` makes, and opens the artifacts panel only when
 *   the gallery accepts it. When it does not, the control says why: the
 *   gallery does not hold the id, the session expired, the server refused,
 *   or the gallery could not be reached.
 */

/** A hex artifact id, as the artifact store mints them. */
const ID = '[0-9a-f]{6,32}';

/** The phrase, whole, inside one text node. */
const PHRASE = new RegExp(`\\bartifact_id\\s*[:=]\\s*(${ID})\\b`, 'gi');

/** The phrase's lead-in, ending a text node whose next sibling is the id in code. */
const LEAD_IN = /\bartifact_id\s*[:=]\s*$/i;

/** An inline code span that holds an id and nothing else. */
const CODE_ID = new RegExp(`^${ID}$`, 'i');

/** Where text is not to be touched. A drawing's text stays SVG: a button cannot sit in it. */
const SKIP = 'a, button, pre, .katex, svg';

/**
 * @param {string} id
 * @param {string} text
 * @returns {HTMLButtonElement}
 */
function control(id, text) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'op-artifact-link';
  b.dataset.artifactId = id.toLowerCase();
  b.title = 'Show this artifact in the workspace';
  b.textContent = text;
  return b;
}

/**
 * Turn every artifact-id phrase under `root` into an artifact control.
 * @param {HTMLElement} root - a rendered agent answer
 * @returns {void}
 */
export function linkArtifactIds(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.parentElement?.closest(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
  });
  /** @type {Text[]} */
  const nodes = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) nodes.push(/** @type {Text} */ (n));

  for (const node of nodes) {
    const text = node.data;
    // The text left after the phrases, which may still end in a lead-in whose
    // id sits in the code span that follows.
    let tail = node;
    PHRASE.lastIndex = 0;
    if (PHRASE.test(text)) {
      PHRASE.lastIndex = 0;
      const frag = document.createDocumentFragment();
      let at = 0;
      for (const m of text.matchAll(PHRASE)) {
        const start = m.index ?? 0;
        frag.append(text.slice(at, start), control(m[1], m[0]));
        at = start + m[0].length;
      }
      tail = document.createTextNode(text.slice(at));
      frag.append(tail);
      node.replaceWith(frag);
    }
    const code = tail.nextSibling;
    const lead = LEAD_IN.exec(tail.data);
    if (
      lead &&
      code instanceof HTMLElement &&
      code.tagName === 'CODE' &&
      CODE_ID.test(code.textContent ?? '')
    ) {
      const b = control(code.textContent ?? '', lead[0]);
      tail.data = tail.data.slice(0, lead.index);
      code.replaceWith(b);
      b.append(code);
    }
  }
}

/**
 * How a request to focus an artifact ended: it is shown, or why it is not.
 * @typedef {'shown' | 'missing' | 'expired' | 'refused' | 'unreachable'} FocusOutcome
 */

/** The control's tooltip for a focus that failed, by outcome. */
export const FOCUS_FAILURE_TITLE = Object.freeze({
  missing: 'This artifact is not in the gallery',
  expired: 'Your session has expired. Sign in again.',
  refused: 'The server refused this request',
  unreachable: 'The gallery could not be reached',
});

/**
 * Route clicks on artifact controls under `root`.
 * @param {HTMLElement} root
 * @param {{ focusArtifact: (id: string) => Promise<FocusOutcome> }} deps -
 *   resolves 'shown' once the artifact is shown, or the reason it cannot be
 * @returns {void}
 */
export function wireArtifactLinks(root, deps) {
  root.addEventListener('click', (e) => {
    const target = /** @type {Element | null} */ (e.target);
    const b =
      target && typeof target.closest === 'function' ? target.closest('.op-artifact-link') : null;
    if (!(b instanceof HTMLButtonElement) || !root.contains(b)) return;
    e.preventDefault();
    const id = b.dataset.artifactId ?? '';
    if (!id) return;
    void deps.focusArtifact(id).then((outcome) => {
      delete b.dataset.missing;
      delete b.dataset.unavailable;
      if (outcome === 'shown') {
        b.title = 'Show this artifact in the workspace';
        return;
      }
      if (outcome === 'missing') b.dataset.missing = 'true';
      else b.dataset.unavailable = outcome;
      b.title = FOCUS_FAILURE_TITLE[outcome];
    });
  });
}
