/**
 * Unit tests for chat-render.js -- the operator-chat message-list renderer.
 *
 * Two concerns:
 *   1. `renderMarkdownInto` -- the chat XSS trust boundary. Model output (incl.
 *      untrusted tool results) must pass through DOMPurify before any HTML
 *      reaches the DOM, and must degrade to inert text when the vendored libs
 *      are absent.
 *   2. `createChatRenderer` -- the view-model: user/agent entries, streamed
 *      text accumulation, the activity-line state machine, first-turn
 *      session_reset suppression, and error rendering.
 *
 * On the XSS tests: the *real* vendored DOMPurify cannot be exercised here --
 * under happy-dom (the vitest env) DOMPurify mis-parses and leaks `<script>`
 * while stripping legitimate tags, so its output is meaningless in this
 * environment (it works correctly in a real browser). We therefore assert the
 * *seam* deterministically: (a) the marked-produced HTML is handed to
 * `DOMPurify.sanitize` and ONLY its return ever reaches `innerHTML` -- so a
 * hostile payload can never reach the DOM un-sanitised (`onlySanitizedReachesDOM`);
 * (b) with a representative stripping sanitiser the rendered DOM carries no
 * executable vector; and (c) with DOMPurify absent the payload is written as
 * inert text, never markup. Together these pin every path by which model HTML
 * could reach the DOM.
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-render.test.mjs
 */

import { test, expect, describe, vi, beforeEach, afterEach } from 'vitest';

import { qs } from '../_support/dom.mjs';

import {
  renderMarkdownInto,
  createChatRenderer,
  buildUserEntry,
  buildAgentEntry,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-render.js';

/** A markdown parser stub that passes text through as-is (raw HTML included, as marked does). */
const passthroughMarked = { parse: /** @param {string} t */ (t) => t };

/** A DOMPurify stub that returns its input unchanged -- for view-model tests where sanitisation isn't under test. */
const identityPurify = { sanitize: /** @param {string} h */ (h) => h };

/**
 * A representative stripping sanitiser: drops `<script>` elements and inline
 * `on*=` event-handler attributes -- the two vectors the XSS payloads probe.
 * Stands in for the real DOMPurify, which happy-dom cannot run faithfully.
 */
const strippingPurify = {
  sanitize: /** @param {string} h */ (h) =>
    h
      .replace(/<script[\s\S]*?<\/script>/gi, '')
      .replace(/<script[^>]*>/gi, '')
      .replace(/\son\w+\s*=\s*"[^"]*"/gi, '')
      .replace(/\son\w+\s*=\s*'[^']*'/gi, '')
      .replace(/\son\w+\s*=\s*[^\s>]+/gi, ''),
};

/** @returns {HTMLElement} a fresh detached container element */
function freshEl() {
  return document.createElement('div');
}

afterEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// renderMarkdownInto -- the sanitisation seam
// ---------------------------------------------------------------------------

describe('renderMarkdownInto', () => {
  test('renders sanitised marked output into innerHTML and highlights code blocks', () => {
    const sanitize = vi.fn(/** @param {string} h */ (h) => h);
    const parse = vi.fn(() => '<p>hi</p><pre><code>x</code></pre>');
    const highlightElement = vi.fn();
    vi.stubGlobal('marked', { parse });
    vi.stubGlobal('DOMPurify', { sanitize });
    vi.stubGlobal('hljs', { highlightElement });

    const el = freshEl();
    renderMarkdownInto(el, '# hi');

    expect(parse).toHaveBeenCalledWith('# hi');
    expect(sanitize).toHaveBeenCalledWith('<p>hi</p><pre><code>x</code></pre>');
    expect(el.innerHTML).toBe('<p>hi</p><pre><code>x</code></pre>');
    // one `pre code` block -> highlighted once
    expect(highlightElement).toHaveBeenCalledTimes(1);
  });

  test('onlySanitizedReachesDOM: hostile markdown flows through sanitize; only its output lands in the DOM', () => {
    // sanitize returns a fixed safe string; if the pipeline leaked, the hostile
    // markup (not this sentinel) would appear in the DOM.
    const sanitize = vi.fn(/** @type {(html: string) => string} */ (() => '<em>safe</em>'));
    vi.stubGlobal('marked', passthroughMarked);
    vi.stubGlobal('DOMPurify', { sanitize });
    vi.stubGlobal('hljs', undefined);

    const el = freshEl();
    const hostile = '<img src=x onerror=alert(1)><script>alert(2)</script>';
    renderMarkdownInto(el, hostile);

    // The hostile HTML was handed to the sanitiser...
    expect(sanitize).toHaveBeenCalledTimes(1);
    expect(sanitize.mock.calls[0][0]).toContain('onerror');
    expect(sanitize.mock.calls[0][0]).toContain('<script>');
    // ...and ONLY the sanitiser's output reached the DOM.
    expect(el.innerHTML).toBe('<em>safe</em>');
    expect(el.querySelector('script')).toBeNull();
    expect(el.querySelector('[onerror]')).toBeNull();
    expect(el.querySelector('img')).toBeNull();
  });

  test('with a stripping sanitiser the rendered DOM carries no script node or on* handler', () => {
    vi.stubGlobal('marked', passthroughMarked);
    vi.stubGlobal('DOMPurify', strippingPurify);
    vi.stubGlobal('hljs', undefined);

    const el = freshEl();
    renderMarkdownInto(
      el,
      '<p>ok</p><img src=x onerror="alert(1)"><script>alert(2)</script>'
    );

    expect(el.querySelector('script')).toBeNull();
    expect(el.querySelector('[onerror]')).toBeNull();
    const img = el.querySelector('img');
    if (img !== null) expect(img.hasAttribute('onerror')).toBe(false);
    // benign content is preserved
    expect(el.textContent).toContain('ok');
  });

  test('degrades to inert textContent when DOMPurify is absent (no unsanitised HTML)', () => {
    vi.stubGlobal('marked', passthroughMarked);
    vi.stubGlobal('DOMPurify', undefined);
    vi.stubGlobal('hljs', undefined);

    const el = freshEl();
    const hostile = '<img src=x onerror=alert(1)>hi';
    renderMarkdownInto(el, hostile);

    // Written as text, so no live nodes and the markup is HTML-escaped.
    expect(el.querySelector('img')).toBeNull();
    expect(el.querySelector('script')).toBeNull();
    expect(el.textContent).toBe(hostile);
    expect(el.innerHTML).toContain('&lt;img');
  });

  test('degrades to inert textContent when marked is absent', () => {
    vi.stubGlobal('marked', undefined);
    vi.stubGlobal('DOMPurify', identityPurify);
    vi.stubGlobal('hljs', undefined);

    const el = freshEl();
    renderMarkdownInto(el, '**bold**');
    expect(el.textContent).toBe('**bold**');
    expect(el.querySelector('strong')).toBeNull();
  });

  test('falls back to text when marked.parse throws', () => {
    vi.stubGlobal('marked', { parse: () => { throw new Error('boom'); } });
    vi.stubGlobal('DOMPurify', identityPurify);
    vi.stubGlobal('hljs', undefined);

    const el = freshEl();
    renderMarkdownInto(el, 'content');
    expect(el.textContent).toBe('content');
  });

  test('renders fenced code even when the global marked singleton is polluted', () => {
    // Regression: scaffold/utils.js calls `marked.use({ renderer, … })` on the
    // shared UMD singleton (loaded on every hub page), and that custom renderer
    // empties chat's fenced blocks to `<pre><code></code></pre>`. chat-render
    // must parse through its own isolated `Marked` instance so page-global
    // config can't reach it. This mock makes instance-vs-singleton behaviour
    // distinguishable: `use()` pollutes only the global `parse`, while the
    // `Marked` instance renders fences faithfully.
    const renderFences = /** @param {string} t */ (t) =>
      t.replace(
        /```(?:\w*)\n([\s\S]*?)```/g,
        (_m, code) => `<pre><code>${code}</code></pre>`
      );
    /** @type {any} */
    const marked = {
      // Fresh (unpolluted) singleton would render fences; `use()` breaks it.
      parse: renderFences,
      use() {
        // scaffold-style global mutation: empty every fenced block.
        marked.parse = /** @param {string} t */ (t) =>
          t.replace(/```(?:\w*)\n[\s\S]*?```/g, '<pre><code></code></pre>');
      },
      Marked: class {
        /** @param {string} t */
        parse(t) {
          return renderFences(t);
        }
      },
    };
    vi.stubGlobal('marked', marked);
    vi.stubGlobal('DOMPurify', identityPurify);
    vi.stubGlobal('hljs', undefined);

    // Simulate the scaffold hub configuring the shared singleton.
    marked.use({ renderer: {} });
    // Sanity: the polluted singleton now empties fenced blocks.
    expect(marked.parse('```\nconsole.log(1)\n```')).toBe(
      '<pre><code></code></pre>'
    );

    const el = freshEl();
    renderMarkdownInto(el, '```\nconsole.log(1)\n```');

    // chat used its own instance, so the fence content survived.
    const code = el.querySelector('pre code');
    expect(code).not.toBeNull();
    expect(code?.textContent).toContain('console.log(1)');
  });
});

// ---------------------------------------------------------------------------
// Pure builders
// ---------------------------------------------------------------------------

describe('DOM builders', () => {
  test('buildUserEntry: operator entry with prefix and plain-text body', () => {
    const entry = buildUserEntry('hello <b>there</b>');
    expect(entry.classList.contains('op-entry')).toBe(true);
    expect(entry.classList.contains('operator')).toBe(true);
    expect(qs(entry, '.op-entry-prefix').textContent).toBe('Operator');
    const body = qs(entry, '.op-entry-body');
    // plain text -- markup is not interpreted
    expect(body.textContent).toBe('hello <b>there</b>');
    expect(body.querySelector('b')).toBeNull();
  });

  test('buildAgentEntry: assistant entry whose body carries osprey-md-rendered', () => {
    const { entry, body } = buildAgentEntry();
    expect(entry.classList.contains('assistant')).toBe(true);
    expect(qs(entry, '.op-entry-prefix').textContent).toBe('Osprey');
    expect(body.classList.contains('op-entry-body')).toBe(true);
    expect(body.classList.contains('osprey-md-rendered')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// createChatRenderer -- view-model
// ---------------------------------------------------------------------------

describe('createChatRenderer', () => {
  /** @type {HTMLElement} */
  let container;

  beforeEach(() => {
    container = freshEl();
    // Sensible default globals for view-model tests; individual tests override.
    vi.stubGlobal('marked', passthroughMarked);
    vi.stubGlobal('DOMPurify', identityPurify);
    vi.stubGlobal('hljs', undefined);
  });

  test('addUserMessage appends an operator entry and bumps the message count', () => {
    const r = createChatRenderer(container);
    expect(r.messageCount()).toBe(0);
    r.addUserMessage('align the orbit');
    expect(r.messageCount()).toBe(1);
    const entry = qs(container, '.op-entry.operator');
    expect(qs(entry, '.op-entry-body').textContent).toBe('align the orbit');
  });

  test('text events create one agent entry and accumulate streamed text', () => {
    const r = createChatRenderer(container);
    r.handleEvent({ type: 'text', content: 'Hello ' });
    r.handleEvent({ type: 'text', content: 'world' });

    const agents = container.querySelectorAll('.op-entry.assistant');
    expect(agents.length).toBe(1);
    expect(qs(container, '.op-entry.assistant .op-entry-body').textContent).toBe('Hello world');
    expect(r.messageCount()).toBe(1);
  });

  test('a result event ends the turn so the next text starts a fresh agent entry', () => {
    const r = createChatRenderer(container);
    r.handleEvent({ type: 'text', content: 'first' });
    r.handleEvent({ type: 'result', is_error: false });
    r.handleEvent({ type: 'text', content: 'second' });

    const agents = container.querySelectorAll('.op-entry.assistant');
    expect(agents.length).toBe(2);
    expect(agents[0].querySelector('.op-entry-body')?.textContent).toBe('first');
    expect(agents[1].querySelector('.op-entry-body')?.textContent).toBe('second');
  });

  describe('activity-line state machine', () => {
    test('thinking marker shows "Thinking…"', () => {
      const r = createChatRenderer(container);
      r.handleEvent({ type: 'thinking' });
      const line = qs(container, '.op-processing');
      expect(line.classList.contains('active')).toBe(true);
      expect(qs(line, '.op-processing-label').textContent).toBe('Thinking…');
    });

    test('tool_use shows "Using <tool_name>…"', () => {
      const r = createChatRenderer(container);
      r.handleEvent({ type: 'tool_use', tool_name: 'Channel Read' });
      expect(qs(container, '.op-processing-label').textContent).toBe('Using Channel Read…');
    });

    test('tool_use without a tool_name falls back to a generic label', () => {
      const r = createChatRenderer(container);
      r.handleEvent({ type: 'tool_use' });
      expect(qs(container, '.op-processing-label').textContent).toBe('Using tool…');
    });

    test('text clears the activity line', () => {
      const r = createChatRenderer(container);
      r.handleEvent({ type: 'thinking' });
      expect(qs(container, '.op-processing').classList.contains('active')).toBe(true);
      r.handleEvent({ type: 'text', content: 'answer' });
      expect(qs(container, '.op-processing').classList.contains('active')).toBe(false);
    });

    test('the activity line stays pinned below the latest entry', () => {
      const r = createChatRenderer(container);
      r.addUserMessage('go');
      r.handleEvent({ type: 'tool_use', tool_name: 'Read' });
      expect(container.lastElementChild?.classList.contains('op-processing')).toBe(true);
    });
  });

  describe('session_reset divider suppression', () => {
    test('negative first-turn case: a reset on an empty list renders no divider', () => {
      const r = createChatRenderer(container);
      r.handleEvent({ type: 'session_reset' });
      expect(container.querySelector('.op-system')).toBeNull();
      expect(container.children.length).toBe(0);
    });

    test('real first turn: user bubble then a frame-0 reset renders no divider', () => {
      // The controller appends the operator bubble, then the fresh session's
      // stream opens with session_reset as frame 0. The operator's own prompt
      // must not count as prior history, so the divider is suppressed.
      const r = createChatRenderer(container);
      r.addUserMessage('align the orbit');
      r.handleEvent({ type: 'session_reset' });
      expect(container.querySelector('.op-system')).toBeNull();
      // the just-added user bubble is still the only entry
      expect(r.messageCount()).toBe(1);
    });

    test('reset on a later turn (after a completed exchange) renders a divider', () => {
      const r = createChatRenderer(container);
      // Turn 1: a full exchange completes.
      r.addUserMessage('read the beam current');
      r.handleEvent({ type: 'text', content: 'The beam current is 500 mA.' });
      r.handleEvent({ type: 'result', is_error: false });
      // Turn 2: new prompt, then the stream re-opens with a session_reset.
      r.addUserMessage('and the orbit?');
      r.handleEvent({ type: 'session_reset' });
      const divider = qs(container, '.op-system');
      expect(divider.textContent).toBe('session reset');
      // the divider is not a message: user + agent + user = 3
      expect(r.messageCount()).toBe(3);
    });

    test('the first reset of a conversation is suppressed even after a later one shows', () => {
      const r = createChatRenderer(container);
      r.handleEvent({ type: 'session_reset' }); // suppressed (empty)
      r.addUserMessage('hi');
      r.handleEvent({ type: 'session_reset' }); // still suppressed (no exchange yet)
      expect(container.querySelector('.op-system')).toBeNull();
      // Complete an exchange, then a reset on the next turn is worth a divider.
      r.handleEvent({ type: 'text', content: 'hello' });
      r.handleEvent({ type: 'result', is_error: false });
      r.addUserMessage('again');
      r.handleEvent({ type: 'session_reset' }); // shown
      expect(container.querySelectorAll('.op-system').length).toBe(1);
    });
  });

  describe('error rendering', () => {
    test('an error event renders an error block with its message', () => {
      const r = createChatRenderer(container);
      r.handleEvent({ type: 'error', message: 'API error: rate limited' });
      const block = qs(container, '.op-error-block');
      expect(block.textContent).toBe('API error: rate limited');
      expect(r.messageCount()).toBe(1);
    });

    test('an error event without a message uses a default', () => {
      const r = createChatRenderer(container);
      r.handleEvent({ type: 'error' });
      expect(qs(container, '.op-error-block').textContent).toBe('An error occurred.');
    });
  });

  test('unknown event types (e.g. system) render nothing', () => {
    const r = createChatRenderer(container);
    r.handleEvent({ type: 'system' });
    r.handleEvent({ type: 'keepalive' });
    expect(container.children.length).toBe(0);
    expect(r.messageCount()).toBe(0);
  });

  test('reset clears the list and all per-conversation state', () => {
    const r = createChatRenderer(container);
    r.addUserMessage('hi');
    r.handleEvent({ type: 'text', content: 'yo' });
    r.reset();
    expect(container.children.length).toBe(0);
    expect(r.messageCount()).toBe(0);
    // after reset, a session_reset is again a first-turn (suppressed)
    r.handleEvent({ type: 'session_reset' });
    expect(container.querySelector('.op-system')).toBeNull();
  });

  test('hostile model text routed via a text event stays inert in the DOM', () => {
    vi.stubGlobal('DOMPurify', strippingPurify);
    const r = createChatRenderer(container);
    r.handleEvent({
      type: 'text',
      content: 'result: <img src=x onerror="steal()"><script>evil()</script>',
    });
    const body = qs(container, '.op-entry.assistant .op-entry-body');
    expect(body.querySelector('script')).toBeNull();
    expect(body.querySelector('[onerror]')).toBeNull();
  });
});
