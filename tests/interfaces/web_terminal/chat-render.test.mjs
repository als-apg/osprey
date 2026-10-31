/**
 * Unit tests for chat-render.js -- the operator-chat message-list renderer.
 *
 * Four concerns:
 *   1. `renderMarkdownInto` -- the chat XSS trust boundary. Model output (incl.
 *      untrusted tool results) must pass through DOMPurify before any HTML
 *      reaches the DOM, and must degrade to inert text when the vendored libs
 *      are absent.
 *   2. The tool vocabulary -- name normalisation and the tool-name-to-phrase
 *      table behind the activity line, including the raw-name fallback that
 *      keeps an unmapped tool visible.
 *   3. `createChatRenderer` -- the view-model: user/agent entries, streamed
 *      text accumulation, the activity-line state machine, first-turn
 *      session_reset suppression, error rendering, and the transcript replay
 *      that repopulates the log after a handoff from the other view.
 *   4. Approval cards in the log -- a card waiting on a call is filed under that
 *      call's row, the row carries the decision, and a replayed call or a second
 *      tab finds the same row.
 *
 * On the XSS tests: the *real* vendored DOMPurify cannot be exercised here --
 * under happy-dom (the vitest env) DOMPurify mis-parses and leaks `<script>`
 * while stripping legitimate tags, so its output is meaningless in this
 * environment (it works correctly in a real browser). We therefore assert the
 * *seam* deterministically: (a) the marked-produced HTML is handed to
 * `DOMPurify.sanitize` and ONLY its return ever reaches `innerHTML` -- so a
 * hostile payload can never reach the DOM un-sanitised (`onlySanitizedReachesDOM`);
 * (b) the streamed and replayed agent paths both route through that seam
 * (a sentinel sanitiser's output is all that lands); and (c) with DOMPurify
 * absent the payload is written as inert text, never markup. Together these
 * pin every path by which model HTML could reach the DOM.
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-render.test.mjs
 */

import { test, expect, describe, vi, beforeEach, afterEach } from 'vitest';

import { qs } from '../_support/dom.mjs';

import { renderMarkdownInto } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-markdown.js';
import { createChatRenderer } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-render.js';
import {
  normaliseToolName,
  toolPhrase,
  activityLabel,
  TOOL_PHRASES,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-activity.js';

/** A markdown parser stub that passes text through as-is (raw HTML included, as marked does). */
const passthroughMarked = { parse: /** @param {string} t */ (t) => t };

/** A DOMPurify stub that returns its input unchanged -- for view-model tests where sanitisation isn't under test. */
const identityPurify = { sanitize: /** @param {string} h */ (h) => h };

/** @returns {HTMLElement} a fresh detached container element */
function freshEl() {
  return document.createElement('div');
}

/** Let the animation frame that paints coalesced streamed text run. */
function flushFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
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
    expect(sanitize).toHaveBeenCalledWith('<p>hi</p><pre><code>x</code></pre>', expect.any(Object));
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
    vi.stubGlobal('marked', {
      parse: () => {
        throw new Error('boom');
      },
    });
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
      t.replace(/```(?:\w*)\n([\s\S]*?)```/g, (_m, code) => `<pre><code>${code}</code></pre>`);
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
    expect(marked.parse('```\nconsole.log(1)\n```')).toBe('<pre><code></code></pre>');

    const el = freshEl();
    renderMarkdownInto(el, '```\nconsole.log(1)\n```');

    // chat used its own instance, so the fence content survived.
    const code = el.querySelector('pre code');
    expect(code).not.toBeNull();
    expect(code?.textContent).toContain('console.log(1)');
  });
});

// ---------------------------------------------------------------------------
// Tool vocabulary
// ---------------------------------------------------------------------------

describe('tool vocabulary', () => {
  test('normaliseToolName folds both spellings of a name onto one key', () => {
    // What the SDK sends, and what operator_session.format_tool_name makes of it.
    expect(normaliseToolName('mcp__osprey__channel_write')).toBe('channel_write');
    expect(normaliseToolName('Channel Write')).toBe('channel_write');
    // Built-ins arrive unprefixed.
    expect(normaliseToolName('Read')).toBe('read');
    // Only the mcp prefix is stripped -- the rest of the name survives.
    expect(normaliseToolName('mcp__osprey__phoebus_drive')).toBe('phoebus_drive');
  });

  test('normaliseToolName strips underscored server names too', () => {
    // Framework servers whose names contain underscores, and any
    // facility-declared server named the same way.
    expect(normaliseToolName('mcp__osprey_workspace__submit_response')).toBe('submit_response');
    expect(normaliseToolName('mcp__osprey_facility_knowledge__list_concepts')).toBe(
      'list_concepts'
    );
    expect(normaliseToolName('mcp__als_custom_srv__do_thing')).toBe('do_thing');
  });

  test('the raw name maps even when the display name is unhelpful', () => {
    expect(
      activityLabel({
        type: 'tool_use',
        // A display name with no row of its own: only the raw name can map.
        tool_name: 'Write Values',
        tool_name_raw: 'mcp__osprey__channel_write',
      })
    ).toBe('Writing control channels…');
  });

  test('a frame carrying only the display name still maps', () => {
    expect(activityLabel({ type: 'tool_use', tool_name: 'Execute File' })).toBe('Running Python…');
  });

  // The capitalise-and-ellipsis mechanism, over the spellings a frame carries:
  // a prefixed framework tool, one whose server name has underscores, and an
  // unprefixed built-in. The table's own shape is the invariant test below.
  test.each([
    ['mcp__osprey__channel_write', 'Writing control channels…'],
    ['mcp__osprey__execute', 'Running Python…'],
    ['mcp__osprey_facility_knowledge__list_concepts', 'Browsing facility knowledge…'],
    ['Bash', 'Running a shell command…'],
  ])('%s reads as "%s"', (raw, expected) => {
    expect(activityLabel({ type: 'tool_use', tool_name_raw: raw })).toBe(expected);
  });

  test('the logbook picture and hybrid search tools have their own phrases', () => {
    expect(TOOL_PHRASES.attachment_view).toBe('viewing a logbook picture');
    expect(TOOL_PHRASES.hybrid_search).toBe('searching the logbook');
    expect(activityLabel({ type: 'tool_use', tool_name_raw: 'mcp__ariel__attachment_view' })).toBe(
      'Viewing a logbook picture…'
    );
    expect(activityLabel({ type: 'tool_use', tool_name_raw: 'mcp__ariel__hybrid_search' })).toBe(
      'Searching the logbook…'
    );
  });

  test('showing a logbook entry has its own phrase', () => {
    expect(activityLabel({ type: 'tool_use', tool_name_raw: 'mcp__ariel__entry_open' })).toBe(
      'Showing a logbook entry…'
    );
  });

  test('keeping a logbook picture has its own phrase', () => {
    expect(
      activityLabel({ type: 'tool_use', tool_name_raw: 'mcp__ariel__attachment_to_artifact' })
    ).toBe('Saving a logbook picture…');
  });

  test('an unmapped tool has no phrase and keeps its raw name', () => {
    const event = /** @type {const} */ ({
      type: 'tool_use',
      tool_name: 'Queue Reorder',
      tool_name_raw: 'mcp__osprey__queue_reorder',
    });
    expect(toolPhrase(event)).toBeNull();
    expect(activityLabel(event)).toBe('Using Queue Reorder…');
  });

  test('a nameless tool_use falls back to a generic label', () => {
    expect(activityLabel({ type: 'tool_use' })).toBe('Using tool…');
  });

  test('every phrase is a lower-case fragment with no punctuation of its own', () => {
    // The table's contract: activityLabel capitalises and appends the ellipsis,
    // so a row must not carry either. (Proper nouns like "Python" are fine
    // mid-phrase -- only the first character is constrained.)
    for (const [name, phrase] of Object.entries(TOOL_PHRASES)) {
      expect(phrase, name).toBe(phrase.trim());
      expect(phrase[0], name).toBe(phrase[0].toLowerCase());
      expect(phrase, name).not.toMatch(/[.…]$/);
    }
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
    r.addUserMessage('align the <b>orbit</b>');
    expect(r.messageCount()).toBe(1);
    const entry = qs(container, '.op-entry.operator');
    // Operator text is plain text: markup in it is not interpreted.
    const bubble = qs(entry, '.op-bubble');
    expect(bubble.textContent).toBe('align the <b>orbit</b>');
    expect(bubble.querySelector('b')).toBeNull();
  });

  test('whole text events each open a paragraph of their own in one agent entry', () => {
    const r = createChatRenderer(container);
    r.handleEvent({ type: 'text', content: 'first' });
    r.handleEvent({ type: 'text', content: 'second' });

    const agents = container.querySelectorAll('.op-entry.assistant');
    expect(agents.length).toBe(1);
    const bodies = agents[0].querySelectorAll('.op-entry-body');
    expect(bodies.length).toBe(2);
    expect(bodies[1].textContent).toBe('second');
    // The shared markdown-content class styles the agent's rendered markdown.
    expect(bodies[0].classList.contains('osprey-md-rendered')).toBe(true);
    expect(r.messageCount()).toBe(1);
  });

  test('streamed deltas accumulate into one text block, painted once per frame', async () => {
    const r = createChatRenderer(container);
    r.handleEvent({ type: 'delta', kind: 'text', content: 'Hello ' });
    r.handleEvent({ type: 'delta', kind: 'text', content: 'world' });
    await flushFrame();

    const agents = container.querySelectorAll('.op-entry.assistant');
    expect(agents.length).toBe(1);
    const body = qs(container, '.op-entry.assistant .op-entry-body');
    expect(body.textContent).toBe('Hello world');
    expect(body.classList.contains('osprey-md-rendered')).toBe(true);
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
    // The loader is the activity line: it stays at the foot of the log for the
    // whole turn, and its label follows what the agent is doing.
    test('thinking marker shows "Thinking…"', () => {
      const r = createChatRenderer(container);
      r.beginTurn();
      r.handleEvent({ type: 'thinking' });
      expect(qs(container, '.op-loader-label').textContent).toBe('Thinking…');
    });

    test("tool_use shows the tool's operator phrase", () => {
      const r = createChatRenderer(container);
      r.beginTurn();
      r.handleEvent({ type: 'tool_use', tool_name: 'Channel Read' });
      expect(qs(container, '.op-loader-label').textContent).toBe('Reading control channels…');
    });

    test('the line stays up through a whole turn, and its label follows each phase', () => {
      const r = createChatRenderer(container);
      r.beginTurn();
      const label = () => qs(container, '.op-loader-label').textContent;
      const pinned = () => container.lastElementChild?.classList.contains('op-loader');

      r.handleEvent({ type: 'delta', kind: 'text', content: 'Reading the channels now.' });
      expect([pinned(), label()]).toEqual([true, 'Writing…']);
      // The call is named as soon as it starts, before its input arrives.
      r.handleEvent({ type: 'tool_start', tool_name: 'Channel Read', tool_use_id: 't1' });
      expect([pinned(), label()]).toEqual([true, 'Reading control channels…']);
      r.handleEvent({ type: 'tool_use', tool_name: 'Channel Read', tool_use_id: 't1', input: {} });
      r.handleEvent({ type: 'tool_result', tool_use_id: 't1', is_error: false, preview: '1.0' });
      r.handleEvent({ type: 'delta', kind: 'text', content: 'Done.' });
      expect([pinned(), label()]).toEqual([true, 'Writing…']);

      r.handleEvent({ type: 'result', is_error: false, duration_ms: 10 });
      expect(container.querySelector('.op-loader')).toBeNull();
    });

    test("a tool's phrase leaves the line when its result arrives", () => {
      const r = createChatRenderer(container);
      r.beginTurn();
      const label = () => qs(container, '.op-loader-label').textContent;
      r.handleEvent({ type: 'tool_use', tool_name: 'Channel Read', tool_use_id: 't1', input: {} });
      expect(label()).toBe('Reading control channels…');
      r.handleEvent({ type: 'tool_result', tool_use_id: 't1', is_error: false, preview: '1.0' });
      expect(label()).toBe('Working…');
    });

    test('the phrase stays while another call of the turn is still running', () => {
      const r = createChatRenderer(container);
      r.beginTurn();
      const label = () => qs(container, '.op-loader-label').textContent;
      r.handleEvent({ type: 'tool_use', tool_name: 'Channel Read', tool_use_id: 't1', input: {} });
      r.handleEvent({ type: 'tool_use', tool_name: 'Channel Write', tool_use_id: 't2', input: {} });
      r.handleEvent({ type: 'tool_result', tool_use_id: 't1', is_error: false, preview: '1.0' });
      expect(label()).not.toBe('Working…');
      r.handleEvent({ type: 'tool_result', tool_use_id: 't2', is_error: false, preview: 'ok' });
      expect(label()).toBe('Working…');
    });

    test("a subagent's own result does not reset the line", () => {
      const r = createChatRenderer(container);
      r.beginTurn();
      r.handleEvent({
        type: 'tool_use',
        tool_name: 'Agent',
        tool_name_raw: 'Agent',
        tool_use_id: 'a1',
        input: { subagent_type: 'channel-finder', description: 'find' },
      });
      r.handleEvent({
        type: 'tool_use',
        tool_name: 'Channel Read',
        tool_use_id: 'n1',
        parent_tool_use_id: 'a1',
        input: {},
      });
      r.handleEvent({
        type: 'tool_result',
        tool_use_id: 'n1',
        parent_tool_use_id: 'a1',
        is_error: false,
        preview: '1.0',
      });
      expect(qs(container, '.op-loader-label').textContent).toBe(
        'channel-finder · Reading control channels…'
      );
    });

    test('"Waiting for the background agent…" ends when that agent reports', () => {
      const r = createChatRenderer(container);
      r.beginTurn();
      const label = () => qs(container, '.op-loader-label').textContent;
      r.handleEvent({
        type: 'tool_use',
        tool_name: 'Agent',
        tool_name_raw: 'Agent',
        tool_use_id: 'a1',
        input: { subagent_type: 'channel-finder', description: 'find', run_in_background: true },
      });
      r.handleEvent({ type: 'system', subtype: 'background_wait' });
      expect(label()).toBe('Waiting for the background agent…');
      r.handleEvent({
        type: 'system',
        subtype: 'agent_done',
        tool_use_id: 'a1',
        status: 'completed',
      });
      expect(label()).toBe('Working…');
    });

    test('the wait goes on while a second background agent is still out', () => {
      const r = createChatRenderer(container);
      r.beginTurn();
      const label = () => qs(container, '.op-loader-label').textContent;
      for (const id of ['a1', 'a2']) {
        r.handleEvent({
          type: 'tool_use',
          tool_name: 'Agent',
          tool_name_raw: 'Agent',
          tool_use_id: id,
          input: { subagent_type: 'channel-finder', description: id },
        });
      }
      r.handleEvent({ type: 'system', subtype: 'background_wait' });
      r.handleEvent({
        type: 'system',
        subtype: 'agent_done',
        tool_use_id: 'a1',
        status: 'completed',
      });
      expect(label()).toBe('Waiting for the background agent…');
      r.handleEvent({
        type: 'system',
        subtype: 'agent_done',
        tool_use_id: 'a2',
        status: 'completed',
      });
      expect(label()).toBe('Working…');
    });

    test("a subagent's call starting names the agent, and draws no row", () => {
      const r = createChatRenderer(container);
      r.beginTurn();
      r.handleEvent({
        type: 'tool_use',
        tool_name: 'Agent',
        tool_name_raw: 'Agent',
        tool_use_id: 'a1',
        input: { subagent_type: 'channel-finder', description: 'find' },
      });
      r.handleEvent({ type: 'tool_start', tool_name: 'Channel Read', parent_tool_use_id: 'a1' });
      expect(qs(container, '.op-loader-label').textContent).toBe(
        'channel-finder · Reading control channels…'
      );
      expect(container.querySelectorAll('.op-tool:not(.op-agent)')).toHaveLength(0);
    });

    test('while the reply streams a drawing, the line says "Drawing…", then "Writing…" again', () => {
      const r = createChatRenderer(container);
      r.beginTurn();
      const label = () => qs(container, '.op-loader-label').textContent;
      r.handleEvent({
        type: 'delta',
        kind: 'text',
        content: 'Quick look:\n\n```svg\n<svg viewBox',
      });
      expect(label()).toBe('Drawing…');
      r.handleEvent({
        type: 'delta',
        kind: 'text',
        content: '="0 0 1 1"></svg>\n```\n\nThat is all',
      });
      expect(label()).toBe('Writing…');
    });

    test('the activity line stays pinned below the latest entry', () => {
      const r = createChatRenderer(container);
      r.addUserMessage('go');
      r.beginTurn();
      r.handleEvent({ type: 'tool_use', tool_name: 'Read' });
      expect(container.lastElementChild?.classList.contains('op-loader')).toBe(true);
    });

    test('while a card waits on the operator, the line says so', () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.beginTurn();
      r.handleEvent({
        type: 'permission_request',
        request_id: 'req-1',
        kind: 'tool',
        tool_name: 'Execute',
        tool_name_raw: 'mcp__python__execute',
        tool_use_id: 'x1',
        input: { code: 'print(1)' },
        reason: null,
        can_allow_session: false,
      });
      expect(qs(container, '.op-loader-label').textContent).toBe('Waiting for your answer');
    });
  });

  describe('session_reset divider suppression', () => {
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
      expect(divider.textContent).toBeTruthy();
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

    test('a reset just before the result of a moving turn divides it from the earlier exchange', () => {
      // A conversation that moves inside a live chat is marked where the move
      // is seen: the reset arrives just before the result closing that turn.
      const r = createChatRenderer(container);
      r.addUserMessage('read the beam current');
      r.handleEvent({ type: 'text', content: 'The beam current is 500 mA.' });
      r.handleEvent({ type: 'result', is_error: false });
      r.addUserMessage('/clear');
      r.handleEvent({ type: 'thinking' });
      r.handleEvent({ type: 'session_reset' });
      r.handleEvent({ type: 'result', is_error: false });

      const dividers = container.querySelectorAll('.op-system');
      expect(dividers.length).toBe(1);
      expect(dividers[0].textContent).toBe('session reset');
      const children = Array.from(container.children);
      const earlier = qs(container, '.op-entry.assistant');
      expect(children.indexOf(dividers[0])).toBeGreaterThan(children.indexOf(earlier));
      // The turn finished: no activity line is left showing.
      expect(container.querySelector('.op-processing.active')).toBeNull();
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

  describe('replay of prior transcript turns', () => {
    /** Three turns in the shape `GET /api/session-chat` serves. */
    const turns = [
      { role: 'user', content: 'read the beam current', timestamp: '2026-09-05T10:00:00Z' },
      {
        role: 'assistant',
        content: 'The beam current is 500 mA.',
        timestamp: '2026-09-05T10:00:02Z',
      },
      { role: 'user', content: 'and the orbit?', timestamp: '2026-09-05T10:00:09Z' },
    ];

    test('renders one entry per turn, in order, and returns the count', () => {
      const r = createChatRenderer(container);
      expect(r.replay(turns)).toBe(3);
      expect(r.messageCount()).toBe(3);

      const entries = container.querySelectorAll('.op-entry');
      expect(entries.length).toBe(3);
      expect(entries[0].classList.contains('operator')).toBe(true);
      expect(entries[1].classList.contains('assistant')).toBe(true);
      expect(entries[2].classList.contains('operator')).toBe(true);
      expect(entries[1].querySelector('.op-entry-body')?.textContent).toBe(
        'The beam current is 500 mA.'
      );
    });

    test('agent turns go through the sanitising markdown path', () => {
      // The sanitiser answers with a sentinel, so only its output can land.
      vi.stubGlobal('DOMPurify', { sanitize: () => '<em>safe</em>' });
      const r = createChatRenderer(container);
      r.replay([
        {
          role: 'assistant',
          content: 'result: <img src=x onerror="steal()"><script>evil()</script>',
        },
      ]);
      const body = qs(container, '.op-entry.assistant .op-entry-body');
      expect(body.innerHTML).toBe('<em>safe</em>');
    });

    test('markdown in an agent turn is rendered, not shown as source', () => {
      vi.stubGlobal('marked', { parse: (/** @type {string} */ t) => `<p><em>${t}</em></p>` });
      const r = createChatRenderer(container);
      r.replay([{ role: 'assistant', content: 'orbit corrected' }]);
      const body = qs(container, '.op-entry.assistant .op-entry-body');
      expect(body.querySelector('em')?.textContent).toBe('orbit corrected');
    });

    test('a session_reset after a replay draws no divider', () => {
      // Replayed turns are the same conversation continued in another window,
      // so the fresh stream's frame-0 reset is not a boundary worth marking.
      const r = createChatRenderer(container);
      r.replay(turns);
      r.handleEvent({ type: 'session_reset' });
      expect(container.querySelector('.op-system')).toBeNull();
      expect(r.messageCount()).toBe(3);
    });

    test('a reset still earns a divider once a live exchange completes above it', () => {
      const r = createChatRenderer(container);
      r.replay(turns);
      r.handleEvent({ type: 'text', content: 'The orbit is flat.' });
      r.handleEvent({ type: 'result', is_error: false });
      r.addUserMessage('thanks');
      r.handleEvent({ type: 'session_reset' });
      expect(container.querySelector('.op-system')).not.toBeNull();
    });

    test('a live text event after a replay opens a fresh agent entry', () => {
      const r = createChatRenderer(container);
      r.replay([{ role: 'assistant', content: 'replayed' }]);
      r.handleEvent({ type: 'text', content: 'live' });
      const agents = container.querySelectorAll('.op-entry.assistant');
      expect(agents.length).toBe(2);
      expect(agents[0].querySelector('.op-entry-body')?.textContent).toBe('replayed');
      expect(agents[1].querySelector('.op-entry-body')?.textContent).toBe('live');
    });

    test('a replayed turn whose history has no end stays open, and the live frames continue it', () => {
      const r = createChatRenderer(container);
      r.replay([
        { role: 'user', content: 'read the orbit' },
        { role: 'assistant', content: 'Reading.', events: [{ type: 'text', content: 'Reading.' }] },
      ]);
      expect(r.hasOpenTurn()).toBe(true);
      expect(container.querySelector('.op-footer')).toBeNull();

      // The feed joins the turn still running.
      r.beginTurn();
      r.handleEvent({ type: 'text', content: 'It is flat.' });
      r.handleEvent({ type: 'result', is_error: false, duration_ms: 4000 });

      const agents = container.querySelectorAll('.op-entry.assistant');
      expect(agents).toHaveLength(1);
      expect([...agents[0].querySelectorAll('.op-entry-body')].map((b) => b.textContent)).toEqual([
        'Reading.',
        'It is flat.',
      ]);
      expect(agents[0].querySelectorAll('.op-footer')).toHaveLength(1);
      expect(r.hasOpenTurn()).toBe(false);
    });

    test('a replayed turn that ended is closed, and a live frame after it opens a new entry', () => {
      const r = createChatRenderer(container);
      r.replay([
        {
          role: 'assistant',
          content: 'Done.',
          events: [
            { type: 'text', content: 'Done.' },
            { type: 'result', is_error: false },
          ],
        },
      ]);
      expect(r.hasOpenTurn()).toBe(false);
      r.beginTurn();
      r.handleEvent({ type: 'text', content: 'Something new.' });

      expect(container.querySelectorAll('.op-entry.assistant')).toHaveLength(2);
    });

    test('a replayed turn with a run of calls and an approval draws exactly what the live turn drew', () => {
      /** A read of one channel, as the stream carries it. */
      const read = (/** @type {string} */ id) => ({
        type: 'tool_use',
        tool_name: 'Channel Read',
        tool_name_raw: 'mcp__osprey__channel_read',
        tool_use_id: id,
        input: { channel: `SR:${id}:BPM` },
      });
      const done = (/** @type {string} */ id, /** @type {boolean} */ isError = false) => ({
        type: 'tool_result',
        tool_use_id: id,
        is_error: isError,
        preview: isError ? 'timeout' : '1.0',
        truncated: false,
      });
      const write = {
        type: 'tool_use',
        tool_name: 'Channel Write',
        tool_name_raw: 'mcp__osprey__channel_write',
        tool_use_id: 'w1',
        input: { channel: 'SR:C01:HCOR', value: 0.5 },
      };
      const events = [
        read('t1'),
        read('t2'),
        read('t3'),
        done('t1'),
        done('t2', true),
        done('t3'),
        write,
        {
          type: 'permission_request',
          request_id: 'req-w1',
          kind: 'tool',
          tool_name: 'Channel Write',
          tool_name_raw: 'mcp__osprey__channel_write',
          tool_use_id: 'w1',
          input: write.input,
          reason: null,
          can_allow_session: false,
        },
        { type: 'permission_resolved', request_id: 'req-w1', allow: true, answers: null },
        done('w1'),
        { type: 'text', content: 'Corrected.' },
        { type: 'result', is_error: false, duration_ms: 53_000 },
      ];

      const live = createChatRenderer(container, { answerPermission: vi.fn() });
      live.beginTurn();
      for (const event of events) live.handleEvent(event);

      const replayed = document.createElement('div');
      createChatRenderer(replayed, { answerPermission: vi.fn() }).replay([
        { role: 'assistant', content: 'Corrected.', events },
      ]);

      expect(replayed.innerHTML).toBe(container.innerHTML);
      // What both drew: the reads folded with the failure on the run row, the
      // write on its own heavier row with the decision, and the footer's counts.
      const run = qs(container, '.op-run');
      expect(qs(run, '.op-row-label').textContent).toBe('Read 3 channels');
      expect(run.dataset.state).toBe('error');
      expect(run.nextElementSibling?.getAttribute('data-kind')).toBe('control-write');
      expect(qs(container, '.op-tool-decision').dataset.decision).toBe('allowed');
      expect(container.querySelector('.op-approval')).toBeNull();
      expect([...container.querySelectorAll('.op-footer-part')].map((p) => p.textContent)).toEqual([
        '53s',
        '4 actions',
        '1 approval',
      ]);
      expect(container.querySelectorAll('.pulse')).toHaveLength(0);
    });

    test('an empty or missing turn list renders nothing', () => {
      const r = createChatRenderer(container);
      expect(r.replay([])).toBe(0);
      expect(r.replay(undefined)).toBe(0);
      expect(container.children.length).toBe(0);
      expect(r.messageCount()).toBe(0);
    });

    test('a turn with no content is skipped and not counted', () => {
      const r = createChatRenderer(container);
      expect(
        r.replay([
          { role: 'user', content: '' },
          { role: 'assistant', content: 'hi' },
        ])
      ).toBe(1);
      expect(r.messageCount()).toBe(1);
      expect(container.querySelectorAll('.op-entry.operator').length).toBe(0);
    });

    test('addAgentMessage appends a finished agent entry and marks the exchange', () => {
      const r = createChatRenderer(container);
      const entry = r.addAgentMessage('The beam current is 500 mA.');
      expect(entry.classList.contains('assistant')).toBe(true);
      expect(qs(container, '.op-entry.assistant .op-entry-body').textContent).toBe(
        'The beam current is 500 mA.'
      );
      expect(r.messageCount()).toBe(1);
      // Unlike a replay, a directly added agent message is a completed
      // exchange, so the next reset is worth a divider.
      r.addUserMessage('and the orbit?');
      r.handleEvent({ type: 'session_reset' });
      expect(container.querySelector('.op-system')).not.toBeNull();
    });
  });

  test('a hostile tool name reaches the activity line as text, never markup', () => {
    // An unmapped name is echoed verbatim, so it is the one place agent-supplied
    // text enters the activity line. The loader's label writes textContent only.
    const r = createChatRenderer(container);
    r.beginTurn();
    const hostile = '<img src=x onerror="steal()"><script>evil()</script>';
    r.handleEvent({ type: 'tool_use', tool_name: hostile });
    const label = qs(container, '.op-loader-label');
    expect(label.querySelector('img')).toBeNull();
    expect(label.querySelector('script')).toBeNull();
    expect(label.textContent).toBe(`Using ${hostile}…`);
  });

  test('hostile model text routed via a text event goes through the sanitiser', () => {
    // The sanitiser answers with a sentinel, so only its output can land.
    vi.stubGlobal('DOMPurify', { sanitize: () => '<em>safe</em>' });
    const r = createChatRenderer(container);
    r.handleEvent({
      type: 'text',
      content: 'result: <img src=x onerror="steal()"><script>evil()</script>',
    });
    const body = qs(container, '.op-entry.assistant .op-entry-body');
    expect(body.innerHTML).toBe('<em>safe</em>');
  });

  test('entering plan mode draws no tool row and keeps the reply text open, while the loader says the agent is planning', async () => {
    const r = createChatRenderer(container);
    r.beginTurn();
    r.handleEvent({ type: 'delta', kind: 'text', content: 'Let me plan ' });
    r.handleEvent({
      type: 'tool_use',
      tool_name: 'Enter Plan Mode',
      tool_name_raw: 'EnterPlanMode',
      tool_use_id: 'p1',
      input: {},
    });
    r.handleEvent({
      type: 'tool_result',
      tool_use_id: 'p1',
      is_error: false,
      preview: 'Entered plan mode. Explore the codebase, then write a plan…',
      truncated: false,
    });
    // While the call is in flight the loader says what is happening, in the
    // chat's own words rather than the tool's name.
    expect(qs(container, '.op-loader-label').textContent).toBe('Planning…');
    r.handleEvent({ type: 'delta', kind: 'text', content: 'this first.' });
    await flushFrame();

    // No row, no run, and the CLI's planning instructions never reach the log.
    expect(container.querySelector('.op-tool, .op-run')).toBeNull();
    expect(container.textContent).not.toContain('Entered plan mode');
    // The text block stays open across the hidden call: one body, not two.
    const bodies = container.querySelectorAll('.op-entry-body');
    expect(bodies).toHaveLength(1);
    expect(bodies[0].textContent).toBe('Let me plan this first.');
  });

  describe('approval cards under their tool rows', () => {
    /** A tool call, as the chat stream carries it. */
    const call = (/** @type {string} */ id, /** @type {Record<string, unknown>} */ extra = {}) => ({
      type: 'tool_use',
      tool_name: 'Execute',
      tool_name_raw: 'mcp__python__execute',
      tool_use_id: id,
      input: { code: 'print(1)' },
      ...extra,
    });
    /** The approval the CLI asks for one call. */
    const ask = (
      /** @type {string} */ toolUseId,
      /** @type {string} */ requestId,
      /** @type {Record<string, unknown>} */ extra = {}
    ) => ({
      type: 'permission_request',
      request_id: requestId,
      kind: 'tool',
      tool_name: 'Execute',
      tool_name_raw: 'mcp__python__execute',
      tool_use_id: toolUseId,
      input: { code: 'print(1)' },
      reason: null,
      can_allow_session: false,
      ...extra,
    });
    /** Another tab's answer to a request, as the feed carries it. */
    const resolved = (
      /** @type {string} */ requestId,
      /** @type {boolean} */ allow,
      /** @type {boolean} */ forSession = false
    ) => ({
      type: 'permission_resolved',
      request_id: requestId,
      allow,
      for_session: forSession,
      answers: null,
    });
    /** A call's result: the error it raised, or the output it printed. */
    const result = (/** @type {string} */ id, /** @type {boolean} */ isError) => ({
      type: 'tool_result',
      tool_use_id: id,
      is_error: isError,
      preview: isError ? 'NameError: name x is not defined' : '1',
      truncated: false,
    });

    /** The tool rows in the log, oldest first. */
    const rows = () => /** @type {HTMLElement[]} */ ([...container.querySelectorAll('.op-tool')]);
    /** The button in the log with this label. */
    const buttonLabelled = (/** @type {string} */ label) =>
      /** @type {HTMLElement} */ (
        [...container.querySelectorAll('button')].find((b) => b.textContent === label)
      );
    /** Let the answer's promise chain run. */
    const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

    test('an approval sits directly under the row of its own call, not another call to the same tool', () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.handleEvent(call('t1'));
      r.handleEvent(call('t2'));
      r.handleEvent(ask('t2', 'req-2'));

      const [first, second] = rows();
      const card = qs(container, '.op-approval');
      expect(card.previousElementSibling).toBe(second);
      expect(card.parentElement).toBe(second.parentElement);
      expect(first.nextElementSibling).toBe(second);
    });

    test('a waiting approval stays in view once the agent moves on: its row never folds away', () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.handleEvent(call('t1'));
      r.handleEvent(call('t2'));
      r.handleEvent(ask('t2', 'req-2'));
      r.handleEvent({ type: 'text', content: 'Waiting on you.' });

      const card = qs(container, '.op-approval');
      expect(card.closest('[hidden]')).toBeNull();
      expect(card.closest('.op-run')).toBeNull();
      // The row that carries the card stands on its own in the turn, under no run.
      expect(card.previousElementSibling).toBe(rows()[1]);
      expect(rows()[1].parentElement?.classList.contains('op-entry')).toBe(true);
    });

    test('once answered, the row keeps the decision in view where the card was', async () => {
      const answerPermission = vi.fn(async () => {});
      const r = createChatRenderer(container, { answerPermission });
      r.handleEvent(call('t1'));
      r.handleEvent(ask('t1', 'req-1'));
      r.handleEvent({ type: 'text', content: 'Done.' });
      qs(container, '.op-btn-primary').click();
      await settle();

      expect(answerPermission).toHaveBeenCalledWith('req-1', { allow: true });
      const badge = qs(container, '.op-tool-decision');
      expect(badge.closest('[hidden]')).toBeNull();
      expect(badge.closest('.op-run')).toBeNull();
    });

    test.each([
      {
        how: 'the primary button allows the call',
        sessionOffer: false,
        act: () => qs(container, '.op-btn-primary').click(),
        answer: { allow: true },
        decision: 'allowed',
        label: 'Allowed by you',
      },
      {
        how: '"Allow for this session" records the session decision',
        sessionOffer: true,
        act: () => buttonLabelled('Allow for this session').click(),
        answer: { allow: true, for_session: true },
        decision: 'session',
        label: 'Allowed for this session',
      },
      {
        how: 'Deny records the denial',
        sessionOffer: false,
        act: () => buttonLabelled('Deny').click(),
        answer: { allow: false },
        decision: 'denied',
        label: 'Denied by you',
      },
      {
        how: 'Enter on the waiting card allows the call',
        sessionOffer: false,
        act: () =>
          qs(container, '.op-approval').dispatchEvent(
            new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })
          ),
        answer: { allow: true },
        decision: 'allowed',
        label: 'Allowed by you',
      },
    ])(
      '$how on the row, and takes the card out of the log',
      async ({ sessionOffer, act, answer, decision, label }) => {
        const answerPermission = vi.fn(async () => {});
        const r = createChatRenderer(container, { answerPermission });
        r.handleEvent(call('t1'));
        r.handleEvent(ask('t1', 'req-1', { can_allow_session: sessionOffer }));
        act();
        await settle();

        expect(answerPermission).toHaveBeenCalledWith('req-1', answer);
        expect(container.querySelector('.op-approval, .op-card-line')).toBeNull();
        const badge = qs(container, '.op-tool-decision');
        expect(badge.dataset.decision).toBe(decision);
        expect(badge.textContent).toBe(label);
      }
    );

    test('a waiting approval takes focus on its primary control where it sits under the row', () => {
      document.body.append(container);
      try {
        const r = createChatRenderer(container, { answerPermission: vi.fn() });
        r.handleEvent(call('t1'));
        r.handleEvent(ask('t1', 'req-1'));
        expect(document.activeElement).toBe(qs(container, '.op-btn-primary'));
      } finally {
        container.remove();
      }
    });

    test('a call allowed and then failed reads as one row: the decision, then the failure in its result', async () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn(async () => {}) });
      r.handleEvent(call('t1'));
      r.handleEvent(ask('t1', 'req-1'));
      qs(container, '.op-btn-primary').click();
      await settle();
      r.handleEvent(result('t1', true));

      expect(rows()).toHaveLength(1);
      const row = qs(container, '.op-tool');
      expect(row.dataset.state).toBe('error');
      const badge = qs(row, '.op-tool-decision');
      expect(badge.textContent).toBe('Allowed by you');
      expect(badge.nextElementSibling?.classList.contains('op-row-status')).toBe(true);
      expect(qs(row, '.op-tool-pre.op-tool-error').textContent).toContain('NameError');
      expect(container.querySelector('.op-approval, .op-card-line')).toBeNull();
    });

    test('an answer given in another tab settles the row the same way', () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.handleEvent(call('t1'));
      r.handleEvent(ask('t1', 'req-1'));
      r.handleEvent(resolved('req-1', false));

      expect(container.querySelector('.op-approval')).toBeNull();
      expect(qs(container, '.op-tool-decision').dataset.decision).toBe('denied');
    });

    test('an approval nobody answered expires: the row says so without blaming the operator, and the card leaves', () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.handleEvent(call('t1'));
      r.handleEvent(ask('t1', 'req-1'));
      r.handleEvent({ ...resolved('req-1', false), expired: true });

      expect(container.querySelector('.op-approval')).toBeNull();
      const badge = qs(container, '.op-tool-decision');
      expect(badge.dataset.decision).toBe('expired');
      expect(badge.textContent).toMatch(/no answer/i);
      expect(badge.textContent).not.toMatch(/by you/i);
    });

    test('a plan nobody approved expires to a kept-planning line, not an approval', () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.handleEvent(
        ask('', 'req-9', {
          kind: 'plan',
          tool_name: 'Plan',
          tool_name_raw: 'ExitPlanMode',
          input: { plan: '1. Read the file.' },
        })
      );
      r.handleEvent({ ...resolved('req-9', false), expired: true });

      const card = qs(container, '.op-approval');
      expect(card.dataset.state).toBe('denied');
      expect(card.textContent).toMatch(/no answer/i);
      expect(container.querySelector('button')).toBeNull();
    });

    test('a question nobody answered expires to a skipped line', () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.handleEvent({
        type: 'permission_request',
        request_id: 'q-1',
        kind: 'question',
        input: {
          questions: [
            {
              question: 'Which lattice?',
              header: 'Lattice',
              multiSelect: false,
              options: [
                { label: 'Top-off', description: '' },
                { label: 'Hybrid', description: '' },
              ],
            },
          ],
        },
      });
      r.handleEvent({ ...resolved('q-1', false), expired: true });

      const card = qs(container, '.op-card');
      expect(card.dataset.state).toBe('skipped');
      expect(card.textContent).toMatch(/no answer/i);
    });

    test('an approval with no row of its own stays in the log as a card and folds to its outcome', async () => {
      const answerPermission = vi.fn(async () => {});
      const r = createChatRenderer(container, { answerPermission });
      r.handleEvent(
        ask('', 'req-9', {
          kind: 'plan',
          tool_name: 'Plan',
          tool_name_raw: 'ExitPlanMode',
          input: { plan: '1. Read the file.' },
        })
      );

      const card = qs(container, '.op-approval');
      expect(card.closest('.op-run, .op-tool')).toBeNull();
      qs(container, '.op-btn-primary').click();
      await settle();

      expect(container.contains(card)).toBe(true);
      expect(card.dataset.state).toBe('allowed');
      expect(container.querySelector('.op-tool-decision')).toBeNull();
    });

    test("a subagent's approval sits under that subagent's own row, and its trace opens to show it", () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.handleEvent({
        type: 'tool_use',
        tool_name: 'Agent',
        tool_name_raw: 'Agent',
        tool_use_id: 'a1',
        input: { subagent_type: 'facility-knowledge', description: 'look it up' },
      });
      r.handleEvent(call('s1', { parent_tool_use_id: 'a1' }));
      r.handleEvent(ask('s1', 'req-s1'));

      const trace = qs(container, '.op-agent-trace');
      const card = qs(container, '.op-approval');
      expect(card.parentElement).toBe(trace);
      expect(card.previousElementSibling).toBe(trace.querySelector('.op-tool'));
      expect(card.closest('[hidden]')).toBeNull();
    });

    test('after a reload, a call still awaiting its answer takes the approval under its replayed row', async () => {
      const answerPermission = vi.fn(async () => {});
      const r = createChatRenderer(container, { answerPermission });
      // The transcript shows the call, not yet answered.
      r.replay([{ role: 'assistant', content: '', events: [call('t1')] }]);
      // The feed opens mid-turn and reports the request still waiting.
      r.beginTurn();
      r.handleEvent(ask('t1', 'req-1'));

      const row = qs(container, '.op-tool');
      expect(qs(container, '.op-approval').previousElementSibling).toBe(row);
      qs(container, '.op-btn-primary').click();
      await settle();

      expect(container.querySelector('.op-approval')).toBeNull();
      expect(qs(row, '.op-tool-decision').dataset.decision).toBe('allowed');
    });

    test('a replayed turn that carries its approval draws the decision on the row, as it was drawn live', () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.replay([
        {
          role: 'assistant',
          content: '',
          events: [
            call('t1'),
            ask('t1', 'req-1'),
            resolved('req-1', true),
            result('t1', true),
            { type: 'result', is_error: false },
          ],
        },
      ]);

      expect(container.querySelector('.op-approval')).toBeNull();
      expect(qs(container, '.op-tool-decision').dataset.decision).toBe('allowed');
      expect(qs(container, '.op-tool').dataset.state).toBe('error');
    });

    test.each([
      {
        told: 'allowed',
        decision: 'allowed',
        label: 'Allowed by you',
        isError: false,
        state: 'done',
      },
      {
        told: 'declined, which failed',
        decision: 'denied',
        label: 'Denied by you',
        isError: true,
        state: 'error',
      },
      {
        told: 'never asked about',
        decision: undefined,
        label: null,
        isError: false,
        state: 'done',
      },
    ])(
      'a replayed call the operator $told shows that on its row, with no card',
      ({ decision, label, isError, state }) => {
        const r = createChatRenderer(container, { answerPermission: vi.fn() });
        r.replay([
          {
            role: 'assistant',
            content: '',
            events: [
              call('t1'),
              { ...result('t1', isError), decision },
              { type: 'result', is_error: false },
            ],
          },
        ]);

        const badge = container.querySelector('.op-tool-decision');
        if (decision === undefined) {
          expect(badge).toBeNull();
        } else {
          expect(/** @type {HTMLElement} */ (badge).dataset.decision).toBe(decision);
          expect(badge?.textContent).toBe(label);
          expect(badge?.nextElementSibling?.classList.contains('op-row-status')).toBe(true);
        }
        expect(container.querySelector('.op-approval')).toBeNull();
        expect(qs(container, '.op-tool').dataset.state).toBe(state);
      }
    );

    test('a decision for a call with no row of its own draws nothing', () => {
      // A question or a plan has no row, and its answer shows no pill live either.
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.replay([
        {
          role: 'assistant',
          content: '',
          events: [
            { ...result('q1', true), decision: 'denied' },
            { type: 'result', is_error: false },
          ],
        },
      ]);

      expect(container.querySelector('.op-tool-decision')).toBeNull();
      expect(container.querySelector('.op-tool')).toBeNull();
      expect(container.querySelector('.op-approval')).toBeNull();
    });

    test('a call replayed while still running finishes when its result arrives', () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.replay([{ role: 'assistant', content: '', events: [call('t1')] }]);
      r.beginTurn();
      r.handleEvent(result('t1', false));

      expect(qs(container, '.op-tool').dataset.state).toBe('done');
    });

    test('a turn that ends while an approval waits takes the card away, and the row keeps what it has', () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.handleEvent(call('t1'));
      r.handleEvent(ask('t1', 'req-1'));
      r.endTurn();

      expect(container.querySelector('.op-approval')).toBeNull();
      expect(container.querySelector('.op-tool-decision')).toBeNull();
      expect(rows()).toHaveLength(1);
      expect(qs(container, '.op-tool').closest('.op-run')).toBeNull();
    });
  });

  describe('settled questions leave a trace', () => {
    const QUESTIONS = [
      {
        question: 'Which lattice?',
        header: 'Lattice',
        multiSelect: false,
        options: [
          { label: 'Top-off', description: '' },
          { label: 'Hybrid', description: '' },
        ],
      },
      {
        question: 'Say "go" when ready?',
        header: 'Ready',
        multiSelect: false,
        options: [
          { label: 'Go', description: '' },
          { label: 'Wait', description: '' },
        ],
      },
    ];

    /** @param {HTMLElement} root */
    const items = (root) =>
      [...root.querySelectorAll('.op-qtrace-item')].map((item) => [
        item.querySelector('.op-qtrace-q')?.textContent,
        item.querySelector('.op-qtrace-a')?.textContent,
      ]);

    test('an answered question folds to a closed line that opens to each question and its answer', async () => {
      const answerPermission = vi.fn(async () => {});
      const r = createChatRenderer(container, { answerPermission });
      r.handleEvent({
        type: 'permission_request',
        request_id: 'q-1',
        kind: 'question',
        input: { questions: QUESTIONS },
      });
      const card = qs(container, '.op-card');
      const key = (/** @type {string} */ k) =>
        card.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true }));
      key('2');
      qs(card, '.op-btn-primary').click();
      key('1');
      qs(card, '.op-btn-primary').click();
      await vi.waitFor(() => expect(card.dataset.state).toBe('answered'));

      expect(answerPermission).toHaveBeenCalledWith('q-1', {
        allow: true,
        answers: { 'Which lattice?': 'Hybrid', 'Say "go" when ready?': 'Go' },
      });
      const head = qs(card, '.op-qtrace-head');
      expect(head.textContent).toBe('Answered · 2 questions');
      expect(qs(card, '.op-qtrace-body').hidden).toBe(true);
      head.click();
      expect(head.getAttribute('aria-expanded')).toBe('true');
      expect(qs(card, '.op-qtrace-body').hidden).toBe(false);
      expect(items(card)).toEqual([
        ['Which lattice?', 'Hybrid'],
        ['Say "go" when ready?', 'Go'],
      ]);
    });

    test('a replayed question is redrawn as its trace, answers read back from the result', () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.replay([
        {
          role: 'assistant',
          content: '',
          events: [
            {
              type: 'tool_use',
              tool_name: 'AskUserQuestion',
              tool_name_raw: 'AskUserQuestion',
              tool_use_id: 'u1',
              input: { questions: QUESTIONS },
            },
            {
              type: 'tool_result',
              tool_use_id: 'u1',
              is_error: false,
              preview:
                'Your questions have been answered: "Which lattice?"="Hybrid, then "top-off"", ' +
                '"Say "go" when ready?"="Go". You can now continue with these answers in mind.',
            },
            { type: 'result', is_error: false },
          ],
        },
      ]);

      const card = qs(container, '.op-card');
      expect(card.dataset.state).toBe('answered');
      expect(items(card)).toEqual([
        ['Which lattice?', 'Hybrid, then "top-off"'],
        ['Say "go" when ready?', 'Go'],
      ]);
      expect(container.querySelector('.op-tool')).toBeNull();
    });

    test('a replayed question whose call ended in an error reads as skipped, with no answers', () => {
      const r = createChatRenderer(container, { answerPermission: vi.fn() });
      r.replay([
        {
          role: 'assistant',
          content: '',
          events: [
            {
              type: 'tool_use',
              tool_name: 'AskUserQuestion',
              tool_name_raw: 'AskUserQuestion',
              tool_use_id: 'u2',
              input: { questions: [QUESTIONS[0]] },
            },
            {
              type: 'tool_result',
              tool_use_id: 'u2',
              is_error: true,
              preview: 'The user declined.',
            },
            { type: 'result', is_error: false },
          ],
        },
      ]);

      const card = qs(container, '.op-card');
      expect(card.dataset.state).toBe('skipped');
      expect(qs(card, '.op-qtrace-head').textContent).toBe('Skipped · Lattice');
      expect(items(card)).toEqual([['Which lattice?', 'No answer']]);
    });
  });

  describe('artifact ids in agent answers', () => {
    /** @param {string} markdown */
    const links = (markdown) => {
      createChatRenderer(container).addAgentMessage(markdown);
      return [...container.querySelectorAll('.op-artifact-link')].map((b) => [
        /** @type {HTMLElement} */ (b).dataset.artifactId,
        b.textContent,
      ]);
    };

    test('the artifact_id phrase becomes a control that reads as written', () => {
      expect(
        links('<p>Tunes (artifact_id: a57934cf00dd), and artifact_id=7AC86013F4B4.</p>')
      ).toEqual([
        ['a57934cf00dd', 'artifact_id: a57934cf00dd'],
        ['7ac86013f4b4', 'artifact_id=7AC86013F4B4'],
      ]);
      expect(container.textContent).toBe(
        'Tunes (artifact_id: a57934cf00dd), and artifact_id=7AC86013F4B4.'
      );
    });

    test('an id in inline code after the phrase is linked; a code block and a link are not', () => {
      expect(
        links(
          '<p>artifact_id: <code>a57934cf00dd</code></p>' +
            '<pre><code>artifact_id: 7ac86013f4b4</code></pre>' +
            '<p><a href="/x">artifact_id: 0123456789ab</a></p>'
        )
      ).toEqual([['a57934cf00dd', 'artifact_id: a57934cf00dd']]);
    });
  });
});
