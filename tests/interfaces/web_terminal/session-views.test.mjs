// @ts-nocheck
// TODO(frontend-hardening Pn): remove & fix types when this interface is retrofitted (P2–P5)
/**
 * Unit tests for session.html's ES modules:
 *
 *   - session-views.js: the four Activity-log view renderers (Agents,
 *     Tool Log, Artifacts, Conversation), each driving output from fixture
 *     JSON through a stubbed `apiFetch`/`showToast`/`cache` context --
 *     no need to boot the full page.
 *   - session.js: the page entry point's refresh-loop view dispatch (nav
 *     click -> activeView change -> the matching session-views.js
 *     renderer runs against the right DOM section) and the initial-load /
 *     periodic-refresh call. The same-origin `osprey-session-change`
 *     receiver's foreign-origin-rejection contract is pinned separately
 *     by the real-browser test_contract_params.py suite (this file does
 *     not duplicate that -- see the module docstring there).
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally), fetch
 * stubbed (never a real network call):
 *   npx vitest run tests/interfaces/web_terminal/session-views.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const VIEWS_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/session-views.js';
const ENTRY_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/session.js';

// ---------------------------------------------------------------------------
// session-views.js -- renderer output from fixture JSON
// ---------------------------------------------------------------------------

describe('session-views renderers', () => {
  /** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/session-views.js')} */
  let Views;

  beforeEach(async () => {
    vi.resetModules();
    Views = await import(VIEWS_PATH);
    document.body.innerHTML = `
      <section id="view-agents"></section>
      <section id="view-toollog"></section>
      <section id="view-artifacts"></section>
      <section id="view-conversation"></section>
    `;
  });

  function ctx({ apiFetch, cache = {} } = {}) {
    return { apiFetch, showToast: vi.fn(), cache };
  }

  describe('renderAgents', () => {
    const FIXTURE = {
      total_events: 5,
      agents: [
        { agent_id: null },
        { agent_id: 'sub-1', agent_type: 'code-reviewer', duration: '2.1s', server: 'python' },
      ],
      tool_calls_by_agent: {
        main: [],
        'sub-1': [
          { tool_name: 'Read', server_name: 'python', is_error: false, timestamp: '2026-01-01T00:00:00Z' },
          { tool_name: 'Bash', server_name: 'controls', is_error: true, timestamp: '2026-01-01T00:00:01Z' },
        ],
      },
    };

    test('renders the root session card and a sub-agent card with its tool rows', async () => {
      const apiFetch = vi.fn().mockResolvedValue(FIXTURE);
      await Views.renderAgents(ctx({ apiFetch }));

      const el = document.getElementById('view-agents');
      const cards = el.querySelectorAll('.agent-card');
      expect(cards.length).toBe(2);
      expect(el.textContent).toContain('ROOT SESSION');
      expect(el.textContent).toContain('code-reviewer');
      expect(el.querySelectorAll('.tool-row').length).toBe(2);
      expect(el.querySelectorAll('.badge-error').length).toBe(1);
    });

    test('caches the fetched data on the passed-in cache object', async () => {
      const cache = {};
      await Views.renderAgents(ctx({ apiFetch: vi.fn().mockResolvedValue(FIXTURE), cache }));
      expect(cache.agents).toEqual(FIXTURE);
    });

    test('shows an empty state and clears the cache when there is no activity', async () => {
      const cache = { agents: FIXTURE };
      await Views.renderAgents(
        ctx({ apiFetch: vi.fn().mockResolvedValue({ agents: [], total_events: 0 }), cache })
      );
      expect(document.getElementById('view-agents').textContent).toContain('NO AGENTS');
      expect(cache.agents).toBeNull();
    });

    test('on fetch failure, toasts and falls back to an error state only when nothing was cached', async () => {
      const showToast = vi.fn();
      const apiFetch = vi.fn().mockRejectedValue(new Error('network down'));
      await Views.renderAgents({ apiFetch, showToast, cache: {} });
      expect(showToast).toHaveBeenCalledWith('Failed to load agents');
      expect(document.getElementById('view-agents').textContent).toContain('ERROR');
    });

    test('on fetch failure with a previously cached render, leaves the stale DOM untouched', async () => {
      document.getElementById('view-agents').innerHTML = '<div class="agent-card">stale</div>';
      const apiFetch = vi.fn().mockRejectedValue(new Error('network down'));
      await Views.renderAgents({ apiFetch, showToast: vi.fn(), cache: { agents: FIXTURE } });
      expect(document.getElementById('view-agents').innerHTML).toContain('stale');
    });

    test('expanding a sub-agent card lazily loads its timeline exactly once', async () => {
      const timelineFetch = vi.fn().mockResolvedValue({
        timeline: [{ kind: 'tool_call', tool: 'Read', arguments: { path: '/x' } }],
      });
      const apiFetch = vi.fn((path) =>
        path.startsWith('/api/session-agent-timeline') ? timelineFetch(path) : Promise.resolve(FIXTURE)
      );
      await Views.renderAgents(ctx({ apiFetch }));

      const el = document.getElementById('view-agents');
      const subCard = [...el.querySelectorAll('.agent-card')].find((c) => c.dataset.agent === 'sub-1');
      subCard.querySelector('.agent-card-header').dispatchEvent(new Event('click'));
      await Promise.resolve();
      await Promise.resolve();

      expect(timelineFetch).toHaveBeenCalledWith('/api/session-agent-timeline?agent_id=sub-1');
      expect(subCard.querySelector('.agent-timeline').textContent).toContain('Read');

      // Collapsing and re-expanding must not re-fetch (dataset.loaded guard).
      subCard.querySelector('.agent-card-header').dispatchEvent(new Event('click'));
      subCard.querySelector('.agent-card-header').dispatchEvent(new Event('click'));
      await Promise.resolve();
      expect(timelineFetch).toHaveBeenCalledTimes(1);
    });
  });

  describe('renderToolLog', () => {
    const FIXTURE = {
      events: [
        { tool_name: 'Read', server_name: 'python', agent_id: 'sub-1', is_error: false, timestamp: '2026-01-01T00:00:00Z' },
        { tool_name: 'Bash', server_name: 'controls', agent_id: 'main', is_error: true, timestamp: '2026-01-01T00:00:01Z' },
      ],
    };

    test('renders one log row per event with server/agent columns', async () => {
      await Views.renderToolLog(ctx({ apiFetch: vi.fn().mockResolvedValue(FIXTURE) }));
      const el = document.getElementById('view-toollog');
      expect(el.querySelectorAll('.log-row').length).toBe(2);
      expect(el.querySelectorAll('.log-row.is-error').length).toBe(1);
      expect(el.querySelector('#filter-agent')).not.toBeNull();
    });

    test('shows an empty state (with a still-usable filter bar) when there are no events', async () => {
      await Views.renderToolLog(ctx({ apiFetch: vi.fn().mockResolvedValue({ events: [] }) }));
      const el = document.getElementById('view-toollog');
      expect(el.textContent).toContain('NO TOOL CALLS');
      expect(el.querySelector('#filter-agent')).not.toBeNull();
    });

    test('toggling the errors-only filter re-fetches with errors_only=true', async () => {
      const apiFetch = vi.fn().mockResolvedValue(FIXTURE);
      await Views.renderToolLog(ctx({ apiFetch }));

      const el = document.getElementById('view-toollog');
      const errCb = el.querySelector('#filter-errors');
      errCb.checked = true;
      errCb.dispatchEvent(new Event('change'));
      await Promise.resolve();

      const lastCall = apiFetch.mock.calls.at(-1)[0];
      expect(lastCall).toContain('errors_only=true');
    });
  });

  describe('renderArtifacts', () => {
    const FIXTURE = {
      totals: { entry_count: 2, total_bytes: 2048, categories: { plot: 1, table: 1 } },
      entries: [
        { category: 'plot', artifact_type: 'plot', title: 'Orbit response', size_bytes: 1024, channels: ['SR:C01'] },
        { category: 'table', artifact_type: 'table', name: 'Results', size_bytes: 1024 },
      ],
    };

    test('renders grouped artifact entries with totals', async () => {
      await Views.renderArtifacts(ctx({ apiFetch: vi.fn().mockResolvedValue(FIXTURE) }));
      const el = document.getElementById('view-artifacts');
      expect(el.querySelectorAll('.artifact-group').length).toBe(2);
      expect(el.textContent).toContain('Orbit response');
      expect(el.textContent).toContain('2.0 KB');
      expect(el.querySelectorAll('.channel-tag').length).toBe(1);
    });

    test('shows an empty state when there are no entries', async () => {
      await Views.renderArtifacts(ctx({ apiFetch: vi.fn().mockResolvedValue({ entries: [] }) }));
      expect(document.getElementById('view-artifacts').textContent).toContain('NO ARTIFACTS');
    });
  });

  describe('renderConversation', () => {
    const FIXTURE = {
      turns: [
        { role: 'user', content: 'What is the beam current?', timestamp: '2026-01-01T00:00:00Z' },
        { role: 'assistant', text: '<b>102 mA</b>' },
      ],
    };

    test('renders one chat bubble per turn, escaping HTML in content', async () => {
      await Views.renderConversation(ctx({ apiFetch: vi.fn().mockResolvedValue(FIXTURE) }));
      const el = document.getElementById('view-conversation');
      expect(el.querySelectorAll('.chat-msg').length).toBe(2);
      expect(el.querySelector('.chat-msg.user')).not.toBeNull();
      // turn.text is escaped, not parsed as markup.
      expect(el.innerHTML).toContain('&lt;b&gt;102 mA&lt;/b&gt;');
    });

    test('shows an empty state when the transcript is empty', async () => {
      await Views.renderConversation(ctx({ apiFetch: vi.fn().mockResolvedValue({ turns: [] }) }));
      expect(document.getElementById('view-conversation').textContent).toContain('NO CONVERSATION');
    });
  });
});

// ---------------------------------------------------------------------------
// session.js -- refresh-loop view dispatch
// ---------------------------------------------------------------------------

describe('session.js refresh-loop view dispatch', () => {
  /** Fixture bodies keyed by the endpoint path each view's apiFetch call hits. */
  const FIXTURES = {
    '/api/session-agents': { total_events: 1, agents: [{ agent_id: null }], tool_calls_by_agent: { main: [] } },
    '/api/session-log': { events: [{ tool_name: 'Read', server_name: 'python', agent_id: 'main', timestamp: '2026-01-01T00:00:00Z' }] },
    '/api/session-summary': { totals: { entry_count: 0, total_bytes: 0, categories: {} }, entries: [] },
    '/api/session-chat': { turns: [{ role: 'user', content: 'hi' }] },
  };

  function stubFetch() {
    vi.stubGlobal(
      'fetch',
      vi.fn((path) => {
        const key = Object.keys(FIXTURES).find((k) => path.startsWith(k));
        return Promise.resolve({
          status: 200,
          ok: true,
          json: () => Promise.resolve(FIXTURES[key] ?? null),
        });
      })
    );
  }

  beforeEach(() => {
    document.body.innerHTML = `
      <header>
        <h1>Activity</h1>
        <osprey-theme-switcher></osprey-theme-switcher>
        <div class="refresh-dot" id="refresh-dot"></div>
      </header>
      <nav id="nav">
        <button class="pill active" data-view="agents">Agents</button>
        <button class="pill" data-view="toollog">Tool Log</button>
        <button class="pill" data-view="artifacts">Artifacts</button>
        <button class="pill" data-view="conversation">Conversation</button>
      </nav>
      <main>
        <section class="view active" id="view-agents"></section>
        <section class="view" id="view-toollog"></section>
        <section class="view" id="view-artifacts"></section>
        <section class="view" id="view-conversation"></section>
      </main>
      <div class="toast" id="toast"></div>
    `;
    stubFetch();
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  async function flush() {
    // Let the initial refreshActive()'s awaited apiFetch chain settle.
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  }

  test('initial load dispatches to the Agents view (default activeView)', async () => {
    await import(ENTRY_PATH);
    await flush();
    expect(document.getElementById('view-agents').textContent).toContain('Events');
    expect(fetch).toHaveBeenCalledWith('/api/session-agents', expect.anything());
  });

  test('clicking a nav pill dispatches refreshActive to the matching view only', async () => {
    await import(ENTRY_PATH);
    await flush();

    document.querySelector('.pill[data-view="toollog"]').dispatchEvent(
      new MouseEvent('click', { bubbles: true })
    );
    await flush();

    expect(document.getElementById('view-toollog').querySelectorAll('.log-row').length).toBe(1);
    expect(document.querySelector('.pill[data-view="toollog"]').classList.contains('active')).toBe(true);
    expect(document.querySelector('.pill[data-view="agents"]').classList.contains('active')).toBe(false);
    expect(document.getElementById('view-toollog').classList.contains('active')).toBe(true);
    expect(document.getElementById('view-agents').classList.contains('active')).toBe(false);
  });

  test('an accepted same-origin osprey-session-change resets the cache and re-dispatches the active view', async () => {
    await import(ENTRY_PATH);
    await flush();

    fetch.mockClear();
    window.dispatchEvent(
      new MessageEvent('message', {
        origin: window.location.origin,
        data: { type: 'osprey-session-change', session_id: 'test-session-42' },
      })
    );
    await flush();

    const [path] = fetch.mock.calls.at(-1);
    expect(path).toContain('/api/session-agents');
    expect(path).toContain('session_id=test-session-42');
  });

  test('a foreign-origin osprey-session-change is ignored (no re-dispatch with a session id)', async () => {
    await import(ENTRY_PATH);
    await flush();

    fetch.mockClear();
    window.dispatchEvent(
      new MessageEvent('message', {
        origin: 'https://evil.example',
        data: { type: 'osprey-session-change', session_id: 'evil-session-999' },
      })
    );
    await flush();

    expect(fetch).not.toHaveBeenCalled();
  });
});
