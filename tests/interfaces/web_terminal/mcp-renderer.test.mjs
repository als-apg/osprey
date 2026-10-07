/**
 * Unit tests for mcp-renderer.js -- the .mcp.json renderer behind
 * config-renderers.js's re-export.
 * Covers:
 *
 *   - `renderMcpJson`'s progressive enhancement: cards render immediately
 *     from raw JSON, then get enriched in-place once the mocked
 *     /api/mcp-servers fetch resolves
 *   - how each tool's Google-style docstring renders in the enriched card
 *     (summary, ARGS rows, RETURNS; a Raises section is dropped)
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/mcp-renderer.test.mjs
 */

import { test, expect, describe, vi, afterEach } from 'vitest';

import { qs } from '../_support/dom.mjs';

import { renderMcpJson } from '../../../src/osprey/interfaces/web_terminal/static/js/mcp-renderer.js';

/**
 * `renderMcpJson` returns `HTMLDivElement | null` (null only on invalid JSON
 * or an empty `mcpServers`); these tests feed it valid multi-server JSON, so
 * assert the non-null case once here rather than re-guarding at every call
 * site.
 * @param {string} jsonString
 * @returns {HTMLDivElement}
 */
function renderContainer(jsonString) {
  const container = renderMcpJson(jsonString);
  if (container === null) throw new Error('expected renderMcpJson to return a container');
  return container;
}

const MCP_JSON = JSON.stringify({
  mcpServers: {
    bluesky: {
      command: 'python',
      args: ['-m', 'osprey.mcp_server.bluesky'],
      env: { BLUESKY_LAUNCH_TOKEN: '${BLUESKY_LAUNCH_TOKEN}' },
    },
  },
}, null, 2);

describe('renderMcpJson', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    delete window.__OSPREY_PREFIX__;
  });

  test('returns null on invalid JSON (matches the other renderers\' parse-guard)', () => {
    expect(renderMcpJson('not json')).toBeNull();
  });

  test('returns null when mcpServers is absent or empty', () => {
    expect(renderMcpJson(JSON.stringify({}))).toBeNull();
    expect(renderMcpJson(JSON.stringify({ mcpServers: {} }))).toBeNull();
  });

  test('renders a basic card synchronously, before any fetch resolves', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {}))); // never resolves

    const container = renderContainer(MCP_JSON);

    const card = qs(container, '.config-mcp-card');
    expect(card).not.toBeNull();
    expect(qs(card, '.config-mcp-card-name').textContent).toBe('bluesky');
    // Loading placeholder present before enrichment lands.
    expect(card.querySelector('.config-mcp-tools-loading')).not.toBeNull();
  });

  test('enriches a card in-place once /api/mcp-servers resolves (fetch mocked)', async () => {
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve([
          {
            name: 'bluesky',
            description: 'Bluesky plan/run control server.',
            tool_count: 1,
            tools: [
              { name: 'launch_run', description: 'Args:\n    plan: The plan name.' },
            ],
          },
        ]),
      })
    ));

    const container = renderContainer(MCP_JSON);
    const card = qs(container, '.config-mcp-card');

    // Let the fetch/.then chain settle.
    await new Promise((resolve) => setTimeout(resolve, 0));
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(qs(card, '.config-mcp-card-desc').textContent).toBe('Bluesky plan/run control server.');
    const badge = qs(card, '.config-mcp-tool-count');
    expect(badge.textContent).toBe('1');
    expect(badge.style.display).not.toBe('none');
    expect(card.querySelector('.config-mcp-tools-loading')).toBeNull();
    const toolItem = qs(card, '.config-mcp-tool-item');
    expect(toolItem).not.toBeNull();
    expect(qs(toolItem, '.config-mcp-tool-name').textContent).toBe('launch_run');
  });

  /**
   * Enrich the bluesky card with one tool carrying `description` and hand back
   * the rendered tool item.
   * @param {string|null|undefined} description
   */
  async function renderTool(description) {
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve([
          { name: 'bluesky', tool_count: 1, tools: [{ name: 'launch_run', description }] },
        ]),
      })
    ));
    const card = qs(renderContainer(MCP_JSON), '.config-mcp-card');
    await new Promise((resolve) => setTimeout(resolve, 0));
    await new Promise((resolve) => setTimeout(resolve, 0));
    return qs(card, '.config-mcp-tool-item');
  }

  /** @param {Element} item @param {string} selector */
  const texts = (item, selector) =>
    [...item.querySelectorAll(selector)].map((el) => el.textContent);

  test.each([[''], [undefined], [null]])('a tool with no docstring (%o) renders its name and no detail', async (description) => {
    const item = await renderTool(description);

    expect(qs(item, '.config-mcp-tool-name').textContent).toBe('launch_run');
    expect(item.querySelector('.config-mcp-tool-summary')).toBeNull();
    expect(item.querySelector('.config-mcp-tool-body')).toBeNull();
  });

  test('a single-line docstring is a summary with no ARGS or RETURNS', async () => {
    const item = await renderTool('Look up a channel by name.');

    expect(qs(item, '.config-mcp-tool-summary').textContent).toBe('Look up a channel by name.');
    expect(qs(item, '.config-mcp-tool-desc-full').textContent).toBe('Look up a channel by name.');
    expect(item.querySelector('.config-mcp-tool-section')).toBeNull();
  });

  test('a multi-line docstring splits into summary, ARGS rows and RETURNS', async () => {
    const item = await renderTool([
      'Search for a channel across the ontology.',
      'Falls back to fuzzy matching when an exact name is not found.',
      '',
      'Args:',
      '    name: The channel name or alias to search for.',
      '    limit: Maximum number of results to return.',
      '',
      'Returns:',
      'A list of matching channel records.',
    ].join('\n'));

    expect(qs(item, '.config-mcp-tool-desc-full').textContent).toBe(
      'Search for a channel across the ontology. Falls back to fuzzy matching when an exact name is not found.'
    );
    expect(texts(item, '.config-mcp-tool-arg-name')).toEqual(['name', 'limit']);
    expect(texts(item, '.config-mcp-tool-arg-desc')).toEqual([
      'The channel name or alias to search for.',
      'Maximum number of results to return.',
    ]);
    expect(qs(item, '.config-mcp-tool-returns').textContent).toBe('A list of matching channel records.');
  });

  test('an arg description continued on the next line stays one arg', async () => {
    const item = await renderTool(['Args:', '    name: The channel name', '        spanning multiple lines.'].join('\n'));

    expect(texts(item, '.config-mcp-tool-arg-name')).toEqual(['name']);
    expect(texts(item, '.config-mcp-tool-arg-desc')).toEqual(['The channel name spanning multiple lines.']);
  });

  test('a Raises section is dropped, not folded into RETURNS', async () => {
    const item = await renderTool([
      'Do a thing.',
      '',
      'Raises:',
      '    ValueError: if the thing cannot be done.',
      '',
      'Returns:',
      'Nothing of note.',
    ].join('\n'));

    expect(qs(item, '.config-mcp-tool-desc-full').textContent).toBe('Do a thing.');
    expect(qs(item, '.config-mcp-tool-returns').textContent).toBe('Nothing of note.');
    expect(item.textContent).not.toContain('ValueError');
  });

  test('prepends window.__OSPREY_PREFIX__ to the /api/mcp-servers fetch (multi-user deployments)', () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const fetchMock = vi.fn(() => new Promise(() => {})); // never resolves
    vi.stubGlobal('fetch', fetchMock);

    renderContainer(MCP_JSON);

    expect(fetchMock).toHaveBeenCalledWith('/u/alice/api/mcp-servers', { cache: 'no-store' });
  });

  test('falls back to "tools not available" when the fetch rejects', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('network down'))));

    const container = renderContainer(MCP_JSON);
    const card = qs(container, '.config-mcp-card');

    await new Promise((resolve) => setTimeout(resolve, 0));
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(card.querySelector('.config-mcp-tools-loading')).toBeNull();
    const fallback = qs(card, '.config-mcp-tools-fallback');
    expect(fallback).not.toBeNull();
    expect(fallback.textContent).toBe('tools not available');
  });
});
