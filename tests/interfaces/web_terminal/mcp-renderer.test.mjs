/**
 * Unit tests for mcp-renderer.js -- the .mcp.json renderer behind
 * config-renderers.js's re-export (same seam as settings-editor.js).
 * Covers:
 *
 *   - `_parseToolDescription`'s Google-style docstring parsing on
 *     multi-line and empty descriptions
 *   - `renderMcpJson`'s progressive enhancement: cards render immediately
 *     from raw JSON, then get enriched in-place once the mocked
 *     /api/mcp-servers fetch resolves
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/mcp-renderer.test.mjs
 */

import { test, expect, describe, vi, afterEach } from 'vitest';

import { qs } from '../_support/dom.mjs';

import {
  renderMcpJson,
  _parseToolDescription,
} from '../../../src/osprey/interfaces/web_terminal/static/js/mcp-renderer.js';

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
      env: { SCAN_TOKEN: '${SCAN_TOKEN}' },
    },
  },
}, null, 2);

describe('_parseToolDescription', () => {
  test('empty/nullish description returns empty summary/args/returns', () => {
    expect(_parseToolDescription('')).toEqual({ summary: '', args: [], returns: '' });
    // Runtime-defensive paths: the declared signature is `string`, but callers in
    // practice may hand back a nullish `description` field, so verify the nullish
    // fallback directly by deliberately passing values outside the declared type.
    expect(_parseToolDescription(/** @type {string} */ (/** @type {unknown} */ (undefined)))).toEqual({ summary: '', args: [], returns: '' });
    expect(_parseToolDescription(/** @type {string} */ (/** @type {unknown} */ (null)))).toEqual({ summary: '', args: [], returns: '' });
  });

  test('a single-line summary with no Args/Returns section', () => {
    const result = _parseToolDescription('Look up a channel by name.');
    expect(result.summary).toBe('Look up a channel by name.');
    expect(result.args).toEqual([]);
    expect(result.returns).toBe('');
  });

  test('a multi-line docstring splits summary, args, and returns', () => {
    const desc = [
      'Search for a channel across the ontology.',
      'Falls back to fuzzy matching when an exact name is not found.',
      '',
      'Args:',
      '    name: The channel name or alias to search for.',
      '    limit: Maximum number of results to return.',
      '',
      'Returns:',
      'A list of matching channel records.',
    ].join('\n');

    const result = _parseToolDescription(desc);

    expect(result.summary).toBe(
      'Search for a channel across the ontology. Falls back to fuzzy matching when an exact name is not found.'
    );
    expect(result.args).toEqual([
      { name: 'name', desc: 'The channel name or alias to search for.' },
      { name: 'limit', desc: 'Maximum number of results to return.' },
    ]);
    expect(result.returns).toBe('A list of matching channel records.');
  });

  test('a multi-line arg description is joined onto the same arg', () => {
    const desc = [
      'Args:',
      '    name: The channel name',
      '        spanning multiple lines.',
    ].join('\n');

    const result = _parseToolDescription(desc);

    expect(result.args).toEqual([
      { name: 'name', desc: 'The channel name spanning multiple lines.' },
    ]);
  });

  test('a Raises section is skipped entirely (not folded into returns)', () => {
    const desc = [
      'Do a thing.',
      '',
      'Raises:',
      '    ValueError: if the thing cannot be done.',
      '',
      'Returns:',
      'Nothing of note.',
    ].join('\n');

    const result = _parseToolDescription(desc);

    expect(result.summary).toBe('Do a thing.');
    expect(result.returns).toBe('Nothing of note.');
  });
});

describe('renderMcpJson', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
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
