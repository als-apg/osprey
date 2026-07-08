// @ts-nocheck
// TODO(frontend-hardening Pn): remove & fix types when this interface is retrofitted (P2–P5)
/**
 * Unit tests for the Scaffold Gallery pure utilities (scaffold/utils.js).
 *
 * Pure-logic/DOM guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/scaffold-utils.test.mjs
 *
 * Covers YAML front-matter parsing (valid/missing/malformed), Python
 * docstring front-matter + flow-diagram extraction, category-routing set
 * membership, and the rendering helpers.
 *
 * NOTE: imported by RELATIVE path, not an absolute `/design-system/js/*`-style
 * specifier — this module lives under web_terminal, not design-system, so no
 * alias applies. Mirrors tests/interfaces/design_system/js/dom.test.js.
 */

import { test, expect, describe } from 'vitest';

import {
  AGENT_MODEL_OPTIONS,
  CATEGORY_HELP,
  BEHAVIOR_CATEGORIES,
  BEHAVIOR_NAMES,
  SAFETY_CATEGORIES,
  CONFIG_NAMES,
  configureMarked,
  iconForCategory,
  parseFrontMatter,
  extractPythonDocstringFrontMatter,
  renderHighlightedCode,
  renderFlowDiagram,
  renderSourceToggle,
  renderFrontMatterTable,
} from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/utils.js';

describe('parseFrontMatter', () => {
  test('valid front matter is parsed into a fields object and the body is separated', () => {
    const content = '---\nname: my-agent\nmodel: sonnet\n---\n# Body\n\nSome content here.';
    const { frontMatter, body } = parseFrontMatter(content);
    expect(frontMatter).toEqual({ name: 'my-agent', model: 'sonnet' });
    expect(body).toBe('# Body\n\nSome content here.');
  });

  test('quoted values (single or double) have the quotes stripped', () => {
    const content = '---\ndescription: "a quoted value"\nlabel: \'single quoted\'\n---\nbody';
    const { frontMatter } = parseFrontMatter(content);
    expect(frontMatter).toEqual({ description: 'a quoted value', label: 'single quoted' });
  });

  test('missing front matter (no --- delimiters) returns null frontMatter and the content as-is', () => {
    const content = '# Just a markdown file\n\nNo front matter here.';
    const { frontMatter, body } = parseFrontMatter(content);
    expect(frontMatter).toBeNull();
    expect(body).toBe(content);
  });

  test('malformed front matter (unterminated block) falls back to null frontMatter, full content as body', () => {
    const content = '---\nname: broken\nThere is no closing delimiter.';
    const { frontMatter, body } = parseFrontMatter(content);
    expect(frontMatter).toBeNull();
    expect(body).toBe(content);
  });

  test('an empty front-matter block (no parseable key: value lines) yields null frontMatter', () => {
    const content = '---\njust some prose, not a yaml key\n---\nbody text';
    const { frontMatter, body } = parseFrontMatter(content);
    expect(frontMatter).toBeNull();
    expect(body).toBe('body text');
  });

  test('a front-matter block with no body after it yields an empty body', () => {
    const content = '---\nname: solo\n---\n';
    const { frontMatter, body } = parseFrontMatter(content);
    expect(frontMatter).toEqual({ name: 'solo' });
    expect(body).toBe('');
  });
});

describe('extractPythonDocstringFrontMatter', () => {
  test('extracts front matter and a Flow diagram from a module docstring', () => {
    const content = [
      '"""',
      '---',
      'name: my_hook',
      'event: PreToolUse',
      '---',
      'Some docstring prose.',
      '',
      '## Flow',
      '```',
      'A -> B -> C',
      '```',
      '"""',
      '',
      'def main():',
      '    pass',
    ].join('\n');

    const result = extractPythonDocstringFrontMatter(content);
    expect(result.frontMatter).toEqual({ name: 'my_hook', event: 'PreToolUse' });
    expect(result.flowDiagram).toBe('A -> B -> C');
    expect(result.body).toBe('Some docstring prose.');
    expect(result.sourceCode).toBe(content);
  });

  test('handles a docstring with no Flow section (flowDiagram stays null)', () => {
    const content = '"""\n---\nname: plain\n---\nJust prose, no flow.\n"""\n';
    const result = extractPythonDocstringFrontMatter(content);
    expect(result.frontMatter).toEqual({ name: 'plain' });
    expect(result.flowDiagram).toBeNull();
    expect(result.body).toBe('Just prose, no flow.');
  });

  test('a leading shebang line before the docstring is tolerated', () => {
    const content = '#!/usr/bin/env python3\n"""\n---\nname: shebanged\n---\nbody\n"""\n';
    const result = extractPythonDocstringFrontMatter(content);
    expect(result.frontMatter).toEqual({ name: 'shebanged' });
  });

  test('content with no docstring at all returns the null/empty defaults, preserving sourceCode', () => {
    const content = 'def main():\n    pass\n';
    const result = extractPythonDocstringFrontMatter(content);
    expect(result.frontMatter).toBeNull();
    expect(result.body).toBe('');
    expect(result.flowDiagram).toBeNull();
    expect(result.sourceCode).toBe(content);
  });
});

describe('category-routing set membership', () => {
  test('BEHAVIOR_CATEGORIES routes agents/skills/rules/output-styles to the Behavior tab', () => {
    expect(BEHAVIOR_CATEGORIES.has('agents')).toBe(true);
    expect(BEHAVIOR_CATEGORIES.has('skills')).toBe(true);
    expect(BEHAVIOR_CATEGORIES.has('rules')).toBe(true);
    expect(BEHAVIOR_CATEGORIES.has('output-styles')).toBe(true);
    expect(BEHAVIOR_CATEGORIES.has('hooks')).toBe(false);
  });

  test('BEHAVIOR_NAMES routes the claude-md config artifact to the Behavior tab', () => {
    expect(BEHAVIOR_NAMES.has('claude-md')).toBe(true);
    expect(BEHAVIOR_NAMES.has('mcp-json')).toBe(false);
  });

  test('SAFETY_CATEGORIES routes hooks to the Safety tab, and nothing else', () => {
    expect(SAFETY_CATEGORIES.has('hooks')).toBe(true);
    expect(SAFETY_CATEGORIES.has('agents')).toBe(false);
  });

  test('CONFIG_NAMES routes mcp-json/settings-json to the Config tab', () => {
    expect(CONFIG_NAMES.has('mcp-json')).toBe(true);
    expect(CONFIG_NAMES.has('settings-json')).toBe(true);
    expect(CONFIG_NAMES.has('claude-md')).toBe(false);
  });

  test('CATEGORY_HELP and AGENT_MODEL_OPTIONS are populated, non-empty', () => {
    expect(Object.keys(CATEGORY_HELP).length).toBeGreaterThan(0);
    expect(CATEGORY_HELP.hooks).toMatch(/before or after Claude uses a tool/i);
    expect(AGENT_MODEL_OPTIONS).toContain('sonnet');
  });
});

describe('iconForCategory', () => {
  test('returns a distinct icon per known category, case-insensitively', () => {
    expect(iconForCategory('Hooks')).not.toBe(iconForCategory('agents'));
    expect(iconForCategory('hooks')).toBe(iconForCategory('HOOKS'));
  });

  test('unknown or missing categories fall back to the default document icon', () => {
    expect(iconForCategory('something-unrecognized')).toBe(iconForCategory(undefined));
  });
});

describe('configureMarked', () => {
  test('is a no-op (does not throw) when the vendored `marked` global is absent', () => {
    expect(typeof marked).toBe('undefined');
    expect(() => configureMarked()).not.toThrow();
    // Idempotent: calling it again is still safe.
    expect(() => configureMarked()).not.toThrow();
  });
});

describe('rendering helpers', () => {
  test('renderHighlightedCode produces a <pre><code> block with the given text and language class', () => {
    const pre = renderHighlightedCode('print(1)', 'python');
    expect(pre.tagName).toBe('PRE');
    const code = pre.querySelector('code');
    expect(code.className).toBe('language-python');
    expect(code.textContent).toBe('print(1)');
  });

  test('renderFlowDiagram renders the diagram text inside a labeled pre block', () => {
    const section = renderFlowDiagram('A -> B');
    expect(section.className).toBe('prompts-flow-diagram');
    expect(section.textContent).toContain('FLOW');
    expect(section.querySelector('pre code').textContent).toBe('A -> B');
  });

  test('renderSourceToggle wraps highlighted source in a collapsible toggle, starting collapsed', () => {
    const container = renderSourceToggle('x = 1', 'python');
    const content = container.querySelector('.prompts-source-content');
    expect(content.classList.contains('expanded')).toBe(false);
    expect(content.querySelector('code').textContent).toBe('x = 1');
  });

  test('renderFrontMatterTable renders one row per field, with tools/model/safety_layer as pills', () => {
    const table = renderFrontMatterTable({
      model: 'sonnet',
      tools: 'Read, Edit',
      safety_layer: '2',
      description: 'plain text value',
    });
    const rows = table.querySelectorAll('.prompts-fm-row');
    expect(rows.length).toBe(4);
    expect(table.querySelectorAll('.prompts-fm-pill-accent').length).toBe(1);
    expect(table.querySelectorAll('.prompts-fm-pill-shield')[0].textContent).toContain('Layer 2');
    expect(table.textContent).toContain('plain text value');
  });
});
