/**
 * Unit tests for the Scaffold Gallery pure utilities (scaffold/utils.js).
 *
 * Pure-logic/DOM guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/scaffold-utils.test.mjs
 *
 * Covers YAML front-matter parsing (valid/missing/malformed), Python
 * docstring front-matter + flow-diagram extraction, the category-help table's
 * alignment with the displayed categories, and the rendering helpers. Which
 * tab each artifact lands on is pinned at initScaffoldGallery in
 * scaffold-view.test.mjs.
 *
 * NOTE: imported by RELATIVE path, not an absolute `/design-system/js/*`-style
 * specifier — this module lives under web_terminal, not design-system, so no
 * alias applies. Mirrors tests/interfaces/design_system/js/dom.test.js.
 */

import { test, expect, describe } from 'vitest';

import { qs } from '../_support/dom.mjs';

import {
  CATEGORY_HELP,
  BEHAVIOR_CATEGORIES,
  BEHAVIOR_CATEGORY_OVERRIDES,
  BEHAVIOR_CATEGORY_REMAPS,
  BEHAVIOR_PINNED_CATEGORIES,
  SAFETY_CATEGORIES,
  parseFrontMatter,
  extractPythonDocstringFrontMatter,
  renderHighlightedCode,
  renderFlowDiagram,
  renderSourceToggle,
  renderFrontMatterTable,
} from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/utils.js';

describe('parseFrontMatter', () => {
  test.each([
    ['valid front matter is split from the body',
      '---\nname: my-agent\nmodel: sonnet\n---\n# Body\n\nSome content here.',
      { name: 'my-agent', model: 'sonnet' }, '# Body\n\nSome content here.'],
    ['single and double quotes are stripped from values',
      '---\ndescription: "a quoted value"\nlabel: \'single quoted\'\n---\nbody',
      { description: 'a quoted value', label: 'single quoted' }, 'body'],
    ['no --- delimiters: null front matter, content as-is',
      '# Just a markdown file\n\nNo front matter here.',
      null, '# Just a markdown file\n\nNo front matter here.'],
    ['an unterminated block: null front matter, content as-is',
      '---\nname: broken\nThere is no closing delimiter.',
      null, '---\nname: broken\nThere is no closing delimiter.'],
    ['a block with no key: value line: null front matter',
      '---\njust some prose, not a yaml key\n---\nbody text',
      null, 'body text'],
    ['a block with nothing after it: empty body',
      '---\nname: solo\n---\n',
      { name: 'solo' }, ''],
  ])('%s', (_label, content, expectedFrontMatter, expectedBody) => {
    const { frontMatter, body } = parseFrontMatter(content);
    expect(frontMatter).toEqual(expectedFrontMatter);
    expect(body).toBe(expectedBody);
  });
});

describe('extractPythonDocstringFrontMatter', () => {
  const WITH_FLOW = [
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

  test.each([
    ['front matter and a Flow diagram from a module docstring', WITH_FLOW,
      { frontMatter: { name: 'my_hook', event: 'PreToolUse' }, flowDiagram: 'A -> B -> C', body: 'Some docstring prose.' }],
    ['a docstring with no Flow section', '"""\n---\nname: plain\n---\nJust prose, no flow.\n"""\n',
      { frontMatter: { name: 'plain' }, flowDiagram: null, body: 'Just prose, no flow.' }],
    ['a leading shebang line', '#!/usr/bin/env python3\n"""\n---\nname: shebanged\n---\nbody\n"""\n',
      { frontMatter: { name: 'shebanged' } }],
    ['no docstring at all', 'def main():\n    pass\n',
      { frontMatter: null, flowDiagram: null, body: '' }],
  ])('%s', (_label, content, expected) => {
    const result = extractPythonDocstringFrontMatter(content);
    expect(result).toMatchObject(expected);
    expect(result.sourceCode).toBe(content);
  });
});

describe('category help', () => {
  // CATEGORY_HELP is keyed by DISPLAY category, so a key that matches no
  // reachable display category is dead text, and a display category with no
  // key renders a section with no `?` at all -- silently. The displayed set is
  // every raw category a gallery admits after the Behavior remaps, plus the
  // per-name overrides, plus `config` (mcp-json and settings-json carry no "/"
  // in their canonical name, so the service reports them under that catch-all).
  test('its keys are exactly the display categories a gallery can render, pinned ones included', () => {
    const displayed = new Set(['config']);
    for (const raw of [...BEHAVIOR_CATEGORIES, ...SAFETY_CATEGORIES]) {
      displayed.add(BEHAVIOR_CATEGORY_REMAPS[raw] || raw);
    }
    for (const override of Object.values(BEHAVIOR_CATEGORY_OVERRIDES)) {
      displayed.add(override);
    }

    expect(new Set(Object.keys(CATEGORY_HELP))).toEqual(displayed);
    expect(BEHAVIOR_PINNED_CATEGORIES.filter((cat) => !displayed.has(cat))).toEqual([]);
  });
});

describe('rendering helpers', () => {
  test('renderHighlightedCode produces a <pre><code> block with the given text and language class', () => {
    const pre = renderHighlightedCode('print(1)', 'python');
    expect(pre.tagName).toBe('PRE');
    const code = qs(pre, 'code');
    expect(code.className).toBe('language-python');
    expect(code.textContent).toBe('print(1)');
  });

  test('renderFlowDiagram renders the diagram text inside a labeled pre block', () => {
    const section = renderFlowDiagram('A -> B');
    expect(section.className).toBe('prompts-flow-diagram');
    expect(section.textContent).toContain('FLOW');
    expect(qs(section, 'pre code').textContent).toBe('A -> B');
  });

  test('renderSourceToggle wraps highlighted source in a collapsible toggle, starting collapsed', () => {
    const container = renderSourceToggle('x = 1', 'python');
    const content = qs(container, '.prompts-source-content');
    expect(content.classList.contains('expanded')).toBe(false);
    expect(qs(content, 'code').textContent).toBe('x = 1');
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
