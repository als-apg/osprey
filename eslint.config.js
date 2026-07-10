import js from '@eslint/js';
import globals from 'globals';
import noTsNocheck from './tools/eslint/no-ts-nocheck.js';

export default [
  // (1) Leading SOLE-KEY global ignore — must be the ONLY key in this object so it applies globally.
  { ignores: ['**/vendor/**', '**/*.min.js', 'docs/**', '**/.venv/**'] },

  // (2) Base recommended rules.
  js.configs.recommended,

  // (3) Interface + test JS: browser + vendor globals, house-style rules at error.
  {
    files: ['src/osprey/interfaces/**/*.js', 'tests/**/*.{js,mjs}'],
    languageOptions: {
      globals: {
        ...globals.browser,
        Plotly: 'readonly',
        marked: 'readonly',
        hljs: 'readonly',
        katex: 'readonly',
        Terminal: 'readonly',
        FitAddon: 'readonly',
        WebLinksAddon: 'readonly',
      },
    },
    rules: {
      'no-var': 'error',
      'prefer-const': 'error',
      eqeqeq: ['error', 'always', { null: 'ignore' }],
    },
  },

  // (4) max-lines on PRODUCTION interface JS only (exclude test files).
  {
    files: ['src/osprey/interfaces/**/*.js'],
    ignores: ['**/*.test.*'],
    rules: {
      'max-lines': ['error', { max: 450, skipComments: true, skipBlankLines: true }],
    },
  },

  // (5) Root config files: node globals.
  {
    files: ['vitest.config.js', 'eslint.config.js'],
    languageOptions: {
      globals: { ...globals.node },
    },
  },

  // (6) Ban `// @ts-nocheck` across interface + test JS. Under checkJs a
  //     `// @ts-check` header is a redundant no-op, but a leading `// @ts-nocheck`
  //     opts a file out of tsc entirely; this rule keeps a new one from landing
  //     silently. See tools/eslint/no-ts-nocheck.js and CONTRIBUTING.md.
  {
    files: ['src/osprey/interfaces/**/static/js/**/*.js', 'tests/**/*.{js,mjs}'],
    plugins: { local: { rules: { 'no-ts-nocheck': noTsNocheck } } },
    rules: { 'local/no-ts-nocheck': 'error' },
  },

  // (7) Shrink-only allowlist for local/no-ts-nocheck: the test files still
  //     carrying a leading `// @ts-nocheck` because they are not yet type-clean.
  //     This block IS the list — it may ONLY shrink: retrofit a file and delete
  //     its row, never add one.
  {
    files: [
      'tests/interfaces/artifacts/logbook.test.mjs',
      'tests/interfaces/artifacts/preview-content.test.mjs',
      'tests/interfaces/artifacts/preview.test.mjs',
      'tests/interfaces/artifacts/print.test.mjs',
      'tests/interfaces/artifacts/render.test.mjs',
      'tests/interfaces/artifacts/security_render.test.mjs',
      'tests/interfaces/artifacts/state.test.mjs',
      'tests/interfaces/artifacts/timeseries.test.mjs',
      'tests/interfaces/artifacts/types.test.mjs',
    ],
    rules: { 'local/no-ts-nocheck': 'off' },
  },
];
