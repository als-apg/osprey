import js from '@eslint/js';
import globals from 'globals';

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
      eqeqeq: 'error',
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

  // --- Legacy exemptions (shrink-only ratchet; see CONTRIBUTING.md). Each list is
  //     exactly today's violators and may only shrink as P2–P5 retrofit each interface.
  //     Rules stay at `error` everywhere else; these are file-scoped `off` only. ---
  {
    files: [
      'src/osprey/interfaces/artifacts/static/js/print.js',
      'src/osprey/interfaces/design_system/static/js/theme-boot.js',
      'src/osprey/interfaces/design_system/static/js/theme-manager.js',
      'src/osprey/interfaces/okf_panel/static/js/app.js',
      'src/osprey/interfaces/web_terminal/static/js/app.js',
      'src/osprey/interfaces/web_terminal/static/js/panel-manager.js',
      'src/osprey/interfaces/web_terminal/static/js/session-views.js',
      'src/osprey/interfaces/web_terminal/static/js/session.js',
      'tests/interfaces/artifacts/security_render.test.mjs',
      'tests/interfaces/web_terminal/scaffold-edit.test.mjs',
      'tests/interfaces/web_terminal/scaffold-view.test.mjs',
    ],
    rules: { 'no-unused-vars': 'off' },
  },
  {
    files: [
      'src/osprey/interfaces/lattice_dashboard/static/js/render.js',
      'src/osprey/interfaces/lattice_dashboard/static/js/settings.js',
      'src/osprey/interfaces/okf_panel/static/js/app.js',
      'src/osprey/interfaces/web_terminal/static/js/mcp-renderer.js',
      'src/osprey/interfaces/web_terminal/static/js/panel-manager.js',
      'src/osprey/interfaces/web_terminal/static/js/scaffold/utils.js',
      'src/osprey/interfaces/web_terminal/static/js/settings.js',
    ],
    rules: { eqeqeq: 'off' },
  },
  {
    files: [
      'src/osprey/interfaces/artifacts/static/js/logbook.js',
    ],
    rules: { 'max-lines': 'off' },
  },
];
