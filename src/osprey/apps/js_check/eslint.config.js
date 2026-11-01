// `osprey app check` runs this over an application's panel/, whose repository has no
// ESLint config of its own. Browser globals and the house rules.
import js from '@eslint/js';
import globals from 'globals';

export default [
  js.configs.recommended,
  // Fires on the declare-a-safe-default-then-assign-in-try idiom the panels use.
  { rules: { 'no-useless-assignment': 'off' } },
  {
    languageOptions: {
      globals: {
        ...globals.browser,
        Plotly: 'readonly',
        marked: 'readonly',
        DOMPurify: 'readonly',
        hljs: 'readonly',
        katex: 'readonly',
      },
    },
    rules: {
      'no-var': 'error',
      'prefer-const': 'error',
      eqeqeq: ['error', 'always', { null: 'ignore' }],
    },
  },
];
