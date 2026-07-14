// Self-test for the local/no-ts-nocheck rule.
//
// Vitest globals are off in this repo, so RuleTester — which runs its cases
// synchronously and throws an AssertionError on the first failure when no
// global `describe`/`it` is present — is wrapped in a test() call for vitest
// to execute and report. Deleting the rule's report() call makes the invalid
// cases throw "Should have 1 error but had 0"; that is the rule's mutation
// guard (see the task's acceptance gate).
import { test } from 'vitest';
import { RuleTester } from 'eslint';

import rule from '../../tools/eslint/no-ts-nocheck.js';

const ruleTester = new RuleTester({
  languageOptions: { ecmaVersion: 2022, sourceType: 'module' },
});

test("local/no-ts-nocheck matches tsc's real @ts-nocheck grammar", () => {
  ruleTester.run('no-ts-nocheck', rule, {
    valid: [
      // @ts-check is the opposite directive — always allowed.
      { code: '// @ts-check\nexport const a = 1;\n' },
      // Block form: tsc does NOT honor it, so the rule ignores it.
      { code: '/* @ts-nocheck */\nexport const a = 1;\n' },
      // A mid-line prose mention is not a directive.
      { code: 'export const a = 1; // see @ts-nocheck in the docs\n' },
    ],
    invalid: [
      // Canonical directive on line 1.
      { code: '// @ts-nocheck\nexport const a = 1;\n', errors: [{ messageId: 'noTsNocheck' }] },
      // No space after `//` — tsc still honors it.
      { code: '//@ts-nocheck\nexport const a = 1;\n', errors: [{ messageId: 'noTsNocheck' }] },
      // NOT line 1: a banner comment then the directive on line 2.
      { code: '// banner\n// @ts-nocheck\nexport const a = 1;\n', errors: [{ messageId: 'noTsNocheck' }] },
    ],
  });
});
