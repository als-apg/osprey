/**
 * ESLint rule: ban `// @ts-nocheck` line directives.
 *
 * `tsconfig.json` type-checks every file under its include glob with
 * `checkJs`, so a surviving `// @ts-check` header is a redundant no-op — but
 * a leading `// @ts-nocheck` opts a file out of `tsc` entirely, and nothing
 * else caps that list. This rule closes that hole by reporting the directive
 * wherever `tsc` actually honors it.
 *
 * Matches `tsc` 5.9's real acceptance grammar (verified): it honors a
 * `// @ts-nocheck` LINE comment anywhere in the leading comment block — not
 * just line 1, and with or without a space after `//`. It does NOT honor the
 * block form `/* @ts-nocheck *\/` or a mid-line prose mention. So this rule
 * fires on every Line comment whose trimmed text starts with `@ts-nocheck`
 * and ignores Block comments and prose. Over-matching (a comment that merely
 * starts with the directive) is the safe direction; under-matching would make
 * the ban bypassable.
 */

/** @type {import('eslint').Rule.RuleModule} */
const rule = {
  meta: {
    type: 'problem',
    docs: {
      description:
        'ban `// @ts-nocheck`, which silently opts a file out of the type checker',
    },
    schema: [],
    messages: {
      noTsNocheck:
        '`// @ts-nocheck` opts this file out of tsc. Type-clean the file instead; the grandfather allowlist may only shrink.',
    },
  },
  create(context) {
    const { sourceCode } = context;
    return {
      Program() {
        for (const comment of sourceCode.getAllComments()) {
          if (comment.type === 'Line' && /^@ts-nocheck\b/.test(comment.value.trim())) {
            context.report({
              // ESLint comments are tokens, not estree nodes; cast so the
              // report descriptor type-checks under checkJs. `report` reads
              // the node's `loc`, which comments carry.
              node: /** @type {import('estree').Node} */ (/** @type {unknown} */ (comment)),
              messageId: 'noTsNocheck',
            });
          }
        }
      },
    };
  },
};

export default rule;
