/* OSPREY Web Terminal — Word-Level Diff Utilities
 *
 * Provides word-by-word inline highlighting for unified diffs.
 * Used by prompts-gallery.js to enhance the diff view with
 * GitHub-style changed-word highlighting within changed lines.
 *
 * Exports:
 *   tokenize(line)              — split diff line payload into whitespace-delimited tokens
 *   computeWordDiff(old, new)   — LCS-based word diff returning ops array
 *   groupChangeBlocks(lines)    — group consecutive del/add runs into change blocks
 *   renderWordsIntoLine(el, raw, ops, side) — populate a diff-line div with word spans
 */

/**
 * Split a diff line payload (after stripping the +/- sigil) into tokens.
 * Preserves exact text: tokens.join('') === payload.
 *
 * @param {string} line - Raw diff line (e.g., "+the new text here")
 * @returns {string[]} Array of whitespace and non-whitespace tokens
 */
export function tokenize(line) {
  const payload = line.slice(1); // strip leading +/-/space sigil
  return payload.match(/\S+|\s+/g) || [];
}

/**
 * Compute a word-level diff between two token arrays using LCS (longest common subsequence).
 * Returns an array of operations: {op: 'keep'|'delete'|'insert', value: string}.
 *
 * Uses O(N*M) DP table with Uint16Array for memory efficiency.
 * Typical diff lines are 10-80 tokens so performance is trivial.
 *
 * @param {string[]} oldTokens - Tokens from the deleted line
 * @param {string[]} newTokens - Tokens from the added line
 * @returns {{op: string, value: string}[]}
 */
export function computeWordDiff(oldTokens, newTokens) {
  const N = oldTokens.length;
  const M = newTokens.length;

  // Build LCS table
  const dp = new Array(N + 1);
  for (let i = 0; i <= N; i++) {
    dp[i] = new Uint16Array(M + 1);
  }

  for (let i = 1; i <= N; i++) {
    for (let j = 1; j <= M; j++) {
      if (oldTokens[i - 1] === newTokens[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  // Backtrack to produce ops
  const ops = [];
  let i = N;
  let j = M;

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && oldTokens[i - 1] === newTokens[j - 1]) {
      ops.push({ op: 'keep', value: oldTokens[i - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      ops.push({ op: 'insert', value: newTokens[j - 1] });
      j--;
    } else {
      ops.push({ op: 'delete', value: oldTokens[i - 1] });
      i--;
    }
  }

  ops.reverse();
  return ops;
}

/**
 * Scan unified diff lines and group consecutive `-` runs followed by `+` runs
 * into change blocks. Unpaired del/add lines and context/hunk lines pass through
 * as separate block types.
 *
 * @param {string[]} rawLines - Array of unified diff lines
 * @returns {{type: string, lines?: string[], delLines?: string[], addLines?: string[]}[]}
 */
export function groupChangeBlocks(rawLines) {
  const blocks = [];
  let i = 0;

  while (i < rawLines.length) {
    const line = rawLines[i];

    if (line.startsWith('-')) {
      // Collect consecutive del lines
      const delLines = [];
      while (i < rawLines.length && rawLines[i].startsWith('-')) {
        delLines.push(rawLines[i]);
        i++;
      }

      // Collect consecutive add lines immediately following
      const addLines = [];
      while (i < rawLines.length && rawLines[i].startsWith('+')) {
        addLines.push(rawLines[i]);
        i++;
      }

      if (addLines.length > 0) {
        // Paired change block
        blocks.push({ type: 'change', delLines, addLines });
      } else {
        // Unpaired deletions
        for (const dl of delLines) {
          blocks.push({ type: 'del', lines: [dl] });
        }
      }
    } else if (line.startsWith('+')) {
      // Unpaired addition (no preceding del run)
      blocks.push({ type: 'add', lines: [line] });
      i++;
    } else if (line.startsWith('@@')) {
      blocks.push({ type: 'hunk', lines: [line] });
      i++;
    } else {
      blocks.push({ type: 'context', lines: [line] });
      i++;
    }
  }

  return blocks;
}

/**
 * Populate a .prompts-diff-line div with word-level spans.
 *
 * @param {HTMLElement} lineEl - The div.prompts-diff-line element
 * @param {string} rawLine - The raw diff line (e.g., "-old text")
 * @param {{op: string, value: string}[]} ops - Word diff operations
 * @param {'del'|'add'} side - Which side of the diff this line represents
 */
export function renderWordsIntoLine(lineEl, rawLine, ops, side) {
  // Clear any existing content
  lineEl.textContent = '';

  // Add the sigil (+/-) as a bare text node
  const sigil = rawLine.charAt(0);
  lineEl.appendChild(document.createTextNode(sigil));

  for (const op of ops) {
    if (op.op === 'keep') {
      const span = document.createElement('span');
      span.className = 'diff-word-keep';
      span.textContent = op.value;
      lineEl.appendChild(span);
    } else if (op.op === 'delete' && side === 'del') {
      const span = document.createElement('span');
      span.className = 'diff-word-del';
      span.textContent = op.value;
      lineEl.appendChild(span);
    } else if (op.op === 'insert' && side === 'add') {
      const span = document.createElement('span');
      span.className = 'diff-word-add';
      span.textContent = op.value;
      lineEl.appendChild(span);
    }
    // Skip ops that don't belong to this side
  }
}
