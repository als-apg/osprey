---
workflow: commit-organization
category: code-quality
applies_when: [before_commit, multiple_changes, feature_complete]
estimated_time: 20-30 minutes
ai_ready: true
related: [pre-merge-cleanup, release-workflow]
---

# Commit Organization Workflow

This guide helps you organize uncommitted Git changes into atomic, logical commits with corresponding CHANGELOG entries.

## 🤖 AI Quick Start

**Paste this prompt to your AI assistant (Cursor/Copilot):**

```
I have multiple uncommitted changes. Following @docs/workflows/commit-organization.md,
help me organize them into atomic, logical commits.

CRITICAL REQUIREMENTS:
- Each commit must have EXACTLY ONE logical purpose
- Tests MUST be included with the code they test (same commit)
- CHANGELOG entries must be added ONE AT A TIME (right before each commit)
- Never add all CHANGELOG entries at once
- Files must be grouped by feature/fix, not by type
- Dependencies must be committed before features that need them

Step 1 - Analyze my changes:
Run: git diff --stat && git diff --name-status
Then for each file:
- Identify what changed and why
- Categorize: Feature, Fix, Refactor, Docs, Config, Test
- Note which files are related (serve same purpose)
- Identify dependencies (what enables what)

Step 2 - Create commit groups:
- Group by FUNCTION, not file type (source + tests + docs together)
- One feature = one commit (include all related files)
- One fix = one commit (source + regression test)
- Ensure no commit mixes unrelated changes
- Determine order: dependencies first, then features

Step 3 - Generate COMMIT_STRATEGY.md:
Create complete strategy following the template in the guide

Step 4 - Execution Guidance:
After I review the strategy, guide me through EACH commit individually:

For Commit N:
1. "Add this CHANGELOG entry: [show exact entry for THIS commit only]"
2. "Stage these files: git add [exact file list]"
3. "Verify staged: git diff --cached --name-only"
4. "Commit: git commit -m '[exact message]'"
5. "Verify: git log -1 --stat"
6. Wait for my confirmation before moving to next commit

REMEMBER: One CHANGELOG entry at a time, never all at once!
```

## 🎯 Problem & Solution

**Problem**: After development work, you have many uncommitted changes. Committing everything at once creates monolithic, unreviewable history that's difficult to review, debug, or selectively revert.

**Solution**: Analyze changes, group them logically, create a commit strategy document, then execute systematically. Each commit becomes a self-contained, reviewable unit of work.

**Why This Matters:**
- **Reviewability**: Smaller, focused commits are easier to review and understand
- **Debugging**: `git bisect` works better with atomic commits
- **Selective Reversion**: Can revert specific changes without affecting unrelated work
- **Cherry-picking**: Can apply specific fixes to other branches without pulling in unrelated features
- **Clear History**: Git log tells a coherent story of how the codebase evolved

## 📋 Quick Start

```bash
# 1. Understand what changed
git status                    # See what's modified
git diff --stat               # See change statistics
git diff                      # Review actual changes

# 2. Plan commits (use AI assistant - paste prompt from section above)
#    - Group related files (source + tests + docs)
#    - Determine commit order (dependencies first)
#    - Create COMMIT_STRATEGY.md

# 3. Execute commits ONE AT A TIME
#    For each commit:
#    a. Edit CHANGELOG.md (add THIS commit's entry only)
#    b. Stage files: git add file1.py test1.py CHANGELOG.md
#    c. Verify: git diff --cached --name-only
#    d. Commit: git commit -m "type: description"
#    e. Verify: git log -1 --stat

# 4. Final verification
git log --oneline -5          # Review commit history
git log --stat -5             # Verify files in each commit

# 5. Push when ready
git push origin feature/branch-name
```

**Complexity Guide:**
- **Simple (1-2 changes)**: Skip strategy document, commit directly with CHANGELOG
- **Medium (3-5 changes)**: Brief notes or mental plan
- **Complex (6+ changes)**: Full COMMIT_STRATEGY.md with AI assistance

**Critical Checkpoints:**
- ✅ Understood purpose of every changed file?
- ✅ Commits atomic (single logical purpose)?
- ✅ Each commit includes tests with code?
- ✅ CHANGELOG entries added one at a time, not all at once?
- ✅ Verified staged files before each commit?

## 🎯 Core Principles

### Atomic Commits
One commit = one logical change. Each commit should:
- Have a single, clear purpose
- Be understood independently
- Be revertable without breaking other features
- Pass all tests
- Include tests and docs for the change

### CHANGELOG Synchronization

**CRITICAL: One CHANGELOG Entry Per Commit**

Add CHANGELOG entries **one at a time**, right before staging each commit. DO NOT write all CHANGELOG entries at the start.

**Why?** Each commit should be complete and self-documenting:
- ✅ **Good**: Commit includes code + tests + docs + CHANGELOG entry
- ❌ **Bad**: First commit has all CHANGELOG entries, subsequent commits have none

**What happens if you add all entries at once:**
1. First commit contains all CHANGELOG entries for features not yet committed
2. Subsequent commits change code but don't update CHANGELOG
3. Each commit is incomplete (CHANGELOG doesn't match actual changes in that commit)
4. If you cherry-pick or revert a commit, CHANGELOG becomes inaccurate
5. Git history loses its self-documenting property

**Correct Pattern:**
```bash
# For each commit:
1. Edit CHANGELOG.md (add THIS commit's entry only)
2. git add [source files] [test files] [doc files]
3. git add CHANGELOG.md
4. git commit -m "message"
5. Repeat for next commit
```

**CHANGELOG Entry Guidelines:**
- **Timeless**: No emojis, no "TODO", no "WIP"
- **Professional**: No casual language
- **Concise**: Key points only, not implementation details
- **User-focused**: What changed from user's perspective
- **Categorized**: Added/Changed/Fixed/Removed/Deprecated

## 🔄 Workflow

### Step 0: Pre-Flight Check (Optional)

Before organizing commits, run pre-merge cleanup checks to catch obvious issues:

```bash
./scripts/premerge_check.sh main  # See pre-merge-cleanup.md
```

Fix any BLOCKER issues (debug code, hardcoded secrets, missing docstrings) first.

### Step 1: Analyze All Changes

```bash
git status                   # Overview
git diff --stat              # File statistics
git diff --name-status       # Show modification type (M/A/D)
git diff                     # Detailed changes
git diff path/to/file.py    # View specific file
```

**For each file, identify:**
1. **What changed**: Technical modification
2. **Why it changed**: Purpose/reason
3. **Category**: Feature, fix, refactor, docs, config, test
4. **Dependencies**: Related to other changes?
5. **User impact**: Needs CHANGELOG entry?

### Step 2: Group Into Logical Commits

**Grouping Principles:**
- **Feature-based**: All files for one feature (source + tests + docs + config)
- **Fix-based**: All files to fix one bug (source + regression test)
- **Refactor**: Pure refactors with no behavior change
- **Documentation**: Doc-only changes
- **Configuration**: Config changes for same feature

**Decision Tree:**
```
For each change:
├─ Complete feature? → Group all files together (source + tests + docs)
├─ Bug fix? → Group all files needed to fix this bug
├─ Refactor (no behavior change)? → Group related refactors
├─ Documentation only? → Group related doc changes
├─ Enables other changes? → Commit FIRST (dependencies)
└─ Independent changes? → Create separate commits
```

**Anti-patterns:**
- ❌ One commit per file (typically too granular)
- ❌ All changes in one commit (too coarse)
- ❌ Mixing unrelated features
- ❌ Mixing features with unrelated fixes
- ❌ Separating tests from code they test

### Step 3: Determine Commit Order

**Ordering Principles:**
1. Dependencies first (base functionality before dependent features)
2. Infrastructure before features (config/setup before code using it)
3. Fixes before enhancements
4. Breaking changes clearly marked
5. Tests with code (never commit broken tests)

### Step 4: Write CHANGELOG Entries

**Categories:**
- `### Added` - New features/capabilities
- `### Changed` - Changes to existing functionality
- `### Fixed` - Bug fixes
- `### Removed` - Removed features
- `### Deprecated` - Features marked for removal

**Format:**
```markdown
### [Category]
- **[Component]**: [Description]
  - [Optional sub-point]
```

**Length Guidelines:**
- **1 line**: Minor fixes/changes
- **2-3 lines**: Standard features/changes
- **4-5 lines**: Major features (with sub-bullets)

**Include:** User-visible changes, API changes, behavior changes, new features, bug fixes
**Exclude:** Internal refactors, test-only changes, minor cleanup, typo fixes in comments

### Step 5: Create Commit Strategy Document

Create `COMMIT_STRATEGY.md` in project root using template below.

### Step 6: Review Strategy

**Checklist:**
- ✅ Each commit atomic (single purpose)
- ✅ Correct order (dependencies respected)
- ✅ Each has CHANGELOG entry
- ✅ CHANGELOG entries professional and concise
- ✅ No mixed unrelated changes
- ✅ Clear commit messages
- ✅ File groupings make sense
- ✅ All changes accounted for

### Step 7: Execute Commits

**Critical**: Add CHANGELOG entry for EACH commit individually, right before staging files. DO NOT batch all entries at start! [[memory:11682122]]

**For each commit:**

1. **Add CHANGELOG entry for THIS commit only**
2. **Stage files**: `git add file1.py test1.py CHANGELOG.md`
3. **Verify**: `git diff --cached --name-only` (check all intended files, CHANGELOG included)
4. **Commit**: `git commit -m "type: Brief description"`
5. **Verify**: `git log -1 --stat` (check message, files)
6. **Repeat** for next commit

**If you make a mistake:**
```bash
git reset --soft HEAD~1  # Undo last commit, keep changes staged
git reset HEAD~1         # Undo last commit, unstage changes
```

## 📋 Commit Strategy Document Template

```markdown
# Commit Strategy - [Brief Description]

**Date**: [YYYY-MM-DD]
**Total Files Changed**: [N]
**Total Commits Planned**: [N]

## Overview

[2-3 sentence summary of all changes]

**Categories:** Features: [N], Fixes: [N], Refactors: [N], Docs: [N]

## File Inventory

- `path/to/file1.py` - [Brief description]
- `path/to/file2.py` - [Brief description]

## Commit Plan

### Commit 1: [Title]

**Type**: Feature|Fix|Refactor|Docs|Config

**Files**:
- `path/to/file1.py`
- `path/to/file2.py`

**CHANGELOG Entry**:
```
### [Category]
- **[Component]**: [Description]
```

**Commit Message**:
```
[type]: [Short description]

[Optional explanation]
```

**Dependencies**: None | Requires Commit N

---

### Commit 2: [Title]
[... repeat for each commit]

## Execution Plan

For each commit (in order):
1. Add CHANGELOG entry for THIS commit only
2. `git add [files] CHANGELOG.md`
3. `git diff --cached --name-only` (verify)
4. `git commit -m "[message]"`
5. `git log -1 --stat` (verify)

**Critical**: Add CHANGELOG entries one at a time!

## Post-Execution

- [ ] Verify commits: `git log --oneline`
- [ ] Verify CHANGELOG: `git diff origin/main CHANGELOG.md`
- [ ] Run tests
- [ ] Delete COMMIT_STRATEGY.md
- [ ] Push: `git push origin [branch]`
```

## 🔧 Common Patterns

### New Feature
```
Commit: Feature implementation
Files: source + tests + docs + config
CHANGELOG: ### Added - describe feature
Message: "feat: Add [feature] with [capability]"
```

### Bug Fix
```
Commit: Fix with test
Files: source fix + regression test
CHANGELOG: ### Fixed - what was broken
Message: "fix: Correct [issue] in [component]"
```

### Refactoring
```
Commit: Related refactors
Files: refactored sources + updated tests
CHANGELOG: ### Changed (if user-visible) or omit
Message: "refactor: Simplify [component]"
```

### Breaking Change
```
Commit: All changes for breaking change
Files: source + tests + docs + migration guide
CHANGELOG: ### Changed with migration path
Message: "refactor!: [change] BREAKING CHANGE: [description]"
```

## ❌ Common Mistakes

### Mistake 1: Adding All CHANGELOG Entries at Once

**Wrong:**
```bash
# Edit CHANGELOG.md and add entries for all 5 commits
git add CHANGELOG.md
git commit -m "feat: Feature A"  # ← CHANGELOG has entries for B, C, D, E too!
```

**Right:**
```bash
# Edit CHANGELOG.md, add ONLY entry for Feature A
git add fileA.py CHANGELOG.md
git commit -m "feat: Feature A"  # ← CHANGELOG matches this commit

# Edit CHANGELOG.md, add ONLY entry for Feature B
git add fileB.py CHANGELOG.md
git commit -m "feat: Feature B"  # ← CHANGELOG matches this commit
```

**Why this is critical:** Each commit should be complete and self-documenting. [[memory:11682122]]

### Mistake 2: Mixing Unrelated Changes

**Wrong:** `git add fix_bug_A.py new_feature_B.py refactor_C.py`

**Right:** Separate commits for bug fix, feature, and refactor

### Mistake 3: Forgetting to Stage CHANGELOG.md

Always run `git diff --cached --name-only` to verify CHANGELOG.md is staged before committing.

### Mistake 4: Separating Tests from Code

Tests verify code correctness. Always commit tests with the code they test.

### Mistake 5: Vague Commit Messages

Use descriptive messages: `"fix: Correct timeout handling in API client"` not `"fix stuff"`

### Mistake 6: Not Verifying Before Committing

Always verify with `git diff --cached` before committing and `git log -1 --stat` after.

## ⚠️ Edge Cases & Advanced Techniques

### File Modified for Multiple Reasons

Use `git add -p [file]` to stage hunks interactively for different commits.

### Splitting a Large Commit After Creation

```bash
git reset HEAD~1     # Undo commit, keep changes
# Stage and commit separately
git add file_A.py tests/test_A.py CHANGELOG.md
git commit -m "feat: Add feature A"
```

### Combining Small Commits

```bash
git rebase -i HEAD~3  # Squash last 3 commits

# In editor:
# pick abc123 first commit
# squash def456 second commit
# squash ghi789 third commit
```

### Reordering Commits

```bash
git rebase -i HEAD~5  # Reorder commits in editor
```

### Committed to Wrong Branch

```bash
git branch feature/new-feature  # Create correct branch
git reset --hard HEAD~1         # Reset wrong branch
git checkout feature/new-feature
```

### Working with Stashes

```bash
git stash push -m "WIP: organizing commits"
# ... do other work ...
git stash pop  # Restore work
```

## 📚 Examples

### Example 1: Single Feature

**Changes:**
- `src/capabilities/channel_write.py` (new, implementation)
- `src/capabilities/__init__.py` (register capability)
- `tests/capabilities/test_channel_write.py` (new, tests)
- `docs/source/capabilities/channel_write.rst` (new, docs)

**Strategy: Single atomic commit with all 4 files**

```markdown
### Commit 1: Add channel write capability

**CHANGELOG**:
### Added
- Channel write capability: Direct value assignment to control system channels
  - Automatic boundary validation and approval workflow integration

**Message**:
feat: Add channel write capability for direct PV writes
```

### Example 2: Multiple Fixes

**Changes:** 3 independent bug fixes

**Strategy: Three separate commits**

```markdown
### Commit 1: Fix gateway approval detection
Files: gateway.py + test_gateway.py
CHANGELOG: ### Fixed - Gateway approval detection: Enhanced pattern matching

### Commit 2: Fix error node logging
Files: error_node.py + test_error_node.py
CHANGELOG: ### Fixed - Error node logging: Removed duplicate messages

### Commit 3: Fix CLI parsing
Files: main.py + test_main.py
CHANGELOG: ### Fixed - CLI command parsing: Handle quoted arguments
```

### Example 3: Feature with Prerequisite Refactor

**Changes:** Refactored base class + new feature using it

**Strategy: Two commits (refactor first, then feature)**

```markdown
### Commit 1: Refactor base class
Files: base.py + test_base.py
CHANGELOG: ### Changed - Core base class: Refactored for plugin support
Dependencies: Must come before Commit 2

### Commit 2: Add plugin feature
Files: feature.py + test_feature.py + feature.rst
CHANGELOG: ### Added - Plugin system: Extensible architecture
Dependencies: Requires Commit 1
```

### Example 4: Complex Multi-Feature Scenario

**Scenario:** Bug fix, refactor, new feature, docs (13 files total)

**Strategy: 4 commits in dependency order**

```markdown
### Commit 1: Fix data analysis bug (FIRST)
Files: src/capabilities/data_analysis.py + tests/
CHANGELOG: ### Fixed - Data analysis: Correct boundary handling

### Commit 2: Refactor helper utilities (SECOND - enables feature)
Files: src/utils/helpers.py + tests/
CHANGELOG: ### Changed - Utility helpers: Refactored for extensibility

### Commit 3: Add data export capability (THIRD - main feature)
Files: src/capabilities/data_export.py + tests/ + env.example
CHANGELOG: ### Added - Data export: Export results to CSV, JSON, HDF5

### Commit 4: Document data export (FOURTH)
Files: docs/ + examples/
No CHANGELOG entry (docs only)
Message: docs: Add comprehensive data export documentation
```

## 🔍 Troubleshooting

**"Too many changes, where do I start?"**
→ Run `git diff --stat`, group by directory, start with smallest independent change

**"Are changes related or independent?"**
→ Related if: one depends on other, serve same feature, would break if separated
→ Independent if: work standalone, serve different use cases, can revert separately

**"Conflicts after reordering commits?"**
→ Edit files to resolve, `git add`, `git rebase --continue` or `git rebase --abort`

**"CHANGELOG entry in wrong commit?"**
→ `git reset HEAD~1`, fix CHANGELOG, recommit

**"Accidentally committed debug code?"**
→ `git reset HEAD~1`, remove debug code, recommit
→ Prevention: Run pre-merge cleanup BEFORE organizing commits

## 🎯 Golden Rules

1. **Understand first**: Analyze all changes thoroughly before planning
2. **One purpose per commit**: Single, clear logical purpose
3. **Group by function**: Related files for one feature go together
4. **Order matters**: Dependencies first, features later
5. **CHANGELOG mandatory**: Every user-visible change needs entry
6. **One entry per commit**: Add right before staging for that commit [[memory:11682122]]
7. **Tests with code**: Never separate tests from code they test
8. **Verify everything**: Check staged files and commits at each step
9. **When in doubt, split**: Make separate commits if unsure

---

## ✅ Final Verification

Before pushing:

```bash
# Check commit history
git log --oneline -5
git log --stat -5

# Verify CHANGELOG in each commit
for commit in $(git log --oneline HEAD~5..HEAD | cut -d' ' -f1); do
  git diff $commit~1 $commit --name-only | grep -q CHANGELOG.md && echo "✓ $commit" || echo "✗ $commit"
done

# Final checklist
```

**Final Checklist:**
- [ ] All commits have clear messages
- [ ] Each commit is atomic (single purpose)
- [ ] Commits in logical order (dependencies first)
- [ ] Tests included with code
- [ ] CHANGELOG entry in same commit as change
- [ ] No mixed unrelated changes
- [ ] All tests pass
- [ ] No debug code or secrets

**If everything checks out:** `git push origin feature/branch-name`

**Goal**: Clear, reviewable commit history where each commit is a focused, purposeful, complete chapter in your codebase's story.

## 📖 Quick Git Reference

**Essential commands:**
```bash
# Viewing
git status / git diff --stat / git diff / git diff --cached

# Staging
git add file.py / git add -p file.py / git add -A

# Committing
git commit -m "message" / git commit / git commit --amend

# Reviewing
git log --oneline / git log --stat / git show abc123

# Undoing
git reset HEAD~1 / git reset --soft HEAD~1 / git restore --staged file.py

# Advanced
git rebase -i HEAD~5 / git stash / git cherry-pick abc123
```

**Commit message types:** `feat:` `fix:` `refactor:` `docs:` `test:` `chore:` `perf:`

**CHANGELOG categories:** Added, Changed, Fixed, Removed, Deprecated

## See Also

**Before organizing commits:**
- [ai-code-review.md](ai-code-review.md) — Review AI-generated code first
- [pre-merge-cleanup.md](pre-merge-cleanup.md) — Scan for blockers

**While organizing commits:**
- [comments.md](comments.md) — Writing clear inline comments
- [docstrings.md](docstrings.md) — Documenting functions and classes

**After organizing commits:**
- [testing-workflow.md](testing-workflow.md) — Comprehensive testing strategy
- [update-documentation.md](update-documentation.md) — Building and updating docs
- [release-workflow.md](release-workflow.md) — Preparing for release
