---
workflow: migration-workflow
category: release-management
applies_when: [breaking_changes, api_changes, version_bump]
estimated_time: 15-30 minutes
ai_ready: true
related: [release-workflow, pre-merge-cleanup]
---

# Migration Documentation Workflow

This document describes the process for creating machine-readable migration documents that enable downstream facilities to update their implementations when OSPREY releases new versions.

## Overview

OSPREY serves multiple scientific facilities, each with their own implementation based on the Control Assistant tutorial. When OSPREY releases breaking changes, facilities need to update their code. Migration documents automate this process.

### The Migration Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Generate  │────▶│    Audit    │────▶│  Validate   │────▶│   Release   │
│  (LLM #1)   │     │  (LLM #2)   │     │  (LLM #3)   │     │  (Human)    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                    │
      ▼                   ▼                   ▼                    ▼
   Draft YAML         Audit Report      Validation Report     Final YAML
```

## When to Create Migration Documents

Create a migration document when a release includes:

- **Breaking API changes** (method renames, signature changes)
- **Import path changes** (module reorganization)
- **Class/type renames**
- **Configuration schema changes**
- **Behavioral changes** (same API, different behavior)
- **Removals** (deprecated features finally removed)

Skip migration documents for:

- Bug fixes that don't change API
- Documentation-only changes
- Internal refactoring with no public API impact
- New features that don't affect existing code

## Workflow Steps

### Step 1: Generate Migration Draft

After all PRs are merged and before tagging the release:

```bash
cd /path/to/osprey

# Generate the migration prompt
python -m osprey.assist.tasks.migrate.authoring.tools.migrate generate \
    --from 0.9.5 \
    --to 0.9.6

# Output:
# ✓ Generation prompt saved to: v0.9.6.prompt.md
```

**Manual LLM Invocation:**

1. Open the generated prompt file: `v0.9.6.prompt.md`
2. Feed it to Claude (or GPT-4) via:
   - Claude Code: `cat v0.9.6.prompt.md | claude`
   - Web interface: Copy/paste the prompt
   - API: Use your preferred integration
3. Save the YAML output to: `v0.9.6.yml`

**Quality Tip:** Use Claude Opus or GPT-4 for generation; these changes are important enough to warrant the best model.

### Step 2: Audit the Draft

Use a **different LLM instance** to audit the generated document:

```bash
# Generate the audit prompt
python -m osprey.assist.tasks.migrate.authoring.tools.migrate audit v0.9.6.yml

# Output:
# ✓ Audit prompt saved to: v0.9.6.audit.prompt.md
```

**Why a different instance?**

Auditing with the same instance that generated the document creates confirmation bias. The auditor should approach the document fresh.

**Manual LLM Invocation:**

1. Open: `v0.9.6.audit.prompt.md`
2. Feed to a **fresh** Claude/GPT-4 session
3. Review the audit report for:
   - `PASS`: Proceed to validation
   - `PASS_WITH_WARNINGS`: Review warnings, decide if acceptable
   - `NEEDS_REVISION`: Fix issues and re-audit

### Step 3: Validate Against Tutorial

Simulate applying the migration to catch practical issues:

```bash
# Generate the validation prompt
python -m osprey.assist.tasks.migrate.authoring.tools.migrate validate v0.9.6.yml

# Output:
# ✓ Validation prompt saved to: v0.9.6.validate.prompt.md
```

**Manual LLM Invocation:**

1. Open: `v0.9.6.validate.prompt.md`
2. Feed to Claude/GPT-4
3. Review the validation report:
   - Check that all patterns find the expected code
   - Verify simulated replacements look correct
   - Note any files that might be missed

### Step 4: Human Review and Release

Before tagging the release:

1. **Review the final YAML** in `v0.9.6.yml`
2. **Move to versions directory and commit:**
   ```bash
   mv v0.9.6.yml src/osprey/assist/tasks/migrate/versions/
   git add src/osprey/assist/tasks/migrate/versions/v0.9.6.yml
   git commit -m "docs: Add migration document for v0.9.6"
   ```
3. **Continue with release workflow** (release-workflow.md)

## Integration with Release Workflow

Add this checkpoint to **Step 0A** of the release workflow:

```
STEP 0A - Review CHANGELOG (CRITICAL - DO THIS FIRST):
...existing steps...

5. Check for breaking changes:
   - Does CHANGELOG contain "Changed" or "Removed" items affecting public API?
   - If YES: Follow migration-workflow.md before proceeding
   - If NO: Continue to Step 0B
```

## Migration Document Schema

See `src/osprey/assist/tasks/migrate/schema.yml` for the full schema definition.

### Quick Reference: Change Types

| Type | Use For | Automatable? |
|------|---------|--------------|
| `method_rename` | Function/method name changes | Usually yes |
| `signature_change` | Parameter additions/removals/renames | Partial |
| `import_change` | Import path changes | Yes |
| `class_rename` | Class/type name changes | Yes |
| `config_change` | Configuration file changes | Usually yes |
| `behavior_change` | Same API, different behavior | No |
| `removal` | Deprecated features removed | Partial |

### Risk Levels

| Level | Meaning | User Impact |
|-------|---------|-------------|
| `high` | Breaking change | Code won't run without fix |
| `medium` | Runtime risk | May cause errors in some paths |
| `low` | Warning only | Code still works |

## Downstream Usage

### Coding Assistant Integration (Recommended)

OSPREY provides coding assistant integrations via the `osprey claude` command.

**Install the integration:**

```bash
osprey claude install migrate
```

This installs the appropriate integration for your coding assistant (Claude Code, Cursor, etc.).

**With Claude Code:**

Simply ask Claude to upgrade your project:

```
User: Upgrade my project to OSPREY 0.9.6
```

The skill will:
1. Check for clean git state
2. Detect your current OSPREY version
3. Load the migration document
4. Show a dry-run of all changes
5. Apply changes after your confirmation
6. Run validation commands
7. Provide a summary and next steps

**Manual Instructions:**

Follow the tool-agnostic instructions in [`src/osprey/assist/tasks/migrate/instructions.md`](../assist/tasks/migrate/instructions.md).

### CLI Tool

The CLI tool can also be used directly:

```bash
# Check what migrations are available
python -m osprey.assist.tasks.migrate.authoring.tools.migrate status

# Preview changes (dry run)
python -m osprey.assist.tasks.migrate.authoring.tools.migrate apply v0.9.6.yml \
    --target /path/to/my-facility \
    --dry-run

# Apply changes
python -m osprey.assist.tasks.migrate.authoring.tools.migrate apply v0.9.6.yml \
    --target /path/to/my-facility
```

## Best Practices

### For Migration Authors

1. **Be specific with search patterns** - Use word boundaries (`\b`), handle whitespace
2. **Test patterns mentally** - Would this match the new value? False positives?
3. **Provide realistic examples** - From Control Assistant, not abstract foo/bar
4. **Document manual review reasons** - Help downstream developers understand
5. **Order dependent changes** - Note if change A must happen before B

### For Auditors

1. **Check completeness** - Every CHANGELOG item should map to a migration entry
2. **Verify accuracy** - Old values existed, new values exist
3. **Challenge automatability** - Is it really safe to auto-replace everywhere?
4. **Consider edge cases** - Strings, comments, subclasses, type annotations

### For Validators

1. **Apply changes mentally** - Walk through each replacement
2. **Check cascading effects** - If A changes, what else breaks?
3. **Verify imports resolve** - After migration, all imports should work
4. **Test file coverage** - Are all affected files matched?

## Troubleshooting

### "Pattern finds no matches"

- Check if tutorial code doesn't use this API (optional feature)
- Pattern might be too specific
- Code might already be migrated (version mismatch)

### "Too many false positives"

- Add word boundaries: `\bread_pv\b` instead of `read_pv`
- Add context: `\.read_pv\(` to require method call syntax
- Exclude paths: Add file patterns to narrow scope

### "Audit/validation disagrees with generation"

- Trust the auditor/validator - they have fresh perspective
- Regenerate if needed with more specific CHANGELOG context
- Consider manual editing for complex cases

## Files Reference

All migration files are in `src/osprey/assist/tasks/migrate/`:

```
src/osprey/assist/tasks/migrate/
├── instructions.md              # Tool-agnostic migration workflow (for downstream users)
├── schema.yml                   # Migration document schema
├── versions/
│   └── v0.9.6.yml               # Actual migration documents (shipped with package)
└── authoring/                   # For OSPREY maintainers
    ├── README.md                # How to create migrations
    ├── prompts/
    │   ├── generate.md          # Prompt for LLM #1 (generation)
    │   ├── audit.md             # Prompt for LLM #2 (audit)
    │   └── validate.md          # Prompt for LLM #3 (validation)
    ├── tools/
    │   └── migrate.py           # CLI tool for migration workflow
    └── examples/
        └── v0.9.6.yml           # Example migration document format
```

### Tool Integration

```
src/osprey/assist/integrations/
└── claude_code/
    └── migrate/SKILL.md         # Claude Code skill wrapper
```

Install with: `osprey claude install migrate`
