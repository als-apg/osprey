---
workflow: migrate-project
category: maintenance
applies_when: [version_upgrade, migration, update_osprey]
estimated_time: 15-60 minutes
ai_ready: true
skill_description: "Migrate OSPREY project to new version with AI-assisted three-way merge"
allowed_tools: ["Read", "Glob", "Grep", "Bash", "Edit", "Write"]
related: [pre-merge-cleanup, testing-workflow]
---

# OSPREY Project Migration

This task guides you through migrating an OSPREY project to a new framework version
while preserving your facility customizations.

## Overview

OSPREY projects may need migration when:
- The framework version is updated
- Template changes introduce new features
- Configuration schemas evolve

The migration process uses **three-way diffs** to intelligently merge changes:
1. **Old Vanilla**: Original template from when project was created
2. **Your Project**: Current project with your customizations
3. **New Vanilla**: Template from the new OSPREY version

## Prerequisites

- OSPREY project with a `.osprey-manifest.json` file
- (If no manifest) Run `osprey migrate init` first

## Workflow

### Step 1: Check Migration Status

```bash
osprey migrate check
```

This shows:
- Your project's OSPREY version
- Currently installed OSPREY version
- Whether migration is needed

### Step 2: Run Migration Analysis

```bash
osprey migrate run --dry-run
```

This creates a `_migration/` directory with:
- File classifications (auto-copy, preserve, merge)
- Merge prompts for files needing attention
- Summary of all changes

### Step 3: Review File Classifications

Files are classified into categories:

| Category | Meaning | Action |
|----------|---------|--------|
| **AUTO_COPY** | Template changed, you didn't | Safely update from new template |
| **PRESERVE** | You modified, template unchanged | Keep your version |
| **MERGE** | Both changed | Needs manual/AI merge |
| **NEW** | Only in new template | Add to your project |
| **DATA** | User data directories | Always preserved |

### Step 4: Merge Conflicting Files

For each file in `_migration/merge_required/`:

1. Read the merge prompt (shows all three versions)
2. Identify your customizations
3. Identify template updates
4. Create merged version preserving both

#### Common Merge Patterns

**config.yml**:
- Preserve your provider, model, and path settings
- Add new configuration sections from template
- Update deprecated field names

**registry.py**:
- Preserve custom capabilities
- Update import statements if changed
- Add new framework components

**Capability files**:
- Preserve business logic customizations
- Update method signatures if API changed
- Add new required methods

### Step 5: Apply Changes

```bash
osprey migrate run --apply
```

This will:
- Auto-copy safe files
- Add new template files
- Leave merge files for you to handle manually

### Step 6: Verify and Clean Up

```bash
# Check configuration validity
osprey health

# Run tests if available
pytest tests/

# Remove migration artifacts
rm -rf _migration/
```

## AI-Assisted Merge

When using Claude Code or similar AI tools, provide context:

```
Help me merge this OSPREY configuration file.

Current file: [paste your config]
New template: [paste from _migration/merge_required/]

Preserve:
- My provider/model settings
- My custom paths
- My facility-specific comments

Update:
- New configuration sections
- Renamed/restructured fields
```

## Troubleshooting

### No manifest found
```bash
osprey migrate init
```
This creates a manifest by detecting settings from your project.

### Exact version recreation fails
```bash
osprey migrate run --use-current-version
```
Uses current OSPREY for comparison (less accurate but works offline).

### Migration breaks something
Your original files are never modified during `--dry-run`.
Check `_migration/preserved/` for what was kept.

## Best Practices

1. **Always run --dry-run first** to preview changes
2. **Review merge prompts carefully** before applying
3. **Test after migration** with `osprey health`
4. **Commit before and after** to enable easy rollback
5. **Keep the manifest** for future migrations

## Reference

### Manifest File

`.osprey-manifest.json` contains:
```json
{
  "schema_version": "1.0.0",
  "creation": {
    "osprey_version": "0.10.6",
    "timestamp": "2025-01-30T...",
    "template": "control_assistant",
    "registry_style": "extend"
  },
  "init_args": { ... },
  "reproducible_command": "osprey init ...",
  "file_checksums": { ... }
}
```

### Commands Reference

| Command | Description |
|---------|-------------|
| `osprey migrate check` | Check if migration needed |
| `osprey migrate init` | Create manifest for existing project |
| `osprey migrate run` | Run migration (dry-run by default) |
| `osprey migrate run --apply` | Apply safe changes |
| `osprey migrate run --use-current-version` | Skip exact version recreation |
