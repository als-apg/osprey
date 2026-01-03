# OSPREY Assist

Coding assistant integrations for OSPREY. Provides tool-agnostic task instructions that can be installed for different AI coding assistants (Claude Code, Cursor, GitHub Copilot, etc.).

## Installation

```bash
# List available tasks
osprey tasks list

# Show task details
osprey tasks show migrate

# Install a task as a Claude Code skill
osprey claude install migrate

# Output:
# Installed Claude Code skill to .claude/skills/migrate/SKILL.md
# Usage: Ask Claude "Upgrade my project to OSPREY 0.9.6"
```

## Directory Structure

```
src/osprey/assist/
├── README.md                           # This file
│
├── tasks/                              # Tool-agnostic task definitions
│   └── {task-name}/
│       ├── instructions.md             # Core logic (any AI can follow this)
│       └── {task-specific files}       # Data, schemas, examples
│
└── integrations/                       # Tool-specific wrappers
    ├── claude_code/
    │   └── {task}/SKILL.md             # Claude Code skill wrapper
    ├── cursor/
    │   └── {task}.cursorrules          # Cursor rules (future)
    └── generic/
        └── {task}.md                   # Copy-paste prompts (future)
```

## The Two-Layer Pattern

### Layer 1: Task Instructions

Located in `tasks/{name}/instructions.md`. These are **tool-agnostic** - plain markdown that any AI assistant can follow. They contain:

- Step-by-step workflow
- Decision logic
- References to data files
- Validation criteria

**No tool-specific syntax or assumptions.** Works with Claude Code, Cursor, ChatGPT, or any other AI.

### Layer 2: Tool Wrappers

Located in `integrations/{tool}/`. These are **thin wrappers** that:

1. Add tool-specific metadata (e.g., Claude Code skill frontmatter)
2. Point to the task instructions
3. Handle tool-specific invocation patterns

The wrapper should be minimal - just metadata + a reference to the instructions.

## Available Tasks

| Task | Description | Skill Wrapper |
|------|-------------|---------------|
| `migrate` | Upgrade downstream projects to newer OSPREY versions | Yes |
| `testing-workflow` | Comprehensive testing guide (unit, integration, e2e) | No |
| `commit-organization` | Organize changes into atomic commits | No |
| `pre-merge-cleanup` | Detect loose ends before merging PRs | No |
| `ai-code-review` | Review and cleanup AI-generated code | No |
| `docstrings` | Write clear Sphinx-compatible docstrings | No |
| `comments` | Write purposeful inline comments | No |
| `release-workflow` | Create releases with proper versioning | No |
| `update-documentation` | Keep docs in sync with code changes | No |
| `channel-finder-database-builder` | Build channel finder databases | No |
| `channel-finder-pipeline-selection` | Select appropriate CF pipelines | No |

Tasks with "Skill Wrapper: Yes" are automatically discovered by Claude Code when installed.
Tasks with "Skill Wrapper: No" provide instructions that can be referenced manually.

## Adding a New Task

### 1. Create the task directory

```bash
mkdir -p src/osprey/assist/tasks/my-task
```

### 2. Write `instructions.md`

Create `src/osprey/assist/tasks/my-task/instructions.md` with:

```markdown
# My Task

## Overview
Brief description of what this task accomplishes.

## Pre-requisites
What needs to be true before starting.

## Workflow

### Step 1: ...
### Step 2: ...
### Step 3: ...

## Validation
How to verify the task was completed correctly.

## Troubleshooting
Common issues and solutions.
```

**Guidelines:**
- Use imperative language ("Run this command", not "You should run")
- Be specific about file paths, commands, patterns
- Include examples where helpful
- No tool-specific syntax (no `@file` references, no skill metadata)

### 3. Create tool wrappers

For Claude Code, create `src/osprey/assist/integrations/claude_code/my-task/SKILL.md`:

```yaml
---
name: osprey-my-task
description: >
  Brief description for Claude to decide when to use this skill.
  Include keywords users might say.
allowed-tools: Read, Glob, Grep, Bash, Edit
---

# My Task

Follow the instructions in [instructions.md](../../../tasks/my-task/instructions.md).
```

### 4. Register with CLI (optional)

Tasks in `src/osprey/assist/tasks/` are automatically discovered by the CLI.
To add a Claude Code skill wrapper, create the integration in `src/osprey/assist/integrations/claude_code/{task}/SKILL.md`.

## Adding a New Tool Integration

### 1. Create the integration directory

```bash
mkdir -p src/osprey/assist/integrations/{tool_name}
```

### 2. Create wrappers for each task

Each tool has its own wrapper format:

| Tool | Wrapper Format | Install Location |
|------|---------------|------------------|
| Claude Code | SKILL.md with YAML frontmatter | `.claude/skills/{task}/SKILL.md` |
| Cursor | .cursorrules files | `.cursorrules` or `.cursor/rules/` |
| Generic | Plain markdown prompts | (printed to console) |

### 3. Update the CLI

Installation logic is in `src/osprey/cli/claude_cmd.py` (for Claude Code) and `src/osprey/cli/tasks_cmd.py` (for task browsing).

## Design Principles

1. **Tool-agnostic core** - Task logic lives in `tasks/`, not in tool wrappers
2. **Thin integrations** - Wrappers are metadata + pointers, not logic
3. **Single source of truth** - Update `instructions.md`, all tools get the update
4. **Ships with package** - `pip install osprey` includes all assist content
5. **Extensible** - Easy to add tasks or tools without modifying existing code

## Related Documentation

- [Migration Task](tasks/migrate/instructions.md) - Migration workflow for upgrading downstream projects
- [Migration Authoring](tasks/migrate/authoring/README.md) - CLI tools and prompts for creating migrations
