"""AI-Assisted Development Workflow Files.

This package contains markdown workflow files that guide AI coding assistants
through common development tasks in Osprey projects.

These workflows are designed to be:
- Referenced via @-mentions in AI coding assistants (Cursor, Claude Code, etc.)
- Exported to your project using 'osprey workflows export'
- Version-locked with your installed Osprey version

Available workflows:
- testing-workflow.md: Choose appropriate test types (unit/integration/e2e)
- commit-organization.md: Organize changes into atomic commits
- pre-merge-cleanup.md: Scan for common issues before committing
- docstrings.md: Generate proper docstrings
- comments.md: Add strategic code comments
- update-documentation.md: Identify documentation updates needed
- ai-code-review.md: Review AI-generated code
- channel-finder-pipeline-selection.md: Choose the right pipeline
- channel-finder-database-builder.md: Build high-quality databases
- release-workflow.md: Complete release process

Usage:
    # Export workflows to your project
    osprey workflows export

    # Then reference in AI assistant
    @src/osprey/workflows/testing-workflow.md What type of test should I write?
"""

__all__ = []
