# Developer Workflows

Quick access to development workflows optimized for AI-assisted coding.

## 🎯 Overview

These workflows are battle-tested guides for common development tasks in Osprey. Each guide includes:
- Step-by-step instructions
- Checklists and decision trees
- Code examples and commands
- AI assistant integration prompts

## 🚀 Quick Links by Situation

### Before Committing Code

**About to commit?** Start here to ensure quality:

- **[AI Code Review](ai-code-review.md)** ⭐ **Essential for AI-Generated Code**
  - Critical review of AI-generated implementations
  - Detect redundancy, unused code, and API inconsistencies
  - Refactoring guidance for over-engineered solutions
  - Assumes you have uncommitted changes to review

- **[Testing Workflow](testing-workflow.md)** ⭐ **Essential**
  - Unit vs integration vs e2e tests
  - Cost-aware testing strategy (e2e tests: $0.10-$0.25)
  - When to mock vs use real services
  - Test coverage guidelines

- **[Pre-Merge Cleanup](pre-merge-cleanup.md)** ⭐ **Most Used**
  - Scan for debug code, commented code, hardcoded secrets
  - Verify CHANGELOG entries, docstrings, and type hints
  - Check for TODOs, orphaned references, and test coverage

- **[Commit Organization](commit-organization.md)** ⭐ **Most Used**
  - Organize uncommitted changes into atomic, logical commits
  - Create commit strategy with CHANGELOG entries
  - Follow systematic execution plan

### While Writing Code

**Need documentation guidance?**

- **[Docstrings Guide](docstrings.md)**
  - Professional Sphinx-format docstrings
  - Parameter, return value, and exception documentation
  - Examples and cross-references

- **[Comments Guide](comments.md)**
  - When and how to write inline comments
  - Strategic comments vs. over-documentation
  - Framework-specific patterns

### After Code Changes

**Modified public APIs or behavior?**

- **[Update Documentation](update-documentation.md)**
  - Systematic workflow for keeping docs in sync with code
  - Detect all documentation locations affected by changes
  - Quick decision tree for documentation scope

### Before Release

**Ready to cut a release?**

- **[Release Workflow](release-workflow.md)** ⭐ **Most Used**
  - Complete release process from testing to PyPI
  - Version update checklist
  - Test architecture and execution

## 🤖 Using with AI Assistants (Cursor/Copilot)

### Basic Usage

Simply reference workflows in your prompts using the `@` mention:

```
@docs/workflows/README.md I'm about to commit. What should I check?
```

The AI will guide you through the appropriate workflow.

### Workflow-Specific Prompts

Each workflow guide includes a **"🤖 AI Quick Start"** section at the top with detailed prompts tailored to that specific workflow. Simply @ mention the workflow file and follow the prompt structure shown.

### Example Usage

Simply @ mention the workflow and ask for guidance:

```
@docs/workflows/ai-code-review.md Review my uncommitted AI-generated code
```

```
@docs/workflows/pre-merge-cleanup.md Scan my uncommitted changes
```

```
@docs/workflows/commit-organization.md Help me organize my commits
```

```
@docs/workflows/release-workflow.md Guide me through releasing v0.9.8
```

The AI will read the workflow file (which includes detailed prompt structures in the "AI Quick Start" section) and guide you through the process.

## 📊 Workflow Categories

### Code Quality
- [AI Code Review](ai-code-review.md) - Critical review and refactoring of AI-generated code
- [Testing Workflow](testing-workflow.md) - Cost-effective testing strategy
- [Pre-Merge Cleanup](pre-merge-cleanup.md) - Systematic pre-commit review
- [Commit Organization](commit-organization.md) - Atomic commit strategy

### Documentation
- [Update Documentation](update-documentation.md) - Keep docs synchronized
- [Docstrings](docstrings.md) - Professional docstring writing
- [Comments](comments.md) - Strategic inline comments

### Release Management
- [Release Workflow](release-workflow.md) - Complete release process

## 🎓 Learning Path for New Contributors

**Week 1: Code Quality Basics**
1. Read [Testing Workflow](testing-workflow.md) - understand test types
2. Read [Docstrings Guide](docstrings.md)
3. Read [Comments Guide](comments.md)
4. Read [AI Code Review](ai-code-review.md) - critical for AI-assisted development
5. Use [Pre-Merge Cleanup](pre-merge-cleanup.md) before first commit

**Week 2: Professional Workflow**
6. Practice [Commit Organization](commit-organization.md)
7. Learn [Update Documentation](update-documentation.md) workflow
8. Make first meaningful contribution

**Week 3+: Advanced Topics**
7. Shadow a release using [Release Workflow](release-workflow.md)
8. Mentor others using these workflows

## 💡 Tips for Success

### For Individual Contributors
- **Bookmark this page** - Come back whenever starting a task
- **Use AI integration** - Let AI assistants guide you through workflows
- **Review AI-generated code** - Always run AI Code Review workflow after AI assistance
- **Start small** - Focus on Pre-Merge Cleanup first, expand from there
- **Build habits** - These workflows become second nature with practice
- **Budget review time** - AI code review often takes 2x the generation time (this is normal)

### For Reviewers
- **Reference workflows in PR comments** - Link to specific sections
- **Check for workflow compliance** - Did contributor follow pre-merge cleanup?
- **Encourage adoption** - Positive reinforcement when workflows are followed

### For Maintainers
- **Keep workflows updated** - As project evolves, workflows should too
- **Gather feedback** - What works? What's confusing?
- **Lead by example** - Use workflows consistently

## 🔧 Automation Tools

### Available Now
- **Pre-Merge Check Script** - Automated scanning (see Pre-Merge Cleanup guide)

### Coming Soon
- CLI integration: `osprey workflow list/show/run`
- IDE snippets and templates
- Workflow analytics and metrics

## 📚 Additional Resources

### Related Documentation
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - High-level contribution guidelines
- [TESTING_GUIDE.md](../../TESTING_GUIDE.md) - Comprehensive testing documentation
- [CHANGELOG.md](../../CHANGELOG.md) - See examples of good CHANGELOG entries

### External Resources
- [Conventional Commits](https://www.conventionalcommits.org/) - Commit message format
- [Keep a Changelog](https://keepachangelog.com/) - CHANGELOG best practices
- [Sphinx Documentation](https://www.sphinx-doc.org/) - RST and documentation syntax

## 🤝 Contributing to Workflows

Found an issue or have a suggestion? These workflows are living documents:

1. **Report issues** - File a GitHub issue with label `documentation`
2. **Suggest improvements** - PR welcome for workflow enhancements
3. **Share experiences** - What worked? What was confusing?

## 📝 Workflow Metadata

Each workflow guide includes YAML frontmatter with:
- `workflow`: Unique identifier
- `category`: Code quality, documentation, or release
- `applies_when`: When to use this workflow
- `estimated_time`: How long it takes
- `ai_ready`: Whether AI integration is available
- `related`: Links to related workflows

This metadata enables future automation and better AI integration.

---

**Last Updated**: December 2024

**Maintained By**: Osprey Core Team

**Questions?** Open an issue or ask in PR comments.

