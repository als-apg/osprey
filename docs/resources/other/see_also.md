# CCLI Source Section Design Guide

## Overview
The CCLI (Cross-Component Link Index) source section provides navigation links to related documentation and references. Based on the RST seealso directive pattern, here's how to design effective source sections.

## Structure Pattern

### Basic Format
```rst
.. seealso::

   :doc:`relative/path/to/document`
       Brief description of what this link provides
   
   :doc:`another/relative/path`
       Description of the second reference
   
   :doc:`third/reference/path`
       Description explaining the relevance
```

### Key Components

1. **Directive Declaration**: Always start with `.. seealso::`
2. **Empty Line**: Required after the directive
3. **Indented Links**: Use 3-space indentation for link entries
4. **Path Format**: Use `:doc:` role with relative paths from current file
5. **Descriptions**: Provide 7-space indented descriptions under each link

## Design Principles

### Content Organization
- **API References First**: Link to relevant API documentation
- **Related Concepts**: Include conceptually related guides
- **Framework Patterns**: Reference broader framework documentation

### Description Guidelines
- Keep descriptions concise (one line preferred)
- Focus on **what** the link provides, not **why** to read it
- Use action-oriented language ("API reference for...", "Guide to...")

## Example Analysis

From the attached example:
```rst
.. seealso::

   :doc:`../../api_reference/01_core_framework/05_prompt_management`
       API reference for prompt system classes and functions
   
   :doc:`03_registry-and-discovery`
       Component registration and discovery patterns
   
   :doc:`../01_understanding-the-framework/02_convention-over-configuration`
       Framework conventions and patterns
```

### What Works Well
- **Clear hierarchy**: API reference → related concepts → framework patterns
- **Descriptive titles**: Each description clearly states the content type
- **Logical progression**: From specific implementation to general patterns

### Path Navigation
- `../../` moves up two directory levels
- `../` moves up one directory level  
- Relative paths from current document location

## Best Practices

1. **Limit to 3-5 links**: Avoid overwhelming readers
2. **Prioritize relevance**: Most directly related content first
3. **Maintain consistency**: Use similar description patterns across documents
4. **Test links**: Ensure all paths resolve correctly
5. **Update regularly**: Keep links current as documentation evolves

## Common Mistakes to Avoid

- Don't use absolute paths
- Don't include external URLs in seealso sections
- Don't duplicate links already in main content
- Don't use vague descriptions like "More information"
- Don't forget proper indentation (causes rendering issues)
