#!/usr/bin/env python3
"""
Analyze potentially unreferenced files to determine if they're truly unused.
"""

from pathlib import Path
import re

# List of potentially unreferenced files from the report
UNREFERENCED_FILES = [
    "about_dialog.py",
    "advanced_cache_invalidation.py",
    "analytics_dashboard.py",
    "automated_feedback_integration.py",
    "base_worker.py",
    "checkpointer_manager.py",
    "collapsible_widget.py",
    "conversation_context.py",
    "conversation_display.py",
    "conversation_history.py",
    "conversation_management.py",
    "conversation_manager.py",
    "dead_code_analyzer.py",
    "event_bus.py",
    "help_dialog.py",
    "image_display.py",
    "llm_client.py",
    "message_formatter.py",
    "message_handlers.py",
    "model_config_dialog.py",
    "multi_project_orchestrator.py",
    "multi_project_router.py",
    "orchestration_ui.py",
    "orchestration_worker.py",
    "plot_viewer.py",
    "project_context_manager.py",
    "project_control.py",
    "project_discovery.py",
    "project_manager.py",
    "routing_analytics.py",
    "routing_cache.py",
    "routing_feedback.py",
    "routing_ui.py",
    "semantic_context_analyzer.py",
    "settings_dialog.py",
    "settings_manager.py",
    "version_info.py",
]

def search_for_usage(filename: str, search_dir: Path) -> dict:
    """Search for usage of a module across all Python files.

    Returns dict with:
    - direct_imports: files that directly import this module
    - class_usage: files that use classes from this module
    - function_usage: files that use functions from this module
    """
    module_name = filename.replace('.py', '')
    results = {
        'direct_imports': [],
        'class_usage': [],
        'function_usage': [],
        'string_references': []
    }

    # Search patterns
    import_patterns = [
        rf'from\s+.*{module_name}\s+import',
        rf'import\s+.*{module_name}',
        rf'from\s+\.{module_name}\s+import',
        rf'from\s+osprey\.interfaces\.pyqt\.{module_name}\s+import',
    ]

    for py_file in search_dir.rglob('*.py'):
        if py_file.name == filename:
            continue

        try:
            content = py_file.read_text(encoding='utf-8')

            # Check for imports
            for pattern in import_patterns:
                if re.search(pattern, content):
                    results['direct_imports'].append(str(py_file.relative_to(search_dir)))
                    break

            # Check for string references (like in __all__ or dynamic imports)
            if module_name in content and py_file not in results['direct_imports']:
                # Check if it's in a string
                if f'"{module_name}"' in content or f"'{module_name}'" in content:
                    results['string_references'].append(str(py_file.relative_to(search_dir)))

        except Exception as e:
            pass

    return results

def categorize_file(filename: str) -> str:
    """Categorize a file based on its name and purpose."""
    if filename in ['dead_code_analyzer.py', 'remove_unused_imports.py', 'unreferenced_files_analysis.py']:
        return 'UTILITY_SCRIPT'
    elif filename in ['launcher.py', 'gui.py']:
        return 'ENTRY_POINT'
    elif filename.endswith('_dialog.py'):
        return 'UI_DIALOG'
    elif filename.endswith('_worker.py'):
        return 'WORKER_THREAD'
    elif filename.endswith('_manager.py'):
        return 'MANAGER'
    elif filename.endswith('_ui.py'):
        return 'UI_HANDLER'
    elif 'display' in filename or 'viewer' in filename or 'widget' in filename:
        return 'UI_COMPONENT'
    elif 'router' in filename or 'orchestrator' in filename:
        return 'ROUTING_LOGIC'
    elif 'analytics' in filename or 'feedback' in filename or 'cache' in filename:
        return 'FEATURE_MODULE'
    else:
        return 'OTHER'

def main():
    pyqt_dir = Path(__file__).parent

    print("=" * 80)
    print("UNREFERENCED FILES ANALYSIS")
    print("=" * 80)
    print()

    categorized = {}
    for filename in UNREFERENCED_FILES:
        category = categorize_file(filename)
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(filename)

    # Analyze each category
    for category in sorted(categorized.keys()):
        print(f"\n{'='*80}")
        print(f"{category}")
        print(f"{'='*80}")

        for filename in sorted(categorized[category]):
            print(f"\n{filename}:")

            usage = search_for_usage(filename, pyqt_dir)

            if usage['direct_imports']:
                print(f"  ✓ USED - Direct imports found in:")
                for file in usage['direct_imports'][:5]:  # Show first 5
                    print(f"    - {file}")
                if len(usage['direct_imports']) > 5:
                    print(f"    ... and {len(usage['direct_imports']) - 5} more")
            elif usage['string_references']:
                print(f"  ? MAYBE USED - String references in:")
                for file in usage['string_references'][:3]:
                    print(f"    - {file}")
            else:
                print(f"  ✗ POTENTIALLY UNUSED - No direct imports found")

                # Check if it's a known pattern
                if category == 'UTILITY_SCRIPT':
                    print(f"    → Keep: Utility script for development/analysis")
                elif category == 'ENTRY_POINT':
                    print(f"    → Keep: Entry point file")
                elif category == 'UI_DIALOG':
                    print(f"    → Review: May be instantiated dynamically")
                elif category == 'UI_COMPONENT':
                    print(f"    → Review: May be used in UI construction")
                elif category == 'FEATURE_MODULE':
                    print(f"    → Review: Feature module, may be optional")
                else:
                    print(f"    → Review: Check if truly unused")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_used = 0
    total_maybe = 0
    total_unused = 0

    for filename in UNREFERENCED_FILES:
        usage = search_for_usage(filename, pyqt_dir)
        if usage['direct_imports']:
            total_used += 1
        elif usage['string_references']:
            total_maybe += 1
        else:
            total_unused += 1

    print(f"\nTotal files analyzed: {len(UNREFERENCED_FILES)}")
    print(f"  ✓ Actually used (has imports): {total_used}")
    print(f"  ? Maybe used (string refs): {total_maybe}")
    print(f"  ✗ Potentially unused: {total_unused}")

if __name__ == '__main__':
    main()
