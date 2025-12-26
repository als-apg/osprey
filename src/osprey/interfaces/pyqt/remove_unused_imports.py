#!/usr/bin/env python3
"""
Script to automatically remove unused imports from all Python files in the pyqt directory.
"""

import re
from pathlib import Path
from typing import List, Tuple


def remove_unused_imports_from_file(filepath: Path, unused_imports: List[Tuple[str, int]]) -> bool:
    """Remove unused imports from a file.

    Args:
        filepath: Path to the file.
        unused_imports: List of (import_statement, line_number) tuples.

    Returns:
        True if file was modified, False otherwise.
    """
    if not unused_imports:
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Sort by line number in reverse to avoid index shifting
        unused_imports_sorted = sorted(unused_imports, key=lambda x: x[1], reverse=True)

        modified = False
        for import_stmt, line_num in unused_imports_sorted:
            if line_num <= len(lines):
                line_idx = line_num - 1
                line = lines[line_idx]

                # Check if this is a simple import line we can safely remove
                if line.strip().startswith(('import ', 'from ')):
                    # Don't remove if it's part of a multi-line import
                    if '(' not in line or ')' in line:
                        del lines[line_idx]
                        modified = True
                        print(f"  Removed line {line_num}: {line.strip()}")

        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True

        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main entry point."""
    # Read the dead code report
    report_file = Path(__file__).parent / 'DEAD_CODE_REPORT.txt'

    if not report_file.exists():
        print("Error: DEAD_CODE_REPORT.txt not found!")
        return

    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse unused imports section
    unused_section_start = content.find("UNUSED IMPORTS")
    unused_section_end = content.find("POTENTIALLY UNUSED DEFINITIONS")

    if unused_section_start == -1 or unused_section_end == -1:
        print("Error: Could not find unused imports section in report!")
        return

    unused_section = content[unused_section_start:unused_section_end]

    # Parse file-by-file
    current_file = None
    file_imports = {}

    for line in unused_section.split('\n'):
        line = line.strip()

        # Check if this is a filename
        if line.endswith('.py:'):
            current_file = line[:-1]  # Remove the colon
            file_imports[current_file] = []

        # Check if this is an import line
        elif line.startswith('Line '):
            if current_file:
                # Parse: "Line 123: import something"
                match = re.match(r'Line (\d+): (.+)', line)
                if match:
                    line_num = int(match.group(1))
                    import_stmt = match.group(2)
                    file_imports[current_file].append((import_stmt, line_num))

    # Process each file
    pyqt_dir = Path(__file__).parent
    total_files = 0
    total_imports_removed = 0

    for filename, imports in file_imports.items():
        if not imports:
            continue

        filepath = pyqt_dir / filename
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue

        print(f"\nProcessing {filename}...")
        if remove_unused_imports_from_file(filepath, imports):
            total_files += 1
            total_imports_removed += len(imports)
            print(f"  ✓ Removed {len(imports)} unused import(s)")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files modified: {total_files}")
    print(f"  Total imports removed: {total_imports_removed}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
