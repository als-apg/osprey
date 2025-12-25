#!/usr/bin/env python3
"""
Dead Code Analyzer for PyQt GUI Directory
Analyzes Python files to identify:
- Unused imports
- Unused functions
- Unused classes
- Unreferenced files
"""

import ast
from pathlib import Path
from collections import defaultdict


class CodeAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze code usage."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.imports = {}  # name -> (module, line)
        self.from_imports = {}  # name -> (module, line)
        self.definitions = {}  # name -> (type, line)
        self.usages = set()  # names used in code
        self.current_class = None
        self.current_function = None

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = (alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ''
        for alias in node.names:
            if alias.name == '*':
                continue
            name = alias.asname if alias.asname else alias.name
            self.from_imports[name] = (f"{module}.{alias.name}", node.lineno)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not self.current_class:
            self.definitions[node.name] = ('function', node.lineno)
        else:
            # Method in a class
            self.definitions[f"{self.current_class}.{node.name}"] = ('method', node.lineno)

        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node):
        if not self.current_class:
            self.definitions[node.name] = ('function', node.lineno)
        else:
            # Method in a class
            self.definitions[f"{self.current_class}.{node.name}"] = ('method', node.lineno)

        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_ClassDef(self, node):
        self.definitions[node.name] = ('class', node.lineno)
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.usages.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Track attribute access
        if isinstance(node.value, ast.Name):
            self.usages.add(node.value.id)
        self.generic_visit(node)


def analyze_file(filepath: Path) -> Dict:
    """Analyze a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))
        analyzer = CodeAnalyzer(str(filepath))
        analyzer.visit(tree)

        return {
            'imports': analyzer.imports,
            'from_imports': analyzer.from_imports,
            'definitions': analyzer.definitions,
            'usages': analyzer.usages,
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory recursively."""
    return sorted(directory.rglob('*.py'))


def analyze_directory(directory: Path) -> Dict:
    """Analyze all Python files in directory."""
    files = find_python_files(directory)
    results = {}

    for filepath in files:
        rel_path = filepath.relative_to(directory)
        results[str(rel_path)] = analyze_file(filepath)

    return results


def find_unused_imports(analysis: Dict) -> Dict[str, List[Tuple[str, int]]]:
    """Find unused imports in each file."""
    unused = {}

    for filepath, data in analysis.items():
        if not data.get('success'):
            continue

        file_unused = []
        usages = data['usages']

        # Check regular imports
        for name, (module, line) in data['imports'].items():
            if name not in usages:
                file_unused.append((f"import {module} (as {name})" if module != name else f"import {name}", line))

        # Check from imports
        for name, (module, line) in data['from_imports'].items():
            if name not in usages:
                file_unused.append((f"from ... import {name}", line))

        if file_unused:
            unused[filepath] = sorted(file_unused, key=lambda x: x[1])

    return unused


def find_unused_definitions(analysis: Dict) -> Dict[str, List[Tuple[str, str, int]]]:
    """Find potentially unused functions and classes."""
    # First, collect all definitions and their usages across files
    all_definitions = defaultdict(list)  # name -> [(filepath, type, line)]
    all_usages = set()

    for filepath, data in analysis.items():
        if not data.get('success'):
            continue

        for name, (def_type, line) in data['definitions'].items():
            all_definitions[name].append((filepath, def_type, line))

        all_usages.update(data['usages'])

    # Find definitions that are never used
    unused = {}

    for filepath, data in analysis.items():
        if not data.get('success'):
            continue

        file_unused = []

        for name, (def_type, line) in data['definitions'].items():
            # Skip special methods and private methods that might be used dynamically
            base_name = name.split('.')[-1]
            if base_name.startswith('__') and base_name.endswith('__'):
                continue
            if base_name.startswith('_'):
                # Private, might be used internally
                continue

            # Check if used anywhere
            if name not in all_usages and base_name not in all_usages:
                file_unused.append((name, def_type, line))

        if file_unused:
            unused[filepath] = sorted(file_unused, key=lambda x: x[2])

    return unused


def find_unreferenced_files(analysis: Dict, directory: Path) -> List[str]:
    """Find Python files that are never imported."""
    # Get all module names that are imported
    imported_modules = set()

    for filepath, data in analysis.items():
        if not data.get('success'):
            continue

        for module, line in data['imports'].values():
            imported_modules.add(module.split('.')[0])

        for module, line in data['from_imports'].values():
            parts = module.split('.')
            if parts:
                imported_modules.add(parts[0])

    # Check which files are never imported
    unreferenced = []

    for filepath in analysis.keys():
        # Convert filepath to module name
        module_path = filepath.replace('.py', '').replace('/', '.').replace('\\', '.')
        if module_path.startswith('.'):
            module_path = module_path[1:]

        # Skip __init__.py and main entry points
        if filepath.endswith('__init__.py'):
            continue
        if filepath in ['launcher.py', 'gui.py']:
            continue

        # Check if this module is imported anywhere
        parts = module_path.split('.')
        is_imported = False

        for part in parts:
            if part in imported_modules:
                is_imported = True
                break

        # Also check the full module path
        for imported in imported_modules:
            if module_path in imported or imported in module_path:
                is_imported = True
                break

        if not is_imported:
            unreferenced.append(filepath)

    return sorted(unreferenced)


def generate_report(analysis: Dict, directory: Path) -> str:
    """Generate a comprehensive dead code report."""
    report = []
    report.append("=" * 80)
    report.append("DEAD CODE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nDirectory: {directory}")
    report.append(f"Total files analyzed: {len(analysis)}\n")

    # Unused imports
    report.append("\n" + "=" * 80)
    report.append("UNUSED IMPORTS")
    report.append("=" * 80)

    unused_imports = find_unused_imports(analysis)
    if unused_imports:
        for filepath, imports in sorted(unused_imports.items()):
            report.append(f"\n{filepath}:")
            for import_stmt, line in imports:
                report.append(f"  Line {line}: {import_stmt}")
    else:
        report.append("\nNo unused imports found!")

    # Unused definitions
    report.append("\n\n" + "=" * 80)
    report.append("POTENTIALLY UNUSED DEFINITIONS")
    report.append("=" * 80)
    report.append("(Note: May include false positives for dynamically called code)")

    unused_defs = find_unused_definitions(analysis)
    if unused_defs:
        for filepath, defs in sorted(unused_defs.items()):
            report.append(f"\n{filepath}:")
            for name, def_type, line in defs:
                report.append(f"  Line {line}: {def_type} '{name}'")
    else:
        report.append("\nNo unused definitions found!")

    # Unreferenced files
    report.append("\n\n" + "=" * 80)
    report.append("POTENTIALLY UNREFERENCED FILES")
    report.append("=" * 80)
    report.append("(Files that are never imported - may be entry points or utilities)")

    unreferenced = find_unreferenced_files(analysis, directory)
    if unreferenced:
        for filepath in unreferenced:
            report.append(f"  {filepath}")
    else:
        report.append("\nNo unreferenced files found!")

    # Summary statistics
    report.append("\n\n" + "=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)

    total_unused_imports = sum(len(imports) for imports in unused_imports.values())
    total_unused_defs = sum(len(defs) for defs in unused_defs.values())

    report.append(f"\nTotal unused imports: {total_unused_imports}")
    report.append(f"Total potentially unused definitions: {total_unused_defs}")
    report.append(f"Total potentially unreferenced files: {len(unreferenced)}")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def main():
    """Main entry point."""
    # Analyze current directory
    current_dir = Path(__file__).parent

    print("Analyzing Python files in pyqt directory...")
    print(f"Directory: {current_dir}\n")

    analysis = analyze_directory(current_dir)

    # Generate report
    report = generate_report(analysis, current_dir)

    # Print to console
    print(report)

    # Save to file
    report_file = current_dir / 'DEAD_CODE_REPORT.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n\nReport saved to: {report_file}")


if __name__ == '__main__':
    main()
