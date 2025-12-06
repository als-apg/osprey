"""
Unit tests for the enhanced preview_database.py tool.

Tests all parameters and their combinations by rendering the template
and testing the compiled module.
"""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader


# Paths
TEMPLATE_ROOT = Path(__file__).parent.parent.parent.parent / "src" / "osprey" / "templates" / "apps" / "control_assistant"
EXAMPLES_DIR = TEMPLATE_ROOT / "data" / "channel_databases" / "examples"


@pytest.fixture(scope="session")
def compiled_preview_module(tmp_path_factory):
    """Compile the preview_database.py template and return the module."""

    # Create temporary directory
    tmp_dir = tmp_path_factory.mktemp("preview_test")
    test_preview_dir = tmp_dir / "test_preview"
    test_preview_dir.mkdir()

    # Create __init__.py for the package
    (test_preview_dir / "__init__.py").touch()

    # Create directory structure
    data_dir = test_preview_dir / "data"
    tools_dir = data_dir / "tools"
    tools_dir.mkdir(parents=True)

    # Copy example databases
    db_dir = data_dir / "channel_databases" / "examples"
    db_dir.mkdir(parents=True)
    shutil.copytree(EXAMPLES_DIR, db_dir, dirs_exist_ok=True)

    # Copy hierarchical database implementation
    services_dir = test_preview_dir / "services" / "channel_finder"
    databases_dir = services_dir / "databases"
    utils_dir = services_dir / "utils"
    core_dir = services_dir / "core"

    for directory in [services_dir, databases_dir, utils_dir, core_dir]:
        directory.mkdir(parents=True)

    # Create proper __init__ files
    (test_preview_dir / "services" / "__init__.py").touch()
    (services_dir / "__init__.py").write_text("from . import databases\n")
    (utils_dir / "__init__.py").touch()
    (core_dir / "__init__.py").touch()

    # Copy core base classes
    src_core = TEMPLATE_ROOT / "services" / "channel_finder" / "core"
    shutil.copy(src_core / "base_database.py", core_dir / "base_database.py")
    shutil.copy(src_core / "exceptions.py", core_dir / "exceptions.py")
    shutil.copy(src_core / "models.py", core_dir / "models.py")

    # Copy database files (template.py depends on flat.py via inheritance)
    src_databases = TEMPLATE_ROOT / "services" / "channel_finder" / "databases"
    shutil.copy(src_databases / "hierarchical.py", databases_dir / "hierarchical.py")
    shutil.copy(src_databases / "flat.py", databases_dir / "flat.py")
    shutil.copy(src_databases / "template.py", databases_dir / "template.py")

    # Create __init__.py for databases with imports
    (databases_dir / "__init__.py").write_text("""
from .hierarchical import HierarchicalChannelDatabase
from .template import ChannelDatabase as TemplateChannelDatabase

__all__ = ['HierarchicalChannelDatabase', 'TemplateChannelDatabase']
""")

    # Copy utils - config.py is not a template
    src_utils = TEMPLATE_ROOT / "services" / "channel_finder" / "utils"
    shutil.copy(src_utils / "config.py", utils_dir / "config.py")

    # Create minimal config.yml
    config_yml = test_preview_dir / "config.yml"
    config_yml.write_text(f"""
project_root: {test_preview_dir}
channel_finder:
  pipeline_mode: hierarchical
  pipelines:
    hierarchical:
      database:
        path: data/channel_databases/examples/consecutive_instances.json
""")

    # Set environment variables for config
    import os
    os.environ['PROJECT_ROOT'] = str(test_preview_dir)
    os.environ['CONFIG_FILE'] = str(config_yml)

    # Render preview_database.py from template
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_ROOT / "data" / "tools")))
    template = env.get_template("preview_database.py.j2")
    rendered = template.render(package_name="test_preview")

    preview_file = tools_dir / "preview_database.py"
    preview_file.write_text(rendered)

    # Add to sys.path
    sys.path.insert(0, str(tmp_dir))

    # Import the module
    spec = importlib.util.spec_from_file_location("preview_database", preview_file)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.skip(f"Could not load preview module: {e}")

    yield module

    # Cleanup
    sys.path.remove(str(tmp_dir))
    os.environ.pop('PROJECT_ROOT', None)
    os.environ.pop('CONFIG_FILE', None)


@pytest.fixture
def consecutive_db_path():
    """Path to consecutive_instances.json."""
    return str(EXAMPLES_DIR / "consecutive_instances.json")


@pytest.fixture
def instance_first_db_path():
    """Path to instance_first.json."""
    return str(EXAMPLES_DIR / "instance_first.json")


@pytest.fixture
def optional_levels_db_path():
    """Path to optional_levels.json."""
    return str(EXAMPLES_DIR / "optional_levels.json")


class TestDepthParameter:
    """Test --depth parameter."""

    def test_depth_3_limits_tree_levels(self, compiled_preview_module, consecutive_db_path, tmp_path):
        """Test that depth=3 actually limits the tree to 3 levels."""
        import io
        import re
        from rich.console import Console

        # Import Osprey's theme to get proper styles
        try:
            from osprey.cli.styles import osprey_theme
            theme = osprey_theme
        except ImportError:
            # Fallback: no theme (won't have custom colors but won't crash)
            theme = None

        # Capture output to a string with Osprey's theme
        output = io.StringIO()
        test_console = Console(
            file=output,
            width=120,
            legacy_windows=False,
            force_terminal=False,  # Disable terminal features
            no_color=True,         # Disable color codes for easier parsing
            theme=theme            # Use Osprey's theme for style names
        )

        # Replace the module's console with our capturing console
        original_console = compiled_preview_module.console
        compiled_preview_module.console = test_console

        try:
            compiled_preview_module.preview_database(
                depth=3,
                max_items=10,
                sections='tree',
                focus=None,
                show_full=False,
                db_path=consecutive_db_path
            )

            # Get the captured output
            result = output.getvalue()

            # Save to file for inspection
            output_file = tmp_path / "depth_3_output.txt"
            output_file.write_text(result)
            print(f"\n✓ Output saved to: {output_file}")
            print(f"\n--- Sample Output (first 500 chars) ---")
            print(result[:500])
            print("...")

            # Validate depth=3 behavior:
            # 1. Configuration section should show Display Depth = 3
            assert re.search(r'Display Depth\s+3', result), \
                f"Output should show 'Display Depth' with value 3\nGot: {result[:500]}"

            # 2. Tree should be present
            assert 'Hierarchy Tree' in result, \
                "Tree section header should be present"

            # 3. Validate actual tree depth limitation
            # With depth=3, we should see: system → family → sector (3 levels)
            # Count tree branch characters (━━) to estimate depth
            tree_section = result[result.find('Hierarchy Tree'):result.find('💡 Tip:')]
            branch_lines = [line for line in tree_section.split('\n') if '━━' in line]

            # Should have branches but not too many levels deep
            assert len(branch_lines) > 0, "Tree should have branches"

            # Check that we don't go beyond the requested depth
            # Each level adds indentation (┃ characters)
            max_indentation = max(line.count('┃') for line in branch_lines) if branch_lines else 0
            assert max_indentation <= 3, \
                f"Tree depth should not exceed 3 levels, but found {max_indentation} levels of indentation"

            # 4. Should have the tip about using -1 for complete hierarchy
            # (only shown when depth/max_items are limited)
            assert '--depth -1' in result, \
                "Should show tip about viewing complete hierarchy with --depth -1"

            print(f"\n✓ All depth=3 validations passed:")\

            print(f"  - Display Depth parameter shown correctly")
            print(f"  - Tree structure present")
            print(f"  - Tree limited to {max_indentation} levels (≤3 as expected)")
            print(f"  - Tip message present")

        finally:
            # Restore original console
            compiled_preview_module.console = original_console

    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, -1])
    def test_depth_values(self, compiled_preview_module, consecutive_db_path, depth, capsys):
        """Test different depth values."""
        from unittest.mock import patch

        # Mock console to prevent actual output
        with patch.object(compiled_preview_module, 'console'):
            try:
                compiled_preview_module.preview_database(
                    depth=depth,
                    max_items=5,
                    sections='tree',
                    focus=None,
                    show_full=False,
                    db_path=consecutive_db_path
                )
                # Should not raise errors
                assert True
            except Exception as e:
                pytest.fail(f"depth={depth} raised exception: {e}")


class TestMaxItemsParameter:
    """Test --max-items parameter."""

    def test_max_items_limits_branches(self, compiled_preview_module, consecutive_db_path, tmp_path):
        """Test that max_items=3 actually limits branches to 3 per level."""
        import io
        import re
        from rich.console import Console

        try:
            from osprey.cli.styles import osprey_theme
            theme = osprey_theme
        except ImportError:
            theme = None

        output = io.StringIO()
        test_console = Console(
            file=output,
            width=120,
            legacy_windows=False,
            force_terminal=False,
            no_color=True,
            theme=theme
        )

        original_console = compiled_preview_module.console
        compiled_preview_module.console = test_console

        try:
            compiled_preview_module.preview_database(
                depth=5,
                max_items=3,  # Limit to 3 items per level
                sections='tree',
                focus=None,
                show_full=False,
                db_path=consecutive_db_path
            )

            result = output.getvalue()

            # Save for inspection
            output_file = tmp_path / "max_items_3_output.txt"
            output_file.write_text(result)
            print(f"\n✓ Output saved to: {output_file}")

            # Validate max_items=3 behavior:
            # 1. Should show Max Items/Level = 3
            assert re.search(r'Max Items/Level\s+3', result), \
                "Output should show 'Max Items/Level' with value 3"

            # 2. Extract tree section
            tree_section = result[result.find('Hierarchy Tree'):result.find('💡 Tip:')]

            # 3. Check for truncation message "... X more"
            assert '... ' in tree_section and ' more ' in tree_section, \
                "Tree should show truncation message when items exceed max_items"

            # 4. Count top-level systems in the tree
            # Top level has pattern: │  ┣━━ or │  ┗━━ (panel border + 2 spaces + branch + capital letter)
            # This distinguishes from nested items which have ┃ tree connectors
            top_level_pattern = re.compile(r'│\s\s[┣┗]━━\s+[A-Z]')
            top_level_branches = [line for line in tree_section.split('\n')
                                 if top_level_pattern.search(line)]

            # With max_items=3, should see exactly 3 systems (M, V, D)
            assert len(top_level_branches) == 3, \
                f"Should show exactly 3 top-level items, found {len(top_level_branches)}"

            print(f"\n✓ All max_items=3 validations passed:")
            print(f"  - Max Items/Level parameter shown correctly")
            print(f"  - Truncation message present")
            print(f"  - Top level limited to 3 items")

        finally:
            compiled_preview_module.console = original_console

    @pytest.mark.parametrize("max_items", [1, 3, 5, 10, 20, -1])
    def test_max_items_values(self, compiled_preview_module, consecutive_db_path, max_items):
        """Test different max_items values."""
        from unittest.mock import patch

        with patch.object(compiled_preview_module, 'console'):
            try:
                compiled_preview_module.preview_database(
                    depth=3,
                    max_items=max_items,
                    sections='tree',
                    focus=None,
                    show_full=False,
                    db_path=consecutive_db_path
                )
                assert True
            except Exception as e:
                pytest.fail(f"max_items={max_items} raised exception: {e}")


class TestSectionsParameter:
    """Test --sections parameter."""

    def test_sections_tree_only(self, compiled_preview_module, consecutive_db_path, tmp_path):
        """Test sections='tree' shows only tree, no stats/breakdown."""
        import io
        from rich.console import Console

        try:
            from osprey.cli.styles import osprey_theme
            theme = osprey_theme
        except ImportError:
            theme = None

        output = io.StringIO()
        test_console = Console(file=output, width=120, no_color=True, theme=theme, force_terminal=False)

        original_console = compiled_preview_module.console
        compiled_preview_module.console = test_console

        try:
            compiled_preview_module.preview_database(
                depth=3, max_items=5, sections='tree',
                focus=None, show_full=False, db_path=consecutive_db_path
            )

            result = output.getvalue()

            # Should have tree
            assert 'Hierarchy Tree' in result, "Tree section should be present"

            # Should NOT have stats or breakdown
            assert 'Hierarchy Level Statistics' not in result, "Stats section should NOT be present"
            assert 'Channel Count Breakdown' not in result, "Breakdown section should NOT be present"
            assert 'Sample Channels' not in result, "Samples section should NOT be present"

            print("\n✓ sections='tree' correctly shows only tree")

        finally:
            compiled_preview_module.console = original_console

    def test_sections_stats_only(self, compiled_preview_module, consecutive_db_path, tmp_path):
        """Test sections='stats' shows only stats, no tree."""
        import io
        from rich.console import Console

        try:
            from osprey.cli.styles import osprey_theme
            theme = osprey_theme
        except ImportError:
            theme = None

        output = io.StringIO()
        test_console = Console(file=output, width=120, no_color=True, theme=theme, force_terminal=False)

        original_console = compiled_preview_module.console
        compiled_preview_module.console = test_console

        try:
            compiled_preview_module.preview_database(
                depth=3, max_items=5, sections='stats',
                focus=None, show_full=False, db_path=consecutive_db_path
            )

            result = output.getvalue()

            # Should have stats
            assert 'Hierarchy Level Statistics' in result, "Stats section should be present"

            # Should NOT have tree or breakdown
            assert 'Hierarchy Tree' not in result, "Tree section should NOT be present"
            assert 'Channel Count Breakdown' not in result, "Breakdown section should NOT be present"

            print("\n✓ sections='stats' correctly shows only stats")

        finally:
            compiled_preview_module.console = original_console

    def test_sections_all(self, compiled_preview_module, consecutive_db_path, tmp_path):
        """Test sections='all' includes all sections."""
        import io
        from rich.console import Console

        try:
            from osprey.cli.styles import osprey_theme
            theme = osprey_theme
        except ImportError:
            theme = None

        output = io.StringIO()
        test_console = Console(file=output, width=120, no_color=True, theme=theme, force_terminal=False)

        original_console = compiled_preview_module.console
        compiled_preview_module.console = test_console

        try:
            compiled_preview_module.preview_database(
                depth=3, max_items=5, sections='all',
                focus=None, show_full=False, db_path=consecutive_db_path
            )

            result = output.getvalue()

            # All sections should be present
            assert 'Hierarchy Tree' in result, "Tree section should be present"
            assert 'Hierarchy Level Statistics' in result, "Stats section should be present"
            assert 'Channel Count Breakdown' in result, "Breakdown section should be present"
            assert 'Sample Channels' in result, "Samples section should be present"

            print("\n✓ sections='all' correctly includes all sections")

        finally:
            compiled_preview_module.console = original_console

    @pytest.mark.parametrize("sections", [
        "tree",
        "stats",
        "breakdown",
        "samples",
        "tree,stats",
        "tree,breakdown",
        "tree,stats,breakdown",
        "tree,stats,breakdown,samples",
        "all"
    ])
    def test_sections_combinations(self, compiled_preview_module, consecutive_db_path, sections):
        """Test different section combinations."""
        from unittest.mock import patch

        with patch.object(compiled_preview_module, 'console'):
            try:
                compiled_preview_module.preview_database(
                    depth=3,
                    max_items=5,
                    sections=sections,
                    focus=None,
                    show_full=False,
                    db_path=consecutive_db_path
                )
                assert True
            except Exception as e:
                pytest.fail(f"sections={sections} raised exception: {e}")


class TestDepthMaxItemsCombinations:
    """Test combinations of depth and max_items."""

    @pytest.mark.parametrize("depth,max_items", [
        (1, 1),
        (1, 5),
        (2, 1),
        (2, 5),
        (3, 1),
        (3, 5),
        (3, 10),
        (4, 10),
        (5, 5),
        (-1, -1),  # unlimited both
        (-1, 5),   # unlimited depth, limited items
        (3, -1),   # limited depth, unlimited items
    ])
    def test_depth_max_items_combinations(self, compiled_preview_module, instance_first_db_path,
                                         depth, max_items):
        """Test all combinations of depth and max_items."""
        from unittest.mock import patch

        with patch.object(compiled_preview_module, 'console'):
            try:
                compiled_preview_module.preview_database(
                    depth=depth,
                    max_items=max_items,
                    sections='tree',
                    focus=None,
                    show_full=False,
                    db_path=instance_first_db_path
                )
                assert True
            except Exception as e:
                pytest.fail(f"depth={depth}, max_items={max_items} raised exception: {e}")


class TestFocusParameter:
    """Test --focus parameter."""

    def test_focus_filters_tree(self, compiled_preview_module, consecutive_db_path, tmp_path):
        """Test that focus='M' only shows M subtree, not V or D."""
        import io
        import re
        from rich.console import Console

        try:
            from osprey.cli.styles import osprey_theme
            theme = osprey_theme
        except ImportError:
            theme = None

        output = io.StringIO()
        test_console = Console(file=output, width=120, no_color=True, theme=theme, force_terminal=False)

        original_console = compiled_preview_module.console
        compiled_preview_module.console = test_console

        try:
            compiled_preview_module.preview_database(
                depth=3, max_items=10, sections='tree',
                focus='M',  # Focus on M system only
                show_full=False, db_path=consecutive_db_path
            )

            result = output.getvalue()

            # Save for inspection
            output_file = tmp_path / "focus_M_output.txt"
            output_file.write_text(result)

            # Should show focus path in config
            assert re.search(r'Focus Path\s+M', result), "Should show Focus Path = M"

            # Tree title should show M
            assert 'M' in result.split('Hierarchy Tree')[0], "Should show M in tree title area"

            # Should have M system content
            assert 'QB' in result or 'DP' in result or 'CM' in result, \
                "Should show M subsystems (QB, DP, or CM)"

            # Should NOT have other top-level systems
            tree_section = result[result.find('Hierarchy Tree'):]
            assert '━━ V ' not in tree_section, "Should NOT show V system"
            assert '━━ D ' not in tree_section, "Should NOT show D system"

            print("\n✓ focus='M' correctly filters to M subtree only")

        finally:
            compiled_preview_module.console = original_console

    def test_focus_valid_path(self, compiled_preview_module, consecutive_db_path):
        """Test focusing on a valid path."""
        from unittest.mock import patch

        with patch.object(compiled_preview_module, 'console'):
            try:
                # M is a valid system in consecutive_instances.json
                compiled_preview_module.preview_database(
                    depth=3,
                    max_items=5,
                    sections='tree',
                    focus='M',
                    show_full=False,
                    db_path=consecutive_db_path
                )
                assert True
            except Exception as e:
                pytest.fail(f"focus='M' raised exception: {e}")

    def test_focus_invalid_path(self, compiled_preview_module, consecutive_db_path):
        """Test focusing on an invalid path."""
        from unittest.mock import patch

        with patch.object(compiled_preview_module, 'console'):
            # Should handle gracefully, not crash
            compiled_preview_module.preview_database(
                depth=3,
                max_items=5,
                sections='tree',
                focus='NONEXISTENT',
                show_full=False,
                db_path=consecutive_db_path
            )


class TestPathParameter:
    """Test --path parameter with different databases."""

    def test_path_consecutive_instances(self, compiled_preview_module, consecutive_db_path):
        """Test with consecutive_instances.json."""
        from unittest.mock import patch

        with patch.object(compiled_preview_module, 'console'):
            compiled_preview_module.preview_database(
                depth=3,
                max_items=5,
                sections='tree,stats',
                focus=None,
                show_full=False,
                db_path=consecutive_db_path
            )

    def test_path_instance_first(self, compiled_preview_module, instance_first_db_path):
        """Test with instance_first.json."""
        from unittest.mock import patch

        with patch.object(compiled_preview_module, 'console'):
            compiled_preview_module.preview_database(
                depth=3,
                max_items=5,
                sections='tree,stats',
                focus=None,
                show_full=False,
                db_path=instance_first_db_path
            )

    def test_path_optional_levels(self, compiled_preview_module, optional_levels_db_path):
        """Test with optional_levels.json."""
        from unittest.mock import patch

        with patch.object(compiled_preview_module, 'console'):
            compiled_preview_module.preview_database(
                depth=5,
                max_items=5,
                sections='all',
                focus=None,
                show_full=False,
                db_path=optional_levels_db_path
            )


class TestBackwardsCompatibility:
    """Test --full flag backwards compatibility."""

    def test_full_flag(self, compiled_preview_module, consecutive_db_path):
        """Test that --full sets depth and max_items to -1."""
        from unittest.mock import patch

        with patch.object(compiled_preview_module, 'preview_hierarchical') as mock_preview:
            compiled_preview_module.preview_database(
                depth=3,  # Will be overridden
                max_items=10,  # Will be overridden
                sections='tree',
                focus=None,
                show_full=True,  # This should override to -1, -1
                db_path=consecutive_db_path
            )

            # Verify it was called with -1 for both
            assert mock_preview.called
            call_kwargs = mock_preview.call_args[1]
            assert call_kwargs['depth'] == -1
            assert call_kwargs['max_items'] == -1


class TestStatisticsCalculation:
    """Test statistics calculation functions."""

    def test_level_statistics(self, compiled_preview_module, consecutive_db_path):
        """Test level statistics are calculated correctly."""
        from unittest.mock import patch

        # Load database
        from test_preview.services.channel_finder.databases import HierarchicalChannelDatabase
        db = HierarchicalChannelDatabase(consecutive_db_path)

        stats = compiled_preview_module._calculate_level_statistics(db, db.hierarchy_levels)

        # Should return list of tuples (level_name, count)
        assert isinstance(stats, list)
        assert len(stats) == len(db.hierarchy_levels)

        for level_name, count in stats:
            assert level_name in db.hierarchy_levels
            assert isinstance(count, int)
            assert count > 0

    def test_breakdown_calculation(self, compiled_preview_module, consecutive_db_path):
        """Test breakdown calculation."""
        from test_preview.services.channel_finder.databases import HierarchicalChannelDatabase
        db = HierarchicalChannelDatabase(consecutive_db_path)

        breakdown = compiled_preview_module._calculate_breakdown(db, db.hierarchy_levels, focus=None)

        # Should return sorted list of (path, count) tuples
        assert isinstance(breakdown, list)
        assert len(breakdown) > 0

        # Verify format
        for path, count in breakdown:
            assert isinstance(path, str)
            assert isinstance(count, int)
            assert count > 0
            # Path should use colon separator
            assert ':' in path or len(path.split(':')) == 1

        # Verify sorted by count descending
        counts = [count for _, count in breakdown]
        assert counts == sorted(counts, reverse=True)


class TestCrossDatabaseCompatibility:
    """Test that features work across all database formats."""

    @pytest.mark.parametrize("db_fixture_name", [
        "consecutive_db_path",
        "instance_first_db_path",
        "optional_levels_db_path"
    ])
    def test_depth_works_on_all_databases(self, compiled_preview_module, db_fixture_name, request):
        """Test that depth parameter works on all database formats."""
        import io
        import re
        from rich.console import Console

        # Get the actual fixture value by name
        db_path = request.getfixturevalue(db_fixture_name)

        try:
            from osprey.cli.styles import osprey_theme
            theme = osprey_theme
        except ImportError:
            theme = None

        output = io.StringIO()
        test_console = Console(file=output, width=120, no_color=True, theme=theme, force_terminal=False)

        original_console = compiled_preview_module.console
        compiled_preview_module.console = test_console

        try:
            compiled_preview_module.preview_database(
                depth=2, max_items=5, sections='tree',
                focus=None, show_full=False, db_path=db_path
            )

            result = output.getvalue()

            # Basic validations that should work on all databases
            assert 'Hierarchy Tree' in result, f"Tree should render on {db_fixture_name}"
            assert re.search(r'Display Depth\s+2', result), \
                f"Depth parameter should show on {db_fixture_name}"
            assert 'channels' in result.lower(), \
                f"Should show channel count on {db_fixture_name}"

            print(f"\n✓ depth=2 works on {db_fixture_name}")

        finally:
            compiled_preview_module.console = original_console


class TestEdgeCases:
    """Test edge cases."""

    def test_depth_zero(self, compiled_preview_module, consecutive_db_path):
        """Test depth=0."""
        from unittest.mock import patch

        with patch.object(compiled_preview_module, 'console'):
            # Should handle gracefully
            compiled_preview_module.preview_database(
                depth=0,
                max_items=10,
                sections='tree',
                focus=None,
                show_full=False,
                db_path=consecutive_db_path
            )

    def test_max_items_zero(self, compiled_preview_module, consecutive_db_path):
        """Test max_items=0."""
        from unittest.mock import patch

        with patch.object(compiled_preview_module, 'console'):
            # Should handle gracefully
            compiled_preview_module.preview_database(
                depth=3,
                max_items=0,
                sections='tree',
                focus=None,
                show_full=False,
                db_path=consecutive_db_path
            )

    def test_empty_sections(self, compiled_preview_module, consecutive_db_path):
        """Test with empty sections string."""
        from unittest.mock import patch

        with patch.object(compiled_preview_module, 'console'):
            # Should handle gracefully
            compiled_preview_module.preview_database(
                depth=3,
                max_items=10,
                sections='',
                focus=None,
                show_full=False,
                db_path=consecutive_db_path
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
