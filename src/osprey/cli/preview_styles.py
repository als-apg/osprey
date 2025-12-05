#!/usr/bin/env python3
"""Interactive theme preview and testing tool for Osprey CLI.

This tool helps you visualize and test different color themes before applying them.
You can preview the default theme, create custom themes, and see real-world examples.

Usage:
    python src/osprey/cli/preview_styles.py              # Preview default theme
    python src/osprey/cli/preview_styles.py --compare    # Compare multiple themes
    python src/osprey/cli/preview_styles.py --custom     # Create custom theme interactively
"""

import sys
from pathlib import Path

# Ensure we can import from the local src directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.panel import Panel  # isort: skip
from rich.table import Table  # isort: skip
from rich.text import Text  # isort: skip

from osprey.cli.styles import (  # isort: skip
    OSPREY_THEME,
    ColorTheme,
    Messages,
    Styles,
    console,
    set_theme,
)


# ============================================================================
# PREDEFINED THEMES FOR COMPARISON
# ============================================================================

THEMES = {
    "osprey": ColorTheme(
        primary="#9370DB",  # Purple
        accent="#00cccc",  # Teal
        command="#ff9500",  # Orange
        path="#999999",  # Gray
        info="#00aaff",  # Cyan
    ),
    "ocean": ColorTheme(
        primary="#0077be",  # Ocean blue
        accent="#00ccaa",  # Sea green
        command="#ff8800",  # Warm orange
        path="#888888",  # Gray
        info="#66ccff",  # Light blue
    ),
    "sunset": ColorTheme(
        primary="#ff6b6b",  # Coral red
        accent="#feca57",  # Golden yellow
        command="#ff9500",  # Orange
        path="#999999",  # Gray
        info="#48dbfb",  # Sky blue
    ),
    "forest": ColorTheme(
        primary="#26de81",  # Forest green
        accent="#20bf6b",  # Dark green
        command="#ff9500",  # Orange
        path="#999999",  # Gray
        info="#4bcffa",  # Cyan
    ),
    "monochrome": ColorTheme(
        primary="#ffffff",  # White
        accent="#bbbbbb",  # Light gray
        command="#888888",  # Gray
        path="#666666",  # Dark gray
        info="#aaaaaa",  # Medium gray
    ),
}


# ============================================================================
# PREVIEW SECTIONS
# ============================================================================


def show_theme_header(theme_name: str, theme: ColorTheme):
    """Show theme name and core colors."""
    console.print()
    console.print("=" * 80, style=Styles.PRIMARY)
    console.print(f"THEME: {theme_name.upper()}", style="bold", justify="center")
    console.print("=" * 80, style=Styles.PRIMARY)
    console.print()


def show_color_swatches(theme: ColorTheme):
    """Show visual swatches of all theme colors."""
    console.print(
        Panel.fit(
            "[bold]COLOR PALETTE[/bold]\n\n"
            "[dim]The 5 core configurable colors that define your theme:[/dim]",
            border_style=Styles.BORDER,
        )
    )
    console.print()

    # Core configurable colors
    colors_table = Table(show_header=True, box=None, padding=(0, 2))
    colors_table.add_column("Color", style="bold", width=20)
    colors_table.add_column("Hex", width=10)
    colors_table.add_column("Swatch", width=10)
    colors_table.add_column("Usage", style=Styles.DIM)

    core_colors = [
        ("PRIMARY", theme.primary, "Brand identity, headers, selections"),
        ("ACCENT", theme.accent, "Interactive elements, highlights"),
        ("COMMAND", theme.command, "Shell commands, actions"),
        ("PATH", theme.path, "File paths, locations"),
        ("INFO", theme.info, "Informational messages (ℹ️)"),
    ]

    for name, hex_val, usage in core_colors:
        swatch = Text("████", style=hex_val)
        colors_table.add_row(name, hex_val, swatch, usage)

    console.print(colors_table)
    console.print()

    # Fixed standard colors
    console.print("[bold]Fixed Standard Colors[/bold] [dim](consistent across themes):[/dim]")
    fixed_table = Table(show_header=False, box=None, padding=(0, 2))
    fixed_table.add_column(width=20)
    fixed_table.add_column(width=10)
    fixed_table.add_column(width=10)
    fixed_table.add_column(style=Styles.DIM)

    fixed_colors = [
        ("SUCCESS", theme.success, "Success messages, confirmations"),
        ("ERROR", theme.error, "Error messages, failures"),
        ("WARNING", theme.warning, "Warnings, cautions"),
    ]

    for name, hex_val, usage in fixed_colors:
        swatch = Text("████", style=hex_val)
        fixed_table.add_row(name, hex_val, swatch, usage)

    console.print(fixed_table)
    console.print()

    # Derived colors (automatically calculated)
    console.print("[bold]Derived Colors[/bold] [dim](auto-calculated from primary):[/dim]")
    derived_table = Table(show_header=False, box=None, padding=(0, 2))
    derived_table.add_column(width=20)
    derived_table.add_column(width=10)
    derived_table.add_column(width=10)
    derived_table.add_column(style=Styles.DIM)

    derived = [
        ("PRIMARY_DARK", theme.primary_dark, "85% brightness"),
        ("PRIMARY_LIGHT", theme.primary_light, "115% brightness"),
    ]

    for name, hex_val, description in derived:
        swatch = Text("████", style=hex_val)
        derived_table.add_row(name, hex_val, swatch, description)

    console.print(derived_table)
    console.print()


def show_status_messages():
    """Show status message examples."""
    console.print(
        Panel.fit(
            "[bold]STATUS MESSAGES[/bold]\n\n"
            "[dim]Standard indicators for success, errors, warnings, and info:[/dim]",
            border_style=Styles.BORDER,
        )
    )
    console.print()

    console.print(Messages.success("Project created successfully"))
    console.print(Messages.error("Failed to connect to database"))
    console.print(Messages.warning("API key not configured - using default"))
    console.print(Messages.info("Starting service initialization..."))
    console.print()


def show_inline_styles():
    """Show inline text styling."""
    console.print(
        Panel.fit(
            "[bold]INLINE TEXT STYLES[/bold]\n\n"
            "[dim]How colors appear in regular text flow:[/dim]",
            border_style=Styles.BORDER,
        )
    )
    console.print()

    console.print(f"[{Styles.PRIMARY}]Primary brand text for emphasis[/{Styles.PRIMARY}]")
    console.print(f"[{Styles.HEADER}]Section Header[/{Styles.HEADER}]")
    console.print(f"[{Styles.SUBHEADER}]Subsection Header[/{Styles.SUBHEADER}]")
    console.print(f"[{Styles.ACCENT}]Accent color for highlights[/{Styles.ACCENT}]")
    console.print(f"Command: [{Styles.COMMAND}]osprey deploy up[/{Styles.COMMAND}]")
    console.print(f"Path: [{Styles.PATH}]/home/user/project[/{Styles.PATH}]")
    console.print(f"[{Styles.DIM}]Secondary dimmed text[/{Styles.DIM}]")
    console.print()


def show_real_world_example():
    """Show a realistic CLI interaction."""
    console.print(
        Panel.fit(
            "[bold]REAL-WORLD EXAMPLE[/bold]\n\n"
            "[dim]How the theme looks in actual CLI usage:[/dim]",
            border_style=Styles.BORDER,
        )
    )
    console.print()

    # Project creation flow
    console.print(f"[{Styles.HEADER}]Creating New Osprey Project[/{Styles.HEADER}]\n")

    console.print(Messages.label_value("Project Name", "my-agent"))
    console.print(Messages.label_value("Template", "chat_basic"))
    console.print(Messages.label_value("Provider", "anthropic"))
    console.print()

    console.print(f"[{Styles.PRIMARY}]Initializing...[/{Styles.PRIMARY}]\n")

    console.print(Messages.success("Created project structure"))
    console.print(Messages.success("Installed dependencies"))
    console.print(Messages.success("Generated configuration"))
    console.print(Messages.warning("Remember to set your API key"))
    console.print()

    console.print(f"[bold {Styles.SUCCESS}]✓ Project ready![/bold {Styles.SUCCESS}]\n")

    console.print(f"[{Styles.SUBHEADER}]Next Steps:[/{Styles.SUBHEADER}]")
    console.print(f"  1. Navigate: [{Styles.COMMAND}]cd my-agent[/{Styles.COMMAND}]")
    console.print(f"  2. Configure: [{Styles.COMMAND}]cp .env.example .env[/{Styles.COMMAND}]")
    console.print(f"  3. Start: [{Styles.COMMAND}]osprey chat[/{Styles.COMMAND}]")
    console.print()


def show_panel_examples():
    """Show panels with different border styles."""
    console.print(
        Panel.fit(
            "[bold]PANELS & BORDERS[/bold]\n\n" "[dim]How panels and borders appear:[/dim]",
            border_style=Styles.BORDER,
        )
    )
    console.print()

    # Status panel
    console.print(
        Panel(
            f"[{Styles.SUCCESS}]✓[/{Styles.SUCCESS}] All services running\n"
            f"[{Styles.WARNING}]⚠[/{Styles.WARNING}] High memory usage detected\n"
            f"[{Styles.INFO}]ℹ[/{Styles.INFO}] Database: postgres:5432\n"
            f"[{Styles.ERROR}]✗[/{Styles.ERROR}] Cache connection failed",
            title=f"[{Styles.HEADER}]Service Status[/{Styles.HEADER}]",
            border_style=Styles.BORDER,
        )
    )
    console.print()


def show_table_example():
    """Show a table with styled content."""
    console.print(
        Panel.fit(
            "[bold]TABLES[/bold]\n\n" "[dim]Configuration and data display:[/dim]",
            border_style=Styles.BORDER,
        )
    )
    console.print()

    table = Table(
        title="Configuration",
        border_style=Styles.BORDER,
        title_style=Styles.HEADER,
    )

    table.add_column("Setting", style=Styles.LABEL, no_wrap=True)
    table.add_column("Value", style=Styles.VALUE)
    table.add_column("Status", justify="center")

    table.add_row("Provider", "anthropic", f"[{Styles.SUCCESS}]✓[/{Styles.SUCCESS}]")
    table.add_row("Model", "claude-3-5-sonnet", f"[{Styles.SUCCESS}]✓[/{Styles.SUCCESS}]")
    table.add_row("API Key", "sk-***", f"[{Styles.WARNING}]⚠[/{Styles.WARNING}]")
    table.add_row("Database", "postgres", f"[{Styles.ERROR}]✗[/{Styles.ERROR}]")

    console.print(table)
    console.print()


def show_banner():
    """Show the Osprey ASCII banner."""
    console.print()
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║    ░█████╗░░██████╗██████╗░██████╗░███████╗██╗░░░██╗      ║
    ║    ██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝╚██╗░██╔╝      ║
    ║    ██║░░██║╚█████╗░██████╔╝██████╔╝█████╗░░░╚████╔╝░      ║
    ║    ██║░░██║░╚═══██╗██╔═══╝░██╔══██╗██╔══╝░░░░╚██╔╝░░      ║
    ║    ╚█████╔╝██████╔╝██║░░░░░██║░░██║███████╗░░░██║░░░      ║
    ║    ░╚════╝░╚═════╝░╚═╝░░░░░╚═╝░░╚═╝╚══════╝░░░╚═╝░░░      ║
    ║                                                           ║
    ║      Command Line Interface for the Osprey Framework      ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(Text(banner, style=Styles.PRIMARY))
    console.print("    [bold]Interactive Menu System[/bold]", style=Styles.ACCENT)
    console.print("    [dim]Use arrow keys to navigate • Press Ctrl+C to exit[/dim]")
    console.print()


# ============================================================================
# MAIN PREVIEW FUNCTIONS
# ============================================================================


def preview_theme(theme_name: str, theme: ColorTheme, show_banner_art: bool = True):
    """Preview a complete theme with all UI elements.

    Args:
        theme_name: Name of the theme
        theme: ColorTheme object to preview
        show_banner_art: Whether to show the ASCII banner
    """
    # Activate the theme
    set_theme(theme)

    # Clear and show header
    console.clear()
    show_theme_header(theme_name, theme)

    # Show banner if requested
    if show_banner_art:
        show_banner()

    # Show all UI elements
    show_color_swatches(theme)
    show_status_messages()
    show_inline_styles()
    show_real_world_example()
    show_panel_examples()
    show_table_example()

    # Footer
    console.print()
    console.print("=" * 80, style=Styles.PRIMARY)
    console.print(f"[{Styles.SUCCESS}]✓[/{Styles.SUCCESS}] Theme preview complete")
    console.print("=" * 80, style=Styles.PRIMARY)
    console.print()


def compare_themes():
    """Show a side-by-side comparison of available themes."""
    console.clear()
    console.print()
    console.print("=" * 80, style=Styles.PRIMARY)
    console.print("THEME COMPARISON", style="bold", justify="center")
    console.print("=" * 80, style=Styles.PRIMARY)
    console.print()

    console.print("[bold]Available Themes:[/bold]\n")

    for i, (name, theme) in enumerate(THEMES.items(), 1):
        # Set theme temporarily to show colors
        set_theme(theme)

        console.print(f"{i}. [{Styles.PRIMARY}]{name.upper()}[/{Styles.PRIMARY}]")
        console.print("   Primary: ", end="")
        console.print(Text("████", style=theme.primary), end="")
        console.print(f" {theme.primary}  ", end="")
        console.print("Accent: ", end="")
        console.print(Text("████", style=theme.accent), end="")
        console.print(f" {theme.accent}")
        console.print()

    # Reset to default
    set_theme(OSPREY_THEME)

    console.print(f"\n[{Styles.INFO}]ℹ[/{Styles.INFO}] To preview a specific theme:")
    console.print(
        f"  [{Styles.COMMAND}]python src/osprey/cli/preview_styles.py --theme <name>[/{Styles.COMMAND}]\n"
    )


def create_custom_theme():
    """Interactively create a custom theme."""
    try:
        import questionary

        from osprey.cli.styles import get_questionary_style
    except ImportError:
        console.print(f"[{Styles.ERROR}]✗[/{Styles.ERROR}] questionary not installed")
        console.print(
            f"Install with: [{Styles.COMMAND}]pip install questionary[/{Styles.COMMAND}]\n"
        )
        return

    console.clear()
    console.print()
    console.print("=" * 80, style=Styles.PRIMARY)
    console.print("CREATE CUSTOM THEME", style="bold", justify="center")
    console.print("=" * 80, style=Styles.PRIMARY)
    console.print()

    style = get_questionary_style()

    console.print("[bold]Enter hex colors for the 5 core theme colors:[/bold]")
    console.print("[dim]Format: #RRGGBB (e.g., #9370DB)[/dim]\n")

    try:
        primary = questionary.text("Primary (brand color):", default="#9370DB", style=style).ask()

        accent = questionary.text("Accent (highlights):", default="#00cccc", style=style).ask()

        command = questionary.text(
            "Command (shell commands):", default="#ff9500", style=style
        ).ask()

        path = questionary.text("Path (file paths):", default="#999999", style=style).ask()

        info = questionary.text("Info (informational):", default="#00aaff", style=style).ask()

        if not all([primary, accent, command, path, info]):
            console.print(f"\n[{Styles.WARNING}]⚠[/{Styles.WARNING}] Theme creation cancelled\n")
            return

        # Create and preview the custom theme
        custom = ColorTheme(
            primary=primary,
            accent=accent,
            command=command,
            path=path,
            info=info,
        )

        preview_theme("custom", custom, show_banner_art=False)

    except KeyboardInterrupt:
        console.print(f"\n[{Styles.WARNING}]⚠[/{Styles.WARNING}] Theme creation cancelled\n")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preview and test Osprey CLI themes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/osprey/cli/preview_styles.py                  # Preview default theme
  python src/osprey/cli/preview_styles.py --theme ocean    # Preview ocean theme
  python src/osprey/cli/preview_styles.py --compare        # Compare all themes
  python src/osprey/cli/preview_styles.py --custom         # Create custom theme
        """,
    )

    parser.add_argument(
        "--theme",
        "-t",
        choices=list(THEMES.keys()),
        default="osprey",
        help="Theme to preview (default: osprey)",
    )
    parser.add_argument("--compare", "-c", action="store_true", help="Compare all available themes")
    parser.add_argument(
        "--custom", action="store_true", help="Create and preview a custom theme interactively"
    )
    parser.add_argument("--no-banner", action="store_true", help="Skip the ASCII banner in preview")

    args = parser.parse_args()

    try:
        if args.custom:
            create_custom_theme()
        elif args.compare:
            compare_themes()
        else:
            theme = THEMES[args.theme]
            preview_theme(args.theme, theme, show_banner_art=not args.no_banner)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Exiting...[/yellow]\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
